"""Marching cubes and smoothing on a density field, and what they cost.

The blocky export is exact and unusable; this path is usable and inexact, and
the tests here are about the inexactness being known. Every function reports
the volume it produced against the volume of the field, and the checks below
are on fields whose volume is known in closed form, so the error is measured
against arithmetic rather than against another approximation.
"""

from __future__ import annotations

import numpy as np
import pytest

from optimization.topology.smooth import (density_grid, marching_surface,
                                          smoothing_table, taubin_smooth,
                                          write_stl)
from physics.fem.mesh import solid_box_mesh

pytestmark = pytest.mark.slow


def solid_field(mesh) -> np.ndarray:
    return np.ones(mesh.n_elements)


def ball_field(mesh, radius: float) -> np.ndarray:
    centre = np.array([mesh.nx * mesh.dx, mesh.ny * mesh.dy, mesh.nz * mesh.dz]) / 2
    distance = np.linalg.norm(mesh.element_centroids() - centre, axis=1)
    # A smooth indicator, so the iso surface is the sphere and not a staircase.
    return np.clip(1.5 - distance / radius, 0.0, 1.0)


def test_the_grid_puts_every_element_where_its_centroid_is():
    mesh = solid_box_mesh(0.4, 0.2, 0.1, 8, 4, 2)
    density = np.arange(mesh.n_elements, dtype=float)
    grid = density_grid(mesh, density)
    assert grid.shape == (8, 4, 2)
    assert sorted(grid.reshape(-1)) == sorted(density)
    cell = np.array([mesh.dx, mesh.dy, mesh.dz])
    index = np.round(mesh.element_centroids() / cell - 0.5).astype(int)
    for e in range(mesh.n_elements):
        assert grid[tuple(index[e])] == density[e]


def test_a_full_field_loses_a_boundary_layer_that_shrinks_with_the_grid():
    """The case with an exact answer, and it is not exact.

    Densities live at cell centres, so the iso surface of a completely solid
    field passes through the centres of the boundary cells and the body comes
    out half a cell small on every face. Measured on a 0.4 by 0.2 by 0.1 m box:
    9.9 percent low at 8 by 4 by 2, then 2.6, 0.67 and 0.17 percent as the grid
    doubles. It is first order in the cell size, and it is why the volume error
    is reported with every surface instead of being assumed away."""
    exact = 0.4 * 0.2 * 0.1
    errors = []
    for divisions in ((8, 4, 2), (16, 8, 4), (32, 16, 8)):
        mesh = solid_box_mesh(0.4, 0.2, 0.1, *divisions)
        surface = marching_surface(mesh, solid_field(mesh), iso_level=0.5,
                                   smoothing_iterations=0)
        assert surface.watertight and surface.n_components == 1
        assert surface.volume_m3 < exact
        errors.append(abs(surface.volume_error_vs_field))
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] == pytest.approx(0.099, abs=0.005)
    assert errors[2] == pytest.approx(0.0067, abs=0.002)


def test_a_sphere_is_recovered_to_a_few_percent():
    """A curved iso surface, where marching cubes is doing real work: the
    volume of the ball the field describes, to a few percent on this grid."""
    mesh = solid_box_mesh(0.2, 0.2, 0.2, 20, 20, 20)
    radius = 0.06
    surface = marching_surface(mesh, ball_field(mesh, radius), iso_level=0.5,
                               smoothing_iterations=0)
    exact = 4.0 / 3.0 * np.pi * radius ** 3
    assert surface.watertight and surface.n_components == 1
    assert surface.volume_m3 == pytest.approx(exact, rel=0.05)


def test_taubin_smoothing_does_not_deflate_the_body():
    """The reason it is Taubin and not Laplacian. Twenty passes of plain
    Laplacian smoothing shrink a sphere measurably; the two-pass scheme keeps
    the volume within a few percent, and the test compares the two."""
    import trimesh

    mesh = solid_box_mesh(0.2, 0.2, 0.2, 20, 20, 20)
    field = ball_field(mesh, 0.06)
    surface = marching_surface(mesh, field, 0.5, smoothing_iterations=0)
    v, f = surface.vertices, surface.triangles
    before = abs(trimesh.Trimesh(v, f, process=False).volume)

    taubin = taubin_smooth(v, f, iterations=20)
    laplacian = taubin_smooth(v, f, iterations=20, lamb=0.5, mu=0.0)
    after_taubin = abs(trimesh.Trimesh(taubin, f, process=False).volume)
    after_laplacian = abs(trimesh.Trimesh(laplacian, f, process=False).volume)

    shrink_taubin = (before - after_taubin) / before
    shrink_laplacian = (before - after_laplacian) / before
    print(f"\nshrink: taubin {shrink_taubin:.1%}, laplacian {shrink_laplacian:.1%}")
    assert shrink_laplacian > shrink_taubin
    assert abs(shrink_taubin) < 0.05


def test_the_iso_level_moves_the_volume_and_the_table_says_by_how_much():
    """The number a designer has to see: a marching cubes body is not the
    volume the optimiser constrained, and the direction depends on the iso
    level."""
    mesh = solid_box_mesh(0.2, 0.2, 0.2, 16, 16, 16)
    field = ball_field(mesh, 0.06)
    rows = smoothing_table(mesh, field, iso_levels=(0.3, 0.5, 0.7),
                           iterations=(0, 10))
    by_iso = {}
    for r in rows:
        by_iso.setdefault(r["iso_level"], []).append(r)
        assert r["watertight"]
    volumes = [by_iso[i][0]["volume_m3"] for i in (0.3, 0.5, 0.7)]
    assert volumes[0] > volumes[1] > volumes[2], volumes
    assert any(abs(r["error_vs_field"]) > 0.05 for r in rows), (
        "the surface matched the field volume everywhere, which would mean "
        "the iso level did nothing")


def test_the_stl_round_trips_and_can_be_volume_meshed(tmp_path):
    """A watertight surface is what makes the topology result analysable as a
    body. This is also where the second order meshing limit lives: on a
    marching cubes surface the mid-side nodes invert (68 elements with a
    nonpositive Jacobian, measured on the cantilever), so the route that
    solves is linear tetrahedra, which are stiff in bending."""
    from nodes import gmsh_node as gm
    if not gm.is_available():
        pytest.skip("gmsh is required")
    from optimization.topology.smooth import tet_mesh_from_stl

    mesh = solid_box_mesh(0.2, 0.2, 0.2, 16, 16, 16)
    surface = marching_surface(mesh, ball_field(mesh, 0.06), 0.5, 10)
    path = write_stl(surface, tmp_path / "ball.stl")
    assert path.exists() and path.stat().st_size > 0

    tets = tet_mesh_from_stl(path, target_size_m=0.012, order=1)
    assert tets.n_elements > 100
    assert tets.connectivity.shape[1] == 4
    span = tets.node_coords.max(axis=0) - tets.node_coords.min(axis=0)
    assert np.allclose(span, 2 * 0.06, rtol=0.15), span


def test_the_extracted_body_sits_where_the_field_does():
    """Position, not just volume, and it was wrong.

    A density belongs to its element's CENTRE, and marching cubes indexes the
    padded array from its corner. The extraction subtracted a whole cell to
    undo the pad, which put the body half an element low on all three axes:
    4.1 mm on the arm links, whose cells are 8.2 mm across. Every volume check
    passed throughout, because a translation does not change a volume. It was
    found when the parts were placed in an assembly and the joints did not
    meet.

    A completely solid field is the case with an exact answer here: its
    surface is the domain box, corner at the origin.
    """
    mesh = solid_box_mesh(0.4, 0.2, 0.1, 8, 4, 2)
    surface = marching_surface(mesh, solid_field(mesh), iso_level=0.5,
                               smoothing_iterations=0)
    low = np.asarray(surface.vertices).min(axis=0)
    high = np.asarray(surface.vertices).max(axis=0)
    assert low == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert high == pytest.approx([0.4, 0.2, 0.1], abs=1e-12)


def test_a_ball_comes_out_centred_on_the_ball():
    """The same defect on a curved surface, where it is not a special case."""
    mesh = solid_box_mesh(0.2, 0.2, 0.2, 20, 20, 20)
    surface = marching_surface(mesh, ball_field(mesh, 0.06), iso_level=0.5,
                               smoothing_iterations=0)
    vertices = np.asarray(surface.vertices)
    centre = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    assert centre == pytest.approx([0.1, 0.1, 0.1], abs=1e-3)


def test_the_extracted_triangles_point_outward():
    """A body whose faces point inward is not a solid.

    Marching cubes orients its triangles by the direction the field
    decreases, and for this field that came out inside out. Every volume in
    this module is read through abs(), so nothing here noticed. What noticed
    was a mesh boolean, which refused the body as "not a volume" and could not
    cut a bolt hole in it.
    """
    import trimesh

    mesh = solid_box_mesh(0.2, 0.2, 0.2, 12, 12, 12)
    surface = marching_surface(mesh, ball_field(mesh, 0.06), 0.5, 10)
    body = trimesh.Trimesh(vertices=surface.vertices, faces=surface.triangles,
                           process=False)
    assert body.is_watertight
    assert body.volume > 0.0, "the signed volume is negative: faces point in"
    assert body.is_winding_consistent
