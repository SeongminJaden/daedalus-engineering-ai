"""Cross-validation of this project's meshing against Gmsh.

The CalculiX comparison generates its deck from this project's own mesh, so a
meshing error is invisible to it: both solvers would agree while computing the
wrong geometry. These tests close that gap by meshing independently and by
measuring geometry with an independent CAD kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.registry import Category, ProblemContext
from nodes import gmsh_node as gm
from nodes.descriptor import CapabilityUnavailable
from nodes.roster import build_roster
from physics.fem.mesh import hollow_rect_mesh, l_bracket_mesh, solid_box_mesh
from physics.fem.solver import solve_linear_elasticity

requires_gmsh = pytest.mark.skipif(
    not gm.is_available(), reason="gmsh is not installed")

E_PA, NU = 71.7e9, 0.33
BOX = (0.20, 0.04, 0.03)


# ------------------------------------------------------------------ the node

def test_availability_is_read_from_the_import_system():
    descriptor = gm.gmsh_descriptor()
    assert descriptor.available is gm.is_available()
    if not descriptor.available:
        assert "unavailable" in descriptor.unavailable_reason


def test_a_missing_mesher_raises_rather_than_returning_a_mesh(monkeypatch):
    monkeypatch.setattr(gm, "_gmsh", lambda: None)
    with pytest.raises(CapabilityUnavailable):
        gm.structured_box_mesh(*BOX, 4, 2, 1)


def test_the_capability_is_registered():
    registry = build_roster()
    assert gm.GMSH_CAPABILITY in registry
    context = ProblemContext(geometry="prismatic_beam",
                             representations=("prismatic_beam",))
    if gm.is_available():
        assert gm.GMSH_CAPABILITY in registry.query(
            context, Category.ANALYSIS).names()


# --------------------------------------------- independent mesh generation

@requires_gmsh
@pytest.mark.parametrize("nx,ny,nz", [(8, 3, 2), (16, 6, 4)])
def test_gmsh_builds_the_same_box_mesh_this_project_does(nx, ny, nz):
    theirs = gm.structured_box_mesh(*BOX, nx, ny, nz)
    ours = solid_box_mesh(*BOX, nx, ny, nz)
    assert theirs.n_elements == ours.n_elements
    assert theirs.n_nodes == ours.n_nodes
    assert theirs.element_volume == pytest.approx(ours.element_volume, rel=1e-12)


@requires_gmsh
@pytest.mark.parametrize("nx,ny,nz", [(8, 3, 2), (16, 6, 4)])
def test_the_solver_gives_the_same_answer_on_an_independently_built_mesh(
        nx, ny, nz):
    """This is the check CalculiX structurally cannot make.

    Node numbering differs between the two meshes, so agreeing to round-off
    means the CONNECTIVITY describes the same solid, not that the arrays match.
    """
    length = BOX[0]
    results = []
    for mesh in (solid_box_mesh(*BOX, nx, ny, nz),
                 gm.structured_box_mesh(*BOX, nx, ny, nz)):
        solution = solve_linear_elasticity(
            mesh, E_PA, NU, mesh.nodes_at_x(0.0), mesh.nodes_at_x(length),
            total_load_n=-100.0, load_direction=1)
        results.append((solution.tip_deflection(),
                        float(np.abs(solution.element_von_mises).max())))
    (ours_d, ours_s), (theirs_d, theirs_s) = results
    assert theirs_d == pytest.approx(ours_d, rel=1e-11)
    assert theirs_s == pytest.approx(ours_s, rel=1e-11)


@requires_gmsh
def test_every_element_is_an_axis_aligned_box_of_the_expected_size():
    """Guards against an inverted or twisted element, which would still solve.

    A permuted node ordering produces a plausible wrong stiffness rather than
    an error, so the conversion identifies nodes by position and this checks
    the result.
    """
    mesh = gm.structured_box_mesh(*BOX, 8, 3, 2)
    cell = np.array([mesh.dx, mesh.dy, mesh.dz])
    for element in mesh.connectivity:
        points = mesh.node_coords[element]
        extent = points.max(axis=0) - points.min(axis=0)
        assert np.allclose(extent, cell, rtol=1e-12)
        # Node 0 is the minimum corner and node 6 the diagonal opposite.
        assert np.allclose(points[0], points.min(axis=0), atol=1e-15)
        assert np.allclose(points[6], points.max(axis=0), atol=1e-15)


# ------------------------------------------------ the validity boundary

@requires_gmsh
def test_a_curved_shape_gives_tetrahedra_that_this_solver_cannot_consume():
    """The limitation is measured here, not merely asserted in a docstring.

    Gmsh meshes a sphere with tetrahedra. This project has no tetrahedral
    element, so such a mesh cannot be run at all, which is why this node's
    reach stops at boxes and extruded prisms.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("sphere")
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 0.01)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        types, _, _ = gmsh.model.mesh.getElements(3)
    finally:
        gmsh.finalize()
    assert gm._HEX8 not in list(types)
    assert 4 in list(types)          # 4 is Gmsh's 4-node tetrahedron


# -------------------------------------------- independent geometry measurement

@requires_gmsh
def test_the_hollow_section_mesh_loses_no_material():
    """The voxel mesh volume against an exact CAD volume, not against itself."""
    length, height, width, wall = 0.30, 0.040, 0.020, 0.002
    exact = gm.hollow_rectangle_volume(length, height, width, wall)
    assert exact == pytest.approx(
        length * (height * width - (height - 2 * wall) * (width - 2 * wall)),
        rel=1e-12)
    for through_wall in (1, 2, 3, 4):
        mesh = hollow_rect_mesh(length, height, width, wall, nx=10,
                                elements_through_wall=through_wall)
        check = gm.check_mesh_volume(mesh, exact)
        assert abs(check.relative_error) < 1e-12


@requires_gmsh
@pytest.mark.parametrize("n,fraction", [(20, 0.40), (15, 0.40), (10, 0.40)])
def test_the_l_bracket_is_exact_when_the_arm_lands_on_a_cell_boundary(
        n, fraction):
    size, width = 0.10, 0.01
    exact = gm.l_bracket_volume(size, size * fraction, width)
    mesh = l_bracket_mesh(size, fraction, width, n=n, nz=2)
    assert abs(gm.check_mesh_volume(mesh, exact).relative_error) < 1e-12


@requires_gmsh
@pytest.mark.parametrize("n,fraction,worst", [(10, 0.35, 0.05),
                                              (12, 0.30, 0.05),
                                              (10, 0.25, 0.05),
                                              (11, 0.30, 0.05)])
def test_the_l_bracket_meshes_a_different_bracket_when_the_arm_does_not(
        n, fraction, worst):
    """A real finding, and exactly the class of error CalculiX cannot see.

    `l_bracket_mesh` rounds the arm to a whole number of cells, so when
    n * thickness_fraction is not an integer the mesh represents a bracket of
    a different thickness. The volume error reaches 17.7% in the cases here.
    A deck generated from this mesh would make two solvers agree beautifully
    on the wrong geometry.
    """
    size, width = 0.10, 0.01
    requested = gm.l_bracket_volume(size, size * fraction, width)
    with pytest.raises(ValueError, match="allow_snapping"):
        l_bracket_mesh(size, fraction, width, n=n, nz=2)
    mesh = l_bracket_mesh(size, fraction, width, n=n, nz=2,
                          allow_snapping=True)
    error = gm.check_mesh_volume(mesh, requested).relative_error
    assert abs(error) > worst, "this case was supposed to be quantised"

    # And the discrepancy is fully explained by the rounded arm thickness.
    realised = gm.realised_l_bracket_thickness(size, fraction, n)
    assert realised != pytest.approx(size * fraction, rel=1e-9)
    explained = gm.l_bracket_volume(size, realised, width)
    assert abs(gm.check_mesh_volume(mesh, explained).relative_error) < 1e-12


def test_the_realised_thickness_helper_matches_what_the_mesh_builds():
    """The helper must reproduce the mesh's rounding, not a tidier version."""
    size, width = 0.10, 0.01
    for n, fraction in ((10, 0.35), (11, 0.30), (10, 0.25), (20, 0.40)):
        mesh = l_bracket_mesh(size, fraction, width, n=n, nz=2,
                              allow_snapping=True)
        realised = gm.realised_l_bracket_thickness(size, fraction, n)
        centroids = mesh.element_centroids()
        # The vertical arm's material stops at the realised thickness.
        tallest = centroids[:, 1].max()
        in_arm = centroids[centroids[:, 1] > tallest - 1e-12]
        assert in_arm[:, 0].max() < realised


# --------------------------------------------------------------- the limit

def test_meshing_agreement_is_not_a_measurement_of_a_real_part():
    method = gm.gmsh_capability_method()
    assert method.evidence == "SIMULATED"
    assert "not a measurement of a real part" in method.notes
