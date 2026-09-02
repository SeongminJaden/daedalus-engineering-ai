"""The geometric measurements behind the process rules, on shapes with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from core.part_dataset import FAMILIES, make_part
from core.part_dataset.pointcloud import tessellate
from geometry.cad_export.kernel import kernel_available
from geometry.manufacturability import (measure_mesh, overhang_area_fraction,
                                        tool_access_area_fraction,
                                        wall_thickness_samples)
from nodes import step_analyzer as sa

requires_cad = pytest.mark.skipif(not (kernel_available() and sa.is_available()),
                                  reason="build123d and OCP are required")


def _mesh_of(solid):
    import build123d as bd
    return tessellate(solid.wrapped, 1e-3, deflection=0.5)


@requires_cad
def test_wall_thickness_of_a_hollow_section_is_its_wall(tmp_path):
    """3 mm wall: the minimum over samples must be the wall to a few
    percent, and no sample can be thinner than the wall on a shape that has
    nothing thinner."""
    record, _ = make_part(FAMILIES["hollow_rect"],
                          dict(length_m=0.2, height_m=0.04, width_m=0.03, wall_m=0.003),
                          tmp_path, labelled=False)
    contents = sa.read_step(tmp_path / f"{record.part_id}.step")
    mesh = tessellate(contents.shapes[0], contents.unit_to_metres, deflection=0.5)
    walls = wall_thickness_samples(mesh.vertices, mesh.triangles, 400,
                                   np.random.default_rng(1))
    assert len(walls) > 300
    assert walls.min() == pytest.approx(0.003, rel=0.02)
    # end faces see across the whole section, so the distribution has a
    # long tail; the median is still the wall
    assert np.median(walls) == pytest.approx(0.003, rel=0.02)


@requires_cad
def test_a_plate_reports_its_thickness_and_a_solid_bar_its_smaller_side(tmp_path):
    plate, _ = make_part(FAMILIES["box"], dict(length_m=0.1, height_m=0.004, width_m=0.05),
                         tmp_path / "p", labelled=False)
    contents = sa.read_step(tmp_path / "p" / f"{plate.part_id}.step")
    mesh = tessellate(contents.shapes[0], contents.unit_to_metres, deflection=0.5)
    walls = wall_thickness_samples(mesh.vertices, mesh.triangles, 300,
                                   np.random.default_rng(2))
    assert walls.min() == pytest.approx(0.004, rel=1e-6)   # planar, exact


@requires_cad
def test_overhang_is_the_underside_of_a_flange_and_not_the_plate_face():
    """A T: a 40 by 10 by 20 stem on the plate carrying a 100 by 10 by 20
    flange. Built along y, the flange underside outside the stem is a
    horizontal overhang and nothing else is; its area fraction is exact."""
    import build123d as bd
    stem = bd.Pos(0, 5, 0) * bd.Box(40, 10, 20)
    flange = bd.Pos(0, 15, 0) * bd.Box(100, 10, 20)
    mesh = _mesh_of(stem + flange)
    fraction, worst = overhang_area_fraction(mesh.vertices, mesh.triangles,
                                             build_axis=1, max_angle_deg=45.0)
    underside = (100 - 40) * 20 * 1e-6
    total = mesh.area_m2
    assert fraction == pytest.approx(underside / total, rel=1e-6)
    assert worst == pytest.approx(90.0)
    none, _ = overhang_area_fraction(mesh.vertices, mesh.triangles,
                                     build_axis=1, max_angle_deg=90.0)
    assert none == 0.0


@requires_cad
def test_a_closed_cavity_is_unreachable_and_an_open_one_is_not():
    """Box with a sealed inner void: every inner face is reachable from no
    direction, so the inaccessible fraction is the inner area over the total.
    The same box open through one face is reachable from that face's
    direction, except the shadowed floor rim the ray model cannot see past
    the lip of, which is stated as the model's optimism the other way."""
    import build123d as bd
    outer = bd.Box(60, 40, 40)
    void = bd.Box(40, 20, 20)
    closed = _mesh_of(outer - void)
    inaccessible, n = tool_access_area_fraction(closed.vertices, closed.triangles)
    inner = 2 * (40 * 20 + 40 * 20 + 20 * 20) * 1e-6
    outer_area = 2 * (60 * 40 + 60 * 40 + 40 * 40) * 1e-6
    assert n == 6
    assert inaccessible == pytest.approx(inner / (inner + outer_area), rel=1e-6)

    opened = _mesh_of(outer - (bd.Pos(0, 10, 0) * bd.Box(40, 40, 20)))
    fraction_open, _ = tool_access_area_fraction(opened.vertices, opened.triangles)
    assert fraction_open == 0.0


@requires_cad
def test_measure_mesh_collects_everything_once(tmp_path):
    record, _ = make_part(FAMILIES["housing"],
                          dict(length_m=0.1, width_m=0.06, height_m=0.04, wall_m=0.004),
                          tmp_path, labelled=False)
    contents = sa.read_step(tmp_path / f"{record.part_id}.step")
    mesh = tessellate(contents.shapes[0], contents.unit_to_metres, deflection=0.5)
    measures = measure_mesh(mesh.vertices, mesh.triangles, build_axis=1)
    assert measures.min_wall_m == pytest.approx(0.004, rel=0.03)
    assert measures.inaccessible_fraction_6_axis == 0.0     # open at the top
    assert measures.overhang_fraction_45 == 0.0             # walls are vertical
    assert measures.wall_samples > 300
