"""Per-process manufacturing rules: cited, measured, and never evidence."""

from __future__ import annotations

import numpy as np
import pytest

from core.part_dataset import FAMILIES, make_part
from core.part_dataset.pointcloud import tessellate
from geometry.cad_export.kernel import kernel_available
from geometry.manufacturability import (PROCESSES, RULE_BASED, RULE_SOURCES,
                                        Process, assess, assess_all)
from geometry.manufacturability.processes import RULES
from nodes import step_analyzer as sa

requires_cad = pytest.mark.skipif(not (kernel_available() and sa.is_available()),
                                  reason="build123d and OCP are required")


def _part(name, params, root):
    record, _ = make_part(FAMILIES[name], params, root, labelled=False)
    contents = sa.read_step(root / f"{record.part_id}.step")
    mesh = tessellate(contents.shapes[0], contents.unit_to_metres, deflection=0.5)
    return record, mesh


def test_every_rule_names_a_source_that_was_read_and_quotes_it():
    for rule in RULES:
        assert rule.source in RULE_SOURCES, rule.id
        assert rule.as_printed.strip(), rule.id
        assert RULE_SOURCES[rule.source].retrieved == "2026-09-03"
        assert RULE_SOURCES[rule.source].url.startswith("https://")
    assert {r.process for r in RULES} == set(PROCESSES)


def test_the_grade_is_rule_based_and_not_an_evidence_level():
    from brain.semantic.evidence import EvidenceLevel
    assert RULE_BASED not in {level.value for level in EvidenceLevel}
    assert "rule_based" in RULE_BASED


@requires_cad
def test_a_thin_walled_tube_fails_the_metal_wall_rules_and_a_thick_one_passes(tmp_path):
    thin, mesh_thin = _part("hollow_rect", dict(length_m=0.1, height_m=0.02, width_m=0.02,
                                                wall_m=0.0006), tmp_path / "thin")
    thick, mesh_thick = _part("hollow_rect", dict(length_m=0.1, height_m=0.02, width_m=0.02,
                                                  wall_m=0.003), tmp_path / "thick")
    r_thin = assess(Process.CNC_MILLING, mesh_thin.vertices, mesh_thin.triangles, thin)
    r_thick = assess(Process.CNC_MILLING, mesh_thick.vertices, mesh_thick.triangles, thick)
    assert [f.rule.id for f in r_thin.failures] == ["cnc_wall_metal"]
    assert r_thick.passes
    assert r_thick.failures == []
    # the tube is open at both ends, so every inner face is reachable
    access = next(f for f in r_thick.findings if f.rule.id == "cnc_access")
    assert access.passes and access.measured == 0.0
    # the same 0.6 mm wall passes SLM's 0.4 mm floor
    slm = assess(Process.SLM, mesh_thin.vertices, mesh_thin.triangles, thin)
    assert next(f for f in slm.findings if f.rule.id == "slm_wall").passes


@requires_cad
def test_the_polymer_rule_replaces_the_metal_rule_for_a_polymer(tmp_path):
    record, mesh = _part("box", dict(length_m=0.05, height_m=0.001, width_m=0.02), tmp_path)
    metal = assess(Process.CNC_MILLING, mesh.vertices, mesh.triangles, record, is_polymer=False)
    polymer = assess(Process.CNC_MILLING, mesh.vertices, mesh.triangles, record, is_polymer=True)
    by_id = lambda r, i: next(f for f in r.findings if f.rule.id == i)
    assert by_id(metal, "cnc_wall_metal").passes            # 1.0 mm >= 0.8
    assert not by_id(metal, "cnc_wall_plastic").assessed
    assert by_id(polymer, "cnc_wall_plastic").passes is False   # 1.0 mm < 1.5
    assert not by_id(polymer, "cnc_wall_metal").assessed


@requires_cad
def test_unassessed_rules_are_listed_and_never_count_as_passes(tmp_path):
    record, mesh = _part("box", dict(length_m=0.1, height_m=0.02, width_m=0.02), tmp_path)
    turning = assess(Process.CNC_TURNING, mesh.vertices, mesh.triangles, record)
    ids = {f.rule.id for f in turning.unassessed}
    assert "turn_axisymmetric" in ids and "turn_hole" in ids     # no holes on a box
    assert "axisymmetric" in turning.not_measured
    sheet = assess(Process.SHEET_METAL, mesh.vertices, mesh.triangles, record)
    assert {f.rule.id for f in sheet.unassessed} >= {"sheet_bend_radius", "sheet_hole_to_bend"}
    assert "unassessed" in sheet.summary()


@requires_cad
def test_a_box_has_no_draft_and_fails_moulding_and_casting_draft_rules(tmp_path):
    """Four vertical walls are parallel to any axis pull: two thirds of the
    surface of a cube drags, and the rule is a fail, not a warning."""
    record, mesh = _part("box", dict(length_m=0.05, height_m=0.05, width_m=0.05), tmp_path)
    im = assess(Process.INJECTION_MOULDING, mesh.vertices, mesh.triangles, record, pull_axis=1)
    cast = assess(Process.DIE_CASTING, mesh.vertices, mesh.triangles, record, pull_axis=1)
    im_draft = next(f for f in im.findings if f.rule.id == "im_draft")
    cast_draft = next(f for f in cast.findings if f.rule.id == "cast_draft")
    assert im_draft.passes is False and im_draft.measured == pytest.approx(4 / 6, rel=1e-6)
    assert cast_draft.passes is False
    assert next(f for f in cast.findings if f.rule.id == "cast_wall_rec").passes  # 50 mm wall


@requires_cad
def test_a_flange_with_small_bolt_holes_fails_the_hole_rules_it_should(tmp_path):
    record, mesh = _part("flange", dict(outer_radius_m=0.05, thickness_m=0.01,
                                        bore_radius_m=0.012, bolt_radius_m=0.001,
                                        bolt_circle_radius_m=0.035, bolt_count=4.0),
                         tmp_path)
    reports = assess_all(mesh.vertices, mesh.triangles, record, build_axis=0, pull_axis=0)
    cnc = next(f for f in reports[Process.CNC_MILLING].findings if f.rule.id == "cnc_hole")
    sls = next(f for f in reports[Process.SLS].findings if f.rule.id == "sls_hole")
    assert cnc.measured == pytest.approx(0.002, rel=1e-6)
    assert cnc.passes is False and sls.passes is True          # 2.0 mm: below 2.5, above 1.5
    assert all(r.grade == RULE_BASED for r in reports.values())


@requires_cad
def test_overhang_rule_flags_a_flange_underside_for_fdm_and_slm():
    import build123d as bd
    stem = bd.Pos(0, 5, 0) * bd.Box(40, 10, 20)
    flange = bd.Pos(0, 15, 0) * bd.Box(100, 10, 20)
    mesh = tessellate((stem + flange).wrapped, 1e-3, deflection=0.5)
    fdm = assess(Process.FDM, mesh.vertices, mesh.triangles, None, build_axis=1)
    slm = assess(Process.SLM, mesh.vertices, mesh.triangles, None, build_axis=1)
    assert next(f for f in fdm.findings if f.rule.id == "fdm_overhang").passes is False
    assert next(f for f in slm.findings if f.rule.id == "slm_overhang").passes is False
    # laid on its side the flange has no horizontal underside off the plate
    fdm_side = assess(Process.FDM, mesh.vertices, mesh.triangles, None, build_axis=2)
    assert next(f for f in fdm_side.findings if f.rule.id == "fdm_overhang").passes
