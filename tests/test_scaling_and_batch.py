"""Material scaling of solver labels, and the resumable batch that uses it.

Scaling is exact in E and density and approximate across Poisson's ratio; the
residual was measured per load case and the bounds are pinned above the
measurements. The batch is checked for the property that matters: a second
run labels only what the first did not.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import pytest

from core.materials import get_material, load_materials
from core.part_dataset import FAMILIES, make_part
from core.part_dataset.batch import (Cell, cells_for, default_cases,
                                     expand_materials, plan, read_cell,
                                     run_batch, run_cell)
from core.part_dataset.labeller import LoadCase, LoadKind, labelling_available
from core.part_dataset.scaling import (POISSON_RESIDUAL_BOUND, UnscalableLabel,
                                       scale_factor, scale_record)
from geometry.cad_export.kernel import kernel_available
from nodes import step_analyzer as sa

pytestmark = pytest.mark.slow
requires_all = pytest.mark.skipif(
    not (kernel_available() and sa.is_available() and labelling_available()),
    reason="needs build123d, OCP, gmsh and CalculiX")


def test_scale_factors_are_the_stated_ratios():
    al, steel = get_material("al_7075_t6"), get_material("ss_304")
    assert scale_factor("density", al, steel) == pytest.approx(
        steel.density_kg_m3 / al.density_kg_m3)
    assert scale_factor("inverse_modulus", al, steel) == pytest.approx(
        al.youngs_modulus_pa / steel.youngs_modulus_pa)
    g = lambda m: m.youngs_modulus_pa / (2 * (1 + m.poisson_ratio))
    assert scale_factor("inverse_shear_modulus", al, steel) == pytest.approx(
        g(al) / g(steel))
    assert scale_factor("none", al, steel) == 1.0
    with pytest.raises(UnscalableLabel):
        scale_factor("magic", al, steel)


def test_every_load_case_has_a_measured_residual_and_the_thermal_stress_is_worst():
    assert set(POISSON_RESIDUAL_BOUND) == set(LoadKind)
    disp, stress = POISSON_RESIDUAL_BOUND[LoadKind.THERMAL_GRADIENT]
    assert stress > 0.05            # measured 5.85 percent
    for kind, (d, s) in POISSON_RESIDUAL_BOUND.items():
        assert 0 < d <= 0.01 and 0 < s <= 0.07, kind


@requires_all
def test_a_scaled_record_matches_a_direct_solve_within_the_bound(tmp_path):
    """Measured: displacements to 0.35 percent and peak stress to 1.05 percent
    across alumina, PEEK and 304 for bending; mass exact."""
    ref, target = get_material("al_7075_t6"), get_material("peek")
    params = dict(length_m=0.2, height_m=0.04, width_m=0.03, wall_m=0.004)
    record, _ = make_part(FAMILIES["hollow_rect"], params, tmp_path, ref, LoadCase())
    direct, _ = make_part(FAMILIES["hollow_rect"], params, tmp_path, target,
                          LoadCase())
    scaled = scale_record(record, ref, target)
    assert scaled.material_id == "peek"
    assert scaled.part_id.endswith("-peek")
    assert scaled.labels["mass_kg"]["value"] == pytest.approx(
        direct.labels["mass_kg"]["value"], rel=1e-9)
    disp_bound, stress_bound = POISSON_RESIDUAL_BOUND[LoadKind.BENDING]
    assert scaled.labels["tip_deflection_m"]["value"] == pytest.approx(
        direct.labels["tip_deflection_m"]["value"], rel=disp_bound)
    assert scaled.labels["max_von_mises_pa"]["value"] == pytest.approx(
        direct.labels["max_von_mises_pa"]["value"], rel=stress_bound)
    item = scaled.labels["tip_deflection_m"]
    assert item["evidence"] == "simulated" and item["derived"] is True
    assert item["scaled_from"] == "al_7075_t6"
    assert item["poisson_residual_bound"] == disp_bound
    assert scaled.labels["max_von_mises_pa"]["poisson_residual_bound"] == stress_bound


@requires_all
def test_scaling_refuses_what_it_cannot_do(tmp_path):
    ref = get_material("al_7075_t6")
    record, _ = make_part(FAMILIES["box"],
                          dict(length_m=0.2, height_m=0.02, width_m=0.02),
                          tmp_path, ref, LoadCase())
    with pytest.raises(UnscalableLabel, match="isotropy"):
        scale_record(record, ref, get_material("cfrp_ud"))
    with pytest.raises(UnscalableLabel, match="not the reference"):
        scale_record(record, get_material("ss_304"), get_material("peek"))
    thermal, _ = make_part(FAMILIES["box"],
                           dict(length_m=0.2, height_m=0.02, width_m=0.02),
                           tmp_path, ref, LoadCase(kind=LoadKind.THERMAL_GRADIENT))
    from core.materials import MissingMaterialValue
    with pytest.raises(MissingMaterialValue, match="thermal expansion"):
        scale_record(thermal, ref, get_material("pla"))      # PLA has no alpha


def test_the_plan_is_arithmetic_on_the_measured_cost():
    cells = cells_for(["box", "hollow_rect"], default_cases())
    assert len(cells) == 10
    bill = plan(cells, samples_per_cell=100, workers=8)
    assert bill["solves"] == 1000
    assert bill["hours_one_worker"] == pytest.approx(1000 * 3.2 / 3600)
    assert bill["hours_at_workers"] == pytest.approx(bill["hours_one_worker"] / 8)
    assert Cell("box", default_cases()[0]).name == "box__bending"


@requires_all
def test_a_second_run_labels_only_what_the_first_did_not(tmp_path):
    cell = Cell("box", LoadCase(total_load_n=-100.0, kind=LoadKind.BENDING))
    first = run_cell(cell, samples=2, root=tmp_path, seed=3)
    assert first.labelled == 2 and first.refused == 0
    second = run_cell(cell, samples=4, root=tmp_path, seed=3)
    assert second.labelled == 2                     # two new, two skipped
    records = read_cell(tmp_path, cell)
    assert len(records) == 4
    assert len({r.part_id for r in records}) == 4
    done = (tmp_path / cell.name / "done").read_text().split()
    assert len(done) == 4
    third = run_cell(cell, samples=4, root=tmp_path, seed=3)
    assert third.labelled == 0                      # nothing left to do
    assert all((tmp_path / cell.name / "step" / f"{r.part_id}.step").exists()
               for r in records)


@requires_all
def test_expansion_writes_one_scaled_record_per_isotropic_material(tmp_path):
    cell = Cell("box", LoadCase(total_load_n=-100.0, kind=LoadKind.BENDING))
    run_cell(cell, samples=1, root=tmp_path, seed=5)
    records = read_cell(tmp_path, cell)
    ref = get_material("al_7075_t6")
    scaled, skipped = expand_materials(records, ref)
    isotropic = [m for m in load_materials().materials
                 if m.material_class.value == "isotropic" and m.id != ref.id]
    assert len(scaled) == len(isotropic)
    assert skipped == []
    assert {r.material_id for r in scaled} == {m.id for m in isotropic}
    assert all(r.labels["tip_deflection_m"]["derived"] for r in scaled)


@requires_all
def test_a_batch_bounded_in_time_stops_and_reports_what_it_did(tmp_path):
    cells = cells_for(["box"], default_cases()[:2])
    report = run_batch(cells, samples_per_cell=1, root=tmp_path, seed=1,
                       stop_after_seconds=0.0)
    assert report.labelled == 0                      # no time, no work, no lie
    report = run_batch(cells, samples_per_cell=1, root=tmp_path, seed=1)
    assert report.labelled == 2 and report.refused == 0
    assert "box__axial" in report.summary()


def test_a_cell_seed_is_the_same_in_another_process():
    """The first version used hash(name), which Python salts per process, so
    every spawned worker and every resume drew different parts and the spec
    could not reproduce the set. crc32 is stable; the value is pinned."""
    import subprocess
    import sys
    cell = Cell("box", LoadCase(total_load_n=-100.0, kind=LoadKind.BENDING))
    here = cell.seed(3)
    code = ("from core.part_dataset.batch import Cell; "
            "from core.part_dataset.labeller import LoadCase, LoadKind; "
            "print(Cell('box', LoadCase(total_load_n=-100.0, "
            "kind=LoadKind.BENDING)).seed(3))")
    root = str(Path(__file__).resolve().parents[1])
    other = int(subprocess.run([sys.executable, "-c", code], check=True,
                               capture_output=True, text=True, cwd=root,
                               env={**os.environ, "PYTHONPATH": root}).stdout)
    assert here == other
    assert here == (3 * 1000003 + zlib.crc32(b"box__bending") % 1000003) % 2 ** 32
