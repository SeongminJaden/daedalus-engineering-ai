"""The surrogate over the industrial dataset, checked on a corpus small enough
to build in a test.

The accuracy numbers that matter come from a real generation run and live in
docs/dataset_spec.md; a run is 74 minutes of solver time and is not in the
repository. What is checked here is everything that could silently make those
numbers a lie: the proxies are the closed forms they claim to be, a scaled
copy never lands on the other side of the split from the part it was scaled
from, the features carry the load case, and a prediction cannot become a
verdict.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from brain.semantic.evidence import EvidenceKind, EvidenceLevel, may_decide
from core.materials import get_material
from core.part_dataset.batch import Cell, expand_materials, read_cell, run_cell
from core.part_dataset.industrial_surrogate import (
    FEATURE_NAMES, TARGET_NAMES, CaseValues, IndustrialSet, axial_proxy_m,
    base_id_of, bending_proxy_m, cache_descriptors, draw_order, format_table,
    holdout_mask, load_run, metrics_by_group, thermal_proxy_m,
    torsion_proxy_rad, train_industrial_surrogate)
from core.part_dataset.labeller import LoadCase, LoadKind, labelling_available
from geometry.cad_export.kernel import kernel_available
from integration.checks import CheckStatus
from nodes import step_analyzer as sa

requires_all = pytest.mark.skipif(
    not (kernel_available() and sa.is_available() and labelling_available()),
    reason="build123d, OCP, gmsh and CalculiX are required")

SAMPLES = 5
ALUMINIUM = "al_7075_t6"


def synthetic_record(part_id: str, family: str, material_id: str,
                     length: float = 0.4, side: float = 0.02,
                     **labels) -> "PartRecord":
    """A record with no CAD behind it, for the checks that need only numbers."""
    from core.part_dataset.engine import SYNTHETIC_PROVENANCE
    from core.part_dataset.schema import (GeometrySummary, PartRecord,
                                          TopologySummary)
    return PartRecord(
        part_id=part_id, material_id=material_id,
        provenance=SYNTHETIC_PROVENANCE.model_copy(update={"generator": family}),
        geometry=GeometrySummary(volume_m3=length * side * side,
                                 surface_area_m2=4 * side * length + 2 * side * side,
                                 bounding_box_m=(length, side, side),
                                 centre_of_mass_m=(0.0, 0.0, 0.0)),
        topology=TopologySummary(solids=1, shells=1, faces=6, edges=12, vertices=8),
        labels=dict(labels))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Two families by two load cases, five parts each, plus scaled copies:
    the smallest thing shaped like a real run."""
    if not (kernel_available() and sa.is_available() and labelling_available()):
        pytest.skip("build123d, OCP, gmsh and CalculiX are required")
    root = tmp_path_factory.mktemp("run")
    cells = [Cell(f, c) for f in ("box", "hollow_rect")
             for c in (LoadCase(total_load_n=-100.0, kind=LoadKind.BENDING),
                       LoadCase(kind=LoadKind.TORSION, torque_nm=2.0))]
    reference = get_material(ALUMINIUM)
    targets = [get_material(m) for m in ("al_6061_t6", "ss_304")]
    (root / "scaled").mkdir()
    for cell in cells:
        run_cell(cell, SAMPLES, root, seed=11)
        records = read_cell(root, cell)
        scaled, _skipped = expand_materials(records, reference, targets)
        (root / "scaled" / f"{cell.name}.jsonl").write_text(
            "".join(json.dumps(r.model_dump(mode="json"), sort_keys=True) + "\n"
                    for r in scaled))
    return root, cells


@pytest.fixture(scope="module")
def corpus(run):
    root, _cells = run
    descriptors = cache_descriptors(root)
    return root, descriptors, load_run(root, descriptors)


# --- the proxies are the closed forms they say they are ----------------------

def test_the_proxies_are_the_textbook_formulas_on_the_bounding_box():
    """A solid square bar, so the fill ratio is one and the bounding box is
    the section: every proxy must reproduce the formula in its docstring."""
    length, side = 0.4, 0.02
    record = synthetic_record("bar", "box", ALUMINIUM, length, side)
    material = get_material(ALUMINIUM)
    e, g = material.youngs_modulus_pa, material.shear_modulus_pa
    second = side ** 4 / 12.0
    polar = side * side * (side ** 2 + side ** 2) / 12.0

    bending = CaseValues(LoadKind.BENDING, 1, -100.0, 0.0, 0.0)
    assert bending_proxy_m(record, material, bending) == pytest.approx(
        100.0 * length ** 3 / (3.0 * e * second), rel=1e-12)

    axial = CaseValues(LoadKind.AXIAL, 1, 1000.0, 0.0, 0.0)
    assert axial_proxy_m(record, material, axial) == pytest.approx(
        1000.0 * length / (e * side * side), rel=1e-12)

    torsion = CaseValues(LoadKind.TORSION, 1, 0.0, 2.0, 0.0)
    assert torsion_proxy_rad(record, material, torsion) == pytest.approx(
        2.0 * length / (g * polar), rel=1e-12)

    thermal = CaseValues(LoadKind.THERMAL_GRADIENT, 1, 0.0, 0.0, 1000.0)
    alpha = material.thermal_expansion_1_k
    assert thermal_proxy_m(record, material, thermal) == pytest.approx(
        alpha * 1000.0 * length ** 2 / 2.0, rel=1e-12)


def test_a_material_without_an_expansion_coefficient_gets_no_thermal_proxy():
    """PLA has no coefficient in its sheet, so the proxy is zero rather than
    a number invented to fill the column."""
    record = synthetic_record("bar", "box", "pla")
    pla = get_material("pla")
    assert pla.thermal_expansion_1_k is None
    assert thermal_proxy_m(record, pla, CaseValues(
        LoadKind.THERMAL_GRADIENT, 1, 0.0, 0.0, 1000.0)) == 0.0


# --- the split ---------------------------------------------------------------

@requires_all
def test_a_scaled_copy_stays_with_the_part_it_was_scaled_from(corpus):
    """The leak this split exists to prevent. A scaled copy differs from its
    source by an exact material factor, so a copy in the test set whose source
    was trained on would report an error the surrogate has not earned."""
    root, _descriptors, data = corpus
    order = draw_order(root)
    mask = holdout_mask(data, order, SAMPLES)
    train, test = data.subset(~mask), data.subset(mask)
    assert len(train) and len(test)
    assert not (set(train.base_ids) & set(test.base_ids))
    for part_id, base in zip(data.part_ids, data.base_ids):
        assert part_id == base or part_id.startswith(base + "-")


@requires_all
def test_every_family_and_load_case_reaches_the_held_out_set(corpus):
    root, _descriptors, data = corpus
    test = data.subset(holdout_mask(data, draw_order(root), SAMPLES))
    assert set(test.families) == set(data.families)
    assert set(test.kinds) == set(data.kinds)


@requires_all
def test_the_corpus_carries_every_material_and_the_right_targets(corpus):
    _root, _descriptors, data = corpus
    assert set(data.materials) == {ALUMINIUM, "al_6061_t6", "ss_304"}
    assert data.x.shape[1] == len(FEATURE_NAMES)
    assert data.y.shape[1] == len(TARGET_NAMES)
    assert np.all(data.y > 0.0)
    assert len(set(data.kinds)) == 2


@requires_all
def test_the_features_distinguish_the_load_cases(corpus):
    """Two rows for the same part under different loads must differ, or the
    model could not tell a twist from a deflection."""
    _root, _descriptors, data = corpus
    by_part: dict[str, list[int]] = {}
    for i, base in enumerate(data.base_ids):
        by_part.setdefault(base, []).append(i)
    kind_columns = [FEATURE_NAMES.index(f"kind_{k}") for k in ("bending", "torsion")]
    rows = [i for group in by_part.values() for i in group]
    assert {tuple(data.x[i, kind_columns]) for i in rows} == {(1.0, 0.0), (0.0, 1.0)}


def test_a_scaled_part_id_is_split_at_the_material():
    """A derived label marks a scaled copy; without one the id is its own base,
    so a solved part whose id happens to end in a material name is not cut."""
    from core.part_dataset.schema import label
    from brain.semantic.evidence import EvidenceKind as EK
    scaled = synthetic_record(
        "box-1234abcd-ss_304", "box", "ss_304",
        mass_kg=label(0.8, "kg", EK.ANALYTICAL, "scaled_from_al_7075_t6",
                      derived=True, scaled_from="al_7075_t6"))
    assert base_id_of(scaled) == "box-1234abcd"
    solved = synthetic_record("box-1234abcd", "box", ALUMINIUM,
                              mass_kg=label(0.8, "kg", EK.ANALYTICAL, "brep"))
    assert base_id_of(solved) == "box-1234abcd"


# --- training and reporting --------------------------------------------------

@requires_all
def test_training_improves_and_reports_held_out_error(corpus):
    root, _descriptors, data = corpus
    mask = holdout_mask(data, draw_order(root), SAMPLES)
    surrogate = train_industrial_surrogate(data.subset(~mask), data.subset(mask),
                                           epochs=300, batch=None)
    assert surrogate.training["final_loss"] < surrogate.training["first_loss"]
    for name in TARGET_NAMES:
        stats = surrogate.test_metrics[name]
        assert math.isfinite(stats["p95_rel_err"])
        assert math.isfinite(stats["spearman"])
    rows = metrics_by_group(surrogate, data.subset(mask), ("families",))
    assert {r["families"] for r in rows} == {"box", "hollow_rect"}
    table = format_table(rows, ("families",))
    assert table.startswith("| families | n | Spearman |")


@requires_all
def test_a_prediction_is_surrogate_and_cannot_become_a_verdict(corpus):
    _root, _descriptors, data = corpus
    surrogate = train_industrial_surrogate(data, data, epochs=50, batch=None)
    prediction = surrogate.predict(data.x[:3])
    assert prediction.evidence_kind is EvidenceKind.SURROGATE
    assert prediction.evidence_level is EvidenceLevel.SURROGATE
    assert not prediction.verified
    check = prediction.screened_check("part", "deflection", 0, 1e-3)
    assert check.status is CheckStatus.SCREENED
    assert check.evidence_kind is EvidenceKind.SURROGATE
    assert "not a verdict" in check.detail
    assert not may_decide(EvidenceLevel.SURROGATE)


@requires_all
def test_the_surrogate_round_trips(corpus, tmp_path):
    from core.part_dataset.industrial_surrogate import IndustrialSurrogate
    _root, _descriptors, data = corpus
    surrogate = train_industrial_surrogate(data, data, epochs=50, batch=None)
    surrogate.save(tmp_path / "model")
    back = IndustrialSurrogate.load(tmp_path / "model")
    assert np.allclose(surrogate.predict_array(data.x), back.predict_array(data.x))
    meta = json.loads((tmp_path / "model" / "meta.json").read_text())
    assert meta["evidence"] == EvidenceLevel.SURROGATE.value


@requires_all
def test_the_descriptor_cache_is_written_once_and_reused(corpus):
    root, descriptors, _data = corpus
    cache = json.loads((root / "descriptors.json").read_text())
    assert set(cache) == set(descriptors)
    again = cache_descriptors(root)
    assert again == descriptors
