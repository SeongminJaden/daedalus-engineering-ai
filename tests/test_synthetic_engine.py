"""The synthetic data engine, checked against the ground truth it was built from.

Synthetic parts are worth having because their truth is known. So every test
here compares what the engine reports against arithmetic or against a
parameter, never against another run of the engine. The one solver
comparison is the box, because a box is the one shape both meshers cover and
therefore the only place the labelling route has a reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from brain.semantic.evidence import EvidenceKind, EvidenceLevel
from core.part_dataset import (FAMILIES, LABEL_CEILING, LoadCase, PartRecord,
                               ProvenanceKind, generate_dataset, label,
                               labelling_available, make_part, part_id_for,
                               read_jsonl, sample_parameters, write_jsonl)
from core.part_dataset.engine import GroundTruthMismatch
from core.part_dataset.store import NotPublishable
from geometry.cad_export.kernel import kernel_available
from nodes import step_analyzer as sa

requires_cad = pytest.mark.skipif(not (kernel_available() and sa.is_available()),
                                  reason="build123d and OCP are required")
requires_solvers = pytest.mark.skipif(not labelling_available(),
                                      reason="needs both gmsh and CalculiX")


# ------------------------------------------------------------ the sampler

@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_sampled_parameters_are_always_admissible(name):
    fam = FAMILIES[name]
    rng = np.random.default_rng(3)
    for _ in range(200):
        params = sample_parameters(fam, rng)
        assert fam.admissible(params)
        for key, (low, high) in fam.bounds.items():
            assert low <= params[key] <= high
        for key in fam.integer_parameters:
            assert params[key] == int(params[key])


def test_sampling_is_deterministic_for_a_seed():
    fam = FAMILIES["plate_with_holes"]
    first = [sample_parameters(fam, np.random.default_rng(11)) for _ in range(3)]
    second = [sample_parameters(fam, np.random.default_rng(11)) for _ in range(3)]
    assert first == second
    other = sample_parameters(fam, np.random.default_rng(12))
    assert other != first[0]


def test_the_part_id_is_a_function_of_the_parameters_alone():
    fam = FAMILIES["box"]
    a = {"length_m": 0.1, "height_m": 0.02, "width_m": 0.03}
    b = {"length_m": 0.1 + 1e-12, "height_m": 0.02, "width_m": 0.03}
    c = {"length_m": 0.1, "height_m": 0.02, "width_m": 0.031}
    assert part_id_for(fam, a) == part_id_for(fam, b)     # same part
    assert part_id_for(fam, a) != part_id_for(fam, c)
    assert part_id_for(fam, a).startswith("box-")


# ------------------------------- the analyzer agrees with the closed form

@requires_cad
@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_the_analyzer_volume_matches_the_closed_form(name, tmp_path):
    """The check every generated part passes before it is stored."""
    fam = FAMILIES[name]
    rng = np.random.default_rng(5)
    for _ in range(3):
        params = sample_parameters(fam, rng)
        record, _ = make_part(fam, params, tmp_path, labelled=False)
        assert record.geometry.volume_m3 == pytest.approx(fam.volume_m3(params),
                                                          rel=1e-6)
        assert record.provenance.kind is ProvenanceKind.SYNTHETIC_PARAMETRIC
        assert record.provenance.generator == name
        assert record.is_publishable


@requires_cad
def test_the_plate_features_match_the_parameters_that_made_them(tmp_path):
    fam = FAMILIES["plate_with_holes"]
    params = dict(length_m=0.12, width_m=0.08, thickness_m=0.008,
                  hole_radius_m=0.004, fillet_radius_m=0.006, hole_count=4.0)
    record, _ = make_part(fam, params, tmp_path, labelled=False)
    holes = [f for f in record.features if f["kind"] == "hole"]
    fillets = [f for f in record.features if f["kind"] == "fillet"]
    assert len(holes) == 4 and len(fillets) == 4
    assert all(h["diameter_m"] == pytest.approx(0.008, abs=1e-9) for h in holes)
    assert all(f["radius_m"] == pytest.approx(0.006, abs=1e-9) for f in fillets)
    assert all(f["surface_kind"] == "cylinder" for f in fillets)


@requires_cad
def test_a_part_whose_recogniser_output_disagrees_is_refused(tmp_path,
                                                             monkeypatch):
    """The engine checks its own ground truth, and a mismatch is a refusal."""
    from core.part_dataset import families

    fam = FAMILIES["plate_with_holes"]
    params = sample_parameters(fam, np.random.default_rng(1))
    lying = families.Family(**{**fam.__dict__,
                               "expected_features": lambda p:
                               families.ExpectedFeatures(hole_count=99)})
    with pytest.raises(GroundTruthMismatch, match="99"):
        make_part(lying, params, tmp_path, labelled=False)


@requires_cad
def test_a_part_whose_volume_disagrees_is_refused(tmp_path):
    from core.part_dataset import families

    fam = FAMILIES["box"]
    params = sample_parameters(fam, np.random.default_rng(1))
    wrong = families.Family(**{**fam.__dict__,
                               "volume_m3": lambda p: 2.0 * fam.volume_m3(p)})
    with pytest.raises(GroundTruthMismatch, match="closed form"):
        make_part(wrong, params, tmp_path, labelled=False)


# ------------------------------------------------ labels are graded, not chosen

def test_a_computed_label_is_simulated_and_cannot_be_talked_up():
    assert LABEL_CEILING is EvidenceLevel.SIMULATED
    item = label(1.0, "m", EvidenceKind.SIMULATION, "calculix")
    assert item["evidence"] == "simulated"
    assert label(1.0, "kg", EvidenceKind.ANALYTICAL, "arith")["evidence"] \
        == "simulated"
    assert label(1.0, "m", EvidenceKind.SURROGATE, "mlp")["evidence"] \
        == "surrogate"
    assert label(1.0, "m", EvidenceKind.PHYSICAL_TEST, "bench")["evidence"] \
        == "experimentally_validated"


def _record_with(labels: dict) -> PartRecord:
    from core.part_dataset import (GeometrySummary, Licence, Provenance,
                                   TopologySummary)
    return PartRecord(
        part_id="x",
        provenance=Provenance(kind=ProvenanceKind.SYNTHETIC_PARAMETRIC,
                              source="test", generator="box",
                              licence=Licence(identifier="Apache-2.0",
                                              redistributable=True)),
        geometry=GeometrySummary(volume_m3=1e-5, surface_area_m2=1e-2,
                                 bounding_box_m=(0.1, 0.01, 0.01),
                                 centre_of_mass_m=(0, 0, 0)),
        topology=TopologySummary(solids=1, shells=1, faces=6, edges=12,
                                 vertices=8),
        labels=labels)


@pytest.mark.parametrize("claimed", ["repeated", "high_confidence",
                                     "experimentally_validated"])
def test_a_record_refuses_a_label_above_the_ceiling(claimed):
    with pytest.raises(ValidationError, match="ceiling"):
        _record_with({"tip_deflection_m": {"value": 1e-4, "evidence": claimed,
                                           "kind": "simulation"}})


def test_a_record_refuses_a_level_that_is_not_on_the_ladder():
    with pytest.raises(ValidationError, match="not a level"):
        _record_with({"mass_kg": {"value": 1.0, "evidence": "measured"}})


def test_a_physical_test_label_is_the_one_exception():
    record = _record_with({"mass_kg": label(1.0, "kg", EvidenceKind.PHYSICAL_TEST,
                                            "scale")})
    assert record.labels["mass_kg"]["evidence"] == "experimentally_validated"


# -------------------------------------------- the labelling route, checked

@requires_cad
@requires_solvers
def test_the_labelled_box_agrees_with_the_warp_hex_solution(tmp_path):
    """The one family with a reference. Same tolerance the general shape
    route was accepted at."""
    from physics.fem.mesh import solid_box_mesh
    from physics.fem.solver import solve_linear_elasticity
    from core.materials.db import get_material

    fam = FAMILIES["box"]
    params = dict(length_m=0.2, height_m=0.04, width_m=0.03)
    material = get_material("al_7075_t6")
    record, report = make_part(fam, params, tmp_path, material,
                               LoadCase(total_load_n=-100.0, direction=1))
    labels = record.labels

    mesh = solid_box_mesh(0.2, 0.04, 0.03, 20, 4, 3)
    reference = solve_linear_elasticity(
        mesh, material.youngs_modulus_pa, material.poisson_ratio,
        mesh.nodes_at_x(0.0), mesh.nodes_at_x(0.2), total_load_n=-100.0,
        load_direction=1).tip_deflection()
    assert labels["tip_deflection_m"]["value"] == pytest.approx(reference,
                                                                rel=0.02)
    assert labels["mass_kg"]["value"] == pytest.approx(
        0.2 * 0.04 * 0.03 * material.density_kg_m3, rel=1e-6)
    for name in ("mass_kg", "tip_deflection_m", "max_displacement_m",
                 "max_von_mises_pa", "load_case"):
        assert labels[name]["evidence"] == "simulated", name
    assert report.fine_nodes > report.coarse_nodes


@requires_cad
@requires_solvers
def test_deflection_converges_and_the_peak_stress_says_it_does_not(tmp_path):
    """Measured before this was written: displacement moved 0.06 percent
    between meshes, peak von Mises 14 percent, because the clamped edge is a
    singularity. The label carries both numbers and the stress carries the
    warning."""
    fam = FAMILIES["hollow_rect"]
    params = dict(length_m=0.2, height_m=0.04, width_m=0.03, wall_m=0.005)
    record, _ = make_part(fam, params, tmp_path)
    tip = record.labels["tip_deflection_m"]
    stress = record.labels["max_von_mises_pa"]
    assert tip["mesh_sensitivity"] < 0.02
    assert "mesh_sensitivity" in stress
    assert "does NOT converge" in stress["note"]


@requires_cad
@requires_solvers
def test_a_small_dataset_round_trips_through_the_store(tmp_path):
    """One part per family, written, read back, and every record checked."""
    out = tmp_path / "parts.jsonl"
    records, report = generate_dataset(len(FAMILIES), seed=0, out_path=out,
                                       step_dir=tmp_path / "step",
                                       stop_on_mismatch=True)
    assert report.generated == len(FAMILIES)
    assert report.refused == []
    assert set(report.per_family) == set(FAMILIES)
    back = read_jsonl(out)
    assert [r.part_id for r in back] == [r.part_id for r in records]
    assert len({r.part_id for r in back}) == len(back)
    for record in back:
        assert record.is_publishable
        assert record.material_id == "al_7075_t6"
        assert record.labels["parameters"]["family"] == record.provenance.generator
        for name, item in record.labels.items():
            if name != "parameters":
                assert item["evidence"] == "simulated", (record.part_id, name)
        assert (tmp_path / "step" / f"{record.part_id}.step").exists()
    print(f"\n{report.summary()}")


@requires_cad
def test_generation_is_reproducible_for_a_seed(tmp_path):
    first, _ = generate_dataset(4, seed=7, labelled=False)
    second, _ = generate_dataset(4, seed=7, labelled=False)
    assert [r.part_id for r in first] == [r.part_id for r in second]
    assert [r.geometry.volume_m3 for r in first] == \
        [r.geometry.volume_m3 for r in second]


# ----------------------------------------------------------------- the store

def test_a_proprietary_record_cannot_be_written_to_a_public_file(tmp_path):
    from core.part_dataset import (GeometrySummary, Licence, Provenance,
                                   TopologySummary)
    private = PartRecord(
        part_id="private",
        provenance=Provenance(kind=ProvenanceKind.PROPRIETARY_LOCAL,
                              source="a customer",
                              licence=Licence(identifier="proprietary",
                                              redistributable=False)),
        geometry=GeometrySummary(volume_m3=1e-5, surface_area_m2=1e-2,
                                 bounding_box_m=(0.1, 0.01, 0.01),
                                 centre_of_mass_m=(0, 0, 0)),
        topology=TopologySummary(solids=1, shells=1, faces=6, edges=12,
                                 vertices=8))
    out = tmp_path / "public.jsonl"
    with pytest.raises(NotPublishable):
        write_jsonl(out, [private])
    assert not out.exists()                    # nothing partial left behind
    assert write_jsonl(out, [private], public=False) == 1
    assert read_jsonl(out)[0].part_id == "private"


def test_a_bad_line_names_itself(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"part_id": "x"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        read_jsonl(bad)
