"""The shape surrogate: a ranker behind the gate, measured against the solver.

Labelling is solver time, so this module trains on a small corpus and the
floors are set below what that corpus size measured on random draws. The
metric is rank order, because ranking is the one thing the surrogate is
allowed to do. The corpus is built once per module and every test reads
from it.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.semantic.evidence import EvidenceKind, EvidenceLevel, derive_level
from core.materials.db import get_material
from core.part_dataset.families import ORIGINAL_FAMILIES
from core.part_dataset import FAMILIES, generate_dataset, sample_parameters
from core.part_dataset.labeller import LoadCase, labelling_available
from core.part_dataset.shape_surrogate import (FEATURE_NAMES, TARGET_NAMES,
                                               ShapeSurrogate,
                                               ShapeTrainingSet, beam_proxy_m,
                                               screen_and_verify_parts,
                                               train_shape_surrogate,
                                               training_set_from)
from geometry.cad_export.kernel import kernel_available
from integration import CheckResult, CheckStatus, SurrogateVerdict
from nodes import step_analyzer as sa

pytestmark = pytest.mark.slow
requires_all = pytest.mark.skipif(
    not (kernel_available() and sa.is_available() and labelling_available()),
    reason="needs build123d, OCP, gmsh and CalculiX")

N_TRAIN, N_TEST = 20, 8


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    if not (kernel_available() and sa.is_available() and labelling_available()):
        pytest.skip("needs build123d, OCP, gmsh and CalculiX")
    root = tmp_path_factory.mktemp("labelled")
    material = get_material("al_7075_t6")
    sets = {}
    for name, n, seed in (("train", N_TRAIN, 1), ("test", N_TEST, 2)):
        records, report = generate_dataset(n, seed=seed, step_dir=root / name,
                                           labelled=True, families=ORIGINAL_FAMILIES)
        assert report.generated >= n - 1, report.summary()
        sets[name] = training_set_from(records, root / name, material)
    sets["root"] = root
    return sets


@pytest.fixture(scope="module")
def surrogate(corpus):
    return train_shape_surrogate(corpus["train"], corpus["test"], seed=0)


# ------------------------------------------------------------- the features

def test_the_beam_proxy_is_a_cantilever_of_the_bounding_box():
    """For a solid box the proxy IS Euler-Bernoulli, to round-off."""
    from core.part_dataset import GeometrySummary, Licence, PartRecord, Provenance, ProvenanceKind, TopologySummary

    material = get_material("al_7075_t6")
    L, h, w = 0.2, 0.04, 0.03
    record = PartRecord(
        part_id="box", provenance=Provenance(
            kind=ProvenanceKind.SYNTHETIC_PARAMETRIC, source="t", generator="box",
            licence=Licence(identifier="Apache-2.0", redistributable=True)),
        geometry=GeometrySummary(volume_m3=L * h * w, surface_area_m2=1.0,
                                 bounding_box_m=(L, h, w),
                                 centre_of_mass_m=(0, 0, 0)),
        topology=TopologySummary(solids=1, shells=1, faces=6, edges=12, vertices=8))
    case = LoadCase(total_load_n=-100.0, direction=1)
    expected = 100.0 * L ** 3 / (3 * material.youngs_modulus_pa * w * h ** 3 / 12)
    assert beam_proxy_m(record, material, case) == pytest.approx(expected, rel=1e-12)
    assert FEATURE_NAMES[-1] == "log_beam_proxy_m"


# ----------------------------------------------------------- the surrogate

@requires_all
def test_the_surrogate_ranks_deflection_and_quotes_its_error(surrogate):
    """A ranker is judged on order, not on raw R2.

    Measured with 20 training and 8 held-out parts over eight random draws:
    Spearman rank correlation never below 0.79 (median 0.90), log-space R2
    never below 0.41 (median 0.81), median relative error up to 0.68. The
    raw R2 on the same draws went as low as 0.29, because three decades of
    deflection let two large parts decide it. With 40 training parts the
    same model reaches log R2 0.97 and Spearman 0.99. The floors here sit
    below the worst 20-part draw.
    """
    m = surrogate.test_metrics["tip_deflection_m"]
    print(f"\ntip deflection held-out: spearman {m['spearman']:.2f} r2_log "
          f"{m['r2_log']:.2f} raw r2 {m['r2']:.2f} median rel "
          f"{m['median_rel_err']:.2f} p95 {m['p95_rel_err']:.2f}; "
          f"trained in {surrogate.training['seconds']:.1f} s")
    assert m["spearman"] > 0.6
    assert m["r2_log"] > 0.3
    assert surrogate.training["final_loss"] < surrogate.training["first_loss"]


@requires_all
def test_without_the_proxy_the_same_model_is_worse_or_unusable(corpus):
    """The ablation, printed rather than pinned: on 40 parts the proxy took
    log R2 from below zero to 0.97, but at 20 parts a lucky draw without it
    can score well, so only the with-proxy floor is asserted elsewhere."""
    import core.part_dataset.shape_surrogate as ss

    stripped = ss.FEATURE_NAMES[:-1]
    train = ShapeTrainingSet(x=corpus["train"].x[:, :-1], y=corpus["train"].y,
                             part_ids=corpus["train"].part_ids,
                             families=corpus["train"].families)
    test = ShapeTrainingSet(x=corpus["test"].x[:, :-1], y=corpus["test"].y,
                            part_ids=corpus["test"].part_ids,
                            families=corpus["test"].families)
    original = ss.FEATURE_NAMES
    ss.FEATURE_NAMES = stripped
    try:
        bare = train_shape_surrogate(train, test, seed=0)
    finally:
        ss.FEATURE_NAMES = original
    m = bare.test_metrics["tip_deflection_m"]
    print(f"\nwithout proxy: spearman {m['spearman']:.2f} r2_log {m['r2_log']:.2f}")
    assert np.isfinite(bare.test_metrics["tip_deflection_m"]["r2"])


@requires_all
def test_a_prediction_is_surrogate_and_cannot_become_a_verdict(surrogate, corpus):
    prediction = surrogate.predict(corpus["test"].x[:3])
    assert prediction.verified is False
    assert prediction.evidence_kind is EvidenceKind.SURROGATE
    assert prediction.evidence_level is EvidenceLevel.SURROGATE
    assert set(prediction.values) == set(TARGET_NAMES)
    assert derive_level([prediction.as_evidence("p")], []) is EvidenceLevel.SURROGATE
    with pytest.raises(SurrogateVerdict):
        CheckResult("link", "deflection", CheckStatus.PASSED, "shape_surrogate",
                    2.0, evidence_kind=prediction.evidence_kind)
    screened = prediction.screened_check("link", "deflection", 0, limit_m=1e-3)
    assert screened.status is CheckStatus.SCREENED
    assert screened.safety_factor is None
    assert "run the solver" in screened.detail


@requires_all
def test_the_surrogate_round_trips(surrogate, corpus, tmp_path):
    surrogate.save(tmp_path / "shape")
    back = ShapeSurrogate.load(tmp_path / "shape", device=surrogate.device)
    a = surrogate.predict_array(corpus["test"].x)
    b = back.predict_array(corpus["test"].x)
    assert np.allclose(a, b, rtol=1e-5)
    assert "surrogate" in (tmp_path / "shape" / "meta.json").read_text()


# ---------------------------------------------------- screen, then verify

@requires_all
def test_screening_returns_only_a_solver_verified_winner(surrogate, corpus):
    """Eight candidates ranked by the surrogate, two solved. The winner's
    numbers are the solver's and the result grades SIMULATED; the surrogate's
    own guess is kept beside it so the two can be compared."""
    rng = np.random.default_rng(7)
    names = ("box", "hollow_rect", "box", "l_bracket", "hollow_rect", "box",
             "stepped_shaft", "l_bracket")
    candidates = [(n, sample_parameters(FAMILIES[n], rng)) for n in names]
    result = screen_and_verify_parts(surrogate, candidates,
                                     deflection_limit_m=1e-2,
                                     step_dir=corpus["root"] / "screen", top_k=2)
    assert result.n_screened == 8 and result.n_verified == 2
    assert result.verified and result.winner is not None
    assert result.evidence_level is EvidenceLevel.SIMULATED
    labels = result.winner.labels
    assert labels["tip_deflection_m"]["evidence"] == "simulated"
    assert abs(labels["tip_deflection_m"]["value"]) == result.solver_deflection_m
    assert result.solver_deflection_m <= 1e-2
    assert result.surrogate_error_on_winner is not None
    print(f"\nscreening: predicted {result.predicted_deflection_m:.3e} solver "
          f"{result.solver_deflection_m:.3e} error "
          f"{result.surrogate_error_on_winner:.0%}")


@requires_all
def test_screening_with_an_impossible_limit_verifies_and_returns_nothing(
        surrogate, corpus):
    """Nothing on the shortlist passes the solver, so nothing is returned,
    and the result grades SURROGATE because no solver number backs a winner."""
    rng = np.random.default_rng(8)
    candidates = [("box", sample_parameters(FAMILIES["box"], rng))
                  for _ in range(3)]
    result = screen_and_verify_parts(surrogate, candidates,
                                     deflection_limit_m=1e-12,
                                     step_dir=corpus["root"] / "none", top_k=1)
    assert result.n_verified == 1
    assert not result.verified and result.winner is None
    assert result.evidence_level is EvidenceLevel.SURROGATE
