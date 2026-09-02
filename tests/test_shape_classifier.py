"""Shape descriptors and family classification, measured against ground truth.

The rules are checked on parts the generator made, whose family is known by
construction, and on the Fusion fixtures, which the generator did not make.
The learned classifier is checked the same way, and then checked for the
thing a learned classifier gets wrong: accepting something it has never seen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brain.semantic.evidence import EvidenceLevel
from core.part_dataset.families import ORIGINAL_FAMILIES
from core.part_dataset import FAMILIES, generate_dataset
from core.part_dataset.classify import (UNKNOWN, NearestNeighbourClassifier,
                                        classification_label, evaluate,
                                        rule_classify)
from core.part_dataset.descriptors import (DESCRIPTOR_NAMES, ShapeDescriptor,
                                           describe_step)
from geometry.cad_export.kernel import kernel_available
from nodes import step_analyzer as sa

requires_cad = pytest.mark.skipif(not (kernel_available() and sa.is_available()),
                                  reason="build123d and OCP are required")

FIXTURES = Path("tests/fixtures/cad")
requires_fixtures = pytest.mark.skipif(not FIXTURES.exists(),
                                       reason="the CAD fixtures are not present")

PER_FAMILY_TRAIN, PER_FAMILY_TEST = 20, 10


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Descriptors for generated parts, split by seed, plus their families."""
    if not (kernel_available() and sa.is_available()):
        pytest.skip("build123d and OCP are required")
    root = tmp_path_factory.mktemp("corpus")
    out = {}
    for name, n, seed in (("train", PER_FAMILY_TRAIN * len(ORIGINAL_FAMILIES), 1),
                          ("test", PER_FAMILY_TEST * len(ORIGINAL_FAMILIES), 2)):
        records, report = generate_dataset(n, seed=seed, step_dir=root / name,
                                           labelled=False, stop_on_mismatch=True, families=ORIGINAL_FAMILIES)
        assert report.refused == []
        descriptors = [describe_step(root / name / f"{r.part_id}.step")[0]
                       for r in records]
        out[name] = (descriptors, [r.provenance.generator for r in records])
    return out


@pytest.fixture(scope="module")
def fixtures():
    if not sa.is_available() or not FIXTURES.exists():
        pytest.skip("OCP and the fixtures are required")
    return {p.name: describe_step(p)[0] for p in sorted(FIXTURES.glob("*.step"))}


# ------------------------------------------------------------- descriptors

@requires_cad
def test_descriptors_are_scale_free(tmp_path):
    """The same proportions at twice the size describe identically."""
    from core.part_dataset import make_part

    fam = FAMILIES["plate_with_holes"]
    small = dict(length_m=0.10, width_m=0.06, thickness_m=0.01,
                 hole_radius_m=0.004, fillet_radius_m=0.003, hole_count=4.0)
    large = {k: (v * 2 if k != "hole_count" else v) for k, v in small.items()}
    a, _ = make_part(fam, small, tmp_path / "a", labelled=False)
    b, _ = make_part(fam, large, tmp_path / "b", labelled=False)
    da = describe_step(tmp_path / "a" / f"{a.part_id}.step")[0].vector()
    db = describe_step(tmp_path / "b" / f"{b.part_id}.step")[0].vector()
    assert np.allclose(da, db, rtol=1e-6, atol=1e-9)


@requires_cad
def test_the_euler_characteristic_counts_inner_loops(corpus):
    """A hollow tube is genus 1 and a plate with n holes is genus n.

    The first version reported 2 for the tube, because its two annular end
    faces are not disks and V - E + F alone does not know that.
    """
    descriptors, families = corpus["train"]
    expected = {"box": 2, "l_bracket": 2, "stepped_shaft": 2, "hollow_rect": 0}
    for d, f in zip(descriptors, families):
        if f in expected:
            assert d["euler"] == expected[f], f
        else:
            assert d["euler"] == 2 - 2 * d["hole_count"], f


def test_the_descriptor_vector_has_a_fixed_order():
    d = ShapeDescriptor(values={n: float(i) for i, n in enumerate(DESCRIPTOR_NAMES)})
    assert d.vector().tolist() == list(range(len(DESCRIPTOR_NAMES)))
    assert len(DESCRIPTOR_NAMES) == len(set(DESCRIPTOR_NAMES))


# ------------------------------------------------------------------- rules

@requires_cad
def test_the_rules_classify_every_generated_part(corpus):
    """On parts built to the families, a rule that misses has a bug."""
    for split in ("train", "test"):
        descriptors, families = corpus[split]
        for d, f in zip(descriptors, families):
            result = rule_classify(d)
            assert result.family == f, (f, result.reason)
            assert result.evidence is EvidenceLevel.SIMULATED
            assert result.decided


@requires_fixtures
def test_the_fusion_plates_classify_as_plates_across_kernels(fixtures):
    """Fixtures A and B were authored in Fusion to the plate's shape. The
    rules never saw Fusion output, so agreement here is a check on both."""
    for name in ("fixtureA.step", "fixtureB.step"):
        result = rule_classify(fixtures[name])
        assert result.family == "plate_with_holes", (name, result.reason)


@requires_fixtures
def test_parts_outside_the_families_are_unknown_with_reasons(fixtures):
    """A chamfer, a cone, a concave blend, a slot: none is a family, and the
    rules say so rather than picking the nearest."""
    for name in ("fixtureC.step", "fixtureD.step", "fixtureE.step",
                 "fixtureF.step", "fixtureG.step"):
        result = rule_classify(fixtures[name])
        assert result.family == UNKNOWN, name
        assert not result.decided
        assert "needs" in result.reason


# ------------------------------------------------- nearest neighbour, graded

@requires_cad
def test_the_learned_classifier_grades_surrogate_and_agrees_with_the_rules(corpus):
    """Measured on a held-out seed: 1.00, against 1.00 for the rules.

    Before compactness was logged it was 0.96, with two thin tubes rejected
    as unknown because their unlogged compactness put them 7.8 standardised
    units from everything. The model has learned what the rules knew, which
    is what it is for; the floor below is loose on purpose, since the rules
    are the answer and this is the check on them."""
    train_d, train_y = corpus["train"]
    test_d, test_y = corpus["test"]
    clf = NearestNeighbourClassifier().fit(train_d, train_y)
    ev = evaluate(clf, test_d, test_y)
    print(f"\nrules {ev.rule_accuracy:.2f}  knn {ev.knn_accuracy:.2f}  "
          f"knn unknown {ev.knn_unknown}  threshold {clf.threshold_:.2f}  "
          f"disagreements {ev.disagreements}")
    assert ev.rule_accuracy == 1.0
    assert ev.knn_accuracy >= 0.9
    for d in test_d:
        result = clf.classify(d)
        assert result.evidence is EvidenceLevel.SURROGATE
        assert result.method == "knn_descriptors"
    # a wrong answer, when it happens, is a rejection and not a wrong family
    for truth, rule, knn in ev.disagreements:
        assert rule == truth
        assert knn == UNKNOWN


@requires_cad
@requires_fixtures
def test_a_cone_is_not_a_box(corpus, fixtures):
    """Open-set rejection, pinned where it was measured to matter.

    At a rejection factor of 1.25 the cone-bearing fixture D was accepted as
    a box with a 0.6 vote. At 1.0 it is unknown. The Fusion plates, which
    ARE plates, are still accepted.
    """
    train_d, train_y = corpus["train"]
    clf = NearestNeighbourClassifier().fit(train_d, train_y)
    assert clf.classify(fixtures["fixtureD.step"]).family == UNKNOWN
    for name in ("fixtureC.step", "fixtureF.step", "fixtureG.step"):
        assert clf.classify(fixtures[name]).family == UNKNOWN, name
    for name in ("fixtureA.step", "fixtureB.step"):
        assert clf.classify(fixtures[name]).family == "plate_with_holes", name


def test_the_classifier_refuses_to_run_unfitted():
    d = ShapeDescriptor(values={n: 0.0 for n in DESCRIPTOR_NAMES})
    with pytest.raises(RuntimeError, match="not been fitted"):
        NearestNeighbourClassifier().classify(d)


def test_a_classification_becomes_a_graded_label():
    from core.part_dataset.classify import Classification

    ruled = Classification("box", "topology_rules", EvidenceLevel.SIMULATED,
                           "6 planar faces")
    learned = Classification("box", "knn_descriptors", EvidenceLevel.SURROGATE,
                             "vote", confidence=0.8, distance=1.0)
    assert classification_label(ruled)["evidence"] == "simulated"
    assert classification_label(learned)["evidence"] == "surrogate"
    assert classification_label(learned)["family"] == "box"


# ------------------------------------------------------------ the capability

def test_the_capability_is_registered_and_says_what_it_will_not_decide():
    from nodes.roster import build_roster
    from nodes.shape_classifier import CLASSIFY_CAPABILITY

    registry = build_roster()
    method = registry.get(CLASSIFY_CAPABILITY).method
    assert "UNKNOWN" in method.notes
    assert "SURROGATE" in method.notes
    assert method.evidence == "SIMULATED"
