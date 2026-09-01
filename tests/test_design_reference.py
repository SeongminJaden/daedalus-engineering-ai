"""Design references may bias a search. They may never gate it.

The central test in this file is not that a reference works, it is that a
reference cannot change a pass into a fail or the other way round. Everything
else here supports that one claim.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core.design_reference import (PROVENANCE_CEILING, DesignReference,
                                   FormTarget, LoadPathHint, Provenance,
                                   RangePrior, ReferenceItem,
                                   describe_influence, forbidden_influence)
from brain.semantic.evidence import LEVEL_CONFIDENCE_CEILING, EvidenceLevel
from optimization.constraints import (build_optimization_problem,
                                      evaluate_design)
from optimization.gradient.slsqp import default_start
from projects.robotic_link.problem import build_mvp_problem


@pytest.fixture(scope="module")
def problem():
    return build_optimization_problem(build_mvp_problem())


def cited(name: str, **kwargs) -> RangePrior:
    return RangePrior(name=name, source="survey of a product class",
                      confidence=0.5, provenance=Provenance.CITED, **kwargs)


# ----------------------------------------------------- provenance is mandatory

def test_a_prior_without_a_source_is_refused():
    """Once stored, an unattributed prior reads exactly like a measured fact."""
    with pytest.raises(ValueError, match="no source"):
        ReferenceItem(name="wall thickness", source="   ", confidence=0.3,
                      provenance=Provenance.CITED)


def test_a_prior_without_a_usable_confidence_is_refused():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="confidence"):
            ReferenceItem(name="x", source="somewhere", confidence=bad,
                          provenance=Provenance.CITED)


def test_an_assumed_prior_cannot_claim_more_than_the_unverified_ceiling():
    """A guess with a confident number attached is the whole failure mode."""
    ceiling = LEVEL_CONFIDENCE_CEILING[EvidenceLevel.UNVERIFIED]
    assert PROVENANCE_CEILING[Provenance.ASSUMED] == ceiling
    with pytest.raises(ValueError, match="caps at"):
        ReferenceItem(name="x", source="a model suggested it",
                      confidence=ceiling + 0.01,
                      provenance=Provenance.ASSUMED)
    ReferenceItem(name="x", source="a model suggested it", confidence=ceiling,
                  provenance=Provenance.ASSUMED)


def test_no_provenance_reaches_repeated_or_above():
    """Reading a reference is not an independent run, so it earns no rank."""
    simulated = LEVEL_CONFIDENCE_CEILING[EvidenceLevel.SIMULATED]
    assert max(PROVENANCE_CEILING.values()) == simulated
    assert simulated < LEVEL_CONFIDENCE_CEILING[EvidenceLevel.REPEATED]


def test_an_assumed_item_is_labelled_as_assumed():
    item = ReferenceItem(name="taper", source="a model suggested it",
                         confidence=0.1, provenance=Provenance.ASSUMED)
    assert item.label().startswith("[ASSUMED] ")
    assert "a model suggested it" in item.label()
    assert "0.10" in item.label()


def test_a_cited_item_is_not_labelled_assumed():
    assert "[ASSUMED]" not in cited("outer_width_m", minimum=0.02,
                                    maximum=0.04).label()


def test_an_empty_range_is_refused():
    with pytest.raises(ValueError, match="empty range"):
        cited("outer_width_m", minimum=0.04, maximum=0.02)


def test_duplicate_item_names_are_refused():
    """A later duplicate would silently win, which is a hard bug to see."""
    item = cited("outer_width_m", minimum=0.02, maximum=0.04)
    with pytest.raises(ValueError, match="two items"):
        DesignReference(name="r", items=(item, item))


def test_a_form_target_must_state_a_target():
    with pytest.raises(ValueError, match="states no target"):
        FormTarget(name="silhouette", source="a catalogue", confidence=0.3,
                   provenance=Provenance.CITED, target="  ")


def test_a_load_path_hint_must_describe_something():
    with pytest.raises(ValueError, match="describes nothing"):
        LoadPathHint(name="diagonal", source="a sketch", confidence=0.3,
                     provenance=Provenance.CITED, description="")


# ------------------------------------------- the physics stays out of reach

def test_the_reference_has_no_api_that_returns_a_modified_problem():
    """The absence is the mechanism, so the absence is what is tested.

    A method that took an OptimizationProblem and returned a new one would be
    the path by which a generated sentence could relax an allowable stress.
    There is no such method, and this fails if one is ever added.
    """
    public = {name for name in dir(DesignReference)
              if not name.startswith("_")}
    assert public == {"items", "of_type", "prior", "assumed_items",
                      "weakest_confidence", "provenance_report",
                      "starting_point", "aesthetic_weights", "as_dict"}

    # And the one method that is handed an OptimizationProblem returns an
    # array, not a problem, so there is nothing for a caller to install.
    reference = DesignReference(name="r")
    op = build_optimization_problem(build_mvp_problem())
    result = reference.starting_point(op, default_start(op))
    assert isinstance(result, np.ndarray)


def test_the_declared_influence_is_exactly_two_things():
    assert len(describe_influence()) == 2
    assert len(forbidden_influence()) == 5


def test_applying_a_reference_leaves_every_physics_field_untouched(problem):
    """Bounds, allowable stress and the deflection limit must be identical."""
    before = dataclasses.replace(problem)
    reference = DesignReference(name="r", items=(
        cited("outer_height_m", minimum=0.05, maximum=0.07),))
    reference.starting_point(problem, default_start(problem))

    assert problem.allowable_stress_pa == before.allowable_stress_pa
    assert problem.max_deflection_m == before.max_deflection_m
    assert np.array_equal(problem.lower, before.lower)
    assert np.array_equal(problem.upper, before.upper)


def test_a_design_that_fails_still_fails_with_a_reference_applied(problem):
    """The central invariant. A prior cannot buy a design a pass.

    The reference points hard at a section that is far too thin, which is
    exactly what a plausible but wrong prior would do. The evaluation must be
    unmoved: the same design is infeasible either way, and by the same amount.
    """
    flimsy = np.array([problem.lower[0], problem.lower[1], problem.lower[2]])
    without = evaluate_design(problem, flimsy)
    assert not without.is_feasible(), "this fixture must start infeasible"

    reference = DesignReference(name="wishful", items=(
        RangePrior(name="wall_thickness_m", source="a model suggested it",
                   confidence=0.2, provenance=Provenance.ASSUMED,
                   minimum=0.0005, maximum=0.0006),))
    reference.starting_point(problem, default_start(problem))
    with_reference = evaluate_design(problem, flimsy)

    assert not with_reference.is_feasible()
    assert with_reference.constraints == without.constraints
    assert with_reference.safety_factor == without.safety_factor


# --------------------------------------------------- what it may actually do

def test_a_reference_moves_the_starting_point(problem):
    plain = default_start(problem)
    reference = DesignReference(name="r", items=(
        cited("outer_height_m", minimum=0.05, maximum=0.07),))
    biased = reference.starting_point(problem, plain)
    assert biased[1] != pytest.approx(plain[1])
    assert biased[1] == pytest.approx(
        np.clip(0.06, problem.lower[1], problem.upper[1]))


def test_an_empty_reference_changes_nothing(problem):
    plain = default_start(problem)
    assert np.array_equal(
        DesignReference(name="r").starting_point(problem, plain), plain)


def test_a_prior_outside_the_bounds_is_clipped_not_honoured(problem):
    """A reference cannot reach a design the bounds already excluded."""
    reference = DesignReference(name="greedy", items=(
        cited("outer_height_m", minimum=9.0, maximum=11.0),))
    biased = reference.starting_point(problem, default_start(problem))
    assert biased[1] <= problem.upper[1]
    assert np.all(biased >= problem.lower) and np.all(biased <= problem.upper)


def test_an_impossible_section_falls_back_instead_of_being_handed_over(problem):
    """A wall thicker than half the section has no cavity and cannot evaluate."""
    plain = default_start(problem)
    reference = DesignReference(name="impossible", items=(
        cited("wall_thickness_m", minimum=0.019, maximum=0.020),))
    biased = reference.starting_point(problem, plain)
    assert problem.is_geometrically_valid(biased)


def test_aesthetic_weights_are_scaled_by_confidence():
    """A guess must pull less hard than a cited convention."""
    reference = DesignReference(name="r", items=(
        FormTarget(name="taper", source="a model suggested it", confidence=0.2,
                   provenance=Provenance.ASSUMED, target="tapered",
                   weight=1.0),
        FormTarget(name="symmetry", source="a product family", confidence=0.6,
                   provenance=Provenance.CITED, target="mirror about y",
                   weight=1.0)))
    weights = reference.aesthetic_weights()
    assert weights["tapered"] == pytest.approx(0.2)
    assert weights["mirror about y"] == pytest.approx(0.6)
    assert weights["tapered"] < weights["mirror about y"]


def test_the_weakest_item_is_what_is_reported_not_the_average():
    """One solid citation must not launder several guesses."""
    reference = DesignReference(name="r", items=(
        cited("outer_width_m", minimum=0.02, maximum=0.04),
        FormTarget(name="taper", source="a model suggested it", confidence=0.1,
                   provenance=Provenance.ASSUMED, target="tapered")))
    assert reference.weakest_confidence() == pytest.approx(0.1)
    assert len(reference.assumed_items) == 1


# ------------------------------------------------------------ what is stored

def test_the_stored_form_keeps_every_provenance(tmp_path):
    """A prior read back out of a database has lost its context.

    So the context is stored with it: source, confidence, provenance and the
    ceiling that capped it, plus an explicit no on physical validation.
    """
    from brain.db import BrainDB
    from brain.episodic.memory import EpisodicMemory

    reference = DesignReference(name="humanoid arm", items=(
        cited("outer_height_m", minimum=0.03, maximum=0.05),
        FormTarget(name="taper", source="a model suggested it", confidence=0.2,
                   provenance=Provenance.ASSUMED, target="tapered")))

    memory = EpisodicMemory(BrainDB(tmp_path / "brain.db"))
    memory.record_run("run-1", "arm", meta=reference.as_dict())
    stored = memory.get_run("run-1")["meta"]

    assert stored["reference"] == "humanoid arm"
    assert stored["physically_validated"] is False
    assert stored["assumed_item_count"] == 1
    assert stored["weakest_confidence"] == pytest.approx(0.2)
    by_name = {item["name"]: item for item in stored["items"]}
    assert by_name["taper"]["assumed"] is True
    assert by_name["taper"]["source"] == "a model suggested it"
    assert by_name["outer_height_m"]["assumed"] is False


def test_the_stored_ceiling_is_simulated_not_higher():
    """Nothing a reference contributes can be recorded as validated."""
    stored = DesignReference(name="r").as_dict()
    assert stored["evidence_ceiling"] == LEVEL_CONFIDENCE_CEILING[
        EvidenceLevel.SIMULATED]
    assert stored["physically_validated"] is False
