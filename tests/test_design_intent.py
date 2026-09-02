"""Design intent, checked by ablation through the solver rather than asserted.

Every outcome below was measured before it was pinned, on the parts named in
each test. The solver is CalculiX through the labeller, two meshes per part,
so each ablation is four solves and one to six seconds.
"""

from __future__ import annotations

import pytest

from brain import Brain
from brain.semantic.evidence import EvidenceLevel
from core.design_reference import DesignReference, Provenance
from core.part_dataset.intent import (Direction, IntentClaim, Outcome, ablate,
                                      ablated_parameters, intent_claims,
                                      record_in_brain)
from core.part_dataset.labeller import labelling_available
from geometry.cad_export.kernel import kernel_available
from nodes import step_analyzer as sa

pytestmark = pytest.mark.slow
requires_all = pytest.mark.skipif(
    not (kernel_available() and sa.is_available() and labelling_available()),
    reason="needs build123d, OCP, gmsh and CalculiX")


def claim(name, **kw) -> IntentClaim:
    fields = dict(name=name, source="the designer's note", confidence=0.2,
                  provenance=Provenance.ASSUMED)
    fields.update(kw)
    return IntentClaim(**fields)


HOLLOW = dict(length_m=0.2, height_m=0.04, width_m=0.03, wall_m=0.003)
SLENDER_BOX = dict(length_m=0.12, height_m=0.012, width_m=0.012)
PLATE = dict(length_m=0.16, width_m=0.08, thickness_m=0.008, hole_radius_m=0.004,
             fillet_radius_m=0.004, hole_count=2.0)


# ------------------------------------------------------------- the claims

def test_a_claim_needs_provenance_a_role_and_a_real_parameter():
    with pytest.raises(ValueError, match="no source"):
        IntentClaim(name="x", source=" ", confidence=0.2,
                    provenance=Provenance.ASSUMED, family="box",
                    parameter="length_m", role="reach")
    with pytest.raises(ValueError, match="caps at"):
        claim("x", confidence=0.9, family="box", parameter="length_m", role="r")
    with pytest.raises(ValueError, match="does not have"):
        claim("x", family="box", parameter="wall_m", role="r")
    with pytest.raises(ValueError, match="states no role"):
        claim("x", family="box", parameter="length_m", role="  ")
    with pytest.raises(ValueError, match="not 1"):
        claim("x", family="box", parameter="length_m", role="r", factor=1.0)


def test_claims_live_in_a_design_reference_beside_other_priors():
    c = claim("wall", family="hollow_rect", parameter="wall_m", role="stiffness")
    reference = DesignReference(name="link", items=(c,))
    assert intent_claims(reference) == (c,)
    assert reference.weakest_confidence() == 0.2
    assert "[ASSUMED]" in reference.provenance_report()[0]
    assert "hollow_rect: wall_m is stiffness" in c.statement()


def test_an_ablation_that_leaves_the_admissible_region_is_refused():
    c = claim("wall", family="hollow_rect", parameter="wall_m", role="r",
              factor=10.0)
    with pytest.raises(ValueError, match="admissible region"):
        ablated_parameters(c, HOLLOW)
    c2 = claim("holes", family="plate_with_holes", parameter="hole_count",
               role="r", factor=2.0)
    assert ablated_parameters(c2, PLATE)["hole_count"] == 4.0


# ----------------------------------------------------- measured outcomes

@requires_all
def test_wall_thickness_is_what_makes_the_link_stiff(tmp_path):
    """Measured: doubling the wall took the deflection to 0.625 of itself,
    a 37 percent drop against a 0.5 percent mesh sensitivity."""
    c = claim("wall", family="hollow_rect", parameter="wall_m",
              role="what makes the link stiff", direction=Direction.DOWN)
    r = ablate(c, HOLLOW, tmp_path)
    assert r.outcome is Outcome.SUPPORTED, r.reason
    assert r.ratio < 0.75
    assert r.noise_floor < 0.02


@requires_all
def test_length_enters_as_a_cube_and_the_solver_agrees_with_beam_theory(tmp_path):
    """Measured ratio 7.98 for a doubling on a slender box, against the 8 of
    Euler-Bernoulli. The claim carries the number and the solver checks it."""
    c = claim("length", family="box", parameter="length_m", role="set by reach",
              direction=Direction.UP, expected_ratio=8.0, tolerance=0.15)
    r = ablate(c, SLENDER_BOX, tmp_path)
    assert r.outcome is Outcome.SUPPORTED, r.reason
    assert r.ratio == pytest.approx(8.0, rel=0.05)


@requires_all
def test_holes_are_clearance_and_not_structure(tmp_path):
    """The same measurement settles two opposite claims. Going from two holes
    to four raised the deflection by 1.8 percent, real against a 0.3 percent
    noise floor. That refutes 'the holes are structural', because 1.8 percent
    is not a structural role, and supports 'the holes are clearance' stated
    with a 10 percent tolerance."""
    structural = claim("holes-structural", family="plate_with_holes",
                       parameter="hole_count", role="structural",
                       direction=Direction.UP, tolerance=0.2)
    clearance = claim("holes-clearance", family="plate_with_holes",
                      parameter="hole_count", role="fastener clearance",
                      direction=Direction.NONE, tolerance=0.10)
    r1 = ablate(structural, PLATE, tmp_path)
    r2 = ablate(clearance, PLATE, tmp_path)
    assert r1.outcome is Outcome.REFUTED, r1.reason
    assert "direction is real and the role is not" in r1.reason
    assert r2.outcome is Outcome.SUPPORTED, r2.reason
    assert r1.ratio == r2.ratio                      # one measurement, two claims


@requires_all
def test_an_effect_inside_the_mesh_noise_teaches_nothing(tmp_path):
    """A 0.1 percent change in height moves the deflection 0.3 percent, which
    is the mesh sensitivity itself. Inconclusive, and recorded as nothing."""
    c = claim("tiny", family="box", parameter="height_m", role="stiffness",
              direction=Direction.DOWN, factor=1.001)
    r = ablate(c, dict(length_m=0.2, height_m=0.02, width_m=0.02), tmp_path)
    assert r.outcome is Outcome.INCONCLUSIVE, r.reason
    with Brain(":memory:") as brain:
        assert record_in_brain(r, brain.semantic, run_id="r1") is None
        assert brain.semantic.by_domain("design_intent") == []


@requires_all
def test_the_brain_grades_intent_by_the_ladder_and_never_above_simulated(tmp_path):
    """A supported ablation is one simulation; a refuted one is a
    counterexample. Independent runs would be needed to climb, and the gate
    to EXPERIMENTALLY_VALIDATED stays shut."""
    c = claim("wall", family="hollow_rect", parameter="wall_m",
              role="what makes the link stiff", direction=Direction.DOWN)
    r = ablate(c, HOLLOW, tmp_path)
    with Brain(":memory:") as brain:
        k = record_in_brain(r, brain.semantic, run_id="run-a")
        assert k.evidence_level is EvidenceLevel.SIMULATED
        assert k.claim_key == c.claim_key()
        # the same run again consolidates rather than duplicating
        k2 = record_in_brain(r, brain.semantic, run_id="run-a")
        assert k2.knowledge_id == k.knowledge_id
        assert len(brain.semantic.by_domain("design_intent")) == 1
        assert k2.evidence_level is EvidenceLevel.SIMULATED   # one run still
        # a refutation lands as a counterexample on the same claim
        bad = claim("wall-up", family="hollow_rect", parameter="wall_m",
                    role="what makes the link stiff", direction=Direction.UP)
        r_bad = ablate(bad, HOLLOW, tmp_path)
        assert r_bad.outcome is Outcome.REFUTED
        k3 = record_in_brain(r_bad, brain.semantic, run_id="run-a")
        assert len(k3.counterexamples) == 1
        assert k3.evidence_level is EvidenceLevel.UNVERIFIED  # no support at all
