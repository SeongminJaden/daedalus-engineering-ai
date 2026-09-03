"""The policy seam: a sentence becomes a problem, and nothing it says is trusted.

A policy proposes and never decides. These tests are mostly injection tests:
a policy that lies, invents a material, returns prose instead of JSON or picks
a strategy that does not exist must be stopped by validation, and a proposal
that survives validation must still be verified by a solver before it means
anything.
"""

from __future__ import annotations

import json

import pytest

from agent.execution import executable_methods
from agent.policy import (InvalidProposal, LanguageModelPolicy, PolicyProposal,
                          RuleBasedPolicy, validate_problem, validate_strategy)
from core.engineering_ir.schema import (BoundaryCondition, Constraints,
                                        EngineeringProblem, Geometry, Load,
                                        Objective, ObjectiveQuantity,
                                        ObjectiveSense, Vec3)

GOAL = ("a 500 mm long link carrying 200 N at the tip, deflection under 1 mm, "
        "100 mm wide and 100 mm tall, safety factor 2")


def problem(**overrides) -> EngineeringProblem:
    base = dict(
        name="test", geometry=Geometry(length_m=0.5, max_width_m=0.1,
                                       max_height_m=0.1),
        material_id="al_7075_t6",
        loads=[Load(magnitude_n=200.0, direction=Vec3(x=0.0, y=-1.0, z=0.0))],
        boundary_conditions=[BoundaryCondition()],
        constraints=Constraints(max_deflection_m=1e-3, min_safety_factor=2.0),
        objectives=[Objective(sense=ObjectiveSense.MINIMIZE,
                              quantity=ObjectiveQuantity.MASS)])
    base.update(overrides)
    return EngineeringProblem(**base)


# --- the rule policy reads what is there and refuses what is not -------------

def test_a_sentence_becomes_a_problem():
    proposal = RuleBasedPolicy().propose_problem(GOAL)
    assert isinstance(proposal, PolicyProposal)
    assert proposal.problem.geometry.length_m == 0.5
    assert proposal.problem.geometry.max_width_m == 0.1
    assert proposal.problem.loads[0].magnitude_n == 200.0
    assert proposal.problem.constraints.max_deflection_m == 1e-3
    assert proposal.verified is False
    assert "read a length" in proposal.rationale


def test_a_mass_in_the_sentence_becomes_a_force():
    proposal = RuleBasedPolicy().propose_problem(
        "a 300 mm long arm holding 5 kg at the tip, safety factor 2")
    assert proposal.problem.loads[0].magnitude_n == pytest.approx(5 * 9.81)


def test_a_goal_missing_a_quantity_is_refused_not_defaulted():
    policy = RuleBasedPolicy()
    with pytest.raises(InvalidProposal, match="does not state a length"):
        policy.propose_problem("make it strong and light")
    with pytest.raises(InvalidProposal, match="does not state a load"):
        policy.propose_problem("a 400 mm long bracket, safety factor 2")
    with pytest.raises(InvalidProposal, match="neither a safety factor"):
        policy.propose_problem("a 400 mm long bracket carrying 100 N")
    with pytest.raises(InvalidProposal, match="empty goal"):
        policy.propose_problem("   ")


def test_a_very_long_goal_is_refused():
    with pytest.raises(InvalidProposal, match="at most"):
        RuleBasedPolicy().propose_problem(
            "a 500 mm long link 200 N safety factor 2 " + "x" * 3000)


def test_the_strategy_follows_from_the_problem_not_from_taste():
    policy = RuleBasedPolicy()
    available = set(executable_methods())
    with_envelope = policy.choose_strategy(problem(), available)
    assert with_envelope.method == "generative_cad"

    no_envelope = policy.choose_strategy(
        problem(geometry=Geometry(length_m=0.5)), available)
    assert no_envelope.method == "parametric_section"
    assert "no design envelope" in no_envelope.rationale

    no_limit = policy.choose_strategy(
        problem(constraints=Constraints(min_safety_factor=2.0)), available)
    assert no_limit.method in ("freeform_topology", "topology_compliance")


def test_a_retry_direction_comes_from_a_measured_failure():
    policy = RuleBasedPolicy()
    assert policy.suggest_retry("the solver returned nothing at 3 mesh sizes",
                                problem()).action == "refine_mesh"
    assert policy.suggest_retry("threshold 0.5 leaves the load path is cut",
                                problem()).action == "raise_volume_fraction"
    unknown = policy.suggest_retry("the coffee machine is empty", problem())
    assert unknown.action == "stop"
    assert "no measured remedy" in unknown.rationale


# --- validation stops the lies ------------------------------------------------

def test_an_invented_material_is_refused():
    with pytest.raises(InvalidProposal, match="not in the database"):
        validate_problem(problem(material_id="unobtainium"))


def test_a_dimension_outside_the_measured_range_is_refused():
    with pytest.raises(InvalidProposal, match="outside the range"):
        validate_problem(problem(geometry=Geometry(length_m=900.0)))


def test_a_load_without_a_direction_never_reaches_validation():
    """The IR schema refuses a zero direction before the policy layer sees it,
    which is the right place for it. The policy keeps its own check as the
    second line, for a problem built some other way."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="unit vector"):
        Load(magnitude_n=100.0, direction=Vec3(x=0.0, y=0.0, z=0.0))


def test_a_problem_that_cannot_bound_mass_is_refused_early():
    """Without a stress limit or a safety factor the optimisation layer
    refuses three calls later. Catching it here names the policy as the
    source."""
    with pytest.raises(InvalidProposal, match="unbounded"):
        validate_problem(problem(constraints=Constraints(max_deflection_m=1e-3)))


def test_a_useless_limit_is_warned_about_rather_than_refused():
    warnings = validate_problem(problem(constraints=Constraints(
        max_deflection_m=0.4, min_safety_factor=2.0)))
    assert any("constrains nothing" in w for w in warnings)


def test_a_strategy_that_does_not_exist_is_refused():
    with pytest.raises(InvalidProposal, match="not executable"):
        validate_strategy("ask_a_friend", set(executable_methods()))


# --- the language model seam, injected with bad answers ----------------------

def test_a_model_policy_without_a_callable_refuses_to_exist():
    with pytest.raises(InvalidProposal, match="completion callable"):
        LanguageModelPolicy()


def test_a_model_that_returns_prose_is_refused():
    policy = LanguageModelPolicy(complete=lambda prompt: "Sure! Here is a design.")
    with pytest.raises(InvalidProposal, match="did not return JSON"):
        policy.propose_problem(GOAL)


def test_a_model_that_invents_a_material_is_refused():
    answer = json.dumps({"length_m": 0.5, "material_id": "adamantium",
                         "load_n": 200.0, "min_safety_factor": 2.0})
    policy = LanguageModelPolicy(complete=lambda prompt: answer)
    with pytest.raises(InvalidProposal, match="not in the database"):
        policy.propose_problem(GOAL)


def test_a_model_that_omits_a_required_value_is_refused():
    answer = json.dumps({"length_m": 0.5, "material_id": "al_7075_t6"})
    policy = LanguageModelPolicy(complete=lambda prompt: answer)
    with pytest.raises(InvalidProposal, match="left out load_n"):
        policy.propose_problem(GOAL)


def test_a_model_that_returns_an_absurd_size_is_refused():
    answer = json.dumps({"length_m": 1200.0, "material_id": "al_7075_t6",
                         "load_n": 200.0, "min_safety_factor": 2.0})
    policy = LanguageModelPolicy(complete=lambda prompt: answer)
    with pytest.raises(InvalidProposal, match="outside the range"):
        policy.propose_problem(GOAL)


def test_a_model_that_answers_correctly_still_returns_an_unverified_proposal():
    answer = json.dumps({"length_m": 0.5, "max_width_m": 0.1,
                         "max_height_m": 0.1, "material_id": "al_7075_t6",
                         "load_n": 200.0, "load_direction": [0.0, -1.0, 0.0],
                         "max_deflection_m": 1e-3, "min_safety_factor": 2.0})
    policy = LanguageModelPolicy(complete=lambda prompt: answer)
    proposal = policy.propose_problem(GOAL)
    assert proposal.verified is False
    assert proposal.origin == "language_model_policy"
    assert "not verified" in proposal.rationale


def test_a_model_that_picks_a_strategy_it_invented_is_refused():
    policy = LanguageModelPolicy(complete=lambda prompt: "magic_solver")
    with pytest.raises(InvalidProposal, match="not executable"):
        policy.choose_strategy(problem(), set(executable_methods()))


def test_the_retry_direction_is_not_the_models_to_give():
    """A retry direction is a statement about which failures this project has
    measured a remedy for. A model has no access to that, so the seam
    delegates and the origin says which policy answered."""
    policy = LanguageModelPolicy(complete=lambda prompt: "refine everything")
    suggestion = policy.suggest_retry("the solver returned nothing", problem())
    assert suggestion.action == "refine_mesh"
    assert suggestion.origin == "rule_based_policy"


# --- the proposal is not a design ---------------------------------------------

@pytest.mark.slow
def test_a_proposal_becomes_a_design_only_when_a_solver_says_so():
    """The end of the chain: a policy proposal is run, and what makes the
    result trustworthy is the CalculiX label on it, not the policy."""
    from agent.execution import execute
    from core.part_dataset.labeller import labelling_available
    from geometry.cad_export.kernel import kernel_available
    from optimization.constraints import build_optimization_problem

    if not (kernel_available() and labelling_available()):
        pytest.skip("build123d, gmsh and CalculiX are required")

    policy = RuleBasedPolicy()
    proposal = policy.propose_problem(GOAL)
    assert proposal.verified is False
    op = build_optimization_problem(proposal.problem)
    choice = policy.choose_strategy(proposal.problem, set(executable_methods()))
    outcome = execute(choice.method, op, candidates=4, top_k=2, seed=0)
    assert outcome.method == choice.method
    assert outcome.cad_record is not None
    assert outcome.cad_record.labels["tip_deflection_m"]["evidence"] == "simulated"
