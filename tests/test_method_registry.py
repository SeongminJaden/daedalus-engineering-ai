"""The method registry and the strategy selection built on it.

The test that carries the weight is `test_beam_theory_is_excluded_at_low_slenderness`.
Phase 7 failed because a cheap beam model was used on a link too stubby for it,
and nothing in the system was in a position to object. The applicability
declarations exist so that the architecture objects automatically, and that is
what is asserted here.
"""

import importlib

import pytest

from agent.strategy import NoApplicableMethod, StrategySelector
from core.registry import (DEFAULT_REGISTRY, EULER_BERNOULLI_SLENDERNESS,
                           Category, Condition, Cost, DuplicateMethod, Fidelity,
                           Method, MethodRegistry, ProblemContext, UnknownMethod,
                           build_default_registry)


def beam_context(slenderness: float) -> ProblemContext:
    return ProblemContext(geometry="prismatic_beam", slenderness=slenderness,
                          material_class="isotropic", needs_stress_field=False)


# --- the honesty gate --------------------------------------------------------

def test_beam_theory_is_excluded_at_low_slenderness():
    """The Phase 7 failure, prevented by declaration rather than by review.

    The MVP link had a slenderness of about 6. Euler-Bernoulli omits shear and
    is measured 2.5% low there, the optimizer sat on exactly that blind spot,
    and the 3D FEM gate rejected the result. Asking the registry for an
    analysis method on that problem must not return the beam model.
    """
    registry = build_default_registry()
    candidates = registry.query(beam_context(6.0), Category.ANALYSIS)

    assert "beam_eb" not in candidates.names()
    assert "beam_eb" in candidates.excluded_names()
    reason, = candidates.reason("beam_eb")
    assert "slenderness" in reason

    # And something valid is still offered, so exclusion is not a dead end.
    assert "beam_timoshenko" in candidates.names()
    assert "fem3d" in candidates.names()


def test_beam_theory_is_offered_where_it_is_valid():
    """The range cuts both ways: a slender beam may use the cheap model."""
    registry = build_default_registry()
    candidates = registry.query(beam_context(EULER_BERNOULLI_SLENDERNESS),
                                Category.ANALYSIS)
    assert "beam_eb" in candidates.names()
    # And it is what a cost-driven router picks, being the cheapest valid one.
    cheapest = registry.cheapest_applicable(
        beam_context(30.0), Category.ANALYSIS)
    assert cheapest.cost is Cost.TRIVIAL


def test_timoshenko_is_excluded_below_its_measured_range():
    """A method does not claim ground it was never measured on."""
    registry = build_default_registry()
    candidates = registry.query(beam_context(2.0), Category.ANALYSIS)
    assert "beam_timoshenko" not in candidates.names()
    assert "fem3d" in candidates.names()


def test_an_unstated_feature_excludes_rather_than_defaults():
    """Silence is not permission.

    A problem whose slenderness nobody stated is exactly the case where a beam
    model must not be assumed valid. The condition fails rather than passing on
    a default, and rather than raising out of the query.
    """
    registry = build_default_registry()
    context = ProblemContext(geometry="prismatic_beam", needs_stress_field=False)
    candidates = registry.query(context, Category.ANALYSIS)
    assert "beam_eb" not in candidates.names()
    assert "does not state" in candidates.reason("beam_eb")[0]


def test_a_stress_field_request_excludes_one_dimensional_models():
    registry = build_default_registry()
    context = ProblemContext(geometry="prismatic_beam", slenderness=30.0,
                             needs_stress_field=True)
    names = registry.query(context, Category.ANALYSIS).names()
    assert "beam_eb" not in names and "beam_timoshenko" not in names
    assert "fem3d" in names


# --- registry mechanics ------------------------------------------------------

def test_every_registered_method_points_at_real_code():
    """A registry entry is a claim that the method exists.

    Registering a stub would be the same failure as a fabricated part number:
    it reads as a record of something real and is not one. This walks every
    declared implementation and imports it.
    """
    for method in DEFAULT_REGISTRY.all():
        target = method.implementation.split(" ")[0]
        assert target, f"{method.name} declares no implementation"
        module_path = target
        while module_path:
            try:
                module = importlib.import_module(module_path)
                break
            except ModuleNotFoundError:
                if "." not in module_path:
                    pytest.fail(f"{method.name} points at {target!r}, "
                                f"which does not import")
                module_path = module_path.rsplit(".", 1)[0]
        # A stub module in this tree is a docstring and nothing else. Check
        # structurally for public definitions rather than for the word "stub",
        # which appears in real modules for unrelated reasons.
        public = [name for name, value in vars(module).items()
                  if not name.startswith("_")
                  and (callable(value) or isinstance(value, type))]
        assert public, (f"{method.name} points at {module_path}, which defines "
                        f"nothing: it is a stub, not an implementation")


def test_duplicate_names_are_rejected():
    registry = MethodRegistry()
    method = Method(name="x", category=Category.ANALYSIS, summary="",
                    inputs=(), outputs=(), fidelity=Fidelity.BEAM,
                    cost=Cost.CHEAP)
    registry.register(method)
    with pytest.raises(DuplicateMethod):
        registry.register(method)


def test_unknown_name_reports_what_is_known():
    with pytest.raises(UnknownMethod) as excinfo:
        DEFAULT_REGISTRY.get("no_such_method")
    assert "beam_eb" in str(excinfo.value)


def test_query_ordering_is_deterministic():
    """Two builds of the registry must answer identically.

    A selector that takes the first candidate would otherwise depend on dict
    insertion order, and a run would not reproduce.
    """
    context = beam_context(30.0)
    first = build_default_registry().query(context).names()
    second = build_default_registry().query(context).names()
    assert first == second
    assert len(set(first)) == len(first)


def test_a_failed_condition_is_reported_with_its_description():
    registry = MethodRegistry()
    registry.register(Method(
        name="picky", category=Category.ANALYSIS, summary="", inputs=(),
        outputs=(), fidelity=Fidelity.BEAM, cost=Cost.CHEAP,
        conditions=(Condition("the moon is full", lambda c: False),)))
    candidates = registry.query(ProblemContext())
    assert candidates.reason("picky") == ("the moon is full",)


# --- strategy selection ------------------------------------------------------

def design_context() -> ProblemContext:
    return ProblemContext(
        geometry="prismatic_beam",
        representations=("prismatic_beam", "voxel_domain"),
        slenderness=6.0, material_class="isotropic",
        has_stress_constraint=True, needs_stress_field=False)


def test_all_three_design_strategies_are_reachable():
    selector = StrategySelector(build_default_registry(), design_context())
    ladder = [m.name for m in selector.ladder()]
    assert ladder == ["parametric_section", "topology_compliance",
                      "topology_stress"]


def test_the_selector_escalates_on_a_stall_and_then_stops():
    selector = StrategySelector(build_default_registry(), design_context(),
                                stall_patience=3)
    first = selector.select(0)
    assert first.name == "parametric_section" and not first.switched

    assert selector.select(2).name == "parametric_section"      # not yet stalled

    second = selector.select(3)
    assert second.name == "topology_compliance" and second.switched
    third = selector.select(3)
    assert third.name == "topology_stress" and third.switched

    # Nothing left to escalate to: it says so rather than cycling.
    last = selector.select(3)
    assert last.name == "topology_stress"
    assert last.exhausted and not last.switched
    assert selector.state.tried == ["parametric_section", "topology_compliance",
                                    "topology_stress"]


def test_strategy_selection_is_deterministic():
    def run():
        selector = StrategySelector(build_default_registry(), design_context(),
                                    stall_patience=2)
        return [selector.select(s).name for s in (0, 0, 2, 2, 2, 2)]
    assert run() == run()


def test_a_stress_constraint_is_required_to_reach_the_stress_method():
    context = ProblemContext(
        geometry="voxel_domain", representations=("voxel_domain",),
        has_stress_constraint=False, needs_stress_field=True, slenderness=6.0)
    selector = StrategySelector(build_default_registry(), context)
    assert "topology_stress" not in [m.name for m in selector.ladder()]


def test_no_applicable_method_raises_rather_than_falling_back():
    """The whole point of the declarations is not to run something anyway."""
    context = ProblemContext(geometry="something_nobody_implemented",
                             representations=("something_nobody_implemented",))
    selector = StrategySelector(build_default_registry(), context)
    with pytest.raises(NoApplicableMethod):
        selector.select(0)


# --- routing into the loop ---------------------------------------------------

def test_a_method_with_no_executor_is_recorded_as_unmet():
    """The gap mechanism, exercised against a restricted executor set.

    When this was written the loop could only run parametric designs, so every
    topology escalation took this path. Phase 14.5 wired the topology
    executors, so with the default set nothing lands here any more; the test
    restricts the set explicitly to keep testing the mechanism rather than the
    state of the executor table. The registry is meant to grow faster than the
    executors, and the next method registered will be unmet until it is wired.

    What must never happen is the escalation being stamped into
    `strategy_used`: the design in that episode did not come from it.
    """
    import numpy as np

    from agent.reasoner import RegistryRoutingReasoner
    from agent.reasoner.base import ReasonerState

    selector = StrategySelector(build_default_registry(), design_context(),
                                stall_patience=2)
    reasoner = RegistryRoutingReasoner(selector,
                                       executable={"parametric_section"})

    settled = ReasonerState(iteration=1, best_x=np.array([0.05, 0.08, 0.003]),
                            best_mass_kg=1.0, best_feasible=True,
                            iterations_without_improvement=0,
                            evaluations_used=10, seconds_used=1.0)
    action = reasoner.decide(settled, [])
    assert action.strategy.startswith("parametric_section:")
    assert reasoner.unmet_recommendations == []

    stalled = ReasonerState(iteration=9, best_x=np.array([0.05, 0.08, 0.003]),
                            best_mass_kg=1.0, best_feasible=True,
                            iterations_without_improvement=5,
                            evaluations_used=90, seconds_used=9.0)
    escalated = reasoner.decide(stalled, [])
    assert escalated.strategy.startswith("parametric_section:")
    assert reasoner.unmet_recommendations == ["topology_compliance"]
    assert "no executor" in escalated.hypothesis
    assert "topology_compliance" in escalated.hypothesis


def test_an_escalation_the_loop_can_run_is_stamped_not_deferred():
    """The gap Phase 14.5 closed, from the reasoner's side."""
    import numpy as np

    from agent.execution import executable_methods
    from agent.reasoner import RegistryRoutingReasoner
    from agent.reasoner.base import ReasonerState

    selector = StrategySelector(build_default_registry(), design_context(),
                                stall_patience=2)
    reasoner = RegistryRoutingReasoner(selector)
    assert "topology_compliance" in reasoner.executable

    stalled = ReasonerState(iteration=9, best_x=np.array([0.05, 0.08, 0.003]),
                            best_mass_kg=1.0, best_feasible=True,
                            iterations_without_improvement=5,
                            evaluations_used=90, seconds_used=9.0)
    reasoner.decide(ReasonerState(iteration=1, best_x=stalled.best_x,
                                  best_mass_kg=1.0, best_feasible=True,
                                  iterations_without_improvement=0,
                                  evaluations_used=1, seconds_used=0.1), [])
    escalated = reasoner.decide(stalled, [])
    assert escalated.strategy.startswith("topology_compliance:")
    assert not (set(reasoner.unmet_recommendations) & executable_methods())


def test_the_routing_reasoner_stamps_a_method_it_can_execute():
    import numpy as np

    from agent.reasoner import RegistryRoutingReasoner
    from agent.reasoner.base import ReasonerState

    selector = StrategySelector(build_default_registry(), design_context(),
                                stall_patience=2)
    reasoner = RegistryRoutingReasoner(selector)
    state = ReasonerState(iteration=1, best_x=np.array([0.05, 0.08, 0.003]),
                          best_mass_kg=1.0, best_feasible=True,
                          iterations_without_improvement=0, evaluations_used=1, seconds_used=0.1)
    action = reasoner.decide(state, [])
    method, _, inner = action.strategy.partition(":")
    assert method in DEFAULT_REGISTRY
    assert inner
    assert "selected from the registry" in action.hypothesis
