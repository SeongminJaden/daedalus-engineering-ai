"""The loop running the design method the registry selected.

Before this layer the selector could recommend a topology method and the loop
would run the parametric solver regardless, recording the recommendation as
unmet. The tests that matter here are the ones asserting that an episode's
`strategy_used` is now TRUE: that the design in the record actually came from
the method the record names.
"""

import numpy as np
import pytest

from agent.execution import DesignOutcome, NoExecutor, execute, executable_methods
from agent.execution.topology import COMPLIANCE_METHOD, STRESS_METHOD, design_domain
from agent.loop.engine import DesignLoop, LoopConfig
from agent.reasoner import RegistryRoutingReasoner
from agent.strategy import StrategySelector
from core.registry import DEFAULT_REGISTRY, ProblemContext
from optimization.constraints import build_optimization_problem
from projects.robotic_link.problem import build_mvp_problem

FAST_TOPOLOGY = {"iterations": 8, "bisection_steps": 2, "divisions": (8, 4, 1)}


@pytest.fixture(scope="module")
def op():
    return build_optimization_problem(build_mvp_problem())


def routing_context(op) -> ProblemContext:
    geometry = op.problem.geometry
    return ProblemContext(
        geometry="prismatic_beam",
        representations=("prismatic_beam", "voxel_domain"),
        slenderness=geometry.length_m / geometry.max_height_m,
        material_class="isotropic", has_stress_constraint=True,
        needs_stress_field=False)


# --- provenance --------------------------------------------------------------

def test_an_outcome_carries_exactly_one_representation():
    """The invariant that makes `strategy_used` checkable.

    A parametric result is a design vector and a topology result is a density
    field. An outcome with neither produced nothing; one with both has
    ambiguous provenance, and the episode log built from it could not be
    audited against the method it names.
    """
    with pytest.raises(ValueError):
        DesignOutcome(method="m", mass_kg=1.0, feasible=True)
    with pytest.raises(ValueError):
        DesignOutcome(method="m", mass_kg=1.0, feasible=True,
                      design_vector=np.zeros(3), density_field=np.zeros(4))

    vector = DesignOutcome(method="m", mass_kg=1.0, feasible=True,
                           design_vector=np.zeros(3))
    field = DesignOutcome(method="m", mass_kg=1.0, feasible=True,
                          density_field=np.zeros(4))
    assert vector.representation == "design_vector"
    assert field.representation == "density_field"


def test_an_unknown_method_is_refused_not_defaulted(op):
    """Falling back would file a design under another method's name."""
    with pytest.raises(NoExecutor):
        execute("no_such_method", op)


def test_the_five_registry_strategies_are_all_executable():
    """The three from before, the generative CAD strategy added with the
    knowledge layer, and the free form topology strategy that needs no family
    at all; the routing reasoner reads this set rather than assuming it."""
    assert executable_methods() == {"parametric_section", COMPLIANCE_METHOD,
                                    STRESS_METHOD, "generative_cad",
                                    "freeform_topology"}


# --- the executors -----------------------------------------------------------

def test_the_topology_executor_returns_a_density_field(op):
    outcome = execute(COMPLIANCE_METHOD, op, **FAST_TOPOLOGY)
    assert outcome.method == COMPLIANCE_METHOD
    assert outcome.density_field is not None
    assert outcome.design_vector is None
    mesh = design_domain(op.problem, FAST_TOPOLOGY["divisions"])
    assert outcome.density_field.size == mesh.n_elements
    assert outcome.mass_kg > 0.0


def test_the_parametric_executor_returns_a_design_vector(op):
    outcome = execute("parametric_section", op, max_iter=30)
    assert outcome.design_vector is not None
    assert outcome.density_field is None
    assert outcome.design_vector.shape == (3,)


def test_the_two_strategies_produce_different_designs(op):
    """Different design spaces, so the loop has something to choose between."""
    parametric = execute("parametric_section", op, max_iter=30)
    topology = execute(COMPLIANCE_METHOD, op, **FAST_TOPOLOGY)
    assert parametric.representation != topology.representation
    assert parametric.mass_kg != pytest.approx(topology.mass_kg)


def test_a_topology_run_needs_a_stated_design_envelope(op):
    """Inventing a design domain would silently change the problem."""
    problem = op.problem.model_copy(deep=True)
    problem.geometry.max_height_m = None
    with pytest.raises(ValueError, match="envelope"):
        design_domain(problem)


# --- the loop ----------------------------------------------------------------

@pytest.fixture(scope="module")
def routed_run(op):
    selector = StrategySelector(DEFAULT_REGISTRY, routing_context(op),
                               stall_patience=1)
    reasoner = RegistryRoutingReasoner(selector)
    config = LoopConfig(max_iterations=4, seed=0, local_max_iter=40,
                        topology_options=FAST_TOPOLOGY)
    result = DesignLoop(op, config, reasoner=reasoner).run()
    return reasoner, result


def test_the_loop_actually_executes_a_topology_strategy(routed_run):
    _, result = routed_run
    methods = {e.design_genome["method"] for e in result.episodes}
    assert any(m in (COMPLIANCE_METHOD, STRESS_METHOD) for m in methods), (
        f"no topology strategy ran; methods were {sorted(methods)}")


def test_strategy_used_names_the_method_that_made_the_design(routed_run):
    """Truthfulness of the record, checked against the design itself.

    A topology episode has to carry a density field and a parametric one a
    design vector. The representation is produced by the executor, so it cannot
    agree with the recorded method unless that method genuinely ran.
    """
    _, result = routed_run
    for episode in result.episodes:
        genome = episode.design_genome
        named = episode.strategy_used.split(":")[0]
        assert genome["method"] == named
        if named in (COMPLIANCE_METHOD, STRESS_METHOD):
            assert genome["representation"] == "density_field"
            assert "volume_fraction" in genome
            assert "wall_thickness_m" not in genome
        else:
            assert genome["representation"] == "design_vector"
            assert "wall_thickness_m" in genome


def test_no_executable_recommendation_goes_unmet(routed_run):
    """The gap this phase closed, asserted as a number."""
    reasoner, _ = routed_run
    unmet = set(reasoner.unmet_recommendations)
    assert not (unmet & executable_methods()), (
        f"the loop can execute {sorted(unmet & executable_methods())} but "
        f"recorded them as unmet")


def test_the_loop_tracks_the_best_across_methods(routed_run):
    _, result = routed_run
    feasible = [e for e in result.episodes if e.feasible]
    assert feasible
    best = min(e.observation["mass_kg"] for e in feasible)
    assert result.best_mass_kg == pytest.approx(best, rel=1e-9)


def test_a_routed_run_is_reproducible(op):
    def once():
        selector = StrategySelector(DEFAULT_REGISTRY, routing_context(op),
                                    stall_patience=1)
        loop = DesignLoop(op, LoopConfig(max_iterations=3, seed=5,
                                         local_max_iter=30,
                                         topology_options=FAST_TOPOLOGY),
                          reasoner=RegistryRoutingReasoner(selector))
        result = loop.run()
        return [(e.strategy_used, round(e.observation["mass_kg"], 9))
                for e in result.episodes]
    assert once() == once()
