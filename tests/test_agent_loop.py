"""Phase 4 verification: the autonomous design loop.

The load-bearing check is convergence agreement: the loop reaches the same
optimum as calling the Phase 3 optimizer directly, by a different route
(multi-start with an explore/exploit policy). Everything else verifies that
each termination condition actually fires and that episodes are recorded in a
schema Phase 5 can read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.evaluator import judge  # noqa: E402
from agent.experiment_manager import ComputeBudget, Episode, EpisodeLog  # noqa: E402
from agent.loop import (  # noqa: E402
    PHASE_CYCLE, DesignLoop, LoopConfig, LoopPhase, TerminationReason,
)
from agent.reasoner import (  # noqa: E402
    Action, ActionKind, HeuristicReasoner, Reasoner, ReasonerState,
)
from optimization.constraints import build_optimization_problem, evaluate_design  # noqa: E402
from optimization.gradient import optimize_slsqp  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402

FAST = dict(local_max_iter=30)


@pytest.fixture(scope="module")
def op():
    return build_optimization_problem(build_mvp_problem())


@pytest.fixture(scope="module")
def direct_optimum(op):
    return optimize_slsqp(op).mass_kg


# --------------------------------------------------------------------------- #
# 2. convergence agreement - the core check
# --------------------------------------------------------------------------- #
def test_loop_converges_to_the_direct_optimum(op, direct_optimum):
    """The loop and a single direct solve must land on the same design."""
    result = DesignLoop(op, LoopConfig(max_iterations=5, seed=1, **FAST)).run()
    assert result.best_evaluation is not None
    rel = abs(result.best_mass_kg - direct_optimum) / direct_optimum
    assert rel < 1e-3, (
        f"loop {result.best_mass_kg:.6f} kg vs direct {direct_optimum:.6f} kg "
        f"(rel {rel:.3e})"
    )


def test_loop_best_is_feasible_and_deflection_limited(op):
    result = DesignLoop(op, LoopConfig(max_iterations=4, seed=1, **FAST)).run()
    assert result.best_evaluation.is_feasible()
    assert "deflection" in result.best_evaluation.active_constraints()


def test_loop_improves_on_the_baseline(op):
    baseline = evaluate_design(op, np.array([0.05, 0.08, 0.005])).mass_kg
    result = DesignLoop(op, LoopConfig(max_iterations=3, seed=1, **FAST)).run()
    assert result.best_mass_kg < baseline * 0.5


# --------------------------------------------------------------------------- #
# 1 & 5 & 6. every termination condition fires
# --------------------------------------------------------------------------- #
def test_terminates_on_target_reached(op):
    result = DesignLoop(
        op, LoopConfig(max_iterations=10, seed=0, target_mass_kg=0.5, **FAST)
    ).run()
    assert result.termination is TerminationReason.TARGET_REACHED
    assert result.best_mass_kg <= 0.5
    assert "target" in result.termination_detail


def test_terminates_on_evaluation_budget(op):
    result = DesignLoop(
        op, LoopConfig(max_iterations=20, seed=0, max_evaluations=5, **FAST)
    ).run()
    assert result.termination is TerminationReason.COMPUTE_BUDGET_EXCEEDED
    assert "evaluation budget" in result.termination_detail
    assert result.budget["evaluations"] >= 5


def test_terminates_on_time_budget(op):
    result = DesignLoop(
        op, LoopConfig(max_iterations=20, seed=0, max_seconds=0.0, **FAST)
    ).run()
    assert result.termination is TerminationReason.COMPUTE_BUDGET_EXCEEDED
    assert "time budget" in result.termination_detail
    assert result.iterations == 0        # fired before spending anything


def test_terminates_on_convergence(op):
    result = DesignLoop(op, LoopConfig(
        max_iterations=30, seed=0, convergence_patience=2,
        convergence_epsilon=1e-3, **FAST)).run()
    assert result.termination is TerminationReason.CONVERGED
    assert "consecutive iterations" in result.termination_detail
    assert result.iterations < 30


def test_terminates_on_unsatisfiable_constraints():
    """A deflection cap no geometry in the box can meet must be detected as
    unsatisfiable, not silently returned as a best-effort design."""
    problem = build_mvp_problem()
    problem.constraints.max_deflection_m = 1e-9     # 1 nanometre
    op = build_optimization_problem(problem)
    result = DesignLoop(op, LoopConfig(
        max_iterations=10, seed=0, unsatisfiable_after=2, **FAST)).run()
    assert result.termination is TerminationReason.CONSTRAINTS_UNSATISFIABLE
    assert result.best_evaluation is None
    assert "no feasible design" in result.termination_detail


def test_terminates_on_user_stop(op):
    result = DesignLoop(
        op, LoopConfig(max_iterations=10, seed=0, **FAST), stop_flag=lambda: True
    ).run()
    assert result.termination is TerminationReason.USER_STOP
    assert result.iterations == 0


def test_stop_flag_mid_run(op):
    """Stop after the first iteration, not before the run."""
    calls = {"n": 0}

    def flag():
        calls["n"] += 1
        return calls["n"] > 1

    result = DesignLoop(
        op, LoopConfig(max_iterations=10, seed=0, **FAST), stop_flag=flag
    ).run()
    assert result.termination is TerminationReason.USER_STOP
    assert result.iterations == 1


def test_reasoner_requested_stop(op):
    class StopReasoner(Reasoner):
        name = "always-stop"

        def decide(self, state, history):
            return Action(kind=ActionKind.STOP, hypothesis="done",
                          strategy="test-stop")

    result = DesignLoop(
        op, LoopConfig(max_iterations=5, **FAST), reasoner=StopReasoner()
    ).run()
    assert result.termination is TerminationReason.USER_STOP
    assert result.iterations == 0


def test_terminates_on_max_iterations(op):
    result = DesignLoop(op, LoopConfig(
        max_iterations=1, seed=0, convergence_patience=999, **FAST)).run()
    assert result.termination is TerminationReason.MAX_ITERATIONS
    assert result.iterations == 1


# --------------------------------------------------------------------------- #
# 3. determinism
# --------------------------------------------------------------------------- #
def test_run_is_reproducible_for_a_seed(op):
    cfg = dict(max_iterations=3, seed=5, **FAST)
    a = DesignLoop(op, LoopConfig(**cfg)).run()
    b = DesignLoop(op, LoopConfig(**cfg)).run()

    assert a.termination is b.termination
    assert a.iterations == b.iterations
    assert a.best_mass_kg == b.best_mass_kg
    assert np.array_equal(a.best_x, b.best_x)
    for ea, eb in zip(a.episodes, b.episodes):
        assert ea.action == eb.action
        assert ea.strategy_used == eb.strategy_used
        assert ea.observation == eb.observation


def test_different_seeds_take_different_paths(op):
    a = DesignLoop(op, LoopConfig(max_iterations=4, seed=1, **FAST)).run()
    b = DesignLoop(op, LoopConfig(max_iterations=4, seed=3, **FAST)).run()
    assert ([e.strategy_used for e in a.episodes]
            != [e.strategy_used for e in b.episodes])


# --------------------------------------------------------------------------- #
# 4. episodes
# --------------------------------------------------------------------------- #
def test_episode_fields_are_populated(op):
    result = DesignLoop(op, LoopConfig(max_iterations=2, seed=1, **FAST)).run()
    for e in result.episodes:
        assert e.id and e.timestamp and e.hypothesis and e.conclusion
        assert e.action in ("exploit", "explore")
        assert e.strategy_used
        assert 0.0 <= e.confidence <= 1.0
        for key in ("outer_width_m", "outer_height_m", "wall_thickness_m",
                    "material_id"):
            assert key in e.design_genome
        for key in ("mass_kg", "max_bending_stress_pa", "tip_deflection_m",
                    "safety_factor", "first_natural_frequency_hz"):
            assert key in e.observation
        assert "active" in e.constraint_status
        assert e.evaluations > 0


def test_episode_lineage_is_recorded(op):
    result = DesignLoop(op, LoopConfig(max_iterations=3, seed=1, **FAST)).run()
    assert result.episodes[0].parent_design_id is None
    for prev, cur in zip(result.episodes, result.episodes[1:]):
        assert cur.parent_design_id == prev.id


def test_episode_jsonl_round_trip(op, tmp_path):
    path = tmp_path / "episodes.jsonl"
    log = EpisodeLog(path)
    result = DesignLoop(
        op, LoopConfig(max_iterations=2, seed=1, **FAST), episode_log=log
    ).run()

    loaded = EpisodeLog.read(path)
    assert len(loaded) == len(result.episodes)
    for written, read_back in zip(result.episodes, loaded):
        assert read_back == written


def test_corrupt_episode_line_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x", "not_a_real_field": 1}\n')
    with pytest.raises(ValueError, match="not a valid episode"):
        EpisodeLog.read(path)


def test_episode_rejects_out_of_range_confidence():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Episode(id="a", iteration=0, timestamp=Episode.now_iso(),
                hypothesis="h", action="exploit", strategy_used="s",
                conclusion="c", confidence=1.5)


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #
def test_budget_reads_the_profile():
    budget = ComputeBudget.from_profile("laptop_4gb")
    assert budget.max_evaluations == 20000
    assert budget.max_seconds == 300
    assert ComputeBudget.from_profile("cloud_a100").max_evaluations == 500000


def test_budget_reports_which_limit_was_hit():
    budget = ComputeBudget(max_evaluations=10, max_seconds=1e9)
    assert not budget.exceeded()
    budget.spend(10)
    assert budget.exceeded()
    assert "evaluation budget" in budget.reason()


def test_budget_rejects_negative_spend():
    with pytest.raises(ValueError):
        ComputeBudget(max_evaluations=10, max_seconds=10).spend(-1)


# --------------------------------------------------------------------------- #
# reasoner policy
# --------------------------------------------------------------------------- #
def _state(**kw):
    base = dict(
        iteration=1, best_x=np.array([0.01, 0.08, 0.001]), best_mass_kg=0.25,
        best_feasible=True, iterations_without_improvement=0,
        evaluations_used=0, seconds_used=0.0,
        lower=np.array([0.01, 0.01, 0.001]), upper=np.array([0.1, 0.1, 0.02]),
    )
    base.update(kw)
    return ReasonerState(**base)


def test_first_iteration_exploits():
    action = HeuristicReasoner(seed=0).decide(_state(best_x=None, best_mass_kg=None,
                                                     best_feasible=False), [])
    assert action.kind is ActionKind.EXPLOIT
    assert action.start_x is None


def test_stall_forces_exploration():
    action = HeuristicReasoner(seed=0, patience=2).decide(
        _state(iterations_without_improvement=3), [])
    assert action.kind is ActionKind.EXPLORE
    assert action.strategy == "explore-on-stall"


def test_explore_probability_extremes():
    always = HeuristicReasoner(seed=0, explore_probability=1.0)
    never = HeuristicReasoner(seed=0, explore_probability=0.0)
    assert always.decide(_state(), []).kind is ActionKind.EXPLORE
    assert never.decide(_state(), []).kind is ActionKind.EXPLOIT


def test_random_starts_stay_in_bounds_and_valid():
    reasoner = HeuristicReasoner(seed=2, explore_probability=1.0)
    state = _state()
    for _ in range(200):
        x = reasoner.decide(state, []).start_x
        assert np.all(x >= state.lower) and np.all(x <= state.upper)
        assert x[2] < min(x[0], x[1]) / 2.0


def test_jittered_exploit_starts_stay_valid():
    reasoner = HeuristicReasoner(seed=3, explore_probability=0.0, jitter=0.8)
    state = _state()
    for _ in range(200):
        x = reasoner.decide(state, []).start_x
        assert np.all(x >= state.lower) and np.all(x <= state.upper)
        assert x[2] < min(x[0], x[1]) / 2.0


def test_reasoner_rejects_bad_settings():
    with pytest.raises(ValueError):
        HeuristicReasoner(explore_probability=1.5)
    with pytest.raises(ValueError):
        HeuristicReasoner(patience=0)


def test_reasoner_is_pluggable(op):
    """The loop must accept any Reasoner - this is the LLM extension point."""
    class AlwaysExplore(Reasoner):
        name = "always-explore"

        def __init__(self):
            self.inner = HeuristicReasoner(seed=0, explore_probability=1.0)

        def decide(self, state, history):
            if state.best_x is None:
                return Action(kind=ActionKind.EXPLOIT, hypothesis="seed",
                              strategy="custom-seed")
            return self.inner.decide(state, history)

    result = DesignLoop(
        op, LoopConfig(max_iterations=3, **FAST), reasoner=AlwaysExplore()
    ).run()
    assert [e.strategy_used for e in result.episodes[1:]] == [
        "explore-scheduled"] * (result.iterations - 1)


# --------------------------------------------------------------------------- #
# state machine + verdicts
# --------------------------------------------------------------------------- #
def test_phase_cycle_matches_the_documented_state_machine():
    assert [p.value for p in PHASE_CYCLE] == [
        "observe", "reason", "plan", "design", "simulate", "evaluate",
        "learn", "update_brain",
    ]


def test_loop_ends_in_done_phase(op):
    loop = DesignLoop(op, LoopConfig(max_iterations=1, **FAST))
    loop.run()
    assert loop.state.phase is LoopPhase.DONE


def test_judge_rejects_infeasible(op):
    ev = evaluate_design(op, np.array([0.01, 0.01, 0.0012]))
    verdict = judge(ev, None, True)
    assert not verdict.feasible and not verdict.is_new_best
    assert "infeasible" in verdict.conclusion


def test_judge_accepts_first_feasible(op):
    ev = evaluate_design(op, np.array([0.05, 0.08, 0.005]))
    verdict = judge(ev, None, True)
    assert verdict.feasible and verdict.is_new_best


def test_judge_keeps_incumbent_without_improvement(op):
    ev = evaluate_design(op, np.array([0.05, 0.08, 0.005]))
    verdict = judge(ev, ev.mass_kg, True)
    assert verdict.feasible and not verdict.is_new_best
    assert "no improvement" in verdict.conclusion
