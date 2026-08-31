"""Generalising which registered method wins on which class of problem.

The claim is about models, not about the world, and the evidence ladder has to
keep saying so no matter how much simulation agrees.
"""

import pytest

from brain import Brain
from brain.semantic.evidence import EvidenceKind, EvidenceLevel
from brain.strategy import derive_method_strategies


def samples_favouring(winner: str, runs: int = 4, loser: str = "parametric_section",
                      problem_class: str = "bracket_stress") -> list[dict]:
    rows = []
    for run in range(runs):
        rows.append(dict(ref=f"e{run}-win", run_id=f"r{run}",
                         problem_class=problem_class, strategy_used=winner,
                         score=1.0 + 0.01 * run, feasible=True))
        rows.append(dict(ref=f"e{run}-lose", run_id=f"r{run}",
                         problem_class=problem_class, strategy_used=loser,
                         score=2.0, feasible=True))
    return rows


def test_a_consistent_winner_is_promoted():
    promoted = derive_method_strategies(samples_favouring("topology_stress"))
    assert len(promoted) == 1
    strategy = promoted[0]
    assert strategy.context == {"problem_class": "bracket_stress"}
    assert "topology_stress" in strategy.statement
    assert "4 of 4" in strategy.statement


def test_nothing_is_promoted_without_a_consistent_winner():
    """No recommendation is the correct output when the data has none."""
    rows = []
    for run in range(4):
        winner = "topology_stress" if run % 2 == 0 else "parametric_section"
        loser = "parametric_section" if run % 2 == 0 else "topology_stress"
        rows.append(dict(ref=f"a{run}", run_id=f"r{run}",
                         problem_class="mixed", strategy_used=winner,
                         score=1.0, feasible=True))
        rows.append(dict(ref=f"b{run}", run_id=f"r{run}",
                         problem_class="mixed", strategy_used=loser,
                         score=2.0, feasible=True))
    assert derive_method_strategies(rows) == []


def test_infeasible_episodes_do_not_count():
    """A design that violates its constraints has no score worth comparing."""
    rows = samples_favouring("topology_stress", runs=2)
    for row in rows:
        row["feasible"] = False
    assert derive_method_strategies(rows) == []


def test_wins_are_counted_per_run_not_per_episode():
    """Otherwise a strategy that simply ran more often would look better.

    Here `parametric_section` wins the only run outright, while
    `topology_stress` appears in six losing episodes. Counting episodes would
    hand it the majority; counting runs gives the right answer.
    """
    rows = [dict(ref="win", run_id="r0", problem_class="c",
                 strategy_used="parametric_section", score=0.5, feasible=True)]
    rows += [dict(ref=f"l{i}", run_id="r0", problem_class="c",
                  strategy_used="topology_stress", score=1.0 + i, feasible=True)
             for i in range(6)]
    promoted = derive_method_strategies(rows)
    assert len(promoted) == 1
    assert "parametric_section" in promoted[0].statement


def test_the_evidence_is_simulation_and_can_never_be_experimental():
    """The gate simulation cannot open, however many runs agree.

    EXPERIMENTALLY_VALIDATED requires physical-test evidence. Nothing in this
    path can produce any, so no amount of agreement between simulated runs may
    reach it.
    """
    for runs in (1, 4, 50):
        strategy, = derive_method_strategies(
            samples_favouring("topology_stress", runs=runs))
        assert all(e.kind is EvidenceKind.SIMULATION for e in strategy.evidence)
        assert strategy.evidence_level is not EvidenceLevel.EXPERIMENTALLY_VALIDATED
        assert "says nothing about physical behaviour" in strategy.statement


def test_a_single_run_is_only_simulated():
    """One run is one run. The level rises only as independent runs accumulate."""
    strategy, = derive_method_strategies(
        samples_favouring("topology_stress", runs=1))
    assert strategy.evidence_level is EvidenceLevel.SIMULATED


def test_derivation_is_deterministic():
    rows = samples_favouring("topology_stress")
    first = [s.statement for s in derive_method_strategies(rows)]
    second = [s.statement for s in derive_method_strategies(list(reversed(rows)))]
    assert first == second


def test_a_promoted_strategy_round_trips_through_the_brain(tmp_path):
    brain = Brain(tmp_path / "brain.db")
    try:
        derive_method_strategies(samples_favouring("topology_stress"),
                                 store=brain.strategies)
        stored = brain.strategies.get("method-for-class:bracket_stress")
        assert stored is not None
        assert "topology_stress" in stored.statement
        found = brain.strategies.applicable({"problem_class": "bracket_stress"})
        assert [s.name for s in found] == ["method-for-class:bracket_stress"]
        assert brain.strategies.applicable({"problem_class": "something_else"}) == []
    finally:
        brain.db.close()
