"""A reasoner that routes design generation through the method registry.

Still a deterministic heuristic, not a language model. It delegates the
explore/exploit schedule to an inner reasoner and adds one thing: the design
method comes from a registry query rather than from a name written into the
code. An LLM-backed policy would replace the ordering rule and route over the
same registry.

**It records what it ran, not what it wanted to run.** The executable set is
whatever `agent.execution` can dispatch, and a method outside it is never
stamped into `strategy_used`: the design in that episode did not come from it,
and the log would be a false record. Such a recommendation goes into the
hypothesis and is counted, so the gap between what the registry routes to and
what the loop executes is a visible number rather than a silent limitation.

That gap was the whole story when this was written: the loop could only run
parametric section designs, so every topology recommendation was unmet. The
execution layer now dispatches the topology strategies too, and the count is
zero for them. The mechanism stays because the registry is meant to grow faster
than the executors, and the next method added will be unmet until someone wires
it.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from agent.reasoner.base import Action, Reasoner, ReasonerState
from agent.reasoner.heuristic import HeuristicReasoner
from agent.strategy import StrategySelector

def _default_executable() -> frozenset[str]:
    """What the loop can actually dispatch, asked rather than assumed.

    Read from the execution layer so the two cannot disagree. A hardcoded list
    here would go stale the moment an executor is added or removed, and it
    would go stale silently in the direction that produces false episode
    records.
    """
    from agent.execution import executable_methods

    return executable_methods()


class RegistryRoutingReasoner(Reasoner):
    """Wraps a reasoner, choosing the design method from the registry."""

    name = "registry-routing"

    def __init__(self, selector: StrategySelector,
                 inner: Reasoner | None = None,
                 executable: Iterable[str] | None = None):
        self.selector = selector
        self.inner = inner if inner is not None else HeuristicReasoner()
        self.executable = frozenset(executable if executable is not None
                                    else _default_executable())
        if not self.executable:
            raise ValueError("at least one method must be executable")
        self.unmet_recommendations: list[str] = []

    def decide(self, state: ReasonerState, history: Sequence) -> Action:
        action = self.inner.decide(state, history)
        choice = self.selector.select(state.iterations_without_improvement)

        if choice.name in self.executable:
            return Action(
                kind=action.kind,
                hypothesis=(f"{action.hypothesis} Design method "
                            f"'{choice.name}' selected from the registry: "
                            f"{choice.reason}."),
                strategy=f"{choice.name}:{action.strategy}",
                start_x=action.start_x)

        # Escalated past what the loop can run. Report it, do not claim it.
        self.unmet_recommendations.append(choice.name)
        fallback = sorted(self.executable)[0]
        return Action(
            kind=action.kind,
            hypothesis=(
                f"{action.hypothesis} The registry recommends escalating to "
                f"'{choice.name}' ({choice.reason}), which this loop has no "
                f"executor for. Continuing with '{fallback}' and recording the "
                f"recommendation as unmet."),
            strategy=f"{fallback}:{action.strategy}",
            start_x=action.start_x)
