"""A reasoner that routes design generation through the method registry.

Still a deterministic heuristic, not a language model. It delegates the
explore/exploit schedule to an inner reasoner and adds one thing: the design
method comes from a registry query rather than from a name written into the
code. An LLM-backed policy would replace the ordering rule and route over the
same registry.

**It records what it ran, not what it wanted to run.** The loop engine
evaluates parametric section designs; it cannot yet execute a topology method.
When the selector escalates to a method the loop has no way to run, the episode
must not be stamped with that method's name, because the design in that episode
did not come from it and the log would be a false record. The recommendation is
written into the hypothesis instead, and counted, so the gap between what the
registry can route to and what the loop can execute is a visible number rather
than a silent limitation.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from agent.reasoner.base import Action, Reasoner, ReasonerState
from agent.reasoner.heuristic import HeuristicReasoner
from agent.strategy import StrategySelector

# What the Phase 4 loop engine can actually evaluate today.
LOOP_EXECUTABLE_METHODS = frozenset({"parametric_section"})


class RegistryRoutingReasoner(Reasoner):
    """Wraps a reasoner, choosing the design method from the registry."""

    name = "registry-routing"

    def __init__(self, selector: StrategySelector,
                 inner: Reasoner | None = None,
                 executable: Iterable[str] = LOOP_EXECUTABLE_METHODS):
        self.selector = selector
        self.inner = inner if inner is not None else HeuristicReasoner()
        self.executable = frozenset(executable)
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
                f"'{choice.name}' ({choice.reason}), which this loop cannot "
                f"execute; it evaluates parametric section designs only. "
                f"Continuing with '{fallback}' and recording the "
                f"recommendation as unmet."),
            strategy=f"{fallback}:{action.strategy}",
            start_x=action.start_x)
