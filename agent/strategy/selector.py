"""Choosing a design-generation strategy by asking the method registry.

This is a **deterministic heuristic, not a language model.** It reads the
registry's declarations and follows a fixed rule. Saying otherwise would be the
same overclaim the Phase 4 reasoner docstring warns about.

What matters architecturally is that the rule routes over *declared* method
metadata rather than a hardcoded switch over three known names. The selector
never mentions `parametric_section` or `topology_stress`; it asks which
design-generation methods apply to this problem and orders them. Registering a
new method makes it reachable here with no change to this file, and an
LLM-backed selector would replace the ordering rule while routing over exactly
the same registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.registry import (Candidates, Category, Method, MethodRegistry,
                           ProblemContext)

# How many non-improving iterations before the selector gives up on the
# current strategy. Small enough that a run does not spend its whole budget on
# a strategy that has stopped paying, large enough that ordinary noise in a
# stochastic search does not trigger a switch.
DEFAULT_STALL_PATIENCE = 3


@dataclass(frozen=True)
class StrategyChoice:
    """Which strategy to use next, and why.

    `reason` is recorded in the episode log, so a run reads back as a sequence
    of routing decisions rather than a sequence of names.
    """

    method: Method
    reason: str
    switched: bool = False
    exhausted: bool = False

    @property
    def name(self) -> str:
        return self.method.name


@dataclass
class SelectorState:
    """What the selector tracks across a run."""

    iterations_without_improvement: int = 0
    current: str | None = None
    tried: list[str] = field(default_factory=list)


class NoApplicableMethod(RuntimeError):
    """No registered design-generation method applies to this problem.

    Raised rather than defaulted. Falling back to some method anyway is exactly
    the failure the applicability declarations exist to prevent.
    """


class StrategySelector:
    """Routes design generation over the registry, escalating when stalled."""

    def __init__(self, registry: MethodRegistry, context: ProblemContext,
                 stall_patience: int = DEFAULT_STALL_PATIENCE):
        self.registry = registry
        self.context = context
        self.stall_patience = stall_patience
        self.state = SelectorState()

    def candidates(self) -> Candidates:
        """Design-generation methods, split into applicable and excluded."""
        return self.registry.query(self.context, Category.DESIGN_GENERATION)

    def ladder(self) -> tuple[Method, ...]:
        """The escalation order: cheapest first, then by descending fidelity.

        Cheap methods run first because a run that can be settled cheaply
        should be. The name is the final tiebreak so the ladder is identical
        across processes; ordering that falls through to dict insertion order
        would make a run irreproducible for no reason.
        """
        applicable = self.candidates().applicable
        return tuple(sorted(applicable,
                            key=lambda m: (int(m.cost), -int(m.fidelity), m.name)))

    def select(self, iterations_without_improvement: int = 0) -> StrategyChoice:
        """Pick the strategy for the next iteration."""
        ladder = self.ladder()
        if not ladder:
            excluded = self.candidates().excluded
            detail = "; ".join(f"{e.method.name}: {', '.join(e.failed)}"
                               for e in excluded) or "the registry is empty"
            raise NoApplicableMethod(
                f"no design-generation method applies to this problem. {detail}")

        self.state.iterations_without_improvement = iterations_without_improvement

        if self.state.current is None:
            chosen = ladder[0]
            self.state.current = chosen.name
            self.state.tried.append(chosen.name)
            return StrategyChoice(
                method=chosen,
                reason=(f"cheapest applicable design-generation method "
                        f"(cost {chosen.cost.name.lower()}, fidelity "
                        f"{chosen.fidelity.name.lower()})"))

        names = [m.name for m in ladder]
        if self.state.current not in names:
            # The problem context changed under us and the current method no
            # longer applies. Restart the ladder rather than carry on with a
            # method the registry has just ruled out.
            chosen = ladder[0]
            self.state.current = chosen.name
            if chosen.name not in self.state.tried:
                self.state.tried.append(chosen.name)
            return StrategyChoice(
                method=chosen, switched=True,
                reason="the previous strategy no longer applies to this problem")

        index = names.index(self.state.current)
        if iterations_without_improvement < self.stall_patience:
            return StrategyChoice(
                method=ladder[index],
                reason=(f"no stall ({iterations_without_improvement} of "
                        f"{self.stall_patience} non-improving iterations)"))

        if index + 1 >= len(ladder):
            return StrategyChoice(
                method=ladder[index], exhausted=True,
                reason=(f"stalled after {iterations_without_improvement} "
                        f"non-improving iterations, and this is the last "
                        f"applicable method: nothing further to escalate to"))

        chosen = ladder[index + 1]
        self.state.current = chosen.name
        if chosen.name not in self.state.tried:
            self.state.tried.append(chosen.name)
        return StrategyChoice(
            method=chosen, switched=True,
            reason=(f"stalled after {iterations_without_improvement} "
                    f"non-improving iterations, escalating from "
                    f"{ladder[index].name} to a higher-fidelity method"))
