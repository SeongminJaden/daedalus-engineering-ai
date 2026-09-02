"""The engineering review: what was designed, what governs, and what is shaky.

A verdict is not a review. A number saying the assembly passes at 1.13 tells a
reader nothing about which check produced it, what that check assumed, or what
was never looked at. This assembles the four things a reviewer actually needs:

  * the governing constraint, because that is what sizes the design and what to
    attack if it needs to be better,
  * the most load-bearing ASSUMPTION in the chain, which is the assumption
    behind the governing check rather than the most alarming one in general,
  * every failure mode that was not assessed, listed rather than omitted,
  * what to verify next, in the order that would change the answer most.
"""

from __future__ import annotations

from dataclasses import dataclass

from .checks import AssemblyStatus, AssemblyVerdict, CheckResult


@dataclass(frozen=True)
class Review:
    """A structured engineering review of one assembly verdict."""

    status: AssemblyStatus
    governing: CheckResult | None
    weakest_assumption: str
    unassessed: tuple[str, ...]
    failures: tuple[str, ...]
    recommendations: tuple[str, ...]
    # Modes a surrogate ranked and no solver has run. Listed separately from
    # the unassessed ones because the remedy differs: these have a candidate
    # answer waiting to be checked, those have nothing.
    screened: tuple[str, ...] = ()

    @property
    def headline(self) -> str:
        if self.governing is None:
            return f"{self.status.value}: nothing was assessed"
        return (f"{self.status.value}: {self.governing.component} "
                f"{self.governing.failure_mode} governs at "
                f"{self.governing.safety_factor:.3f}")


def _recommendations(verdict: AssemblyVerdict) -> tuple[str, ...]:
    """What to do next, ordered by how much it would change the answer."""
    lines: list[str] = []
    for failure in verdict.failures():
        lines.append(
            f"FIX {failure.component} {failure.failure_mode}: at "
            f"{failure.safety_factor:.3f} it does not meet its requirement, "
            f"and the assembly cannot pass until it does")

    governing = verdict.governing()
    if governing is not None and governing.status.value == "passed":
        lines.append(
            f"the design is sized by {governing.component} "
            f"{governing.failure_mode}; improving anything else buys nothing "
            f"until that is relieved")
        if governing.optimistic_assumption:
            lines.append(
                f"VERIFY FIRST the assumption behind it: "
                f"{governing.optimistic_assumption}")

    gaps = verdict.unassessed()
    if gaps:
        lines.append(
            f"{len(gaps)} failure modes were not assessed at all and are "
            f"listed below. None of them is known to be satisfied")
    screened = verdict.screened()
    if screened:
        lines.append(
            f"{len(screened)} failure modes were only screened by a surrogate "
            f"and are listed below. A prediction is not a verdict: run the "
            f"solver on each before concluding anything")
    lines.append(
        "every result here is SIMULATED. Nothing has been physically tested, "
        "and the idealisations of the individual checks multiply, so the real "
        "margin is below the governing factor by an amount none of this "
        "estimates")
    return tuple(lines)


def review(verdict: AssemblyVerdict) -> Review:
    """Build the review from a verdict."""
    governing = verdict.governing()
    weakest = (governing.optimistic_assumption if governing is not None
               else "nothing was assessed, so there is no assumption to name")
    return Review(
        status=verdict.status,
        governing=governing,
        weakest_assumption=weakest or "none recorded for the governing check",
        unassessed=tuple(f"{r.component}/{r.failure_mode}"
                         for r in verdict.unassessed()),
        failures=tuple(f"{r.component}/{r.failure_mode} at "
                       f"{r.safety_factor:.3f}" for r in verdict.failures()),
        recommendations=_recommendations(verdict),
        screened=tuple(f"{r.component}/{r.failure_mode}"
                       for r in verdict.screened()),
    )
