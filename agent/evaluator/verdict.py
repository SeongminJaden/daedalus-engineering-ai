"""agent.evaluator.verdict - judge one experiment's result.

Separated from the physics so that "did this design win" is a policy decision
recorded in the episode log, not something buried in a solver callback.
"""

from __future__ import annotations

from dataclasses import dataclass

from optimization.constraints import Evaluation

# Relative mass improvement below this is noise, not progress. The fp32 kernel
# gives ~1e-7 relative precision, so anything under ~1e-6 is meaningless.
IMPROVEMENT_EPSILON = 1e-6


@dataclass
class Verdict:
    """What the loop concluded about one experiment."""

    feasible: bool
    is_new_best: bool
    conclusion: str
    confidence: float
    active_constraints: list[str]


def judge(
    evaluation: Evaluation,
    incumbent_mass_kg: float | None,
    optimizer_succeeded: bool,
) -> Verdict:
    """Decide whether an experiment produced a new incumbent, and say why.

    Confidence is deliberately modest and explainable rather than invented:
      * an infeasible or failed solve is a confident negative
      * a clear improvement from a converged solve is a confident positive
      * a marginal result gets a middling score
    It is a bookkeeping signal for Phase 5, not a calibrated probability.
    """
    active = evaluation.active_constraints()

    if not evaluation.is_feasible():
        worst = evaluation.worst_violation()
        return Verdict(
            feasible=False,
            is_new_best=False,
            conclusion=(
                f"Rejected: infeasible, worst normalized violation {worst:.3e}."
            ),
            confidence=0.9,
            active_constraints=active,
        )

    if incumbent_mass_kg is None:
        return Verdict(
            feasible=True,
            is_new_best=True,
            conclusion=(
                f"Accepted as first feasible design at "
                f"{evaluation.mass_kg:.6f} kg; active: {active or ['none']}."
            ),
            confidence=0.8 if optimizer_succeeded else 0.5,
            active_constraints=active,
        )

    relative = (incumbent_mass_kg - evaluation.mass_kg) / incumbent_mass_kg
    if relative > IMPROVEMENT_EPSILON:
        return Verdict(
            feasible=True,
            is_new_best=True,
            conclusion=(
                f"New best: {evaluation.mass_kg:.6f} kg, {relative:.3%} lighter "
                f"than the incumbent; active: {active or ['none']}."
            ),
            confidence=0.85 if optimizer_succeeded else 0.6,
            active_constraints=active,
        )

    return Verdict(
        feasible=True,
        is_new_best=False,
        conclusion=(
            f"Feasible at {evaluation.mass_kg:.6f} kg but no improvement "
            f"({relative:+.3%}); incumbent retained."
        ),
        confidence=0.7,
        active_constraints=active,
    )
