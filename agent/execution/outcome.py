"""What a design strategy produces, whatever strategy produced it.

The loop compares designs from different methods, so they have to arrive in a
common currency: a mass, a feasibility verdict, and the constraint values
behind that verdict. What they must NOT lose in the process is where they came
from, which is why the payload is representation-specific and required.

A parametric outcome carries a design vector. A topology outcome carries a
density field. Exactly one, always: that invariant is what makes an episode's
`strategy_used` checkable rather than merely asserted. A record claiming a
topology method produced a design has to be holding a density field, and it
cannot be holding one unless a topology optimiser ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DesignOutcome:
    """One design, its performance, and its provenance."""

    method: str
    mass_kg: float
    feasible: bool
    constraints: dict[str, float] = field(default_factory=dict)
    evaluations: int = 0
    seconds: float = 0.0
    converged: bool = True
    design_vector: np.ndarray | None = None
    density_field: np.ndarray | None = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_vector = self.design_vector is not None
        has_field = self.density_field is not None
        if has_vector == has_field:
            raise ValueError(
                "a design outcome carries exactly one representation: a design "
                "vector from a parametric method or a density field from a "
                "topology method. Carrying neither means nothing was produced; "
                "carrying both means the provenance is ambiguous, and the "
                "episode log would not be checkable.")
        if has_vector:
            self.design_vector = np.asarray(self.design_vector, dtype=float)
        else:
            self.density_field = np.asarray(self.density_field, dtype=float)

    @property
    def representation(self) -> str:
        return "design_vector" if self.design_vector is not None else "density_field"



class OutcomeVerdict:
    """The slice of an evaluation that `judge` reads, from any method.

    `judge` was written against the parametric `Evaluation`, which carries a
    stress, a deflection, a safety factor and a natural frequency. A topology
    outcome has none of those in the same sense, and filling them with zeros so
    the type matches would put numbers into the episode log that no solver
    produced. This exposes only what every method genuinely has: a mass and its
    constraint margins.
    """

    def __init__(self, outcome: "DesignOutcome"):
        self.mass_kg = float(outcome.mass_kg)
        self.constraints = dict(outcome.constraints)
        self._feasible = bool(outcome.feasible)

    def is_feasible(self, tol: float | None = None) -> bool:
        # Imported, never restated. Writing the number here instead cost a
        # loop regression: a local 1e-6 against the project's 1e-4 changed
        # which designs counted as feasible, and so which path a seeded run
        # took.
        from optimization.constraints import FEASIBILITY_TOL

        if not self.constraints:
            return self._feasible
        limit = FEASIBILITY_TOL if tol is None else tol
        return all(v >= -limit for v in self.constraints.values())

    def worst_violation(self) -> float:
        if not self.constraints:
            return 0.0 if self._feasible else float("inf")
        return max(0.0, -min(self.constraints.values()))

    def active_constraints(self, tol: float = 1e-3) -> list[str]:
        return sorted(n for n, v in self.constraints.items() if abs(v) <= tol)
