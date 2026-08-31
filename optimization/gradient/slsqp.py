"""optimization.gradient.slsqp - gradient-based constrained optimization.

SLSQP driven by exact derivatives: the objective gradient and the constraint
Jacobian both come from Phase 2's Warp autodiff, not from finite differences.
That is the point of having made the physics differentiable.

Local method - it converges to the nearest KKT point, not necessarily the
global one. optimization.evolutionary searches globally; the two are compared
against each other on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

# The physics kernel is fp32, so mass carries ~1e-7 relative noise. Asking
# SLSQP for a tighter ftol makes its line search fail on that noise and report
# "Positive directional derivative for linesearch" at a point that is in fact
# converged - the optimum is identical to 9 significant figures either way.
# Match the tolerance to the precision that actually exists.
DEFAULT_FTOL = 1e-7

from optimization.constraints import (
    Evaluation,
    OptimizationProblem,
    constraint_jacobian,
    evaluate_design,
    mass_and_gradient,
)


@dataclass
class OptimizationResult:
    """Shared result shape, so both methods can be compared field by field."""

    method: str
    x: np.ndarray                  # (b, h, t) in metres
    evaluation: Evaluation
    n_evaluations: int
    success: bool
    message: str

    @property
    def mass_kg(self) -> float:
        return self.evaluation.mass_kg

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "outer_width_m": float(self.x[0]),
            "outer_height_m": float(self.x[1]),
            "wall_thickness_m": float(self.x[2]),
            "mass_kg": self.evaluation.mass_kg,
            "max_bending_stress_pa": self.evaluation.max_bending_stress_pa,
            "tip_deflection_m": self.evaluation.tip_deflection_m,
            "safety_factor": self.evaluation.safety_factor,
            "first_natural_frequency_hz": self.evaluation.first_natural_frequency_hz,
            "feasible": self.evaluation.is_feasible(),
            "active_constraints": self.evaluation.active_constraints(),
            "n_evaluations": self.n_evaluations,
            "success": self.success,
        }


def default_start(op: OptimizationProblem) -> np.ndarray:
    """Midpoint of the box, nudged to a valid cavity."""
    x = 0.5 * (op.lower + op.upper)
    if not op.is_geometrically_valid(x):
        x[2] = min(x[2], 0.4 * min(x[0], x[1]))
    return x


def optimize_slsqp(
    op: OptimizationProblem,
    x0: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = DEFAULT_FTOL,
) -> OptimizationResult:
    """Minimize mass subject to the shared constraints, in normalized space."""
    x0 = default_start(op) if x0 is None else op.clip_to_bounds(x0)
    scale = op.scale()
    counter = {"n": 0}

    def objective(u):
        counter["n"] += 1
        mass, grad = mass_and_gradient(op, op.to_physical(u))
        return float(mass), grad * scale        # chain rule into unit space

    def constraints_fun(u):
        x = op.to_physical(u)
        jac = constraint_jacobian(op, x)
        ev = evaluate_design(op, x)
        names = sorted(ev.constraints)
        return (np.array([ev.constraints[n] for n in names]),
                np.vstack([jac[n] * scale for n in names]))

    cache: dict[bytes, tuple] = {}

    def cached(u):
        key = np.asarray(u, dtype=float).tobytes()
        if key not in cache:
            cache[key] = constraints_fun(u)
        return cache[key]

    result = minimize(
        fun=lambda u: objective(u)[0],
        x0=op.to_unit(x0),
        jac=lambda u: objective(u)[1],
        bounds=[(0.0, 1.0)] * 3,
        constraints=[{
            "type": "ineq",
            "fun": lambda u: cached(u)[0],
            "jac": lambda u: cached(u)[1],
        }],
        method="SLSQP",
        options={"maxiter": max_iter, "ftol": tol},
    )

    x = op.clip_to_bounds(op.to_physical(result.x))
    return OptimizationResult(
        method="SLSQP",
        x=x,
        evaluation=evaluate_design(op, x),
        n_evaluations=counter["n"],
        success=bool(result.success),
        message=str(result.message),
    )
