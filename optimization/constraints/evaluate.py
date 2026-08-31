"""optimization.constraints.evaluate - constraint values, batched and single.

Both optimizers score candidates through here, so "feasible" means the same
thing to each of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.solver import evaluate_population
from physics.structural import beam_gradients_many, evaluate_beam

from .problem import CAVITY_MARGIN, OptimizationProblem

CONSTRAINT_NAMES = ("stress", "deflection", "cavity_b", "cavity_h")

# A mass-minimal design sits EXACTLY on its binding constraint, so any solver
# lands within numerical noise of the boundary - SLSQP typically a hair inside,
# a penalty method a hair outside. Declaring PASS/FAIL therefore needs a stated
# tolerance rather than a bare `>= 0`. This is 0.01% of each limit: on the MVP's
# 1 mm deflection cap that is 100 nm, which is far below anything the beam model
# itself resolves. It is a numerical tolerance, NOT an engineering allowance -
# the real margin is the safety factor inside sigma_allow.
FEASIBILITY_TOL = 1e-4


@dataclass
class Evaluation:
    """Metrics plus normalized constraint values for one design."""

    mass_kg: float
    max_bending_stress_pa: float
    tip_deflection_m: float
    safety_factor: float
    first_natural_frequency_hz: float
    constraints: dict[str, float]      # >= 0 means satisfied

    def is_feasible(self, tol: float = FEASIBILITY_TOL) -> bool:
        return all(v >= -tol for v in self.constraints.values())

    def worst_violation(self) -> float:
        return max(0.0, -min(self.constraints.values()))

    def active_constraints(self, tol: float = 1e-3) -> list[str]:
        """Constraints sitting on their limit - the ones actually shaping the
        answer. A slack constraint could be deleted without moving the optimum."""
        return sorted(n for n, v in self.constraints.items() if abs(v) <= tol)


def _cavity_terms(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b, h, t = x[..., 0], x[..., 1], x[..., 2]
    return ((b - 2.0 * t) / b - CAVITY_MARGIN,
            (h - 2.0 * t) / h - CAVITY_MARGIN)


def constraint_values(
    op: OptimizationProblem,
    stress_pa: np.ndarray,
    deflection_m: np.ndarray,
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    """Normalized, feasible-when-non-negative."""
    g_b, g_h = _cavity_terms(np.atleast_2d(x))
    out = {
        "stress": 1.0 - np.asarray(stress_pa) / op.allowable_stress_pa,
        "cavity_b": g_b,
        "cavity_h": g_h,
    }
    if op.max_deflection_m is not None:
        out["deflection"] = 1.0 - np.asarray(deflection_m) / op.max_deflection_m
    return out


def evaluate_design(op: OptimizationProblem, x: np.ndarray) -> Evaluation:
    """One design, on the GPU."""
    x = np.asarray(x, dtype=float)
    m = evaluate_beam([op.genome(x)], op.problem).candidate(0)
    cons = constraint_values(
        op, m["max_bending_stress_pa"], m["tip_deflection_m"], x
    )
    return Evaluation(
        mass_kg=m["mass_kg"],
        max_bending_stress_pa=m["max_bending_stress_pa"],
        tip_deflection_m=m["tip_deflection_m"],
        safety_factor=m["safety_factor"],
        first_natural_frequency_hz=m["first_natural_frequency_hz"],
        constraints={k: float(np.ravel(v)[0]) for k, v in cons.items()},
    )


def evaluate_batch(op: OptimizationProblem, X: np.ndarray):
    """A whole population in one GPU batch.

    Geometrically impossible rows (t >= min(b,h)/2) are never handed to the
    kernel - I would be zero or negative there and the metrics meaningless.
    They come back as NaN metrics with their cavity constraint violated, which
    the penalty in the evolutionary driver turns into a large finite cost.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]

    mass = np.full(n, np.nan)
    stress = np.full(n, np.nan)
    defl = np.full(n, np.nan)

    ok = np.array([op.is_geometrically_valid(row) for row in X])
    if ok.any():
        metrics = evaluate_population(
            [op.genome(row) for row in X[ok]], op.problem
        )
        mass[ok] = metrics.mass_kg
        stress[ok] = metrics.max_bending_stress_pa
        defl[ok] = metrics.tip_deflection_m

    cons = constraint_values(op, stress, defl, X)
    return mass, cons, ok


def constraint_jacobian(
    op: OptimizationProblem, x: np.ndarray
) -> dict[str, np.ndarray]:
    """d(constraint)/dx for one design, using Warp autodiff.

    Chain rule on the normalized forms:
        d/dx [1 - sigma/sigma_allow] = -(1/sigma_allow) * d(sigma)/dx
        d/dx [1 - delta/delta_max]   = -(1/delta_max)   * d(delta)/dx
    The cavity terms are analytic in x and need no kernel.
    """
    x = np.asarray(x, dtype=float)
    b, h, t = (float(v) for v in x)

    wanted = ["max_bending_stress_pa"]
    if op.max_deflection_m is not None:
        wanted.append("tip_deflection_m")
    grads = beam_gradients_many([op.genome(x)], op.problem, wanted)

    def vec(metric):
        g = grads[metric]
        return np.array([float(g[v][0]) for v in
                         ("outer_width_m", "outer_height_m", "wall_thickness_m")])

    out = {
        "stress": -vec("max_bending_stress_pa") / op.allowable_stress_pa,
        # d/db[(b-2t)/b] = 2t/b^2 ; d/dt = -2/b ; d/dh = 0
        "cavity_b": np.array([2.0 * t / b**2, 0.0, -2.0 / b]),
        "cavity_h": np.array([0.0, 2.0 * t / h**2, -2.0 / h]),
    }
    if op.max_deflection_m is not None:
        out["deflection"] = -vec("tip_deflection_m") / op.max_deflection_m
    return out


def mass_and_gradient(op: OptimizationProblem, x: np.ndarray):
    """mass(x) and d(mass)/dx."""
    x = np.asarray(x, dtype=float)
    g = op.genome(x)
    mass = evaluate_beam([g], op.problem).candidate(0)["mass_kg"]
    grads = beam_gradients_many([g], op.problem, ["mass_kg"])["mass_kg"]
    grad = np.array([float(grads[v][0]) for v in
                     ("outer_width_m", "outer_height_m", "wall_thickness_m")])
    return mass, grad
