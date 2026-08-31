"""optimization.evolutionary.differential - gradient-free global search.

scipy's differential_evolution with `vectorized=True`, so the whole population
arrives as one array and goes to the GPU in a single batched evaluation. That
is the concrete payoff of the batch physics kernel: population size costs
almost nothing until VRAM runs out.

Constraints are handled by a quadratic penalty on the normalized violations.
The normalization matters - stress is O(1e8) and deflection O(1e-3) in raw
units, so an un-normalized penalty would effectively enforce only one of them.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from optimization.constraints import OptimizationProblem, evaluate_batch, evaluate_design
from optimization.gradient.slsqp import OptimizationResult

# Mass is O(0.1-2 kg) and normalized violations are O(1), so this makes any
# real violation dominate the objective without making the surface degenerate.
PENALTY_WEIGHT = 1.0e3

# Cost assigned to a geometrically impossible design, which has no metrics.
INFEASIBLE_COST = 1.0e6


def penalized_objective(op: OptimizationProblem, X: np.ndarray) -> np.ndarray:
    """Vectorized objective. X is (3, S) from scipy, or (3,) for a single point."""
    X = np.asarray(X, dtype=float)
    single = X.ndim == 1
    rows = X.reshape(1, -1) if single else X.T          # -> (S, 3)

    mass, cons, ok = evaluate_batch(op, rows)

    violation = np.zeros(rows.shape[0])
    for values in cons.values():
        v = np.asarray(values, dtype=float)
        # Stress and deflection are NaN wherever the geometry is impossible -
        # those rows never reached the kernel. Letting NaN through would make
        # the whole objective NaN across most of the box (the valid region
        # t < min(b,h)/2 is a slice of it), and the search would have nothing
        # to descend. Those rows are already charged INFEASIBLE_COST plus a
        # finite, monotone cavity violation, which is what pulls them back.
        v = np.where(np.isfinite(v), v, 0.0)
        violation += np.maximum(0.0, -v) ** 2

    cost = np.where(ok, np.nan_to_num(mass, nan=INFEASIBLE_COST), INFEASIBLE_COST)
    cost = cost + PENALTY_WEIGHT * violation
    return float(cost[0]) if single else cost


def optimize_differential_evolution(
    op: OptimizationProblem,
    seed: int = 0,
    max_iter: int = 200,
    popsize: int = 25,
    tol: float = 1e-8,
) -> OptimizationResult:
    """Global search over the design box. Deterministic for a given seed."""
    counter = {"n": 0}

    def fun(X):
        X = np.asarray(X, dtype=float)
        counter["n"] += 1 if X.ndim == 1 else X.shape[1]
        return penalized_objective(op, X)

    # polish=False on purpose. The penalty jumps by INFEASIBLE_COST across the
    # geometric-validity boundary, and scipy's L-BFGS-B polish step cannot
    # line-search across that discontinuity - it wandered into t >= min(b,h)/2
    # and returned an unevaluatable design. Keeping DE gradient-free also keeps
    # it genuinely independent of the SLSQP result it is cross-checked against.
    result = differential_evolution(
        fun,
        bounds=list(zip(op.lower, op.upper)),
        seed=seed,
        maxiter=max_iter,
        popsize=popsize,
        tol=tol,
        polish=False,
        vectorized=True,
        init="sobol",
    )

    x = _repair_geometry(op, op.clip_to_bounds(result.x))
    return OptimizationResult(
        method="DifferentialEvolution",
        x=x,
        evaluation=evaluate_design(op, x),
        n_evaluations=counter["n"],
        success=bool(result.success),
        message=str(result.message),
    )


def _repair_geometry(op: OptimizationProblem, x: np.ndarray) -> np.ndarray:
    """Pull an impossible wall back inside the cavity limit.

    A guard, not a fix-up path: with polish disabled the returned point is a
    penalized population member and should already be valid. If it ever is not,
    failing loudly later with an unevaluatable genome would be worse than
    projecting it back onto the feasible geometry here.
    """
    x = np.array(x, dtype=float)
    if op.is_geometrically_valid(x):
        return x
    ceiling = 0.499 * min(x[0], x[1])
    x[2] = min(x[2], max(op.lower[2], ceiling))
    return x
