"""optimization.multi_objective.pareto - Pareto tooling. Stub.

Phase 3 solves a single objective (minimize mass) with the rest expressed as
constraints. Real multi-objective work - mass vs stiffness vs cost, with no
single winner - lands in a later phase. The non-dominated sort below is the
one piece that is genuinely reusable, so it is implemented rather than faked.
"""

from __future__ import annotations

import numpy as np


def non_dominated_mask(objectives: np.ndarray) -> np.ndarray:
    """True where a row is Pareto-optimal. Every column is minimized.

    Row i is dominated when some row j is no worse everywhere and strictly
    better somewhere.
    """
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    n = f.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        no_worse = np.all(f <= f[i], axis=1)
        strictly_better = np.any(f < f[i], axis=1)
        dominated_by = no_worse & strictly_better
        if np.any(dominated_by):
            keep[i] = False
    return keep


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(objectives, dtype=float))[
        non_dominated_mask(objectives)
    ]


def scalarize(objectives: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted-sum scalarization. Placeholder for a real preference model."""
    f = np.atleast_2d(np.asarray(objectives, dtype=float))
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != f.shape[1]:
        raise ValueError(f"expected {f.shape[1]} weights, got {w.shape[0]}")
    return f @ w
