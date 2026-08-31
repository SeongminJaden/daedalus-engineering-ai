"""optimization.multi_objective.pareto: the direct non-dominated filter.

Written during Phase 3 as the one genuinely reusable piece of a multi-objective
future, and kept because it is the simplest correct statement of dominance: one
pass, no bookkeeping, easy to read and therefore easy to trust. Phase 20 uses
it as the independent check on the faster sort in `nsga2`, and that comparison
caught an inverted domination count there.

`nsga2.fast_non_dominated_sort` is what an optimiser should call, since it
returns every front rather than only the first. This returns the first.

VALIDITY: every column is MINIMISED. A maximised objective passed in raw
selects exactly the wrong rows, so convert with `nsga2.to_minimisation` first.
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
