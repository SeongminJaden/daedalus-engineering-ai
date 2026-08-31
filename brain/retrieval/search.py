"""brain.retrieval.search - nearest-neighbour retrieval over the Brain.

Exact brute-force search, vectorized with numpy. Exact is the right default
here: correctness is easy to verify (the tests check it against a naive
per-row loop, a genuinely different code path), and the design store is small.
An ANN index is the swap when it stops being small - same API.
"""

from __future__ import annotations

import numpy as np

from brain.episodic import EpisodicMemory
from brain.semantic import EvidenceLevel, SemanticMemory

from .features import design_vector


def _matrix(designs: list[dict], problem_params: dict | None) -> np.ndarray:
    if not designs:
        return np.zeros((0, 0))
    return np.vstack([
        design_vector(d["genome"], d["metrics"], problem_params) for d in designs
    ])


def nearest_designs(
    episodic: EpisodicMemory,
    query: np.ndarray,
    k: int = 5,
    feasible_only: bool = False,
    problem_params: dict | None = None,
) -> list[tuple[dict, float]]:
    """The k designs closest to `query`, nearest first, with distances.

    Ties break on design_id so the ordering is deterministic - an unstable sort
    would make runs non-reproducible for no reason.
    """
    designs = episodic.designs(feasible_only=feasible_only)
    if not designs:
        return []

    matrix = _matrix(designs, problem_params)
    distances = np.linalg.norm(matrix - np.asarray(query, dtype=float), axis=1)
    order = sorted(
        range(len(designs)),
        key=lambda i: (float(distances[i]), designs[i]["design_id"]),
    )
    return [(designs[i], float(distances[i])) for i in order[:k]]


def retrieve_similar(
    episodic: EpisodicMemory,
    genome: dict | None = None,
    metrics: dict | None = None,
    problem_params: dict | None = None,
    k: int = 5,
    feasible_only: bool = True,
) -> list[tuple[dict, float]]:
    """Past designs numerically similar to the one described.

    Not semantic search - see brain/retrieval/features.py.
    """
    query = design_vector(genome or {}, metrics or {}, problem_params)
    return nearest_designs(episodic, query, k=k, feasible_only=feasible_only,
                           problem_params=problem_params)


def retrieve_knowledge(
    semantic: SemanticMemory,
    domain: str | None = None,
    min_level: EvidenceLevel | None = None,
) -> list:
    """Knowledge for a domain, best-supported first, optionally level-filtered."""
    return semantic.by_domain(domain, min_level=min_level)


def retrieve_strategies(strategy_store, context: dict | None = None,
                        min_confidence: float = 0.0) -> list:
    """Strategies applicable to a context (e.g. {'binding_constraint': ...})."""
    return strategy_store.applicable(context or {}, min_confidence=min_confidence)


def warm_start_from_memory(
    episodic: EpisodicMemory,
    problem_params: dict | None = None,
    target_mass_kg: float | None = None,
) -> np.ndarray | None:
    """A starting design drawn from the lightest feasible thing on record.

    This is what closes the loop: a later run begins where an earlier one
    finished instead of rediscovering it. Returns None on an empty Brain, and
    callers must handle that - a cold Brain is the normal first case.
    """
    best = episodic.best_design()
    if best is None:
        return None
    g = best["genome"]
    try:
        return np.array([
            float(g["outer_width_m"]),
            float(g["outer_height_m"]),
            float(g["wall_thickness_m"]),
        ])
    except (KeyError, TypeError, ValueError):
        return None
