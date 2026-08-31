"""physics.solver.batch - profile-aware batch orchestration.

The evaluator itself launches one kernel for whatever it is handed. This layer
decides how much to hand it at a time, from the active GPU profile, so the same
call works on a 4 GB laptop and on an 80 GB datacenter card without the caller
choosing a batch size.

For the beam kernel the memory per candidate is tiny and the chunk size acts as
a guard rail rather than a tuning knob; the same structure is what keeps a much
heavier Phase 3+ FEM kernel inside its VRAM budget.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from core.design_genome import DesignGenome
from core.profile import load_profile
from physics.structural import BeamMetrics, beam_gradients, evaluate_beam


def resolve_batch_size(profile: str | None = None,
                       batch_size: int | None = None) -> int:
    """Explicit argument wins; otherwise compute.max_batch from the profile."""
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        return batch_size
    cfg = load_profile(profile)
    resolved = int(cfg["compute"]["max_batch"])
    if resolved < 1:
        raise ValueError(
            f"profile {cfg.get('name')!r} has compute.max_batch={resolved}"
        )
    return resolved


def chunked(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def evaluate_population(
    genomes: Iterable[DesignGenome],
    problem,
    profile: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> BeamMetrics:
    """Evaluate any number of candidates, chunked to the profile's batch size.

    Chunking must not change the answer: each candidate is independent, and
    tests assert chunked results match a single full-batch launch.
    """
    genomes = list(genomes)
    if not genomes:
        raise ValueError("no genomes to evaluate")
    size = resolve_batch_size(profile, batch_size)
    parts = [
        evaluate_beam(chunk, problem, device=device)
        for chunk in chunked(genomes, size)
    ]
    return BeamMetrics.concatenate(parts)


def population_gradients(
    genomes: Iterable[DesignGenome],
    problem,
    metric: str,
    profile: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict[str, "list"]:
    """Same chunking, for gradients."""
    import numpy as np

    genomes = list(genomes)
    if not genomes:
        raise ValueError("no genomes to evaluate")
    size = resolve_batch_size(profile, batch_size)
    parts = [
        beam_gradients(chunk, problem, metric, device=device)
        for chunk in chunked(genomes, size)
    ]
    return {
        var: np.concatenate([p[var] for p in parts]) for var in parts[0]
    }
