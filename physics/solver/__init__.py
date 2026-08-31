"""physics.solver - batch orchestration across the active GPU profile."""

from .batch import (
    chunked,
    evaluate_population,
    population_gradients,
    resolve_batch_size,
)

__all__ = [
    "chunked", "evaluate_population", "population_gradients",
    "resolve_batch_size",
]
