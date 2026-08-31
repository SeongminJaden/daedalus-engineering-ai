"""brain.retrieval - numeric-feature similarity search (NOT semantic search)."""

from .features import (
    FEATURE_NAMES,
    FEATURE_SCALES,
    FeatureSpace,
    design_vector,
    problem_vector,
)
from .search import (
    nearest_designs,
    retrieve_knowledge,
    retrieve_similar,
    retrieve_strategies,
    warm_start_from_memory,
)

__all__ = [
    "FEATURE_NAMES", "FEATURE_SCALES", "FeatureSpace", "design_vector",
    "nearest_designs", "problem_vector", "retrieve_knowledge",
    "retrieve_similar", "retrieve_strategies", "warm_start_from_memory",
]
