"""brain.strategy - reusable solution strategies, promoted from evidence."""

from .strategies import (
    PROMOTION_THRESHOLD,
    Strategy,
    StrategyStore,
    derive_stiffness_strategy,
)

__all__ = [
    "PROMOTION_THRESHOLD", "Strategy", "StrategyStore",
    "derive_stiffness_strategy",
]
