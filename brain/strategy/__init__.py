"""brain.strategy - reusable solution strategies, promoted from evidence."""

from .methods import derive_method_strategies
from .strategies import (
    PROMOTION_THRESHOLD,
    Strategy,
    StrategyStore,
    derive_stiffness_strategy,
)

__all__ = [
    "PROMOTION_THRESHOLD", "Strategy", "StrategyStore",
    "derive_method_strategies", "derive_stiffness_strategy",
]
