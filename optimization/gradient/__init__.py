"""optimization.gradient - gradient-based optimizers (exact Warp derivatives)."""

from .slsqp import OptimizationResult, default_start, optimize_slsqp

__all__ = ["OptimizationResult", "default_start", "optimize_slsqp"]
