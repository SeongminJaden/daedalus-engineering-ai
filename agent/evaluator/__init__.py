"""agent.evaluator - judges experiment results into episode conclusions."""

from .verdict import IMPROVEMENT_EPSILON, Verdict, judge

__all__ = ["IMPROVEMENT_EPSILON", "Verdict", "judge"]
