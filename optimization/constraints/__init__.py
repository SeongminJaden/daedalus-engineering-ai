"""optimization.constraints - shared problem definition and constraint scoring."""

from .evaluate import (
    CONSTRAINT_NAMES,
    FEASIBILITY_TOL,
    Evaluation,
    constraint_jacobian,
    constraint_values,
    evaluate_batch,
    evaluate_design,
    mass_and_gradient,
)
from .problem import (
    CAVITY_MARGIN,
    VARIABLE_ORDER,
    OptimizationProblem,
    build_optimization_problem,
)

__all__ = [
    "CAVITY_MARGIN", "CONSTRAINT_NAMES", "FEASIBILITY_TOL", "Evaluation", "OptimizationProblem",
    "VARIABLE_ORDER", "build_optimization_problem", "constraint_jacobian",
    "constraint_values", "evaluate_batch", "evaluate_design",
    "mass_and_gradient",
]
