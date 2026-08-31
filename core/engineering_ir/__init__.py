"""core.engineering_ir - the Engineering IR: the fixed problem definition."""

from .io import load_problem, problem_from_dict, problem_to_dict, save_problem
from .schema import (
    BoundaryCondition,
    BoundaryConditionType,
    BoundaryLocation,
    Constraints,
    EngineeringProblem,
    Geometry,
    Load,
    LoadApplication,
    LoadType,
    Objective,
    ObjectiveQuantity,
    ObjectiveSense,
    SectionType,
    Vec3,
)

__all__ = [
    "BoundaryCondition", "BoundaryConditionType", "BoundaryLocation",
    "Constraints", "EngineeringProblem", "Geometry", "Load",
    "LoadApplication", "LoadType", "Objective", "ObjectiveQuantity",
    "ObjectiveSense", "SectionType", "Vec3",
    "load_problem", "save_problem", "problem_to_dict", "problem_from_dict",
]
