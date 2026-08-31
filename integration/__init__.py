"""integration: whole-assembly design with a conjunctive verdict over every method."""

from .capstone import (KNOWN_UNIMPLEMENTED_MODES, CapstoneResult, JointSpec,
                       build_link_problem, design_joint, routed_methods)
from .coupled import CoupledResult, thermal_structural
from .minimum import (MONOTONICITY_SAMPLES, SizingResult,
                      is_monotonic_increasing, minimum_dimension)
from .multi_review import (DesignEntry, MultiDesignReview, RankBy)
from .checks import (FEASIBILITY_TOLERANCE, AssemblyStatus, AssemblyVerdict,
                     CheckResult, CheckStatus, satisfies)
from .review import Review, review

__all__ = [
    "CoupledResult", "DesignEntry", "MONOTONICITY_SAMPLES",
    "MultiDesignReview", "RankBy", "SizingResult", "is_monotonic_increasing",
    "minimum_dimension", "thermal_structural",
    "AssemblyStatus", "AssemblyVerdict", "CapstoneResult", "CheckResult",
    "CheckStatus", "FEASIBILITY_TOLERANCE", "JointSpec",
    "KNOWN_UNIMPLEMENTED_MODES", "Review", "build_link_problem",
    "design_joint", "review", "routed_methods", "satisfies",
]
