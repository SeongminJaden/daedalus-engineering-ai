"""optimization.multi_objective: Pareto fronts and NSGA-II.

Phase 3 minimised mass with everything else as a constraint. Phase 20 made the
objectives a vector: mass, deflection, stress and material cost, searched with
NSGA-II and returned as an approximated Pareto front so the trade-off is
visible before anyone commits to weights.
"""

from .nsga2 import (
    Nsga2Result,
    Sense,
    constrained_dominates,
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    nsga2,
    to_minimisation,
)
from .objectives import (
    OBJECTIVE_NAMES,
    OBJECTIVE_SENSES,
    MaterialFront,
    build_evaluator,
    material_cost_usd,
    merged_front,
    sweep_materials,
)
from .pareto import non_dominated_mask, pareto_front, scalarize

__all__ = [
    "MaterialFront", "Nsga2Result", "OBJECTIVE_NAMES", "OBJECTIVE_SENSES",
    "Sense", "build_evaluator", "constrained_dominates", "crowding_distance",
    "dominates", "fast_non_dominated_sort", "material_cost_usd",
    "merged_front", "non_dominated_mask", "nsga2", "pareto_front",
    "scalarize", "sweep_materials", "to_minimisation",
]
