"""optimization.constraints.problem - the optimization problem, one definition.

Both optimizers (SLSQP and differential evolution) must be solving *exactly*
the same problem, or agreement between them proves nothing. So the design
space, the allowable stress, the constraint values and the feasibility test all
live here and are shared.

Design variables (SI): x = (b, h, t) = (outer_width_m, outer_height_m,
wall_thickness_m).

Constraints are written normalized and in "feasible when >= 0" form, which is
what SLSQP wants and which keeps a 1e8-scale stress term and a 1e-3-scale
deflection term comparable:

    g_stress     = 1 - sigma_max / sigma_allow
    g_deflection = 1 - delta / delta_max
    g_cavity_b   = (b - 2t)/b - CAVITY_MARGIN
    g_cavity_h   = (h - 2t)/h - CAVITY_MARGIN
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.design_genome import DesignBounds, DesignGenome, HollowRectangleSection
from core.engineering_ir import LoadType
from core.materials import get_material

# Keep the wall strictly off the degenerate t = min(b,h)/2 boundary, where I
# collapses to zero and the metrics blow up.
CAVITY_MARGIN = 1e-3

VARIABLE_ORDER = ("outer_width_m", "outer_height_m", "wall_thickness_m")


@dataclass(frozen=True)
class OptimizationProblem:
    """Everything both optimizers need, resolved from an Engineering IR problem."""

    problem: object
    bounds: DesignBounds
    allowable_stress_pa: float
    max_deflection_m: float | None
    # Smallest clear opening the cavity must leave, and the torque the section
    # carries at the same time as the bending load. Both default to absent, so
    # a problem that states neither behaves exactly as it did before.
    min_clear_bore_m: float | None = None
    min_manufacturing_wall_m: float | None = None
    applied_torque_nm: float = 0.0

    @property
    def lower(self) -> np.ndarray:
        return np.array([getattr(self.bounds, v).min for v in VARIABLE_ORDER])

    @property
    def upper(self) -> np.ndarray:
        return np.array([getattr(self.bounds, v).max for v in VARIABLE_ORDER])

    # --- normalized <-> physical -------------------------------------- #
    # SLSQP conditions badly on raw metres (b ~ 5e-2, t ~ 1e-3, gradients
    # spanning 1e2..1e8). Working in a unit cube fixes that; the chain rule
    # factor is just (upper - lower).
    def to_unit(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.lower) / (self.upper - self.lower)

    def to_physical(self, u: np.ndarray) -> np.ndarray:
        return self.lower + np.asarray(u, dtype=float) * (self.upper - self.lower)

    def scale(self) -> np.ndarray:
        return self.upper - self.lower

    # --- helpers -------------------------------------------------------- #
    def genome(self, x: np.ndarray) -> DesignGenome:
        b, h, t = (float(v) for v in x)
        return DesignGenome(
            section=HollowRectangleSection(
                outer_width_m=b, outer_height_m=h, wall_thickness_m=t
            ),
            material_id=self.problem.material_id,
        )

    def is_geometrically_valid(self, x: np.ndarray) -> bool:
        b, h, t = (float(v) for v in x)
        if b <= 0 or h <= 0 or t <= 0:
            return False
        return t < min(b, h) / 2.0

    def clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float), self.lower, self.upper)


def build_optimization_problem(
    problem, bounds: DesignBounds | None = None
) -> OptimizationProblem:
    """Resolve allowable stress and the design space from the IR problem.

    Allowable stress is the *tighter* of the explicit stress ceiling and the
    yield strength divided by the required safety factor - satisfying only the
    looser of the two would leave the other silently violated.
    """
    bounds = bounds or DesignBounds()
    material = get_material(problem.material_id)
    c = problem.constraints

    candidates = []
    if c.max_stress_pa is not None:
        candidates.append(float(c.max_stress_pa))
    if c.min_safety_factor is not None:
        candidates.append(material.allowable_stress_pa(c.min_safety_factor))
    if not candidates:
        raise ValueError(
            "problem constrains neither max_stress_pa nor min_safety_factor; "
            "mass minimization would be unbounded"
        )

    # The design envelope caps the outer dimensions.
    if problem.geometry.max_width_m is not None:
        bounds = bounds.model_copy(deep=True)
        bounds.outer_width_m.max = min(
            bounds.outer_width_m.max, float(problem.geometry.max_width_m)
        )
    if problem.geometry.max_height_m is not None:
        bounds = bounds.model_copy(deep=True)
        bounds.outer_height_m.max = min(
            bounds.outer_height_m.max, float(problem.geometry.max_height_m)
        )

    return OptimizationProblem(
        problem=problem,
        bounds=bounds,
        allowable_stress_pa=min(candidates),
        max_deflection_m=(
            None if c.max_deflection_m is None else float(c.max_deflection_m)
        ),
        min_clear_bore_m=(
            None if c.min_clear_bore_m is None else float(c.min_clear_bore_m)
        ),
        min_manufacturing_wall_m=(
            None if c.min_manufacturing_wall_m is None
            else float(c.min_manufacturing_wall_m)
        ),
        # Torque loads sum: two torques about the same axis are one torque.
        applied_torque_nm=float(sum(
            load.magnitude_n for load in problem.loads
            if load.type is LoadType.TORQUE)),
    )
