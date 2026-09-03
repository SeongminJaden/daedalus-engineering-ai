"""Optimisation on the three-field design variable, with beta continuation.

The step is a move-limited projected gradient with the volume enforced by a
Lagrange multiplier bisection, as in Phase 14a, but with two differences that
both follow from the projection:

* **The volume constraint is on the physical field, not the design variable.**
  Holding `mean(x)` fixed means nothing once a Heaviside sits between `x` and
  the density the solver sees. The bisection therefore drives
  `mean(x_bar(x))` onto the target. It is still a bisection on a monotone
  function: the filter has non-negative weights and the projection is
  increasing, so the physical volume falls monotonically as the multiplier
  rises.

* **The gradient is chained back through the projection**, so the adjoints
  from Phases 13 and 14a are used exactly as verified and the new part is the
  chain, which is separately finite-difference checked.

The OC update from Phase 13 is not reused here. Its `x * (-dc/dx / lambda)^eta`
form assumes the sensitivity is negative everywhere, which the stress-penalised
gradient is not, and the projected gradient step works for either sign.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .projection import BetaSchedule, DesignTransform
from .simp import MIN_DENSITY, SimpProblem, compliance_and_sensitivity

def initial_design(n_elements: int, target_volume: float,
                   transform: DesignTransform) -> np.ndarray:
    """A uniform design whose PHYSICAL volume is already on the target.

    Starting at a flat 0.5 would put the first iterate off the constraint,
    which then shows up in the reported history as a volume the run never
    intended. The projection is monotone in the design value, so the value that
    lands the physical volume on the target is a bisection away.
    """
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if transform.physical(np.full(n_elements, middle)).mean() < target_volume:
            low = middle
        else:
            high = middle
    return np.full(n_elements, 0.5 * (low + high))


def _measured_volume(physical: np.ndarray, problem_passive) -> float:
    """Mean physical density with the passive regions written in, which is the
    volume the part will actually have."""
    if problem_passive is None:
        return float(physical.mean())
    return float(problem_passive(physical).mean())


def _volume_projected_step(design: np.ndarray, direction: np.ndarray,
                           move_limit: float, target_volume: float,
                           transform: DesignTransform,
                           apply_passive=None) -> np.ndarray:
    """Move-limited step whose PHYSICAL volume lands on the target.

    With passive regions the volume that must land on the target is the one
    including them: the free elements have to make room for the elements held
    solid, or the run quietly delivers a heavier part than was asked for. That
    happened and was measured (0.50 against a requested 0.40) before this
    argument existed.
    """
    lower = np.maximum(design - 2.0 * move_limit, 0.0)
    upper = np.minimum(design + 2.0 * move_limit, 1.0)

    def at(multiplier: float) -> np.ndarray:
        return np.clip(design - move_limit * (direction + multiplier),
                       lower, upper)

    low, high = -10.0, 10.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if _measured_volume(transform.physical(at(middle)), apply_passive) > target_volume:
            low = middle
        else:
            high = middle
        if high - low < 1e-12:
            break
    return at(0.5 * (low + high))


def restore_volume(design: np.ndarray, target_volume: float,
                   transform: DesignTransform, apply_passive=None) -> np.ndarray:
    """Shift a design uniformly so its physical volume is back on the target.

    Sharpening beta changes the map from design variable to physical density,
    so the physical volume moves even though the design did not. Left alone,
    the move-limited step needs several iterations to pull it back, and every
    iterate in between reports a compliance measured at the wrong volume,
    which is not comparable to the others.

    This is a re-parameterisation rather than a design step, so it is not
    subject to the move limit: the design is not being improved, it is being
    expressed in the new map.
    """
    low, high = -1.0, 1.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        shifted = np.clip(design + middle, 0.0, 1.0)
        if _measured_volume(transform.physical(shifted), apply_passive) < target_volume:
            low = middle
        else:
            high = middle
    return np.clip(design + 0.5 * (low + high), 0.0, 1.0)


@dataclass
class ThreeFieldResult:
    """The design variable, the density that was solved, and the history."""

    design: np.ndarray
    density: np.ndarray
    compliance_history: list[float] = field(default_factory=list)
    volume_history: list[float] = field(default_factory=list)
    grey_history: list[float] = field(default_factory=list)
    beta_history: list[float] = field(default_factory=list)
    iterations: int = 0

    @property
    def final_compliance(self) -> float:
        return self.compliance_history[-1]

    @property
    def volume_fraction(self) -> float:
        return self.volume_history[-1]


def _grey(density: np.ndarray, low: float = 0.1, high: float = 0.9) -> float:
    return float(np.mean((density > low) & (density < high)))


def optimize_projected(problem: SimpProblem, max_iterations: int = 120,
                       move_limit: float = 0.1,
                       transform: DesignTransform | None = None,
                       schedule: BetaSchedule | None = None,
                       device: str | None = None,
                       callback=None) -> ThreeFieldResult:
    """Minimise compliance on the three-field formulation."""
    mesh = problem.mesh
    if transform is None:
        transform = DesignTransform.for_mesh(mesh,
                                             problem.filter_radius_elements)
    if schedule is None:
        schedule = BetaSchedule()

    passive = problem.apply_passive if problem.free_mask is not None else None
    design = initial_design(mesh.n_elements, problem.volume_fraction, transform)
    if passive is not None:
        design = restore_volume(design, problem.volume_fraction, transform, passive)
    free = problem.free_mask
    result = ThreeFieldResult(design=design,
                              density=problem.apply_passive(transform.physical(design)))

    for iteration in range(1, max_iterations + 1):
        previous_beta = transform.beta
        beta = schedule.apply(transform, iteration)
        if beta != previous_beta:
            design = restore_volume(design, problem.volume_fraction, transform,
                                    passive)
        # Passive elements are written into the density the solver sees and
        # their sensitivity is dropped, so the projection never moves them.
        density = problem.apply_passive(transform.physical(design))
        compliance, d_physical, _ = compliance_and_sensitivity(problem, density,
                                                               device=device)
        if free is not None:
            d_physical = np.where(free, d_physical, 0.0)
        result.compliance_history.append(float(compliance))
        result.volume_history.append(float(density.mean()))
        result.grey_history.append(_grey(density))
        result.beta_history.append(beta)
        result.iterations = iteration
        if callback is not None:
            callback(iteration, float(compliance), float(density.mean()), beta)

        gradient = transform.chain(d_physical, design)
        direction = gradient / max(np.abs(gradient).max(), 1e-30)
        design = _volume_projected_step(design, direction, move_limit,
                                        problem.volume_fraction, transform,
                                        passive)

    density = problem.apply_passive(transform.physical(design))
    compliance, _, _ = compliance_and_sensitivity(problem, density, device=device)
    result.compliance_history.append(float(compliance))
    result.volume_history.append(float(density.mean()))
    result.grey_history.append(_grey(density))
    result.beta_history.append(transform.beta)
    result.design = design
    result.density = density
    return result


@dataclass
class ProjectedStressResult(ThreeFieldResult):
    """A three-field run carrying an aggregated stress constraint."""

    p_norm_history: list[float] = field(default_factory=list)
    max_stress_history: list[float] = field(default_factory=list)
    best_feasible_design: np.ndarray | None = None
    best_feasible_density: np.ndarray | None = None
    best_feasible_compliance: float | None = None
    best_feasible_p_norm: float | None = None
    best_feasible_max_stress_pa: float | None = None

    @property
    def found_feasible(self) -> bool:
        return self.best_feasible_density is not None

    @property
    def feasible(self) -> bool:
        return bool(self.p_norm_history) and self.p_norm_history[-1] <= 1.0


def optimize_projected_stress(problem, max_iterations: int = 120,
                              move_limit: float = 0.1,
                              penalty_weight: float = 1.0,
                              penalty_growth: float = 1.15,
                              transform: DesignTransform | None = None,
                              schedule: BetaSchedule | None = None,
                              device: str | None = None,
                              callback=None) -> ProjectedStressResult:
    """Compliance minimisation with a stress constraint, on the three-field design.

    `problem` is a `StressProblem`. Same penalty treatment as Phase 14a: both
    gradients normalised by their own largest entry before combining, the
    weight growing while the constraint is violated, and the best feasible
    iterate kept because the method oscillates about the boundary.
    """
    from .stress import p_norm_sensitivity, relaxed_stress

    base = problem.base
    mesh = base.mesh
    if transform is None:
        transform = DesignTransform.for_mesh(mesh, base.filter_radius_elements)
    if schedule is None:
        schedule = BetaSchedule()

    design = initial_design(mesh.n_elements, base.volume_fraction, transform)
    result = ProjectedStressResult(design=design,
                                   density=transform.physical(design))
    weight = float(penalty_weight)

    for iteration in range(1, max_iterations + 1):
        previous_beta = transform.beta
        beta = schedule.apply(transform, iteration)
        if beta != previous_beta:
            design = restore_volume(design, base.volume_fraction, transform)
        density = transform.physical(design)
        compliance, d_compliance, _ = compliance_and_sensitivity(base, density,
                                                                 device=device)
        p_norm, d_p_norm, solution, _ = p_norm_sensitivity(problem, density,
                                                           device)
        relaxed, _ = relaxed_stress(problem, density, solution.displacements)

        result.compliance_history.append(float(compliance))
        result.volume_history.append(float(density.mean()))
        result.grey_history.append(_grey(density))
        result.beta_history.append(beta)
        result.p_norm_history.append(p_norm)
        result.max_stress_history.append(float(relaxed.max()))
        result.iterations = iteration
        if p_norm <= 1.0 and (result.best_feasible_compliance is None
                              or compliance < result.best_feasible_compliance):
            result.best_feasible_design = design.copy()
            result.best_feasible_density = density.copy()
            result.best_feasible_compliance = float(compliance)
            result.best_feasible_p_norm = p_norm
            result.best_feasible_max_stress_pa = float(relaxed.max())
        if callback is not None:
            callback(iteration, float(compliance), p_norm, beta)

        violation = max(0.0, p_norm - 1.0)
        dc = d_compliance / max(np.abs(d_compliance).max(), 1e-30)
        ds = d_p_norm / max(np.abs(d_p_norm).max(), 1e-30)
        gradient = transform.chain(dc + weight * 2.0 * violation * ds, design)
        direction = gradient / max(np.abs(gradient).max(), 1e-30)
        design = _volume_projected_step(design, direction, move_limit,
                                        base.volume_fraction, transform)
        if violation > 0:
            weight *= penalty_growth

    density = transform.physical(design)
    compliance, _, _ = compliance_and_sensitivity(base, density, device=device)
    p_norm, _, solution, _ = p_norm_sensitivity(problem, density, device)
    relaxed, _ = relaxed_stress(problem, density, solution.displacements)
    result.compliance_history.append(float(compliance))
    result.volume_history.append(float(density.mean()))
    result.grey_history.append(_grey(density))
    result.beta_history.append(transform.beta)
    result.p_norm_history.append(p_norm)
    result.max_stress_history.append(float(relaxed.max()))
    if p_norm <= 1.0 and (result.best_feasible_compliance is None
                          or compliance < result.best_feasible_compliance):
        result.best_feasible_design = design.copy()
        result.best_feasible_density = density.copy()
        result.best_feasible_compliance = float(compliance)
        result.best_feasible_p_norm = p_norm
        result.best_feasible_max_stress_pa = float(relaxed.max())
    result.design = design
    result.density = density
    return result
