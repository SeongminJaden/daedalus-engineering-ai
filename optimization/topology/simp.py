"""optimization.topology.simp: density-based topology optimization.

SIMP (Solid Isotropic Material with Penalization) on the Phase 7 structured
mesh, solved with the same matrix-free GPU FEM.

    E_e(x_e) = E_min + x_e^p (E0 - E_min),        p = 3
    minimise  c(x) = U^T K(x) U
    subject to  sum_e x_e v_e <= V_frac * V_domain,   x_min <= x_e <= 1

`E_min` is not cosmetic: a truly void element would make K singular and the
solve would fail rather than return a soft region.

The compliance problem is **self-adjoint**, which is why the sensitivity is a
single closed form needing no extra solve:

    dc/dx_e = -p x_e^(p-1) (E0 - E_min)/E0 * u_e^T Ke0 u_e

That derivative is the thing most worth doubting, so it is checked against
finite differences of the actual objective, element by element.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physics.fem.mesh import Mesh
from physics.fem.solver import solve_linear_elasticity

PENALTY = 3.0
MIN_DENSITY = 1e-3
# E_min / E0. Small enough to be structurally void, large enough that the
# stiffness matrix stays non-singular and CG still converges.
VOID_STIFFNESS_RATIO = 1e-9


@dataclass
class SimpProblem:
    """The design domain, its boundary conditions and the SIMP settings."""

    mesh: Mesh
    youngs_modulus_pa: float
    poisson_ratio: float
    fixed_nodes: np.ndarray
    load_nodes: np.ndarray
    total_load_n: float
    load_direction: int = 1
    volume_fraction: float = 0.4
    penalty: float = PENALTY
    min_density: float = MIN_DENSITY
    filter_radius_elements: float = 1.5
    #: Elements held solid throughout the run, and elements held empty. A
    #: point load applied to elements the optimiser is free to empty leaves a
    #: field whose thresholded part has no load path at all; that was measured
    #: (see optimization/topology/verify.py) before this was added. Both
    #: default to absent, so a problem that states neither behaves exactly as
    #: it did before.
    passive_solid: np.ndarray | None = None
    passive_void: np.ndarray | None = None

    def n_elements(self) -> int:
        return self.mesh.n_elements

    def apply_passive(self, density: np.ndarray) -> np.ndarray:
        """Densities with the passive regions written back in."""
        if self.passive_solid is not None:
            density = np.where(self.passive_solid, 1.0, density)
        if self.passive_void is not None:
            density = np.where(self.passive_void, self.min_density, density)
        return density

    @property
    def free_mask(self) -> np.ndarray | None:
        """Elements the optimiser may move, or None when all of them are."""
        if self.passive_solid is None and self.passive_void is None:
            return None
        free = np.ones(self.mesh.n_elements, dtype=bool)
        if self.passive_solid is not None:
            free &= ~np.asarray(self.passive_solid, dtype=bool)
        if self.passive_void is not None:
            free &= ~np.asarray(self.passive_void, dtype=bool)
        return free

    def free_volume_fraction(self) -> float:
        """The volume fraction the OC step should hit over the free elements
        so that the whole domain lands on `volume_fraction`."""
        free = self.free_mask
        if free is None:
            return self.volume_fraction
        n = self.mesh.n_elements
        solid = 0.0 if self.passive_solid is None else float(np.sum(self.passive_solid))
        target = self.volume_fraction * n - solid
        return float(np.clip(target / max(free.sum(), 1), self.min_density, 1.0))


def stiffness_scale(density: np.ndarray, penalty: float = PENALTY,
                    void_ratio: float = VOID_STIFFNESS_RATIO) -> np.ndarray:
    """E(x)/E0 for the SIMP interpolation."""
    x = np.clip(np.asarray(density, dtype=np.float64), 0.0, 1.0)
    return void_ratio + x ** penalty * (1.0 - void_ratio)


def stiffness_scale_derivative(density: np.ndarray, penalty: float = PENALTY,
                               void_ratio: float = VOID_STIFFNESS_RATIO
                               ) -> np.ndarray:
    x = np.clip(np.asarray(density, dtype=np.float64), 0.0, 1.0)
    return penalty * x ** (penalty - 1.0) * (1.0 - void_ratio)


def solve(problem: SimpProblem, density: np.ndarray, device: str | None = None):
    """One FEM solve at the given density field."""
    return solve_linear_elasticity(
        problem.mesh, problem.youngs_modulus_pa, problem.poisson_ratio,
        fixed_nodes=problem.fixed_nodes, load_nodes=problem.load_nodes,
        total_load_n=problem.total_load_n,
        load_direction=problem.load_direction,
        element_scale=stiffness_scale(density, problem.penalty),
        device=device)


def compliance_and_sensitivity(problem: SimpProblem, density: np.ndarray,
                               device: str | None = None):
    """c(x) and dc/dx, from a single solve (the problem is self-adjoint)."""
    solution = solve(problem, density, device)
    energy = solution.element_strain_energy          # u_e^T Ke0 u_e
    scale = stiffness_scale(density, problem.penalty)
    compliance = float(np.sum(scale * energy))
    sensitivity = -stiffness_scale_derivative(density, problem.penalty) * energy
    return compliance, sensitivity, solution


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def build_filter_weights(mesh: Mesh, radius_elements: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour weights for the density filter, on the structured grid.

    Without a filter the solution checkerboards: alternating solid and void
    elements form an artificially stiff pattern that is a numerical artifact,
    not a structure, and the result also becomes mesh dependent. The filter is
    the standard remedy, a linearly decaying cone of radius r_min.
    """
    if radius_elements <= 0:
        raise ValueError("filter radius must be > 0")
    centroids = mesh.element_centroids()
    cell = np.array([mesh.dx, mesh.dy, mesh.dz])
    radius = radius_elements * float(cell.min())

    # Structured grid, so neighbours are found by index rather than a tree.
    index = np.round(centroids / cell).astype(np.int64)
    lookup = {tuple(row): i for i, row in enumerate(index)}
    reach = int(np.ceil(radius_elements))

    rows, weights = [], []
    offsets = [
        (i, j, k)
        for i in range(-reach, reach + 1)
        for j in range(-reach, reach + 1)
        for k in range(-reach, reach + 1)
    ]
    for e, base in enumerate(index):
        neighbours, w = [], []
        for offset in offsets:
            key = tuple(base + np.array(offset))
            target = lookup.get(key)
            if target is None:
                continue
            distance = float(np.linalg.norm((centroids[target] - centroids[e])))
            if distance > radius:
                continue
            w.append(radius - distance)
            neighbours.append(target)
        rows.append(np.array(neighbours, dtype=np.int64))
        weights.append(np.array(w, dtype=np.float64))
    return rows, weights


def apply_sensitivity_filter(sensitivity: np.ndarray, density: np.ndarray,
                             rows, weights, min_density: float = MIN_DENSITY
                             ) -> np.ndarray:
    """The classic Sigmund sensitivity filter."""
    filtered = np.empty_like(sensitivity)
    x = np.maximum(density, min_density)
    for e, (neighbours, w) in enumerate(zip(rows, weights)):
        numerator = float(np.sum(w * x[neighbours] * sensitivity[neighbours]))
        denominator = float(np.sum(w)) * x[e]
        filtered[e] = numerator / denominator if denominator > 0 else sensitivity[e]
    return filtered


def checkerboard_metric(mesh: Mesh, density: np.ndarray) -> float:
    """Mean absolute density difference between face neighbours.

    A checkerboarded field alternates solid and void, so adjacent elements
    differ by nearly 1. A smooth field gives a small value. This is what makes
    the filter's effect measurable rather than a matter of opinion.
    """
    centroids = mesh.element_centroids()
    cell = np.array([mesh.dx, mesh.dy, mesh.dz])
    index = np.round(centroids / cell).astype(np.int64)
    lookup = {tuple(row): i for i, row in enumerate(index)}
    diffs = []
    for e, base in enumerate(index):
        for axis in range(3):
            offset = np.zeros(3, dtype=np.int64)
            offset[axis] = 1
            target = lookup.get(tuple(base + offset))
            if target is not None:
                diffs.append(abs(density[e] - density[target]))
    return float(np.mean(diffs)) if diffs else 0.0


# --------------------------------------------------------------------------- #
# optimality criteria update
# --------------------------------------------------------------------------- #
def oc_update(density: np.ndarray, sensitivity: np.ndarray,
              volume_fraction: float, min_density: float = MIN_DENSITY,
              move_limit: float = 0.2, damping: float = 0.5,
              tolerance: float = 1e-9, max_bisection: int = 200) -> np.ndarray:
    """Optimality criteria step with a bisection on the volume multiplier.

    The multiplier lambda is found by bisection because the volume is monotone
    in it, which makes the search robust without needing a general NLP solver.
    """
    x = np.asarray(density, dtype=np.float64)
    negative = np.maximum(-sensitivity, 1e-30)     # dc/dx is negative
    n = x.shape[0]
    target = volume_fraction * n

    low, high = 1e-12, 1e12
    new = x.copy()
    for _ in range(max_bisection):
        mid = 0.5 * (low + high)
        candidate = x * (negative / mid) ** damping
        new = np.clip(np.clip(candidate, x - move_limit, x + move_limit),
                      min_density, 1.0)
        if new.sum() > target:
            low = mid
        else:
            high = mid
        if (high - low) / max(high, 1e-30) < tolerance:
            break
    return new


@dataclass
class SimpResult:
    density: np.ndarray
    compliance_history: list[float] = field(default_factory=list)
    volume_history: list[float] = field(default_factory=list)
    change_history: list[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False

    @property
    def final_compliance(self) -> float:
        return self.compliance_history[-1]

    @property
    def volume_fraction(self) -> float:
        return float(self.density.mean())


def optimize(problem: SimpProblem, max_iterations: int = 80,
             tolerance: float = 0.01, device: str | None = None,
             use_filter: bool = True, callback=None,
             move_limit: float = 0.1, damping: float = 0.5,
             objective_tolerance: float = 5e-4,
             objective_window: int = 5) -> SimpResult:
    """Run SIMP to convergence or the iteration cap.

    TWO convergence criteria, and both must hold.

    The textbook criterion is "maximum density change below a tolerance". On its
    own that is a trap: heavier damping makes every step small, the criterion is
    met early, and the run reports convergence at a clearly worse design. It was
    measured here: damping 0.3 stopped after 14 iterations at compliance
    1.78e-2, while a gentler step reached 1.01e-2. The design had not converged,
    the STEP SIZE had.

    So convergence also requires the objective to have stopped improving:
    relative compliance change below `objective_tolerance` across the last
    `objective_window` iterations. The default of 5e-4 means the compliance is
    moving by under 0.05% per iteration, which is settled for design purposes;
    it is not tuned to make a particular run report success, and a run that is
    still improving faster than that honestly reports `converged=False`.
    """
    n = problem.n_elements()
    density = problem.apply_passive(
        np.full(n, problem.volume_fraction, dtype=np.float64))
    free = problem.free_mask
    rows = weights = None
    if use_filter:
        rows, weights = build_filter_weights(problem.mesh,
                                             problem.filter_radius_elements)

    result = SimpResult(density=density)
    for iteration in range(1, max_iterations + 1):
        compliance, sensitivity, _ = compliance_and_sensitivity(
            problem, density, device)
        if use_filter:
            sensitivity = apply_sensitivity_filter(
                sensitivity, density, rows, weights, problem.min_density)

        if free is None:
            updated = oc_update(density, sensitivity, problem.volume_fraction,
                                problem.min_density, move_limit=move_limit,
                                damping=damping)
        else:
            # The OC step runs on the free elements only, aiming at the
            # fraction that leaves the whole domain at the requested one; the
            # passive elements are written back afterwards and never move.
            updated = density.copy()
            updated[free] = oc_update(density[free], sensitivity[free],
                                      problem.free_volume_fraction(),
                                      problem.min_density,
                                      move_limit=move_limit, damping=damping)
            updated = problem.apply_passive(updated)
        change = float(np.abs(updated - density).max())
        density = updated

        result.compliance_history.append(compliance)
        result.volume_history.append(float(density.mean()))
        result.change_history.append(change)
        result.iterations = iteration
        if callback is not None:
            callback(iteration, compliance, float(density.mean()), change)

        history = result.compliance_history
        objective_settled = False
        if len(history) > objective_window:
            window = history[-(objective_window + 1):]
            relative = max(abs(b - a) / max(abs(a), 1e-30)
                           for a, b in zip(window, window[1:]))
            objective_settled = relative < objective_tolerance
        if change < tolerance and objective_settled:
            result.converged = True
            break

    result.density = density
    return result
