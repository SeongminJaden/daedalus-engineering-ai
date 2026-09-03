"""Designing for more than one load case, and measuring what one case costs.

A topology optimised for a single load is a structure that is excellent under
that load and can be very poor under any other. That is the best known trap of
the method, and this module both avoids it (weighted compliance over several
cases) and measures it (the same design evaluated under each case in turn).

WHAT A CASE IS HERE
===================
A set of loaded nodes and a force. The structured solver takes either a total
force spread over the nodes in one direction or an explicit force vector, so a
case can be a transverse load, an axial load, or a couple built from opposing
nodal forces, which is how torsion is expressed. The thermal case of the part
dataset has no counterpart here: the Warp solver has no expansion term, and
inventing one would be a different physics rather than another load.

THE OBJECTIVE
=============
The weighted sum of the compliances, with the weights normalised so they sum
to one. Every case is solved separately at every iteration, so the cost of the
run is the number of cases times the cost of one, and the sensitivity is the
same weighted sum of the per-case sensitivities. There is no aggregation
parameter and nothing to tune: the sum is exact.

WHAT THE WEIGHTS DO NOT DO
==========================
They do not make the design good under a case that is not in the list. The
cross evaluation table is the point of this module: it shows what a single
case design does under the others, and the numbers are usually much worse than
anyone expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from physics.fem.mesh import Mesh
from physics.fem.solver import solve_linear_elasticity

from .simp import (SimpProblem, apply_sensitivity_filter, build_filter_weights,
                   oc_update, stiffness_scale, stiffness_scale_derivative)


@dataclass(frozen=True)
class LoadCase:
    """One load on the design domain, with its share of the objective."""

    name: str
    load_nodes: np.ndarray
    total_load_n: float = 0.0
    load_direction: int = 1
    force_vector: np.ndarray | None = None      # (n_dofs,), overrides the above
    weight: float = 1.0


def couple_force_vector(mesh: Mesh, nodes: np.ndarray, torque_nm: float,
                        axis: int = 0) -> np.ndarray:
    """Nodal forces on a face that sum to a pure couple about `axis`.

    The same construction the part labeller uses: force perpendicular to the
    radius, scaled so the total moment is the torque and the net force is zero.
    """
    coords = np.asarray(mesh.node_coords)[nodes]
    centre = coords.mean(axis=0)
    radius = coords - centre
    other = [i for i in range(3) if i != axis]
    lever = radius[:, other]
    tangent = np.zeros_like(lever)
    tangent[:, 0] = -lever[:, 1]
    tangent[:, 1] = lever[:, 0]
    norms = np.linalg.norm(lever, axis=1)
    total = float(np.sum(norms ** 2))
    if total <= 0.0:
        raise ValueError("the loaded face has no lever arm for a couple")
    scale = torque_nm / total
    forces = np.zeros((len(nodes), 3))
    forces[:, other[0]] = tangent[:, 0] * scale
    forces[:, other[1]] = tangent[:, 1] * scale
    vector = np.zeros(mesh.n_dofs)
    for node, force in zip(nodes, forces):
        vector[3 * int(node):3 * int(node) + 3] = force
    return vector


def solve_case(problem: SimpProblem, density: np.ndarray, case: LoadCase,
               device: str | None = None):
    """One case, on the SIMP-scaled stiffness."""
    return solve_linear_elasticity(
        problem.mesh, problem.youngs_modulus_pa, problem.poisson_ratio,
        fixed_nodes=problem.fixed_nodes,
        load_nodes=None if case.force_vector is not None else case.load_nodes,
        total_load_n=0.0 if case.force_vector is not None else case.total_load_n,
        load_direction=case.load_direction, force_vector=case.force_vector,
        element_scale=stiffness_scale(density, problem.penalty),
        device=device)


def case_compliance(problem: SimpProblem, density: np.ndarray, case: LoadCase,
                    device: str | None = None) -> tuple[float, np.ndarray]:
    """Compliance and sensitivity for one case."""
    solution = solve_case(problem, density, case, device)
    energy = solution.element_strain_energy
    compliance = float(np.sum(stiffness_scale(density, problem.penalty) * energy))
    sensitivity = -stiffness_scale_derivative(density, problem.penalty) * energy
    return compliance, sensitivity


def weighted_compliance(problem: SimpProblem, density: np.ndarray,
                        cases: Sequence[LoadCase], device: str | None = None
                        ) -> tuple[float, np.ndarray, dict[str, float]]:
    """The objective, its gradient, and every case's own compliance."""
    weights = np.array([c.weight for c in cases], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("case weights must be non-negative and not all zero")
    weights = weights / weights.sum()
    total = 0.0
    gradient = np.zeros(problem.n_elements())
    per_case: dict[str, float] = {}
    for weight, case in zip(weights, cases):
        compliance, sensitivity = case_compliance(problem, density, case, device)
        per_case[case.name] = compliance
        total += weight * compliance
        gradient += weight * sensitivity
    return total, gradient, per_case


@dataclass
class MultiLoadResult:
    density: np.ndarray
    objective_history: list[float] = field(default_factory=list)
    per_case_history: list[dict] = field(default_factory=list)
    volume_history: list[float] = field(default_factory=list)
    iterations: int = 0

    @property
    def final_compliance(self) -> float:
        """The weighted objective, so the extraction check can read it like a
        single case result."""
        return self.objective_history[-1]

    @property
    def final_per_case(self) -> dict:
        return self.per_case_history[-1]

    @property
    def volume_fraction(self) -> float:
        return float(self.density.mean())


def optimize_multiload(problem: SimpProblem, cases: Sequence[LoadCase],
                       max_iterations: int = 80, move_limit: float = 0.1,
                       damping: float = 0.5, device: str | None = None,
                       use_filter: bool = True) -> MultiLoadResult:
    """Optimality criteria on the weighted compliance."""
    if not cases:
        raise ValueError("no load cases; a design needs at least one")
    n = problem.n_elements()
    # On the volume constraint from the first iterate, passive regions
    # included; see the note in simp.optimize.
    density = problem.apply_passive(np.full(n, problem.free_volume_fraction()))
    free = problem.free_mask
    rows = weights = None
    if use_filter:
        rows, weights = build_filter_weights(problem.mesh,
                                             problem.filter_radius_elements)
    result = MultiLoadResult(density=density)
    for iteration in range(1, max_iterations + 1):
        physical = problem.apply_passive(density)
        objective, sensitivity, per_case = weighted_compliance(
            problem, physical, cases, device)
        if use_filter:
            sensitivity = apply_sensitivity_filter(sensitivity, physical, rows,
                                                   weights, problem.min_density)
        if free is None:
            density = oc_update(density, sensitivity, problem.volume_fraction,
                                problem.min_density, move_limit=move_limit,
                                damping=damping)
        else:
            updated = density.copy()
            updated[free] = oc_update(density[free], sensitivity[free],
                                      problem.free_volume_fraction(),
                                      problem.min_density, move_limit=move_limit,
                                      damping=damping)
            density = problem.apply_passive(updated)
        result.objective_history.append(objective)
        result.per_case_history.append(per_case)
        result.volume_history.append(float(np.mean(problem.apply_passive(density))))
        result.iterations = iteration
    result.density = problem.apply_passive(density)
    return result


def cross_evaluation(problem: SimpProblem, designs: dict[str, np.ndarray],
                     cases: Sequence[LoadCase], device: str | None = None
                     ) -> list[dict]:
    """Every design under every case: the table that shows the trap.

    One row per design, one column per case, each entry that case's compliance
    on that design, so a single case design can be read against the design that
    carried every case.
    """
    rows = []
    for name, density in designs.items():
        row = {"design": name}
        for case in cases:
            compliance, _ = case_compliance(problem, density, case, device)
            row[case.name] = compliance
        rows.append(row)
    return rows


def format_cross_table(rows: list[dict], baseline: str | None = None) -> str:
    if not rows:
        return ""
    cases = [k for k in rows[0] if k != "design"]
    lines = ["| design | " + " | ".join(cases) + " |", "|" + "---|" * (len(cases) + 1)]
    best = {c: min(r[c] for r in rows) for c in cases}
    for r in rows:
        cells = [f"{r[c]:.3e} ({r[c] / best[c]:.2f}x)" for c in cases]
        lines.append(f"| {r['design']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
