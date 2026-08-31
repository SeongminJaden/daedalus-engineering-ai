"""Executing a topology strategy against the same engineering problem.

The parametric strategy searches section dimensions of a fixed shape. This one
searches a density field over the whole design envelope, so the two produce
genuinely different designs for the same problem and the loop can compare them.

**Mass is minimised by searching the volume fraction, not by asking SIMP to do
it.** SIMP minimises compliance at a fixed volume; it has no mass objective. So
this runs SIMP at a volume fraction, evaluates the resulting field against the
problem's real constraints with the 3D FEM, and bisects the volume fraction to
find the lightest field that still satisfies them. That is a small outer search
around the topology solver, and it is the reason a topology iteration is
expensive: the registry rates these methods HEAVY for exactly this reason.

The stress-constrained variant differs only in which optimiser runs inside the
bisection.
"""

from __future__ import annotations

import time

import numpy as np

from core.materials import get_material
from optimization.topology.simp import SimpProblem, optimize
from optimization.topology.stress import StressProblem, optimize_constrained
from physics.fem.mesh import solid_box_mesh
from physics.fem.solver import solve_linear_elasticity
from optimization.topology.simp import stiffness_scale

from .outcome import DesignOutcome

COMPLIANCE_METHOD = "topology_compliance"
STRESS_METHOD = "topology_stress"

# The envelope is meshed this finely. Small on purpose: every bisection step is
# a full topology run, and the loop has an evaluation budget to respect.
DEFAULT_DIVISIONS = (16, 6, 2)

# Volume fractions the bisection searches between.
MIN_VOLUME_FRACTION = 0.05
MAX_VOLUME_FRACTION = 0.6


def design_domain(problem, divisions: tuple[int, int, int] = DEFAULT_DIVISIONS):
    """The voxel envelope the topology strategies design inside.

    Taken from the Engineering IR geometry limits, so both strategies are
    working the same physical problem. A missing envelope limit is an error
    rather than a guess: inventing a design domain would make the two
    strategies solve different problems while reporting one.
    """
    geometry = problem.geometry
    if geometry.max_height_m is None or geometry.max_width_m is None:
        raise ValueError(
            "topology strategies need the geometry envelope "
            "(max_height_m and max_width_m); the problem states neither, and "
            "assuming one would silently change the problem being solved")
    nx, ny, nz = divisions
    return solid_box_mesh(geometry.length_m, geometry.max_height_m,
                          geometry.max_width_m, nx, ny, nz)


def _evaluate_field(op, mesh, density, material) -> tuple[float, float, float]:
    """Mass, tip deflection and peak stress of a density field."""
    load = op.problem.loads[0]
    magnitude = float(load.magnitude_n)
    solution = solve_linear_elasticity(
        mesh, material.youngs_modulus_pa, material.poisson_ratio,
        fixed_nodes=mesh.nodes_at_x(0.0),
        load_nodes=mesh.nodes_at_x(op.problem.geometry.length_m),
        total_load_n=-magnitude, load_direction=1,
        element_scale=stiffness_scale(density, 3.0))
    displacements = solution.displacements.reshape(-1, 3)
    deflection = float(np.abs(displacements[:, 1]).max())
    element_volume = mesh.dx * mesh.dy * mesh.dz
    mass = float(density.sum() * element_volume * material.density_kg_m3)
    return mass, deflection, float(solution.compliance())


def run(op, method: str = COMPLIANCE_METHOD, iterations: int = 30,
        bisection_steps: int = 4,
        divisions: tuple[int, int, int] = DEFAULT_DIVISIONS,
        **_: object) -> DesignOutcome:
    """Search the lightest density field that still meets the constraints."""
    began = time.monotonic()
    material = get_material(op.problem.material_id)
    mesh = design_domain(op.problem, divisions)
    load = op.problem.loads[0]

    def build(volume_fraction: float) -> SimpProblem:
        return SimpProblem(
            mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
            poisson_ratio=material.poisson_ratio,
            fixed_nodes=mesh.nodes_at_x(0.0),
            load_nodes=mesh.nodes_at_x(op.problem.geometry.length_m),
            total_load_n=-float(load.magnitude_n), load_direction=1,
            volume_fraction=volume_fraction, filter_radius_elements=1.5)

    def solve_at(volume_fraction: float) -> np.ndarray:
        base = build(volume_fraction)
        if method == STRESS_METHOD:
            stress_problem = StressProblem(base=base,
                                           stress_limit_pa=op.allowable_stress_pa,
                                           p_norm=8.0)
            result = optimize_constrained(stress_problem,
                                          max_iterations=iterations)
            if result.found_feasible:
                return result.best_feasible_density
            return result.density
        return optimize(base, max_iterations=iterations).density

    # Bisect on volume fraction: heavier is more likely to satisfy the
    # deflection limit, so feasibility is monotone enough to bisect on.
    low, high = MIN_VOLUME_FRACTION, MAX_VOLUME_FRACTION
    best: tuple[float, np.ndarray, float, float] | None = None
    runs = 0
    for _ in range(bisection_steps):
        middle = 0.5 * (low + high)
        density = solve_at(middle)
        runs += 1
        mass, deflection, compliance = _evaluate_field(op, mesh, density,
                                                       material)
        satisfied = (op.max_deflection_m is None
                     or deflection <= op.max_deflection_m)
        if satisfied:
            if best is None or mass < best[0]:
                best = (mass, density.copy(), deflection, compliance)
            high = middle
        else:
            low = middle

    elapsed = time.monotonic() - began
    if best is None:
        # Nothing in the searched range met the constraint. Report the heaviest
        # attempt as infeasible rather than returning the lightest and calling
        # the shortfall a rounding matter.
        density = solve_at(MAX_VOLUME_FRACTION)
        runs += 1
        mass, deflection, compliance = _evaluate_field(op, mesh, density,
                                                       material)
        margin = (0.0 if op.max_deflection_m is None
                  else op.max_deflection_m - deflection)
        return DesignOutcome(
            method=method, mass_kg=mass, feasible=False,
            constraints={"deflection": float(margin)},
            evaluations=runs, seconds=elapsed, converged=False,
            density_field=density,
            detail={"tip_deflection_m": deflection, "compliance_j": compliance,
                    "volume_fraction": float(density.mean())})

    mass, density, deflection, compliance = best
    margin = (0.0 if op.max_deflection_m is None
              else op.max_deflection_m - deflection)
    return DesignOutcome(
        method=method, mass_kg=mass, feasible=True,
        constraints={"deflection": float(margin)},
        evaluations=runs, seconds=elapsed, converged=True,
        density_field=density,
        detail={"tip_deflection_m": deflection, "compliance_j": compliance,
                "volume_fraction": float(density.mean())})
