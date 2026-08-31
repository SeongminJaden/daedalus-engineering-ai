"""physics.fem.verify: the high-fidelity gate of the multi-fidelity funnel.

Cheap models (surrogate, then beam theory) rank and shortlist candidates. This
module is what a shortlisted design must pass before it is treated as anything
more than a candidate.

READ THIS BEFORE USING A STRESS NUMBER FROM HERE
------------------------------------------------
A perfectly clamped face is a **stress singularity**. Linear elasticity says the
stress at the re-entrant corner of an ideal built-in support is unbounded, so
the peak stress a mesh reports is a function of the mesh, not of the part: it
keeps climbing as you refine, and never converges. Measured on a solid
cantilever here, peak von Mises went 3.485 -> 3.842 -> 4.043 -> 4.314 MPa across
four refinements and was still rising.

So this module reports two different things and is explicit about which is which:

  * `peak_von_mises_pa`  : the raw maximum. **Mesh dependent, does NOT converge.**
                           Useful as a relative comparison between designs meshed
                           the same way. Not a number to certify a part with.
  * `gauge_von_mises_pa` : von Mises at a gauge station offset from the support,
                           by default one section height away. Far enough from the
                           corner that it is in the smooth bending field, and it
                           **does** converge.

Deflection has no such problem and converges cleanly, so it is the trustworthy
stiffness measure.

A real support is never perfectly rigid, and a real part has a fillet. Resolving
what the true peak stress is needs the actual joint geometry, which is outside
this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.profile import load_profile

from .mesh import Mesh, hollow_rect_mesh
from .solver import solve_linear_elasticity

FIDELITY = "fem3d"

# Gauge station, in multiples of the section height away from the support.
DEFAULT_GAUGE_OFFSET_FACTOR = 1.0


@dataclass
class HighFidelityResult:
    """Outcome of a 3D FEM verification run."""

    fidelity: str = FIDELITY
    tip_deflection_m: float = 0.0
    beam_tip_deflection_m: float = 0.0
    deflection_ratio: float = 0.0

    peak_von_mises_pa: float = 0.0          # mesh dependent, see module docstring
    gauge_von_mises_pa: float = 0.0         # convergent measure
    gauge_station_m: float = 0.0
    nominal_beam_stress_at_gauge_pa: float = 0.0
    gauge_agreement: float = 0.0            # gauge / nominal, ~1 validates the solve

    beam_root_stress_pa: float = 0.0
    stress_concentration_factor: float = 0.0

    safety_factor_peak: float = 0.0         # conservative, mesh dependent
    safety_factor_gauge: float = 0.0        # convergent

    n_dofs: int = 0
    iterations: int = 0
    converged: bool = False
    mesh: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["mesh"] = dict(self.mesh)
        d["warnings"] = list(self.warnings)
        return d


def size_mesh_for_budget(
    length_m: float, outer_height_m: float, outer_width_m: float,
    wall_thickness_m: float, max_dofs: int,
    elements_through_wall: int = 2, min_nx: int = 6, max_nx: int = 64,
) -> Mesh:
    """Largest axial refinement that fits the profile's DOF budget.

    The cross-section resolution is fixed by `elements_through_wall` because the
    wall is the feature that must be resolved; the axial count is what gets
    traded against the budget.
    """
    best = None
    for nx in range(max_nx, min_nx - 1, -1):
        mesh = hollow_rect_mesh(length_m, outer_height_m, outer_width_m,
                                wall_thickness_m, nx=nx,
                                elements_through_wall=elements_through_wall)
        if mesh.n_dofs <= max_dofs:
            return mesh
        best = mesh
    if best is None:
        raise ValueError("could not build a mesh")
    # Even the coarsest axial mesh exceeds the budget: return it and let the
    # caller warn, rather than silently returning something unsolvable.
    return hollow_rect_mesh(length_m, outer_height_m, outer_width_m,
                            wall_thickness_m, nx=min_nx,
                            elements_through_wall=elements_through_wall)


def high_fidelity_verify(
    genome,
    problem,
    profile: str | None = None,
    elements_through_wall: int = 2,
    gauge_offset_factor: float = DEFAULT_GAUGE_OFFSET_FACTOR,
    max_dofs: int | None = None,
    device: str | None = None,
) -> HighFidelityResult:
    """Run 3D FEM on one design and report it against the beam-theory result."""
    from core.materials import get_material
    from physics.structural import load_case_from_problem

    if not genome.is_valid():
        raise ValueError(f"invalid genome: {genome.validity_reason()}")

    case = load_case_from_problem(problem)
    material = get_material(problem.material_id)
    section = genome.section
    props = section.section_properties()

    if max_dofs is None:
        cfg = load_profile(profile)
        max_dofs = int(cfg["simulation"].get("fem_max_dofs", 150000))

    mesh = size_mesh_for_budget(
        case.length_m, section.outer_height_m, section.outer_width_m,
        section.wall_thickness_m, max_dofs,
        elements_through_wall=elements_through_wall)

    warnings: list[str] = []
    if mesh.n_dofs > max_dofs:
        warnings.append(
            f"mesh has {mesh.n_dofs} DOFs, above the profile budget {max_dofs}; "
            f"the wall needs {elements_through_wall} elements through thickness"
        )

    root_nodes = mesh.nodes_at_x(0.0)
    tip_nodes = mesh.nodes_at_x(mesh.nx * mesh.dx)
    solution = solve_linear_elasticity(
        mesh, case.youngs_modulus_pa, material.poisson_ratio,
        fixed_nodes=root_nodes, load_nodes=tip_nodes,
        total_load_n=-case.tip_load_n, load_direction=1, device=device)

    if not solution.report.converged:
        warnings.append(
            f"CG did not reach tolerance (residual {solution.report.residual:.2e} "
            f"after {solution.report.iterations} iterations); results are suspect"
        )

    centroids = mesh.element_centroids()
    vm = solution.element_von_mises

    # --- deflection (converges) ---
    tip = abs(solution.tip_deflection())
    inertia = props.i_x_m4
    beam_tip = case.tip_load_n * case.length_m ** 3 / (
        3.0 * case.youngs_modulus_pa * inertia)

    # --- stresses ---
    peak = float(vm.max())
    half_height = section.outer_height_m / 2.0
    beam_root = case.tip_load_n * case.length_m * half_height / inertia

    offset = gauge_offset_factor * section.outer_height_m
    band = (centroids[:, 0] >= offset) & (centroids[:, 0] < offset + mesh.dx * 1.001)
    if not band.any():
        warnings.append(
            f"gauge station at {offset:.4f} m falls outside the mesh; "
            "falling back to the mid-span element"
        )
        band = np.abs(centroids[:, 0] - case.length_m / 2) < mesh.dx
    idx = np.flatnonzero(band)
    j = idx[int(np.argmax(vm[idx]))]
    gauge = float(vm[j])
    gauge_x, gauge_y = float(centroids[j, 0]), float(centroids[j, 1])

    # Beam stress at the gauge element's own station AND fibre, so the
    # comparison is like for like. Comparing an element-centre stress against
    # the extreme-fibre value would look like a 20-30% error that is really
    # just the sampling position.
    moment_at_gauge = case.tip_load_n * (case.length_m - gauge_x)
    nominal = abs(moment_at_gauge * (gauge_y - half_height) / inertia)

    yield_strength = material.yield_strength_pa
    warnings.append(
        "peak_von_mises_pa is mesh dependent: a perfectly clamped face is a "
        "stress singularity, so the peak keeps rising with refinement and never "
        "converges. Use gauge_von_mises_pa for a converged stress, and treat the "
        "peak as a relative indicator only."
    )

    return HighFidelityResult(
        tip_deflection_m=tip,
        beam_tip_deflection_m=beam_tip,
        deflection_ratio=tip / beam_tip if beam_tip else 0.0,
        peak_von_mises_pa=peak,
        gauge_von_mises_pa=gauge,
        gauge_station_m=gauge_x,
        nominal_beam_stress_at_gauge_pa=nominal,
        gauge_agreement=gauge / nominal if nominal else 0.0,
        beam_root_stress_pa=beam_root,
        stress_concentration_factor=peak / beam_root if beam_root else 0.0,
        safety_factor_peak=yield_strength / peak if peak else 0.0,
        safety_factor_gauge=yield_strength / gauge if gauge else 0.0,
        n_dofs=mesh.n_dofs,
        iterations=solution.report.iterations,
        converged=solution.report.converged,
        mesh=mesh.summary(),
        warnings=warnings,
    )
