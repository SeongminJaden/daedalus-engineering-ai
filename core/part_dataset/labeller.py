"""Physical labels for a synthetic part, from the solvers that already exist.

The engine adds no physics. A part goes down the route this project already
verified for general shapes: Gmsh meshes the STEP file with quadratic
tetrahedra, CalculiX solves one cantilever load case, and the numbers come
back in the same terms the in-house solver reports. Mass comes from the
analyzer's B-rep volume and the material's density, which is arithmetic.

Every label is graded SIMULATED. Not because a caller chose that, but because
`core.part_dataset.schema.label` grades by the kind of evidence, and one solve
is a simulation. The grade cannot be raised from here.

WHAT WAS MEASURED BEFORE THIS WAS WRITTEN
=========================================
On a 200 by 40 by 30 mm hollow box, two mesh sizes:

    size 10 mm   3790 nodes   solve 0.4 s   max displacement 3.466e-5 m   peak von Mises 3.17 MPa
    size  6 mm  10292 nodes   solve 1.2 s   max displacement 3.468e-5 m   peak von Mises 3.62 MPa

The displacement moved 0.06 percent between meshes. The peak stress moved 14
percent, and it will keep moving, because a fully clamped face produces a
stress singularity at its edge and the peak does not converge under
refinement.

Two parts in the first fifty five refused to solve at all: a stepped shaft
of radius 12.8 mm meshed at 9.9 mm, and a two hole plate, each with a
quadratic tetrahedron on a curved face whose Jacobian went nonpositive, so
CalculiX returned nothing. The shaft solved at 6.7 mm. Gmsh's high order
optimiser fixed both and then terminated the Python process on a third
part with a C++ exception nothing can catch, so it is not used. Instead the
labeller RETRIES: when the solver returns nothing, the mesh is rebuilt at
0.7 of the size, up to twice, and the size actually used is recorded in
the label's note. A part that fails all three is refused with the solver's
message. A solver that returns nothing is a refusal with the
solver's message in the report, never a silent drop. So every solver label here carries `mesh_sensitivity`, the
relative change between the coarse and the fine mesh, and the stress label
says in its note that the peak is not a converged quantity. A consumer that
ignores the note has been told.

VALIDITY DOMAIN
===============
    Linear elastic, small strain, isotropic. One load case: the x-minimum
    face fully clamped, a total force spread over the x-maximum face nodes.
    That is a cantilever and nothing else; a part that is never loaded that
    way in service has been labelled for a load it does not see. The load
    magnitude is fixed and recorded, so a consumer can scale linearly and
    knows that it may.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from brain.semantic.evidence import EvidenceKind
from core.materials.db import MaterialSpec

from .schema import label


class LoadKind(str, Enum):
    """The five load cases the dataset specification names.

    Each was checked against a closed form before it was allowed to label
    anything; the numbers are in the labeller tests and in dataset_spec.md.
    """

    BENDING = "bending"              # force across the axis on the free face
    AXIAL = "axial"                  # force along the axis on the free face
    TORSION = "torsion"              # torque about the axis on the free face
    COMBINED = "combined"            # bending force and torque together
    THERMAL_GRADIENT = "thermal_gradient"   # temperature rising across the
                                            # section, clamped root, no force


@dataclass(frozen=True)
class LoadCase:
    """One load case, clamped at the x minimum face, acting on the x maximum.

    `direction` is the transverse axis for bending and for the thermal
    gradient (1 is y, 2 is z). `torque_nm` is used by torsion and combined.
    `gradient_k_per_m` and `delta_k` describe the thermal case; the material's
    expansion coefficient comes from the database and a material without one
    is refused.
    """

    total_load_n: float = -100.0
    direction: int = 1              # 1 is y, 2 is z
    fixed_axis: int = 0             # clamp the face at the minimum of this axis
    fixed_side: str = "min"
    loaded_side: str = "max"
    kind: LoadKind = LoadKind.BENDING
    torque_nm: float = 5.0
    gradient_k_per_m: float = 1000.0
    delta_k: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        d = {"load_kind": self.kind.value, "direction": self.direction,
             "fixed": f"{'xyz'[self.fixed_axis]}-{self.fixed_side} face, all dof",
             "loaded": f"{'xyz'[self.fixed_axis]}-{self.loaded_side} face nodes"}
        if self.kind in (LoadKind.BENDING, LoadKind.AXIAL, LoadKind.COMBINED):
            d["total_load_n"] = self.total_load_n
        if self.kind in (LoadKind.TORSION, LoadKind.COMBINED):
            d["torque_nm"] = self.torque_nm
        if self.kind is LoadKind.THERMAL_GRADIENT:
            d["gradient_k_per_m"] = self.gradient_k_per_m
            d["delta_k"] = self.delta_k
        return d


@dataclass
class LabelReport:
    labels: dict[str, dict[str, Any]]
    coarse_nodes: int
    fine_nodes: int
    seconds: float
    solver: str


def labelling_available() -> bool:
    from nodes import calculix as ccx
    from nodes import gmsh_node as gm
    return gm.is_available() and ccx.is_available()


def mesh_sizes_for(bounding_box_m: tuple[float, float, float]
                   ) -> tuple[float, float]:
    """Coarse and fine target element sizes, from the part's longest side.

    Fifteen and twenty two elements along the longest side. Gmsh is allowed
    down to a quarter of the target where the geometry demands it, so a thin
    wall still gets elements across it.
    """
    longest = max(bounding_box_m)
    return longest / 15.0, longest / 22.0


#: How much finer to mesh after the solver rejects a mesh, and how often.
RETRY_FACTOR = 0.7
MAX_RETRIES = 2


def torque_forces(points: np.ndarray, torque_nm: float, axis: int = 0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodal forces on a face that sum to a pure torque about `axis`.

    Each node gets a tangential force proportional to its radius from the
    face centroid, F_i = T r_i / sum(r_j^2), so the moments sum to T exactly
    and the net force is zero to round-off on a symmetric face. Returns the
    forces, the unit tangents and the radii, which the twist measurement
    reuses. Nodes on the axis itself get no force and no tangent.
    """
    centre = points.mean(axis=0)
    r = points - centre
    r[:, axis] = 0.0
    radii = np.linalg.norm(r, axis=1)
    unit_axis = np.zeros(3)
    unit_axis[axis] = 1.0
    tangent = np.cross(unit_axis, r)
    norms = np.linalg.norm(tangent, axis=1)
    on_axis = norms < 1e-12
    unit_tangent = np.zeros_like(tangent)
    unit_tangent[~on_axis] = tangent[~on_axis] / norms[~on_axis, None]
    scale = torque_nm / np.sum(radii[~on_axis] ** 2)
    forces = (scale * radii)[:, None] * unit_tangent
    return forces, unit_tangent, radii


def _nodal_forces(case: LoadCase, mesh, loaded: np.ndarray) -> np.ndarray | None:
    """The force vector per loaded node for the case, or None for the plain
    equal division the deck writer does itself."""
    if case.kind in (LoadKind.BENDING, LoadKind.AXIAL, LoadKind.THERMAL_GRADIENT):
        return None
    points = np.asarray(mesh.node_coords)[loaded]
    forces = np.zeros_like(points)
    if case.kind in (LoadKind.TORSION, LoadKind.COMBINED):
        forces += torque_forces(points, case.torque_nm, case.fixed_axis)[0]
    if case.kind is LoadKind.COMBINED:
        forces[:, case.direction] += case.total_load_n / len(loaded)
    return forces


def _thermal(case: LoadCase, material: MaterialSpec):
    from nodes import calculix as ccx

    if case.kind is not LoadKind.THERMAL_GRADIENT:
        return None
    if material.thermal_expansion_1_k is None:
        raise RuntimeError(
            f"{material.id} has no sourced thermal expansion coefficient; the "
            f"thermal case cannot be labelled for it")
    return ccx.ThermalLoad(expansion_1_k=material.thermal_expansion_1_k,
                           delta_k=case.delta_k,
                           gradient_k_per_m=case.gradient_k_per_m,
                           gradient_axis=case.direction)


def face_twist_rad(mesh, displacements: np.ndarray, loaded: np.ndarray,
                   axis: int = 0) -> float:
    """Rotation of a face about `axis` from its nodal displacements.

    The face's mean displacement is removed first, so a bending deflection
    that translates the whole face does not read as a twist; what remains is
    projected on each node's tangent and divided by its radius.
    """
    u = displacements[loaded]
    u = u - u.mean(axis=0)
    points = np.asarray(mesh.node_coords)[loaded]
    _, unit_tangent, radii = torque_forces(points, 1.0, axis)
    off = radii > 1e-12
    tangential = np.einsum("ij,ij->i", u[off], unit_tangent[off])
    return float(np.mean(tangential / radii[off]))


def _primary_response(case: LoadCase, mesh, result, loaded: np.ndarray) -> float:
    """The one number each case is about: a tip deflection, an elongation,
    a twist, or, for the thermal gradient, the transverse tip deflection."""
    u = result.displacements[loaded]
    if case.kind in (LoadKind.BENDING, LoadKind.COMBINED,
                     LoadKind.THERMAL_GRADIENT):
        return float(u[:, case.direction].mean())
    if case.kind is LoadKind.AXIAL:
        return float(u[:, case.fixed_axis].mean())
    return face_twist_rad(mesh, result.displacements, loaded, case.fixed_axis)


def _solve(step_path: Path, size_m: float, material: MaterialSpec,
           case: LoadCase):
    """Mesh and solve, retrying finer when the mesher or solver fails.

    Returns the mesh, the result, the primary response and the size that
    was actually used, which the caller records because a label produced at
    a different size than the one asked for has to say so.
    """
    from nodes import calculix as ccx
    from nodes import gmsh_node as gm

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        size = size_m * RETRY_FACTOR ** attempt
        try:
            mesh = gm.tetrahedral_mesh_from_step(str(step_path), size, order=2)
        except Exception as exc:  # gmsh raises a bare Exception
            # A boundary mesh that fails at one size (overlapping facets on a
            # face whose thin strip is narrower than the element) usually
            # meshes at the next. Measured on the mount family, seed 0:
            # 12.57 mm fails, 8.80 mm meshes. Retried like a solver rejection.
            last_error = RuntimeError(f"Gmsh: {exc}")
            continue
        fixed = mesh.nodes_at_extreme(case.fixed_axis, case.fixed_side)
        loaded = mesh.nodes_at_extreme(case.fixed_axis, case.loaded_side)
        try:
            result = ccx.solve(mesh, material.youngs_modulus_pa,
                               material.poisson_ratio, fixed, loaded,
                               total_load_n=(case.total_load_n
                                             if case.kind is not
                                             LoadKind.THERMAL_GRADIENT else 0.0),
                               load_direction=(case.fixed_axis
                                               if case.kind is LoadKind.AXIAL
                                               else case.direction),
                               element_type=ccx.ElementType.C3D10,
                               nodal_forces=_nodal_forces(case, mesh, loaded),
                               thermal=_thermal(case, material))
        except RuntimeError as exc:
            last_error = exc
            continue
        if not result.converged:
            last_error = RuntimeError(
                f"CalculiX reported an error on {step_path.name}")
            continue
        primary = _primary_response(case, mesh, result, loaded)
        return mesh, result, primary, size
    raise RuntimeError(
        f"{step_path.name}: the solver returned nothing at {MAX_RETRIES + 1} "
        f"mesh sizes down to {size * 1e3:.2f} mm; last error: {last_error}")


#: What each case's primary label is called, and how it scales with the
#: material under linear elasticity (see scaling.py): displacements go as
#: 1/E, a twist as 1/G, and a thermal deflection as alpha alone.
PRIMARY_LABEL = {
    LoadKind.BENDING: ("tip_deflection_m", "m", "inverse_modulus"),
    LoadKind.AXIAL: ("elongation_m", "m", "inverse_modulus"),
    LoadKind.TORSION: ("twist_rad", "rad", "inverse_shear_modulus"),
    LoadKind.COMBINED: ("tip_deflection_m", "m", "inverse_modulus"),
    LoadKind.THERMAL_GRADIENT: ("thermal_tip_deflection_m", "m", "expansion"),
}


def _sensitivity(fine: float, coarse: float) -> float:
    scale = max(abs(fine), 1e-300)
    return abs(fine - coarse) / scale


def cantilever_labels(step_path: str | Path, volume_m3: float,
                      bounding_box_m: tuple[float, float, float],
                      material: MaterialSpec, case: LoadCase) -> LabelReport:
    """Mass from geometry, response and stress from CalculiX, all SIMULATED.

    The name predates the other load cases; every case is still clamped at
    one end, so it stays. `case.kind` picks bending, axial, torsion, combined
    or a thermal gradient.
    """
    from nodes import calculix as ccx

    step_path = Path(step_path)
    started = time.perf_counter()
    coarse_size, fine_size = mesh_sizes_for(bounding_box_m)
    coarse_mesh, coarse, coarse_tip, coarse_used = _solve(
        step_path, coarse_size, material, case)
    fine_mesh, fine, fine_tip, fine_used = _solve(
        step_path, fine_size, material, case)
    solver = f"calculix {ccx.version() or 'unknown'} C3D10"
    retried = coarse_used != coarse_size or fine_used != fine_size
    mesh_note = (f"quadratic tetrahedra, target {fine_used * 1e3:.2f} mm, "
                 f"{fine_mesh.n_nodes} nodes; coarse control at "
                 f"{coarse_used * 1e3:.2f} mm, {coarse_mesh.n_nodes} nodes"
                 + ("; the mesher or the solver rejected the first mesh and "
                    "a finer one was used" if retried else ""))

    primary_name, primary_unit, primary_scaling = PRIMARY_LABEL[case.kind]
    primary_note = {
        LoadKind.BENDING: "mean displacement of the loaded face in the load direction",
        LoadKind.AXIAL: "mean axial displacement of the loaded face",
        LoadKind.TORSION: "mean tangential displacement over radius on the "
                          "loaded face, a twist angle",
        LoadKind.COMBINED: "mean transverse displacement of the loaded face "
                           "under the bending force, with the torque applied",
        LoadKind.THERMAL_GRADIENT: "mean transverse displacement of the free "
                                   "face under a temperature gradient across "
                                   "the section, no force",
    }[case.kind]
    labels = {
        "mass_kg": label(
            volume_m3 * material.density_kg_m3, "kg", EvidenceKind.ANALYTICAL,
            "brep_volume_times_density",
            note=f"B-rep volume from the STEP analyzer times "
                 f"{material.id} density {material.density_kg_m3:g} kg/m3",
            scaling="density"),
        primary_name: label(
            fine_tip, primary_unit, EvidenceKind.SIMULATION, solver,
            note=primary_note + "; " + mesh_note,
            mesh_sensitivity=_sensitivity(fine_tip, coarse_tip),
            scaling=primary_scaling),
        "max_displacement_m": label(
            fine.max_displacement_magnitude(), "m", EvidenceKind.SIMULATION,
            solver, note=mesh_note,
            mesh_sensitivity=_sensitivity(fine.max_displacement_magnitude(),
                                          coarse.max_displacement_magnitude()),
            scaling=("expansion" if case.kind is LoadKind.THERMAL_GRADIENT
                     else "inverse_shear_modulus" if case.kind is LoadKind.TORSION
                     else "inverse_modulus")),
        "max_von_mises_pa": label(
            fine.max_von_mises_pa(), "Pa", EvidenceKind.SIMULATION, solver,
            note="element peak next to a fully clamped face, which is a stress "
                 "singularity: this value does NOT converge under refinement "
                 "and must not be used to certify anything; " + mesh_note,
            mesh_sensitivity=_sensitivity(fine.max_von_mises_pa(),
                                          coarse.max_von_mises_pa()),
            scaling=("modulus_times_expansion"
                     if case.kind is LoadKind.THERMAL_GRADIENT else "none")),
    }
    if case.kind is LoadKind.COMBINED:
        # the twist under the same combined load, so the record carries both
        # responses the case produces
        loaded_fine = fine_mesh.nodes_at_extreme(case.fixed_axis, case.loaded_side)
        labels["twist_rad"] = label(
            face_twist_rad(fine_mesh, fine.displacements, loaded_fine,
                           case.fixed_axis), "rad",
            EvidenceKind.SIMULATION, solver,
            note="twist of the loaded face under the combined load; " + mesh_note,
            scaling="inverse_shear_modulus")
    labels["load_case"] = {**case.as_dict(), "material_id": material.id,
                           "evidence": labels[primary_name]["evidence"],
                           "kind": EvidenceKind.SIMULATION.value}
    return LabelReport(labels=labels, coarse_nodes=coarse_mesh.n_nodes,
                       fine_nodes=fine_mesh.n_nodes,
                       seconds=time.perf_counter() - started, solver=solver)
