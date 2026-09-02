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
refinement. So every solver label here carries `mesh_sensitivity`, the
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
from pathlib import Path
from typing import Any

import numpy as np

from brain.semantic.evidence import EvidenceKind
from core.materials.db import MaterialSpec

from .schema import label


@dataclass(frozen=True)
class LoadCase:
    """The one cantilever case every family is labelled under."""

    total_load_n: float = -100.0
    direction: int = 1              # 1 is y, 2 is z
    fixed_axis: int = 0             # clamp the face at the minimum of this axis
    fixed_side: str = "min"
    loaded_side: str = "max"

    def as_dict(self) -> dict[str, Any]:
        return {"total_load_n": self.total_load_n, "direction": self.direction,
                "fixed": f"{'xyz'[self.fixed_axis]}-{self.fixed_side} face, "
                         f"all dof",
                "loaded": f"{'xyz'[self.fixed_axis]}-{self.loaded_side} face "
                          f"nodes, force spread evenly"}


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


def _solve(step_path: Path, size_m: float, material: MaterialSpec,
           case: LoadCase):
    from nodes import calculix as ccx
    from nodes import gmsh_node as gm

    mesh = gm.tetrahedral_mesh_from_step(str(step_path), size_m, order=2)
    fixed = mesh.nodes_at_extreme(case.fixed_axis, case.fixed_side)
    loaded = mesh.nodes_at_extreme(case.fixed_axis, case.loaded_side)
    result = ccx.solve(mesh, material.youngs_modulus_pa,
                       material.poisson_ratio, fixed, loaded,
                       total_load_n=case.total_load_n,
                       load_direction=case.direction,
                       element_type=ccx.ElementType.C3D10)
    if not result.converged:
        raise RuntimeError(f"CalculiX reported an error on {step_path.name}")
    tip = float(result.displacements[loaded, case.direction].mean())
    return mesh, result, tip


def _sensitivity(fine: float, coarse: float) -> float:
    scale = max(abs(fine), 1e-300)
    return abs(fine - coarse) / scale


def cantilever_labels(step_path: str | Path, volume_m3: float,
                      bounding_box_m: tuple[float, float, float],
                      material: MaterialSpec, case: LoadCase) -> LabelReport:
    """Mass from geometry, deflection and stress from CalculiX, all SIMULATED."""
    from nodes import calculix as ccx

    step_path = Path(step_path)
    started = time.perf_counter()
    coarse_size, fine_size = mesh_sizes_for(bounding_box_m)
    coarse_mesh, coarse, coarse_tip = _solve(step_path, coarse_size, material,
                                             case)
    fine_mesh, fine, fine_tip = _solve(step_path, fine_size, material, case)
    solver = f"calculix {ccx.version() or 'unknown'} C3D10"
    mesh_note = (f"quadratic tetrahedra, target {fine_size * 1e3:.2f} mm, "
                 f"{fine_mesh.n_nodes} nodes; coarse control at "
                 f"{coarse_size * 1e3:.2f} mm, {coarse_mesh.n_nodes} nodes")

    labels = {
        "mass_kg": label(
            volume_m3 * material.density_kg_m3, "kg", EvidenceKind.ANALYTICAL,
            "brep_volume_times_density",
            note=f"B-rep volume from the STEP analyzer times "
                 f"{material.id} density {material.density_kg_m3:g} kg/m3"),
        "tip_deflection_m": label(
            fine_tip, "m", EvidenceKind.SIMULATION, solver,
            note="mean displacement of the loaded face in the load direction; "
                 + mesh_note,
            mesh_sensitivity=_sensitivity(fine_tip, coarse_tip)),
        "max_displacement_m": label(
            fine.max_displacement_magnitude(), "m", EvidenceKind.SIMULATION,
            solver, note=mesh_note,
            mesh_sensitivity=_sensitivity(fine.max_displacement_magnitude(),
                                          coarse.max_displacement_magnitude())),
        "max_von_mises_pa": label(
            fine.max_von_mises_pa(), "Pa", EvidenceKind.SIMULATION, solver,
            note="element peak next to a fully clamped face, which is a stress "
                 "singularity: this value does NOT converge under refinement "
                 "and must not be used to certify anything; " + mesh_note,
            mesh_sensitivity=_sensitivity(fine.max_von_mises_pa(),
                                          coarse.max_von_mises_pa())),
    }
    labels["load_case"] = {**case.as_dict(), "material_id": material.id,
                           "evidence": labels["tip_deflection_m"]["evidence"],
                           "kind": EvidenceKind.SIMULATION.value}
    return LabelReport(labels=labels, coarse_nodes=coarse_mesh.n_nodes,
                       fine_nodes=fine_mesh.n_nodes,
                       seconds=time.perf_counter() - started, solver=solver)
