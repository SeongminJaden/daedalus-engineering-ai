"""Generating a shape with no family behind it, and labelling it like any other.

Every other generator in this repository searches a parametric family. This
one does not: it takes the design envelope, the load and the supports, runs
topology optimisation, extracts the structure, smooths it to a watertight body
and hands that body to CalculiX. What comes back is a part record whose labels
came from a solve, exactly as a family part's do, and whose shape no family
describes.

WHAT MAKES THIS DIFFERENT FROM THE TOPOLOGY EXECUTOR
====================================================
`agent.execution.topology` returns a density field. A field is not a part, and
the loop records it as a density field with no geometry behind it. This
executor carries the field the rest of the way: threshold, marching cubes,
Taubin smoothing, tetrahedral mesh, solve. The outcome carries a part record
and an STL path, and the numbers in it belong to the body that would be made.

WHAT IT COSTS, AND WHAT IS APPROXIMATE
======================================
    The optimisation itself, one run.
    The extracted body is not the field. Measured in docs/topology_design.md:
    with a low grey fraction the extracted compliance is within three percent,
    with a high one it is not connected at all.
    Second order tetrahedra invert on a marching cubes surface (68 elements
    with a nonpositive Jacobian, measured), so the body is solved with LINEAR
    tetrahedra, which are stiff in bending. Measured against quadratic on
    family parts at the same size: a median of 10.7 percent on the loaded face
    displacement. Every label this executor writes carries that note.

WHAT THE CLASSIFIER SAYS ABOUT THE RESULT
=========================================
UNKNOWN, and that is correct. The rule classifier was measured on five
parametric families and a topology result is none of them. A test pins that
the pipeline still runs: an unrecognised shape is analysed, assessed for
manufacturability and labelled, because nothing downstream may depend on a
part having a family.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from brain.semantic.evidence import EvidenceKind
from core.materials import get_material
from core.part_dataset.schema import (GeometrySummary, PartRecord, Provenance,
                                      ProvenanceKind, TopologySummary, label)
from core.part_dataset.engine import SYNTHETIC_PROVENANCE
from optimization.topology import SimpProblem
from optimization.topology.smooth import (marching_surface, tet_mesh_from_stl,
                                          write_stl)
from optimization.topology.threefield import optimize_projected
from optimization.topology.verify import elements_touching
from physics.fem.mesh import solid_box_mesh

from .outcome import DesignOutcome

METHOD = "freeform_topology"

#: Default discretisation of the envelope. Coarser than the study meshes on
#: purpose: this runs inside a design loop.
DEFAULT_DIVISIONS = (24, 10, 4)
DEFAULT_ITERATIONS = 80
DEFAULT_THRESHOLD = 0.5
DEFAULT_VOLUME_FRACTION = 0.35

#: The element the smoothed body is solved with, and why it is not quadratic.
ELEMENT_NOTE = (
    "linear tetrahedra: second order elements invert on a marching cubes "
    "surface (68 nonpositive Jacobians, measured). Linear tetrahedra are stiff "
    "in bending, by a median of 10.7 percent on the loaded face displacement "
    "against quadratic on family parts at the same size, so this displacement "
    "is a lower bound")

FREEFORM_PROVENANCE = SYNTHETIC_PROVENANCE.model_copy(
    update={"generator": METHOD,
            "source": "daedalus topology optimisation, no parametric family"})


class FreeformFailed(RuntimeError):
    """The pipeline produced no solvable body."""


def _surface_record(part_id: str, surface, tets, material_id: str,
                    labels: dict) -> PartRecord:
    vertices = np.asarray(surface.vertices)
    box = vertices.max(axis=0) - vertices.min(axis=0)
    centre = vertices.mean(axis=0) - vertices.min(axis=0) - box / 2.0
    area = float(np.sum(_triangle_areas(vertices, surface.triangles)))
    return PartRecord(
        part_id=part_id, material_id=material_id,
        provenance=FREEFORM_PROVENANCE,
        geometry=GeometrySummary(volume_m3=surface.volume_m3,
                                 surface_area_m2=area,
                                 bounding_box_m=tuple(float(v) for v in box),
                                 centre_of_mass_m=tuple(float(v) for v in centre)),
        topology=TopologySummary(solids=1, shells=surface.n_components,
                                 faces=int(surface.triangles.shape[0]),
                                 edges=0, vertices=int(vertices.shape[0])),
        labels=labels,
        notes=("a marching cubes body from a density field; its faces are "
               "triangles, not analytic surfaces, so the face count is a "
               "triangle count and there are no edges to report"))


def _triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    b = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def run(op, divisions: tuple[int, int, int] = DEFAULT_DIVISIONS,
        iterations: int = DEFAULT_ITERATIONS, threshold: float = DEFAULT_THRESHOLD,
        volume_fraction: float = DEFAULT_VOLUME_FRACTION,
        smoothing_iterations: int = 10, mesh_size_m: float | None = None,
        step_dir: str | Path | None = None, seed: int = 0,
        **_: object) -> DesignOutcome:
    """Optimise, extract, smooth, mesh, solve, and return a labelled part."""
    began = time.monotonic()
    problem_ir = op.problem
    material = get_material(problem_ir.material_id)
    geometry = problem_ir.geometry
    if geometry.max_height_m is None or geometry.max_width_m is None:
        raise ValueError(
            "a free form run needs the design envelope (max_height_m and "
            "max_width_m); inventing one would change the problem")
    length = float(geometry.length_m)
    mesh = solid_box_mesh(length, float(geometry.max_height_m),
                          float(geometry.max_width_m), *divisions)
    fixed, loaded = mesh.nodes_at_x(0.0), mesh.nodes_at_x(length)
    simp = SimpProblem(
        mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio, fixed_nodes=fixed,
        load_nodes=loaded, total_load_n=-float(problem_ir.loads[0].magnitude_n),
        load_direction=1, volume_fraction=volume_fraction,
        filter_radius_elements=2.5,
        passive_solid=(elements_touching(mesh, loaded)
                       | elements_touching(mesh, fixed)))
    result = optimize_projected(simp, max_iterations=iterations)

    context = tempfile.TemporaryDirectory() if step_dir is None else None
    directory = Path(context.name) if context else Path(step_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        # Islands first, then the surface. A thresholded field routinely
        # leaves material that is attached only through an edge or not at all;
        # it carries no load and would make the body several disconnected
        # shells. The count that was dropped is reported rather than hidden.
        from optimization.topology.export import largest_connected_component

        kept = largest_connected_component(mesh, result.density, threshold)
        dropped = int((result.density >= threshold).sum()
                      - (kept >= threshold).sum())
        surface = marching_surface(mesh, kept, threshold, smoothing_iterations)
        if not surface.watertight or surface.n_components != 1:
            raise FreeformFailed(
                f"the extracted surface is not one closed body "
                f"({surface.n_components} components, watertight "
                f"{surface.watertight}) even after dropping {dropped} "
                f"unattached elements; nothing downstream can use it")
        part_id = f"freeform-{abs(hash((seed, iterations, threshold))) % (16 ** 10):010x}"
        stl = write_stl(surface, directory / f"{part_id}.stl")
        size = mesh_size_m or max(mesh.dx, mesh.dy, mesh.dz) * 0.7
        tets = tet_mesh_from_stl(stl, size, order=1)

        from nodes import calculix as ccx
        x = tets.node_coords[:, 0]
        band = mesh.dx * 0.5
        tet_fixed = np.flatnonzero(x <= x.min() + band)
        tet_loaded = np.flatnonzero(x >= x.max() - band)
        if len(tet_fixed) < 4 or len(tet_loaded) < 4:
            raise FreeformFailed(
                f"the smoothed body has {len(tet_fixed)} nodes at the clamp and "
                f"{len(tet_loaded)} at the load; the faces did not survive "
                f"smoothing and the problem would be singular")
        solution = ccx.solve(tets, material.youngs_modulus_pa,
                             material.poisson_ratio, tet_fixed, tet_loaded,
                             total_load_n=simp.total_load_n, load_direction=1,
                             element_type=ccx.ElementType.C3D4)
        if not solution.converged:
            raise FreeformFailed("CalculiX reported an error on the smoothed body")

        deflection = abs(float(np.mean(solution.displacements[tet_loaded, 1])))
        solver = f"calculix {ccx.version() or 'unknown'} C3D4"
        mass = surface.volume_m3 * material.density_kg_m3
        labels = {
            "tip_deflection_m": label(
                deflection, "m", EvidenceKind.SIMULATION, solver,
                note="mean displacement of the loaded face; " + ELEMENT_NOTE,
                scaling="inverse_modulus"),
            "max_displacement_m": label(
                float(np.abs(solution.displacements).max()), "m",
                EvidenceKind.SIMULATION, solver, note=ELEMENT_NOTE,
                scaling="inverse_modulus"),
            "max_von_mises_pa": label(
                float(solution.max_von_mises_pa()), "Pa",
                EvidenceKind.SIMULATION, solver,
                note="peak on a marching cubes surface; not converged",
                scaling="none"),
            "mass_kg": label(mass, "kg", EvidenceKind.ANALYTICAL,
                             "surface_volume_times_density",
                             note="volume of the smoothed body, which differs "
                                  "from the density field's volume by "
                                  f"{surface.volume_error_vs_field:+.1%}",
                             scaling="density"),
        }
        record = _surface_record(part_id, surface, tets, material.id, labels)

        limit = op.max_deflection_m
        feasible = limit is None or deflection <= limit
        return DesignOutcome(
            method=METHOD, mass_kg=mass, feasible=feasible,
            constraints=({} if limit is None
                         else {"deflection": float(limit - deflection)}),
            evaluations=1, seconds=time.monotonic() - began, converged=True,
            cad_record=record,
            detail={
                "family": None,
                "part_id": part_id,
                "stl_path": str(stl) if context is None else None,
                "elements_optimised": mesh.n_elements,
                "island_elements_dropped": dropped,
                "grey_fraction": float(np.mean((result.density > 0.1)
                                               & (result.density < 0.9))),
                "field_compliance_j": float(result.final_compliance),
                "threshold": threshold,
                "surface": surface.row(),
                "tetrahedra": tets.n_elements,
                "element_type": "C3D4",
                "element_note": ELEMENT_NOTE,
                "tip_deflection_m": deflection,
                "evidence": labels["tip_deflection_m"]["evidence"],
                "classifier": "expected UNKNOWN: no parametric family "
                              "describes this shape",
            })
    finally:
        if context is not None:
            context.cleanup()
