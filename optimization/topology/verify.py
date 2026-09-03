"""Re-solving the shape a topology run actually produced, in another solver.

A SIMP run reports the compliance of a density FIELD, in which every element
carries a stiffness interpolated from its density. The part that comes out is
the field thresholded: elements above the threshold become solid, the rest
disappear. Those are two different structures, and the difference is the
central limitation of topology optimisation as a source of parts. This module
measures it instead of describing it.

HOW
===
The retained voxels are written as their own hexahedral mesh, with the fixed
and loaded nodes carried over by index, and solved by CalculiX with
incompatible modes (C3D8I). Nothing is remeshed and nothing is smoothed, so
the solved body is exactly the body the exporter would write, and any
difference in the answer is a difference in the structure, not in the mesh.

The comparison is compliance, the same quantity SIMP minimises: the work of
the applied load, the sum of force times displacement over the loaded nodes.
CalculiX is also an independent solver from this project's matrix-free FEM, so
the check is a cross-validation as well; it is not a new capability.

WHAT CAN GO WRONG, AND WHAT THE FUNCTIONS DO ABOUT IT
=====================================================
Thresholding can disconnect the structure, and a disconnected load path is a
singular stiffness matrix. `retained_submesh` refuses a threshold that
detaches every loaded node or every fixed node from the largest face-connected
component, with the count in the message, rather than handing CalculiX a
problem it will reject less clearly. A threshold that keeps the load path but
leaves a floating island is solved on the largest component only, and the
dropped volume is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from physics.fem.mesh import Mesh

from .export import largest_connected_component


class DisconnectedAtThreshold(ValueError):
    """The threshold cut the load path, so there is nothing to solve."""


@dataclass
class SubMesh:
    """The retained voxels as a mesh of their own.

    Carries what a CalculiX deck needs (`node_coords`, `connectivity`,
    `n_nodes`, `n_elements`) so it can be passed to the CalculiX node
    unchanged, plus the maps back to the parent mesh.
    """

    node_coords: np.ndarray
    connectivity: np.ndarray
    element_volume: float
    parent_nodes: np.ndarray        # index in the parent mesh, per local node
    parent_elements: np.ndarray     # index in the parent mesh, per local element

    @property
    def n_nodes(self) -> int:
        return int(self.node_coords.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.connectivity.shape[0])

    def local_nodes(self, parent_nodes: np.ndarray) -> np.ndarray:
        """The local indices of parent nodes that survived, in order."""
        lookup = {int(p): i for i, p in enumerate(self.parent_nodes)}
        return np.array([lookup[int(n)] for n in parent_nodes if int(n) in lookup],
                        dtype=int)


def elements_touching(mesh: Mesh, nodes: np.ndarray) -> np.ndarray:
    """Boolean mask of elements with at least one of those nodes.

    The load and support patches: the elements a point load is actually
    applied through. SIMP with a volume constraint routinely leaves them grey,
    and keeping them solid is the standard passive region of the literature.
    """
    wanted = np.zeros(mesh.n_nodes, dtype=bool)
    wanted[np.asarray(nodes, dtype=int)] = True
    return wanted[np.asarray(mesh.connectivity)].any(axis=1)


def retained_submesh(mesh: Mesh, density: np.ndarray, threshold: float,
                     fixed_nodes: np.ndarray, load_nodes: np.ndarray,
                     largest_component_only: bool = True,
                     keep_elements: np.ndarray | None = None
                     ) -> tuple[SubMesh, dict]:
    """The thresholded structure as its own mesh, with a report.

    `keep_elements` forces elements solid whatever their density. It is how a
    passive load or support patch is kept, and the report says how many
    elements it added so the intervention is never invisible.
    """
    density = np.asarray(density, dtype=float).reshape(-1)
    kept = density >= threshold
    forced = 0
    if keep_elements is not None:
        keep_elements = np.asarray(keep_elements, dtype=bool).reshape(-1)
        forced = int((keep_elements & ~kept).sum())
        kept = kept | keep_elements
    n_above = int(kept.sum())
    if n_above == 0:
        raise DisconnectedAtThreshold(
            f"threshold {threshold} keeps no element of {mesh.n_elements}")
    dropped_islands = 0
    if largest_component_only:
        filtered = largest_connected_component(mesh, kept.astype(float), 0.5)
        connected = filtered >= 0.5
        dropped_islands = int(kept.sum() - connected.sum())
        kept = connected

    elements = np.flatnonzero(kept)
    connectivity = np.asarray(mesh.connectivity)[elements]
    parent_nodes = np.unique(connectivity)
    lookup = -np.ones(mesh.n_nodes, dtype=int)
    lookup[parent_nodes] = np.arange(len(parent_nodes))
    sub = SubMesh(node_coords=np.asarray(mesh.node_coords)[parent_nodes],
                  connectivity=lookup[connectivity],
                  element_volume=mesh.element_volume,
                  parent_nodes=parent_nodes, parent_elements=elements)

    kept_fixed = sub.local_nodes(fixed_nodes)
    kept_load = sub.local_nodes(load_nodes)
    if len(kept_load) == 0 or len(kept_fixed) == 0:
        raise DisconnectedAtThreshold(
            f"threshold {threshold} leaves {len(kept_fixed)} of "
            f"{len(fixed_nodes)} fixed nodes and {len(kept_load)} of "
            f"{len(load_nodes)} loaded nodes attached to the structure; the "
            f"load path is cut and there is nothing to solve")
    report = {"threshold": float(threshold),
              "elements_above_threshold": n_above,
              "elements_forced_solid": forced,
              "elements_solved": sub.n_elements,
              "island_elements_dropped": dropped_islands,
              "fixed_nodes_kept": len(kept_fixed), "fixed_nodes": len(fixed_nodes),
              "load_nodes_kept": len(kept_load), "load_nodes": len(load_nodes)}
    return sub, report


@dataclass
class ExtractedCheck:
    """What the extracted part does, next to what the field promised."""

    threshold: float
    field_compliance_j: float
    part_compliance_j: float
    field_volume_fraction: float
    part_volume_fraction: float
    mass_kg: float
    max_displacement_m: float
    peak_von_mises_pa: float
    report: dict = field(default_factory=dict)

    @property
    def compliance_ratio(self) -> float:
        """Part over field. Above one means the part is softer than the field
        the optimiser was reporting."""
        return self.part_compliance_j / self.field_compliance_j

    def row(self) -> dict:
        return {"threshold": self.threshold,
                "field_compliance_j": self.field_compliance_j,
                "part_compliance_j": self.part_compliance_j,
                "compliance_ratio": self.compliance_ratio,
                "field_volume_fraction": self.field_volume_fraction,
                "part_volume_fraction": self.part_volume_fraction,
                "mass_kg": self.mass_kg,
                "max_displacement_m": self.max_displacement_m,
                "peak_von_mises_pa": self.peak_von_mises_pa,
                **self.report}


def compliance_from(displacements: np.ndarray, load_nodes: np.ndarray,
                    total_load_n: float, load_direction: int) -> float:
    """Work of the applied load: force times displacement, summed.

    The load is divided equally over the loaded nodes, exactly as both solvers
    apply it, so this is the same functional SIMP minimises.
    """
    per_node = total_load_n / max(len(load_nodes), 1)
    return float(abs(np.sum(per_node * displacements[load_nodes, load_direction])))


def verify_extracted(problem, density: np.ndarray, threshold: float,
                     field_compliance_j: float, density_kg_m3: float,
                     element_type=None, keep_directory: Path | None = None,
                     keep_patches: bool = False) -> ExtractedCheck:
    """Solve the thresholded part in CalculiX and compare it with the field.

    `keep_patches` forces the load and support elements solid before
    thresholding, which is what makes the extracted part solvable at all on a
    problem where the optimiser left them grey.
    """
    from nodes import calculix as ccx

    element_type = element_type or ccx.ElementType.C3D8I
    keep = None
    if keep_patches:
        keep = (elements_touching(problem.mesh, problem.load_nodes)
                | elements_touching(problem.mesh, problem.fixed_nodes))
    sub, report = retained_submesh(problem.mesh, density, threshold,
                                   problem.fixed_nodes, problem.load_nodes,
                                   keep_elements=keep)
    fixed = sub.local_nodes(problem.fixed_nodes)
    loaded = sub.local_nodes(problem.load_nodes)
    result = ccx.solve(sub, problem.youngs_modulus_pa, problem.poisson_ratio,
                       fixed, loaded, total_load_n=problem.total_load_n,
                       load_direction=problem.load_direction,
                       element_type=element_type, keep_directory=keep_directory)
    if not result.converged:
        raise RuntimeError(f"CalculiX reported an error at threshold {threshold}")
    displacements = result.displacements
    # The load is spread over the loaded nodes that survived, which is what
    # the deck did too, so the work is computed with the same division.
    compliance = compliance_from(displacements, loaded, problem.total_load_n,
                                 problem.load_direction)
    volume = sub.n_elements * problem.mesh.element_volume
    stress = result.element_stress
    peak = float(np.max(_von_mises(stress))) if stress is not None else float("nan")
    return ExtractedCheck(
        threshold=float(threshold), field_compliance_j=float(field_compliance_j),
        part_compliance_j=compliance,
        field_volume_fraction=float(np.mean(np.asarray(density, dtype=float))),
        part_volume_fraction=sub.n_elements / problem.mesh.n_elements,
        mass_kg=volume * density_kg_m3,
        max_displacement_m=float(np.max(np.abs(displacements))),
        peak_von_mises_pa=peak, report=report)


def _von_mises(stress: np.ndarray) -> np.ndarray:
    """Von Mises from the six components CalculiX prints, per element."""
    s = np.asarray(stress, dtype=float)
    xx, yy, zz, xy, xz, yz = (s[:, i] for i in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                   + 3.0 * (xy ** 2 + xz ** 2 + yz ** 2))


def threshold_table(problem, density: np.ndarray, field_compliance_j: float,
                    density_kg_m3: float, thresholds=(0.3, 0.5, 0.7),
                    keep_patches: bool = False) -> list[dict]:
    """One row per threshold, and a row saying so where it disconnects."""
    rows = []
    for t in thresholds:
        try:
            rows.append(verify_extracted(problem, density, t, field_compliance_j,
                                         density_kg_m3,
                                         keep_patches=keep_patches).row())
        except DisconnectedAtThreshold as exc:
            rows.append({"threshold": float(t), "refused": str(exc)})
    return rows


def format_table(rows: list[dict]) -> str:
    head = ("| threshold | part mass kg | volume fraction | field compliance J | "
            "part compliance J | ratio | peak von Mises Pa |")
    lines = [head, "|" + "---|" * 7]
    for r in rows:
        if "refused" in r:
            lines.append(f"| {r['threshold']} | disconnected | | | | | |")
            continue
        lines.append(
            f"| {r['threshold']} | {r['mass_kg']:.4f} | "
            f"{r['part_volume_fraction']:.3f} | {r['field_compliance_j']:.4e} | "
            f"{r['part_compliance_j']:.4e} | {r['compliance_ratio']:.2f} | "
            f"{r['peak_von_mises_pa']:.3e} |")
    return "\n".join(lines)
