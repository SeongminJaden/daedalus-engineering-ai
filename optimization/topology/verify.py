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


@dataclass
class StressCheck:
    """What a stress-constrained run claims, next to what a solver reads.

    `physical_peak_verified` is always False and is a field rather than a
    docstring so that a caller cannot skip it. The peak von Mises at a
    re-entrant corner is a singularity: it does not converge under refinement,
    so no number here is the peak stress of the part. What can be said is
    whether the constrained design's own measure sits under the limit, and
    what an independent solver reads on the extracted body at a stated mesh.
    """

    stress_limit_pa: float
    p_norm: float
    design_max_relaxed_pa: float
    extracted_peak_pa: dict            # mesh description to peak von Mises
    physical_peak_verified: bool = False

    @property
    def design_satisfies_limit(self) -> bool:
        return self.design_max_relaxed_pa <= self.stress_limit_pa

    def summary(self) -> str:
        reads = ", ".join(f"{k} {v / 1e6:.1f} MPa"
                          for k, v in self.extracted_peak_pa.items())
        return (f"limit {self.stress_limit_pa / 1e6:.0f} MPa, p-norm "
                f"{self.p_norm:.3f}, design relaxed peak "
                f"{self.design_max_relaxed_pa / 1e6:.1f} MPa; extracted part "
                f"reads {reads}. The peak sits at a re-entrant corner and does "
                f"not converge under refinement, so none of these is the peak "
                f"stress of the part.")


def stress_check(stress_problem, result, density_kg_m3: float,
                 thresholds=(0.3, 0.5)) -> StressCheck:
    """Compare a stress-constrained design with what CalculiX reads on it."""
    peaks: dict = {}
    for threshold in thresholds:
        try:
            check = verify_extracted(stress_problem.base, result.density, threshold,
                                     result.final_compliance, density_kg_m3)
        except DisconnectedAtThreshold:
            continue
        peaks[f"voxel mesh at threshold {threshold}"] = check.peak_von_mises_pa
    return StressCheck(stress_limit_pa=float(stress_problem.stress_limit_pa),
                       p_norm=float(result.final_p_norm),
                       design_max_relaxed_pa=float(result.final_max_stress_pa),
                       extracted_peak_pa=peaks)


@dataclass
class VolumeSearchStep:
    volume_fraction: float
    threshold: float
    mass_kg: float
    tip_displacement_m: float
    feasible: bool
    note: str = ""

    def row(self) -> dict:
        return self.__dict__.copy()


@dataclass
class VolumeSearch:
    """The lightest volume fraction whose EXTRACTED part meets a displacement
    limit, and every step that got there."""

    limit_m: float
    steps: list[VolumeSearchStep] = field(default_factory=list)
    best: VolumeSearchStep | None = None
    density: np.ndarray | None = None
    seconds: float = 0.0

    def summary(self) -> str:
        if self.best is None:
            return (f"no volume fraction in the bracket produced an extracted "
                    f"part under {self.limit_m:.4g} m after {len(self.steps)} "
                    f"runs")
        return (f"volume fraction {self.best.volume_fraction:.4f}, extracted "
                f"part {self.best.mass_kg:.4f} kg at {self.best.tip_displacement_m:.4e} m "
                f"against a limit of {self.limit_m:.4g} m, {len(self.steps)} runs")


def tip_displacement_of_extracted(problem, density: np.ndarray, threshold: float,
                                  element_type=None) -> float:
    """Mean displacement of the loaded face on the extracted part, in the load
    direction. The same quantity the part labeller reports, so it can be
    compared with a deflection requirement."""
    from nodes import calculix as ccx

    element_type = element_type or ccx.ElementType.C3D8I
    sub, _report = retained_submesh(problem.mesh, density, threshold,
                                    problem.fixed_nodes, problem.load_nodes)
    fixed = sub.local_nodes(problem.fixed_nodes)
    loaded = sub.local_nodes(problem.load_nodes)
    result = ccx.solve(sub, problem.youngs_modulus_pa, problem.poisson_ratio,
                       fixed, loaded, total_load_n=problem.total_load_n,
                       load_direction=problem.load_direction,
                       element_type=element_type)
    if not result.converged:
        raise RuntimeError("CalculiX reported an error on the extracted part")
    return abs(float(np.mean(result.displacements[loaded, problem.load_direction])))


def search_volume_fraction(build_problem, runner, limit_m: float,
                           density_kg_m3: float, low: float = 0.05,
                           high: float = 0.6, steps: int = 6,
                           threshold: float = 0.5, iterations: int = 100
                           ) -> VolumeSearch:
    """Bisect the volume fraction against the EXTRACTED part's displacement.

    The field's compliance is not the part's, which was measured before this
    existed, so the test at every step is a CalculiX solve of the thresholded
    body rather than anything the optimiser reports. `build_problem` takes a
    volume fraction and returns a SimpProblem, so the caller owns the mesh,
    the loads, the passive regions and any manufacturing projection.

    One full optimisation per step. Six steps on a 1536 element cantilever is
    about five minutes.
    """
    import time

    started = time.perf_counter()
    search = VolumeSearch(limit_m=float(limit_m))

    def evaluate(fraction: float) -> VolumeSearchStep:
        problem = build_problem(fraction)
        result = runner(problem, max_iterations=iterations)
        try:
            displacement = tip_displacement_of_extracted(problem, result.density,
                                                         threshold)
        except (DisconnectedAtThreshold, RuntimeError) as exc:
            step = VolumeSearchStep(volume_fraction=fraction, threshold=threshold,
                                    mass_kg=float("nan"),
                                    tip_displacement_m=float("nan"),
                                    feasible=False, note=str(exc)[:120])
            search.steps.append(step)
            return step
        kept = int((result.density >= threshold).sum())
        mass = kept * problem.mesh.element_volume * density_kg_m3
        step = VolumeSearchStep(volume_fraction=fraction, threshold=threshold,
                                mass_kg=mass, tip_displacement_m=displacement,
                                feasible=displacement <= limit_m)
        search.steps.append(step)
        if step.feasible and (search.best is None
                              or step.mass_kg < search.best.mass_kg):
            search.best = step
            search.density = result.density
        return step

    top = evaluate(high)
    if not top.feasible:
        search.seconds = time.perf_counter() - started
        return search                      # the envelope cannot meet the limit
    lower, upper = low, high
    for _ in range(steps - 1):
        middle = 0.5 * (lower + upper)
        if evaluate(middle).feasible:
            upper = middle
        else:
            lower = middle
    search.seconds = time.perf_counter() - started
    return search


def format_search(search: VolumeSearch) -> str:
    lines = ["| volume fraction | extracted mass kg | tip displacement m | meets the limit |",
             "|" + "---|" * 4]
    for step in search.steps:
        mass = "disconnected" if np.isnan(step.mass_kg) else f"{step.mass_kg:.4f}"
        displacement = ("" if np.isnan(step.tip_displacement_m)
                        else f"{step.tip_displacement_m:.4e}")
        lines.append(f"| {step.volume_fraction:.4f} | {mass} | {displacement} | "
                     f"{'yes' if step.feasible else 'no'} |")
    return "\n".join(lines)
