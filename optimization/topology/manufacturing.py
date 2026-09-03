"""Manufacturing constraints inside the topology loop, and their price.

A topology result that cannot be made is a picture. These are the constraints
that change what the optimiser is allowed to produce, applied to the density
field itself so that the compliance the run reports is the compliance of a
field that already obeys them.

WHAT IS HERE, AND HOW EXACT EACH ONE IS
=======================================
symmetry
    Exact. The field is averaged with its mirror image about a plane of the
    design domain, and so is the sensitivity, so the run is a search over
    symmetric fields with the correct gradient.

additive support (overhang)
    Langelaar's layer filter: an element can only be solid if something below
    it in the build direction is, taken as a smooth maximum over the three
    supporting neighbours. It is applied to the density the solver sees, and
    its exact chain rule is NOT applied to the sensitivity; the gradient the
    optimiser follows is the unfiltered one. That makes it a projection rather
    than a fully consistent constraint, the runs converge more slowly, and the
    overhang of the result is measured afterwards rather than assumed.

casting pull direction (no undercut)
    Exact as a projection and honest about what it forbids: along the pull
    axis, material may only be a single run starting at the mould face, so
    there is nothing behind anything and no internal void. Applied by a
    cumulative minimum down each column.

minimum member size
    NOT a constraint here. It is a measurement: the filter radius sets the
    thinnest member the optimiser can produce, and `member_size_study` runs
    the same problem at several radii and measures the thinnest wall of the
    extracted part, so the relation is a table rather than a claim.

The price of each is measured by `constraint_study`: the same problem with the
constraint off and on, reporting compliance, mass and the geometric quantity
the constraint was meant to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from physics.fem.mesh import Mesh

from .smooth import density_grid, marching_surface

#: A projection takes the (nx, ny, nz) density grid and returns another.
GridProjection = Callable[[np.ndarray], np.ndarray]


def _to_grid(mesh: Mesh, density: np.ndarray) -> np.ndarray:
    return density_grid(mesh, density)


def _from_grid(mesh: Mesh, grid: np.ndarray) -> np.ndarray:
    cell = np.array([mesh.dx, mesh.dy, mesh.dz])
    index = np.round(mesh.element_centroids() / cell - 0.5).astype(int)
    return grid[index[:, 0], index[:, 1], index[:, 2]]


def as_element_projection(mesh: Mesh, projection: GridProjection):
    """A grid projection, wrapped to act on the element density vector."""
    def apply(density: np.ndarray) -> np.ndarray:
        return _from_grid(mesh, projection(_to_grid(mesh, density)))
    return apply


# ------------------------------------------------------------------ symmetry

def mirror_grid(grid: np.ndarray, axis: int) -> np.ndarray:
    """The field averaged with its mirror image about the mid plane."""
    return 0.5 * (grid + np.flip(grid, axis=axis))


def symmetry_projection(axis: int) -> GridProjection:
    return lambda grid: mirror_grid(grid, axis)


def symmetry_error(mesh: Mesh, density: np.ndarray, axis: int) -> float:
    """Largest difference between the field and its mirror, as a fraction."""
    grid = _to_grid(mesh, density)
    return float(np.max(np.abs(grid - np.flip(grid, axis=axis))))


# ------------------------------------------------------- additive support

def support_filter(grid: np.ndarray, build_axis: int = 1,
                   smooth_p: float = 40.0) -> np.ndarray:
    """Langelaar's layer filter: nothing floats.

    Layer by layer up the build axis, an element's printable density is the
    smaller of its own and the support available beneath it, where the support
    is a smooth maximum over the element below and its in-layer neighbours. The
    first layer sits on the plate and is unfiltered.
    """
    moved = np.moveaxis(grid, build_axis, 0)
    out = np.empty_like(moved)
    out[0] = moved[0]
    for layer in range(1, moved.shape[0]):
        below = out[layer - 1]
        neighbours = [below]
        for axis in (0, 1):
            neighbours.append(np.roll(below, 1, axis=axis))
            neighbours.append(np.roll(below, -1, axis=axis))
        stack = np.stack(neighbours)
        # Smooth maximum, so the filter has a derivative even though the
        # gradient is not chained through it here.
        support = np.log(np.sum(np.exp(smooth_p * stack), axis=0)) / smooth_p
        support = np.clip(support, 0.0, 1.0)
        out[layer] = np.minimum(moved[layer], support)
    return np.moveaxis(out, 0, build_axis)


def support_projection(build_axis: int = 1) -> GridProjection:
    return lambda grid: support_filter(grid, build_axis)


# ------------------------------------------------------- casting direction

def pull_filter(grid: np.ndarray, pull_axis: int = 1,
                from_high: bool = True) -> np.ndarray:
    """No undercut along the pull axis: material is one run from the face.

    Down each column, the density is replaced by the running minimum from the
    mould face inward, so material can never reappear behind a gap. That
    forbids internal voids and re-entrant features along that axis, which is
    exactly what an undercut is.
    """
    moved = np.moveaxis(grid, pull_axis, 0)
    if from_high:
        moved = moved[::-1]
    out = np.minimum.accumulate(moved, axis=0)
    if from_high:
        out = out[::-1]
    return np.moveaxis(out, 0, pull_axis)


def pull_projection(pull_axis: int = 1, from_high: bool = True) -> GridProjection:
    return lambda grid: pull_filter(grid, pull_axis, from_high)


def undercut_fraction(mesh: Mesh, density: np.ndarray, pull_axis: int = 1,
                      threshold: float = 0.5, from_high: bool = True) -> float:
    """Fraction of solid elements that a mould could not release."""
    grid = _to_grid(mesh, density) >= threshold
    released = pull_filter(grid.astype(float), pull_axis, from_high) >= 0.5
    solid = int(grid.sum())
    return float((grid & ~released).sum() / solid) if solid else 0.0


# --------------------------------------------------------------- the price

@dataclass
class ConstraintOutcome:
    name: str
    compliance_j: float
    volume_fraction: float
    mass_kg: float
    grey_fraction: float
    overhang_fraction_45: float
    undercut_fraction: float
    symmetry_error: float
    min_wall_m: float
    seconds: float

    def row(self) -> dict:
        return self.__dict__.copy()


def measure_field(mesh: Mesh, density: np.ndarray, density_kg_m3: float,
                  build_axis: int = 1, pull_axis: int = 1,
                  symmetry_axis: int = 2) -> dict:
    """The geometric quantities the constraints are about, on one field."""
    from geometry.manufacturability import measure_mesh

    surface = marching_surface(mesh, density, 0.5, smoothing_iterations=10)
    measures = measure_mesh(surface.vertices, surface.triangles,
                            build_axis=build_axis)
    volume_fraction = float(np.mean(np.asarray(density, dtype=float)))
    return {"volume_fraction": volume_fraction,
            "mass_kg": volume_fraction * mesh.n_elements * mesh.element_volume
                       * density_kg_m3,
            "grey_fraction": float(np.mean((density > 0.1) & (density < 0.9))),
            "overhang_fraction_45": float(measures.overhang_fraction_45),
            "undercut_fraction": undercut_fraction(mesh, density, pull_axis),
            "symmetry_error": symmetry_error(mesh, density, symmetry_axis),
            "min_wall_m": (float(measures.min_wall_m)
                           if measures.min_wall_m is not None else float("nan"))}


def format_table(rows: list[dict]) -> str:
    lines = ["| constraint | compliance J | mass kg | overhang 45 | undercut | "
             "symmetry error | min wall mm |", "|" + "---|" * 7]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['compliance_j']:.4e} | {r['mass_kg']:.3f} | "
            f"{r['overhang_fraction_45']:.2f} | {r['undercut_fraction']:.2f} | "
            f"{r['symmetry_error']:.3f} | {r['min_wall_m'] * 1e3:.1f} |")
    return "\n".join(lines)


def constraint_study(build_problem, runner, density_kg_m3: float,
                     iterations: int = 100, build_axis: int = 1,
                     pull_axis: int = 1, symmetry_axis: int = 2) -> list[dict]:
    """The same problem with each constraint off and on.

    `build_problem` takes a projection (or None) and returns a SimpProblem, so
    the caller owns the mesh, the loads and the passive regions and this
    function only changes the constraint.
    """
    import time

    cases = [("none", None),
             ("symmetry", symmetry_projection(symmetry_axis)),
             ("additive support", support_projection(build_axis)),
             ("casting pull", pull_projection(pull_axis))]
    rows = []
    for name, projection in cases:
        problem = build_problem(None if projection is None
                                else as_element_projection(problem_mesh(build_problem),
                                                           projection))
        started = time.perf_counter()
        result = runner(problem, max_iterations=iterations)
        row = {"name": name, "compliance_j": float(result.final_compliance),
               "seconds": time.perf_counter() - started}
        row.update(measure_field(problem.mesh, result.density, density_kg_m3,
                                 build_axis, pull_axis, symmetry_axis))
        rows.append(row)
    return rows


def problem_mesh(build_problem) -> Mesh:
    """The mesh a problem builder produces, for wrapping grid projections."""
    return build_problem(None).mesh


def member_size_study(build_problem, runner, radii, density_kg_m3: float,
                      iterations: int = 100) -> list[dict]:
    """Filter radius against the thinnest member that comes out.

    The relation is measured on the extracted surface with the same wall
    thickness ray cast the manufacturability rules use, so the number is
    comparable with a process minimum wall.
    """
    import time

    rows = []
    for radius in radii:
        problem = build_problem(None)
        problem.filter_radius_elements = radius
        started = time.perf_counter()
        result = runner(problem, max_iterations=iterations)
        row = {"name": f"radius {radius}", "filter_radius_elements": float(radius),
               "cell_size_m": float(problem.mesh.dx),
               "compliance_j": float(result.final_compliance),
               "seconds": time.perf_counter() - started}
        row.update(measure_field(problem.mesh, result.density, density_kg_m3))
        rows.append(row)
    return rows
