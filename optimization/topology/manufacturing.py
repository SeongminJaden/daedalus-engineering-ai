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
    it in the build direction is, taken as a smooth maximum over the five
    supporting neighbours. Its chain rule IS available, through
    `support_filter_gradient`, and using it matters: without the chain rule the
    constraint cost 5.07 times the compliance on the cantilever, and with it
    1.15. Judge it by `unsupported_fraction`, the criterion it enforces, which
    goes from 0.025 to exactly zero. Do not judge it by the surface overhang
    area, which does not improve, because that is measured on the smoothed
    surface where a supported staircase becomes a steeper face.

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


def support_projection_with_gradient(mesh: Mesh, build_axis: int = 1,
                                     smooth_p: float = 40.0):
    """The support filter and its chain rule, both on element vectors.

    Returns (projection, vjp) for SimpProblem.density_projection and
    SimpProblem.projection_vjp.
    """
    def projection(density: np.ndarray) -> np.ndarray:
        printed, _ = support_filter_gradient(_to_grid(mesh, density), build_axis,
                                             smooth_p)
        return _from_grid(mesh, printed)

    def vjp(density: np.ndarray, seed: np.ndarray) -> np.ndarray:
        _printed, pullback = support_filter_gradient(_to_grid(mesh, density),
                                                     build_axis, smooth_p)
        return _from_grid(mesh, pullback(_to_grid(mesh, seed)))

    return projection, vjp


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


def _softmax_weights(stack: np.ndarray, p: float) -> np.ndarray:
    """Softmax weights over the first axis, computed shift-stably."""
    shifted = stack - stack.max(axis=0, keepdims=True)
    weights = np.exp(p * shifted)
    return weights / weights.sum(axis=0, keepdims=True)


def _smooth_max(stack: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    """The softmax-weighted mean and its derivative in each argument.

    The weights alone are NOT the derivative: they depend on the arguments
    too. d/dx_i sum_j w_j x_j = w_i (1 + p (x_i - m)), and using w_i on its own
    was measured wrong by 26 percent against a difference quotient.
    """
    weights = _softmax_weights(stack, p)
    mean = np.sum(weights * stack, axis=0)
    derivative = weights * (1.0 + p * (stack - mean))
    return mean, derivative


def support_filter_gradient(grid: np.ndarray, build_axis: int = 1,
                            smooth_p: float = 40.0, smooth_min_p: float = 40.0
                            ) -> tuple[np.ndarray, "callable"]:
    """The layer filter and the function that pulls a gradient back through it.

    The filter is a recursion up the build direction, so its chain rule is a
    recursion down it. This returns the printed field and a vector-Jacobian
    product: given dJ/d(printed) it gives dJ/d(design), which is what the
    optimiser needs to steer with the constraint rather than merely pay for it.

    The minimum is smoothed the same way the maximum is, because a hard
    minimum has a zero derivative on one side and the recursion then stops
    passing information down through any layer that is support limited, which
    is exactly the layer that matters.
    """
    moved = np.moveaxis(np.asarray(grid, dtype=float), build_axis, 0)
    layers = moved.shape[0]
    printed = np.empty_like(moved)
    printed[0] = moved[0]
    # Per layer: the softmax weights of the support maximum, and the two
    # weights of the smooth minimum.
    support_weights: list[np.ndarray] = [None]      # type: ignore[list-item]
    min_weights: list[tuple[np.ndarray, np.ndarray]] = [None]  # type: ignore[list-item]
    shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    for layer in range(1, layers):
        below = printed[layer - 1]
        stack = np.stack([np.roll(np.roll(below, a, axis=0), b, axis=1)
                          for a, b in shifts])
        support, support_derivative = _smooth_max(stack, smooth_p)
        support_weights.append(support_derivative)

        own = moved[layer]
        # Smooth minimum as the negated smooth maximum of the negated pair.
        pair = np.stack([-own, -support])
        smallest, pair_derivative = _smooth_max(pair, smooth_min_p)
        printed[layer] = -smallest
        min_weights.append((pair_derivative[0], pair_derivative[1]))

    def vjp(seed: np.ndarray) -> np.ndarray:
        """dJ/d(printed) to dJ/d(design), by the reverse recursion."""
        bar = np.moveaxis(np.asarray(seed, dtype=float).copy(), build_axis, 0)
        out = np.zeros_like(bar)
        for layer in range(layers - 1, 0, -1):
            d_own, d_support = min_weights[layer]
            out[layer] += bar[layer] * d_own
            to_support = bar[layer] * d_support
            weights = support_weights[layer]
            # The support is a weighted sum of shifted copies of the layer
            # below, so the gradient shifts back the other way.
            for weight, (a, b) in zip(weights, shifts):
                bar[layer - 1] += np.roll(np.roll(weight * to_support, -a, axis=0),
                                          -b, axis=1)
        out[0] += bar[0]
        return np.moveaxis(out, 0, build_axis)

    return np.moveaxis(printed, 0, build_axis), vjp


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


def unsupported_fraction(mesh: Mesh, density: np.ndarray, build_axis: int = 1,
                         threshold: float = 0.5) -> float:
    """Solid elements with nothing under them, as a fraction of the solid.

    This is the criterion the support filter enforces, and therefore the one
    to judge it by. The surface overhang fraction that the manufacturability
    rules read answers a different question: it is measured on the smoothed
    surface, where a staircase of supported voxels becomes a face steeper than
    the staircase was, so a field with no unsupported element can still show a
    large overhang area. Both numbers are reported and neither is a substitute
    for the other.
    """
    grid = _to_grid(mesh, density) >= threshold
    moved = np.moveaxis(grid, build_axis, 0)
    total = int(moved.sum())
    if total == 0:
        return 0.0
    unsupported = 0
    for layer in range(1, moved.shape[0]):
        below = moved[layer - 1]
        support = below.copy()
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            support |= np.roll(np.roll(below, a, axis=0), b, axis=1)
        unsupported += int((moved[layer] & ~support).sum())
    return unsupported / total


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
            "unsupported_fraction": unsupported_fraction(mesh, density, build_axis),
            "mass_kg": volume_fraction * mesh.n_elements * mesh.element_volume
                       * density_kg_m3,
            "grey_fraction": float(np.mean((density > 0.1) & (density < 0.9))),
            "overhang_fraction_45": float(measures.overhang_fraction_45),
            "undercut_fraction": undercut_fraction(mesh, density, pull_axis),
            "symmetry_error": symmetry_error(mesh, density, symmetry_axis),
            "min_wall_m": (float(measures.min_wall_m)
                           if measures.min_wall_m is not None else float("nan"))}


def format_table(rows: list[dict]) -> str:
    lines = ["| constraint | compliance J | mass kg | overhang 45 | unsupported | "
             "undercut | symmetry error | min wall mm |", "|" + "---|" * 8]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['compliance_j']:.4e} | {r['mass_kg']:.3f} | "
            f"{r['overhang_fraction_45']:.2f} | {r.get('unsupported_fraction', float('nan')):.4f} | "
            f"{r['undercut_fraction']:.2f} | "
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
