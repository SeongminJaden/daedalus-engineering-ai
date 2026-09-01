"""physics.fem.mesh - structured hexahedral meshes.

Axis convention, matching the beam model:
    x -> along the link (root at x=0, tip at x=L)
    y -> section height, and the direction the tip load acts
    z -> section width

A hollow rectangular prism is meshed as a structured grid over the bounding box
with the cavity cells switched off, so only the wall carries material. Keeping
the grid structured is what lets every element share one stiffness matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Local node offsets in (i, j, k) matching element.NODE_SIGNS ordering.
NODE_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.int64)


@dataclass
class Mesh:
    """A structured hex mesh with only the active (material) cells kept."""

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    connectivity: np.ndarray     # (n_elements, 8) global node indices
    node_coords: np.ndarray      # (n_nodes, 3) in metres
    origin: np.ndarray

    @property
    def n_elements(self) -> int:
        return int(self.connectivity.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.node_coords.shape[0])

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    @property
    def element_volume(self) -> float:
        return self.dx * self.dy * self.dz

    def nodes_where(self, mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask)

    def nodes_at_x(self, x: float, tol: float | None = None) -> np.ndarray:
        tol = tol if tol is not None else 0.25 * self.dx
        return self.nodes_where(np.abs(self.node_coords[:, 0] - x) <= tol)

    def element_centroids(self) -> np.ndarray:
        return self.node_coords[self.connectivity].mean(axis=1)

    def summary(self) -> dict:
        return {
            "n_elements": self.n_elements,
            "n_nodes": self.n_nodes,
            "n_dofs": self.n_dofs,
            "grid": (self.nx, self.ny, self.nz),
            "cell_size_m": (self.dx, self.dy, self.dz),
        }


def _build(active: np.ndarray, nx: int, ny: int, nz: int,
           dx: float, dy: float, dz: float,
           origin: np.ndarray | None = None) -> Mesh:
    """Turn a boolean (nx, ny, nz) cell mask into a Mesh with compact numbering."""
    if not active.any():
        raise ValueError("mesh has no active cells")
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)

    cells = np.argwhere(active)                                  # (n_e, 3)
    # (n_e, 8, 3) grid indices of each element's nodes
    node_ijk = cells[:, None, :] + NODE_OFFSETS[None, :, :]

    ny1, nz1 = ny + 1, nz + 1
    flat = (node_ijk[..., 0] * ny1 * nz1
            + node_ijk[..., 1] * nz1
            + node_ijk[..., 2])

    used, inverse = np.unique(flat.reshape(-1), return_inverse=True)
    connectivity = inverse.reshape(flat.shape).astype(np.int64)

    i = used // (ny1 * nz1)
    rem = used % (ny1 * nz1)
    j = rem // nz1
    k = rem % nz1
    coords = np.column_stack([i * dx, j * dy, k * dz]).astype(np.float64) + origin

    return Mesh(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                connectivity=connectivity, node_coords=coords, origin=origin)


def solid_box_mesh(length_m: float, height_m: float, width_m: float,
                   nx: int, ny: int, nz: int) -> Mesh:
    """A fully solid box. Used by the patch test and the beam-limit study."""
    for n in (nx, ny, nz):
        if n < 1:
            raise ValueError("subdivisions must be >= 1")
    active = np.ones((nx, ny, nz), dtype=bool)
    return _build(active, nx, ny, nz,
                  length_m / nx, height_m / ny, width_m / nz)


def hollow_rect_mesh(
    length_m: float,
    outer_height_m: float,
    outer_width_m: float,
    wall_thickness_m: float,
    nx: int,
    elements_through_wall: int = 2,
) -> Mesh:
    """A hollow rectangular prism: structured grid with the cavity removed.

    The cross-section resolution is driven by `elements_through_wall` rather
    than a single global count. A thin wall is the feature that has to be
    resolved, and a uniform "N cells across the section" rule would put almost
    no elements inside it.
    """
    if wall_thickness_m <= 0 or wall_thickness_m >= min(outer_height_m,
                                                        outer_width_m) / 2:
        raise ValueError(
            f"wall_thickness_m={wall_thickness_m} does not leave a cavity in a "
            f"{outer_height_m} x {outer_width_m} section"
        )
    if elements_through_wall < 1:
        raise ValueError("elements_through_wall must be >= 1")
    if nx < 1:
        raise ValueError("nx must be >= 1")

    cell = wall_thickness_m / elements_through_wall
    ny = max(2 * elements_through_wall + 1, int(round(outer_height_m / cell)))
    nz = max(2 * elements_through_wall + 1, int(round(outer_width_m / cell)))
    dy = outer_height_m / ny
    dz = outer_width_m / nz

    # Cell centres, to decide wall vs cavity.
    yc = (np.arange(ny) + 0.5) * dy
    zc = (np.arange(nz) + 0.5) * dz
    in_cavity = (
        (yc[:, None] > wall_thickness_m)
        & (yc[:, None] < outer_height_m - wall_thickness_m)
        & (zc[None, :] > wall_thickness_m)
        & (zc[None, :] < outer_width_m - wall_thickness_m)
    )
    section = ~in_cavity                                   # (ny, nz) wall mask
    active = np.broadcast_to(section[None, :, :], (nx, ny, nz)).copy()
    return _build(active, nx, ny, nz, length_m / nx, dy, dz)


#: How far n * thickness_fraction may sit from a whole number of cells before
#: the requested bracket is considered unbuildable on that grid.
ARM_SNAP_TOLERANCE = 1e-9


def realised_arm_thickness(size_m: float, thickness_fraction: float,
                           n: int) -> float:
    """The arm thickness `l_bracket_mesh` can actually build on an n cell grid.

    The arm is a whole number of cells, so a requested fraction that does not
    land on a cell boundary cannot be built exactly. This reports what would
    be built instead of leaving it to be discovered as a volume error.
    """
    return max(1, int(round(n * thickness_fraction))) * (size_m / n)


def _usable_grid_sizes(thickness_fraction: float, n: int,
                       window: int = 12) -> list[int]:
    """Nearby n values on which the requested fraction IS buildable."""
    return [candidate for candidate in range(max(4, n - window), n + window + 1)
            if abs(candidate * thickness_fraction
                   - round(candidate * thickness_fraction))
            <= ARM_SNAP_TOLERANCE]


def l_bracket_mesh(size_m: float, thickness_fraction: float, width_m: float,
                   n: int, nz: int = 2, allow_snapping: bool = False) -> Mesh:
    """An L-shaped bracket: the standard stress-constrained benchmark.

    The domain is a square with the upper-right quadrant removed, leaving a
    **re-entrant corner**. Linear elasticity puts a stress concentration there,
    which is exactly the feature a compliance objective ignores and a stress
    constraint has to deal with. It is the same physics Phase 7 found at the
    clamped root of the cantilever.

    Axes follow the project convention: x along the long arm, y vertical, z the
    out-of-plane width.

    THE ARM IS A WHOLE NUMBER OF CELLS. When `n * thickness_fraction` is not an
    integer the requested bracket cannot be built on that grid, and this used
    to round silently: n=10 with a fraction of 0.25 produced a 0.020 arm
    instead of 0.025, a 17.7 percent volume error, and n=16 with 0.4 produced
    0.0375 instead of 0.040. That is invisible to a solver comparison whose
    mesh comes from here, because both solvers then agree on the wrong solid.
    It was found by measuring against an independent CAD volume.

    So an unbuildable request now RAISES. Pass `allow_snapping=True` to accept
    the rounded arm deliberately, which is reasonable when the exact thickness
    does not matter, and use `realised_arm_thickness` to see what that will be.
    """
    if not 0.0 < thickness_fraction < 1.0:
        raise ValueError("thickness_fraction must be in (0, 1)")
    if n < 4:
        raise ValueError("n must be at least 4")

    cell = size_m / n
    exact_arm = n * thickness_fraction
    arm = max(1, int(round(exact_arm)))
    if not allow_snapping and abs(exact_arm - arm) > ARM_SNAP_TOLERANCE:
        usable = _usable_grid_sizes(thickness_fraction, n)
        raise ValueError(
            f"a thickness fraction of {thickness_fraction} needs "
            f"{exact_arm:.4f} cells on an n={n} grid, so the arm would be "
            f"rounded to {arm} and the bracket built would be "
            f"{realised_arm_thickness(size_m, thickness_fraction, n):.6f} m "
            f"thick instead of {size_m * thickness_fraction:.6f} m. "
            f"Use one of n={usable} instead, or pass allow_snapping=True to "
            f"accept the rounded arm on purpose")
    active = np.zeros((n, n, nz), dtype=bool)
    # Vertical arm: full height on the left.
    active[:arm, :, :] = True
    # Horizontal arm: full width along the bottom.
    active[:, :arm, :] = True
    return _build(active, n, n, nz, cell, cell, width_m / nz)
