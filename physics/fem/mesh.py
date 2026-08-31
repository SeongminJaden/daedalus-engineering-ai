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
