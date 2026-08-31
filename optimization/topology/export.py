"""optimization.topology.export: turning a density field into a surface.

The Phase 9 boundary applies and is not softened here: a topology result is
**not** a clean STEP. It arrives as a density field, and recovering analytic
faces from one is surface reconstruction.

What this does instead is exact for what it is: threshold the field, then emit
the boundary faces of the surviving voxels. The volume is exactly the retained
voxel volume, so it is always known. It is also unmistakably blocky, which is
the honest appearance of a voxel model.

A NOTE ON WATERTIGHTNESS, because the reported flag is often False and that is
not a bug. Two voxels can meet along an edge or at a corner with no shared
face. Face-connectivity filtering removes material that is *only* attached that
way to the rest of the structure, but a diagonal contact INSIDE an otherwise
connected body still leaves a non-manifold edge, and a mesh with non-manifold
edges is not a closed surface. Measured on the cantilever result here, the
structure is fully face-connected (264 of 264 voxels survive the filter) and the
surface is still non-manifold at diagonal contacts.

The volume is reported from the voxel count in that case rather than from the
mesh, so the number stays exact. But it is one more concrete reason this output
is a design concept and not a manufacturable body: closing those contacts means
smoothing or remeshing, which changes the shape, and that is a separate step
this module does not pretend to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physics.fem.mesh import Mesh

# Faces of a unit cube, as (offset direction, four corner offsets).
_FACES = [
    ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
    ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
    ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
]


@dataclass
class MeshExportReport:
    path: Path
    threshold: float
    retained_elements: int
    total_elements: int
    volume_m3: float
    watertight: bool

    @property
    def retained_fraction(self) -> float:
        return self.retained_elements / max(self.total_elements, 1)

    def as_dict(self) -> dict:
        return {"path": str(self.path), "threshold": self.threshold,
                "retained_elements": self.retained_elements,
                "total_elements": self.total_elements,
                "retained_fraction": self.retained_fraction,
                "volume_m3": self.volume_m3, "watertight": self.watertight}


def largest_connected_component(mesh: Mesh, density: np.ndarray,
                                threshold: float = 0.5) -> np.ndarray:
    """Keep only the largest FACE-connected blob of solid elements.

    A thresholded SIMP field routinely contains material joined to the rest only
    through an edge or a corner. Such a join carries no load (two voxels meeting
    at an edge have no shared face to transmit stress through) and it makes the
    voxel surface non-manifold, so the exported mesh is not watertight and its
    volume is undefined.

    Face connectivity is the right rule precisely because it matches what can
    actually carry load.
    """
    from scipy import ndimage

    density = np.asarray(density, dtype=np.float64).reshape(-1)
    grid = np.zeros((mesh.nx, mesh.ny, mesh.nz), dtype=bool)
    cell = np.array([mesh.dx, mesh.dy, mesh.dz])
    index = np.round(mesh.element_centroids() / cell - 0.5).astype(np.int64)
    for e in range(mesh.n_elements):
        if density[e] >= threshold:
            grid[tuple(index[e])] = True

    # 6-connectivity: faces only, not edges or corners.
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(grid, structure=structure)
    if count == 0:
        return density.copy()
    sizes = ndimage.sum(grid, labels, range(1, count + 1))
    keep = int(np.argmax(sizes)) + 1

    out = density.copy()
    for e in range(mesh.n_elements):
        if density[e] >= threshold and labels[tuple(index[e])] != keep:
            out[e] = 0.0
    return out


def voxel_surface(mesh: Mesh, density: np.ndarray, threshold: float = 0.5):
    """Vertices and triangles of the thresholded density field's boundary."""
    density = np.asarray(density, dtype=np.float64).reshape(-1)
    if density.shape[0] != mesh.n_elements:
        raise ValueError(
            f"density has {density.shape[0]} entries for {mesh.n_elements} "
            "elements")

    cell = np.array([mesh.dx, mesh.dy, mesh.dz])
    centroids = mesh.element_centroids()
    index = np.round(centroids / cell - 0.5).astype(np.int64)
    solid = {tuple(index[e]): e for e in range(mesh.n_elements)
             if density[e] >= threshold}
    if not solid:
        raise ValueError(
            f"no element survives the threshold {threshold}; nothing to export")

    vertices: dict[tuple, int] = {}
    triangles: list[tuple[int, int, int]] = []

    def vertex_id(corner) -> int:
        if corner not in vertices:
            vertices[corner] = len(vertices)
        return vertices[corner]

    for cell_index in solid:
        base = np.array(cell_index)
        for direction, corners in _FACES:
            if tuple(base + np.array(direction)) in solid:
                continue                      # interior face, not on the surface
            ids = [vertex_id(tuple(base + np.array(c))) for c in corners]
            triangles.append((ids[0], ids[1], ids[2]))
            triangles.append((ids[0], ids[2], ids[3]))

    coordinates = np.zeros((len(vertices), 3), dtype=np.float64)
    for corner, vid in vertices.items():
        coordinates[vid] = np.array(corner, dtype=np.float64) * cell
    return coordinates, np.array(triangles, dtype=np.int64), len(solid)


def export_stl(mesh: Mesh, density: np.ndarray, path: str | Path,
               threshold: float = 0.5,
               largest_component_only: bool = True) -> MeshExportReport:
    """Write the thresholded topology as STL, and check it is a closed solid.

    `largest_component_only` drops material that is not face-connected to the
    main structure. Such material cannot carry load and makes the surface
    non-manifold; keeping it would produce an STL whose volume is undefined.
    """
    import trimesh

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if largest_component_only:
        density = largest_connected_component(mesh, density, threshold)
    vertices, faces, retained = voxel_surface(mesh, density, threshold)

    surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    watertight = bool(surface.is_watertight)
    surface.export(str(path))

    expected_volume = retained * mesh.element_volume
    if watertight:
        volume = abs(float(surface.volume))
        if abs(volume - expected_volume) / expected_volume > 1e-6:
            raise ValueError(
                f"exported volume {volume:.6g} m^3 does not match the retained "
                f"voxel volume {expected_volume:.6g} m^3")
    else:
        volume = expected_volume

    return MeshExportReport(path=path, threshold=threshold,
                            retained_elements=retained,
                            total_elements=mesh.n_elements,
                            volume_m3=volume, watertight=watertight)


def grey_fraction(density: np.ndarray, low: float = 0.1,
                  high: float = 0.9) -> float:
    """Share of elements that are neither solid nor void.

    SIMP leaves intermediate densities, and they have no physical meaning: no
    material is 40% present. A high grey fraction means the thresholded shape
    differs substantially from the field that was optimized, so this number
    should be reported alongside any topology result.
    """
    x = np.asarray(density, dtype=np.float64)
    return float(np.mean((x > low) & (x < high)))
