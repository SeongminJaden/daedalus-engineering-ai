"""From a voxel field to a surface a designer could work with, and the price.

The blocky export is exact: its volume is the retained voxel volume by
construction. It is also unusable as a part, and its staircase corners are
stress raisers that no real machine would produce. This module makes the other
kind of surface, and measures what it costs.

WHAT IT DOES
============
Marching cubes on the density field at an iso level, then Taubin smoothing,
which alternates a shrinking pass and an expanding one so the body does not
deflate the way plain Laplacian smoothing does. Both steps change the volume,
and the change is the number that matters, so every function here returns it
rather than reporting success.

WHAT IT IS NOT
==============
It is not surface fitting and it does not produce a STEP. Recovering analytic
faces from a marching cubes mesh is surface reconstruction, and the honest
statement of why it is not attempted is that nothing in this repository can do
it: build123d can sew a shell from triangles, but a shell with thousands of
planar facets is a STEP file that says the same thing the STL says, at ten
times the size, and no downstream CAD system would treat it as a solid worth
editing. What comes out is an STL, with the volume error stated.

WHAT THE ISO LEVEL MEANS
========================
The same thing the threshold means for the voxel export, and the volume error
is measured against the same reference: the volume of the density field,
integral of rho over the domain, which is what the optimiser was constraining.
A marching cubes surface at iso 0.5 does not have that volume, and the
difference is the first row of every table here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physics.fem.mesh import Mesh


@dataclass
class SmoothedSurface:
    vertices: np.ndarray
    triangles: np.ndarray
    iso_level: float
    smoothing_iterations: int
    volume_m3: float
    field_volume_m3: float
    voxel_volume_m3: float
    watertight: bool
    n_components: int

    @property
    def volume_error_vs_field(self) -> float:
        return (self.volume_m3 - self.field_volume_m3) / self.field_volume_m3

    @property
    def volume_error_vs_voxels(self) -> float:
        return (self.volume_m3 - self.voxel_volume_m3) / self.voxel_volume_m3

    def row(self) -> dict:
        return {"iso_level": self.iso_level,
                "smoothing_iterations": self.smoothing_iterations,
                "volume_m3": self.volume_m3,
                "field_volume_m3": self.field_volume_m3,
                "voxel_volume_m3": self.voxel_volume_m3,
                "error_vs_field": self.volume_error_vs_field,
                "error_vs_voxels": self.volume_error_vs_voxels,
                "watertight": self.watertight,
                "n_components": self.n_components,
                "n_triangles": int(self.triangles.shape[0])}


def density_grid(mesh: Mesh, density: np.ndarray) -> np.ndarray:
    """The element densities as an (nx, ny, nz) array, in grid order."""
    density = np.asarray(density, dtype=float).reshape(-1)
    cell = np.array([mesh.dx, mesh.dy, mesh.dz])
    index = np.round(mesh.element_centroids() / cell - 0.5).astype(int)
    grid = np.zeros((mesh.nx, mesh.ny, mesh.nz), dtype=float)
    grid[index[:, 0], index[:, 1], index[:, 2]] = density
    return grid


def taubin_smooth(vertices: np.ndarray, triangles: np.ndarray,
                  iterations: int = 10, lamb: float = 0.5,
                  mu: float = -0.53) -> np.ndarray:
    """Taubin smoothing: a shrinking pass and an expanding one, repeated.

    Plain Laplacian smoothing shrinks a closed body every pass, so ten passes
    quietly remove several percent of the volume. The expanding pass with
    mu just below -lambda is what keeps that from happening; the residual
    change is measured by the caller rather than assumed to be zero.
    """
    vertices = np.asarray(vertices, dtype=float).copy()
    triangles = np.asarray(triangles, dtype=int)
    n = vertices.shape[0]
    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]],
                       triangles[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    counts = np.bincount(edges[:, 0], minlength=n).astype(float)
    counts[counts == 0] = 1.0

    def pass_(points: np.ndarray, factor: float) -> np.ndarray:
        summed = np.zeros_like(points)
        np.add.at(summed, edges[:, 0], points[edges[:, 1]])
        average = summed / counts[:, None]
        return points + factor * (average - points)

    for _ in range(iterations):
        vertices = pass_(vertices, lamb)
        vertices = pass_(vertices, mu)
    return vertices


def marching_surface(mesh: Mesh, density: np.ndarray, iso_level: float = 0.5,
                     smoothing_iterations: int = 10) -> SmoothedSurface:
    """Marching cubes at `iso_level`, smoothed, with its volume measured."""
    from skimage import measure
    import trimesh

    grid = density_grid(mesh, density)
    # Pad with the empty value so the surface closes at the domain boundary;
    # an open surface has no volume and every number below would be a lie.
    padded = np.pad(grid, 1, mode="constant", constant_values=0.0)
    spacing = (mesh.dx, mesh.dy, mesh.dz)
    vertices, faces, _normals, _values = measure.marching_cubes(
        padded, level=iso_level, spacing=spacing)
    # Undo the pad offset AND put the surface where the field is. A density
    # lives at its element's CENTRE, so unpadded element j sits at
    # (j + 0.5) * d, while marching cubes returns padded index j + 1 at
    # (j + 1) * d. Subtracting a full cell put the whole body half an element
    # low on every axis: the volume was right and the position was not, which
    # is invisible in a volume check and wrong the moment the part is placed
    # in an assembly.
    vertices = vertices - 0.5 * np.array(spacing)
    if smoothing_iterations:
        vertices = taubin_smooth(vertices, faces, smoothing_iterations)

    surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Point the triangles OUTWARD. Marching cubes orients its faces by the
    # direction the field decreases, which for this field comes out inside
    # out: the body's signed volume was negative and every mesh boolean
    # refused it as "not a volume". Nothing that only measured a magnitude
    # ever saw it, because the volume was read through abs().
    trimesh.repair.fix_normals(surface)
    field_volume = float(np.sum(np.asarray(density, dtype=float))
                         * mesh.element_volume)
    voxel_volume = float(np.sum(np.asarray(density, dtype=float) >= iso_level)
                         * mesh.element_volume)
    return SmoothedSurface(
        vertices=np.asarray(surface.vertices), triangles=np.asarray(surface.faces),
        iso_level=float(iso_level), smoothing_iterations=int(smoothing_iterations),
        volume_m3=abs(float(surface.volume)), field_volume_m3=field_volume,
        voxel_volume_m3=voxel_volume, watertight=bool(surface.is_watertight),
        n_components=int(surface.body_count))


def write_stl(surface: SmoothedSurface, path: str | Path) -> Path:
    import trimesh
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=surface.vertices, faces=surface.triangles,
                    process=False).export(str(path))
    return path


def smoothing_table(mesh: Mesh, density: np.ndarray,
                    iso_levels=(0.3, 0.5, 0.7),
                    iterations=(0, 5, 10, 20)) -> list[dict]:
    """Volume error against the field for each iso level and smoothing pass."""
    return [marching_surface(mesh, density, iso, n).row()
            for iso in iso_levels for n in iterations]


def format_table(rows: list[dict]) -> str:
    lines = ["| iso | smoothing | volume m3 | error vs field | error vs voxels | "
             "watertight | bodies | triangles |", "|" + "---|" * 8]
    for r in rows:
        lines.append(
            f"| {r['iso_level']} | {r['smoothing_iterations']} | "
            f"{r['volume_m3']:.6e} | {r['error_vs_field']:+.1%} | "
            f"{r['error_vs_voxels']:+.1%} | {r['watertight']} | "
            f"{r['n_components']} | {r['n_triangles']} |")
    return "\n".join(lines)


def tet_mesh_from_stl(path: str | Path, target_size_m: float, order: int = 2):
    """Volume mesh of a smoothed surface, for re-solving it.

    Gmsh reclassifies the STL facets into one surface, sews them into a shell
    and fills it. It only works on a watertight surface, which the blocky voxel
    export is not, and that asymmetry is the point: smoothing is what makes the
    topology result analysable as a body rather than as a field.
    """
    from nodes.gmsh_node import GMSH_TET10_TO_CALCULIX, TetMesh, _TET4, _TET10, _session

    with _session("stl_tetrahedra") as gmsh:
        gmsh.merge(str(path))
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.mesh.classifySurfaces(np.pi, True, True, np.pi)
        gmsh.model.mesh.createGeometry()
        surfaces = [s[1] for s in gmsh.model.getEntities(2)]
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_size_m * 0.1)
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        tags, flat, _ = gmsh.model.mesh.getNodes()
        types, _, node_tags = gmsh.model.mesh.getElements(3)
        types = list(types)
        wanted = _TET10 if order == 2 else _TET4
        if wanted not in types:
            raise RuntimeError(f"Gmsh produced element types {types}, not tetrahedra")
        coords = np.asarray(flat, dtype=float).reshape(-1, 3)
        lookup = {int(tag): i for i, tag in enumerate(tags)}
        per = 10 if order == 2 else 4
        raw = np.asarray(node_tags[types.index(wanted)]).reshape(-1, per)
        connectivity = np.vectorize(lookup.get)(raw)
    if order == 2:
        connectivity = connectivity[:, list(GMSH_TET10_TO_CALCULIX)]
    return TetMesh(node_coords=coords, connectivity=connectivity)
