"""Re-checking a smoothed shape structurally, because smoothing does not know.

Smoothing moves material. Usually it helps, by rounding a stress
concentration that the voxel boundary invented rather than the loading
demanded. It can also thin a member the optimiser sized deliberately. Nothing
about a smoother surface says which happened, so the only honest position is
to solve it again.

This routes the smoothed surface through the general shape path that already
exists: a volume mesh of tetrahedra, handed to CalculiX. That path is already
verified; nothing new is being trusted here except the geometry.

WHY THERE IS NO BEFORE AND AFTER STRESS COMPARISON HERE. The obvious design
is to solve the blocky shape, solve the smoothed one, and reject the smoothing
if the peak stress rose. That was the plan, and measurement does not support
it.

On a FIXED geometry, refining only the mesh moves the peak von Mises by a
third:

        tetrahedra      peak von Mises      tip displacement
               751            2.50 MPa             0.00113 mm
              2278            2.07 MPa             0.00134 mm
              6162            1.82 MPa             0.00112 mm
             15043            1.71 MPa             0.00114 mm

Peak stress on linear tetrahedra at these densities is not converged; it keeps
climbing as the mesh finds the corner. Smoothing also changes the mesh, so a
before and after difference is dominated by remeshing rather than by the
geometry change. Measured across smoothing passes the peak appeared to rise 42
percent and then fall back, which is mesh noise wearing the costume of a
finding.

Displacement is far steadier across the same refinement, because it is an
integral of the solution rather than an extremum of its derivative.

So this module reports what the smoothed shape does ON ITS OWN TERMS, and
offers `mesh_sensitivity` so a caller can see whether a stress number is
trustworthy before believing it. It does NOT return a smoothing verdict,
because the measurement that verdict would rest on is not stable enough to
carry it.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Revalidation:
    """What the smoothed shape does under the same load."""

    peak_von_mises_pa: float
    max_displacement_m: float
    tetrahedra: int
    nodes: int
    enclosed_volume_m3: float

    def safety_factor(self, yield_strength_pa: float) -> float:
        if self.peak_von_mises_pa <= 0.0:
            raise ValueError(
                "the solve reported no stress, so a safety factor computed "
                "from it would be meaningless rather than infinite")
        return yield_strength_pa / self.peak_von_mises_pa


def tet_mesh_from_surface(vertices: np.ndarray, faces: np.ndarray,
                          size_factor: float = 1.0):
    """Fill a closed triangle surface with tetrahedra.

    Raises if the surface is not closed. An open surface can still be meshed
    into something, and that something is not the part.
    """
    import gmsh
    import trimesh

    from nodes.gmsh_node import TetMesh

    surface = trimesh.Trimesh(vertices=np.asarray(vertices),
                              faces=np.asarray(faces))
    if not surface.is_watertight:
        raise ValueError(
            "the surface is not watertight, so it does not enclose a solid "
            "and any volume mesh built from it would be filling a shape that "
            "is not the part")

    directory = Path(tempfile.mkdtemp())
    stl = directory / "surface.stl"
    surface.export(stl)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(stl))
        gmsh.model.mesh.classifySurfaces(np.pi / 4.0, True, True)
        gmsh.model.mesh.createGeometry()
        entities = gmsh.model.getEntities(2)
        loop = gmsh.model.geo.addSurfaceLoop([e[1] for e in entities])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", size_factor)
        gmsh.model.mesh.generate(3)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tags = tags[order]
        points = coords.reshape(-1, 3)[order]
        index = {int(t): i for i, t in enumerate(tags)}

        types, ids, nodes = gmsh.model.mesh.getElements(3)
        cells = []
        for element_type, batch, flat in zip(types, ids, nodes):
            if element_type != 4:            # 4 node tetrahedron
                continue
            for row in flat.reshape(len(batch), -1):
                cells.append([index[int(v)] for v in row])
        if not cells:
            raise RuntimeError(
                "gmsh produced no tetrahedra from this surface, so there is "
                "nothing to solve")
    finally:
        gmsh.finalize()

    return TetMesh(node_coords=points,
                   connectivity=np.array(cells, dtype=np.int64))


def revalidate(vertices: np.ndarray, faces: np.ndarray,
               youngs_modulus_pa: float, poisson_ratio: float,
               total_load_n: float, fixed_axis: int = 0,
               load_axis: int = 0, load_direction: int = 1,
               size_factor: float = 1.0,
               grip_fraction: float = 0.05) -> Revalidation:
    """Solve the smoothed shape and report what it does.

    The support and the load are taken as the slabs of nodes at each end along
    `fixed_axis`, which is how the original cantilever was posed. The fraction
    is deliberately a slab rather than a single face of nodes: a point support
    on a tetrahedral mesh produces a singularity, and the peak stress then
    reports the singularity rather than the part.
    """
    from nodes import calculix

    mesh = tet_mesh_from_surface(vertices, faces, size_factor)
    coordinate = mesh.node_coords[:, fixed_axis]
    span = coordinate.max() - coordinate.min()
    if span <= 0.0:
        raise ValueError("the shape has no extent along the fixed axis")

    grip = grip_fraction * span
    fixed = np.flatnonzero(coordinate <= coordinate.min() + grip)
    loaded = np.flatnonzero(coordinate >= coordinate.max() - grip)
    if fixed.size == 0 or loaded.size == 0:
        raise ValueError(
            "the support or the load slab is empty, so the problem is not "
            "posed; widen grip_fraction or refine the mesh")

    result = calculix.solve(
        mesh, youngs_modulus_pa, poisson_ratio, fixed, loaded, total_load_n,
        load_direction=load_axis if load_direction >= 0 else load_axis,
        element_type=calculix.ElementType.C3D4)

    if not result.converged:
        raise RuntimeError(
            "CalculiX did not converge on the smoothed shape, so its stress "
            "is not a result. A non converged solve reported as a number "
            "would look like a part that passed")

    return Revalidation(
        peak_von_mises_pa=float(result.max_von_mises_pa()),
        max_displacement_m=float(np.max(np.abs(result.displacements))),
        tetrahedra=int(mesh.connectivity.shape[0]),
        nodes=int(mesh.node_coords.shape[0]),
        enclosed_volume_m3=float(abs(surface_volume(vertices, faces))))


def mesh_sensitivity(vertices: np.ndarray, faces: np.ndarray,
                     youngs_modulus_pa: float, poisson_ratio: float,
                     total_load_n: float, coarse: float = 1.0,
                     fine: float = 0.5, **kwargs) -> dict:
    """Solve the same geometry twice at different mesh densities.

    A peak stress that moves substantially between the two is not a property
    of the part, and should not be compared with anything. Returned as numbers
    rather than a verdict, because how much movement is tolerable depends on
    what the number is for.
    """
    low = revalidate(vertices, faces, youngs_modulus_pa, poisson_ratio,
                     total_load_n, size_factor=coarse, **kwargs)
    high = revalidate(vertices, faces, youngs_modulus_pa, poisson_ratio,
                      total_load_n, size_factor=fine, **kwargs)
    return {
        "coarse": low,
        "fine": high,
        "stress_change": abs(high.peak_von_mises_pa - low.peak_von_mises_pa)
        / low.peak_von_mises_pa,
        "displacement_change": abs(high.max_displacement_m
                                   - low.max_displacement_m)
        / low.max_displacement_m,
    }


def surface_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    from .organic import enclosed_volume_m3

    return enclosed_volume_m3(vertices, faces)
