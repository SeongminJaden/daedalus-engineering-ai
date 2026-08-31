"""geometry.cad_export.mesh_fallback: the non-parametric path.

WHY THIS EXISTS, STATED PLAINLY

"Always STEP" is a promise this project can keep only for **parametric** solids.
A hollow box is a handful of planar faces, so its B-rep is exact and the STEP
file is clean.

Topology-optimized and organic shapes are a different problem. They come out of
the optimizer as a density field or an implicit surface, not as faces and edges.
Turning that into a clean B-rep needs surface reconstruction and refitting, and
the usual result is either a NURBS patchwork that no downstream CAD system is
happy with, or thousands of tiny facets pretending to be a solid. Neither is a
STEP file anyone should machine from.

So the honest split is:

  parametric shapes  -> B-rep -> STEP, exact, guaranteed  (hollow_rect.py)
  organic shapes     -> mesh  -> STL, approximate, NOT a clean STEP  (here)

`export_stl` tessellates a B-rep, which is useful for visualisation and printing
today. `mesh_from_density_field` is the entry point for the topology case and is
a deliberate stub: implementing it badly would produce files that look like
manufacturable geometry and are not.
"""

from __future__ import annotations

from pathlib import Path

from .kernel import Kernel, require_kernel


def export_stl(solid, path: str | Path, kernel: Kernel | None = None,
               tolerance: float = 1e-3, angular_tolerance: float = 0.1) -> Path:
    """Tessellate a B-rep and write STL.

    The STL is an approximation with a stated linear tolerance, in millimetres,
    because that is what a triangle mesh is. It is not a substitute for the
    STEP file when dimensional accuracy matters.
    """
    kernel = kernel or require_kernel()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if kernel.name == "build123d":
        kernel.module.export_stl(solid, str(path), tolerance=tolerance,
                                 angular_tolerance=angular_tolerance)
    else:
        kernel.module.exporters.export(solid, str(path), exportType="STL",
                                       tolerance=tolerance,
                                       angularTolerance=angular_tolerance)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"STL export produced no file at {path}")
    return path


def stl_volume_m3(path: str | Path, scale_to_m: float = 1e-3) -> float:
    """Volume of a written STL, via trimesh. For checking a tessellation.

    A closed mesh has a well defined volume, so this catches a tessellation that
    lost the cavity or left the surface open. It will not match the B-rep
    exactly: faceting a curved surface always loses a little, and for a box it
    should match closely.
    """
    import trimesh

    mesh = trimesh.load_mesh(str(path))
    if not mesh.is_watertight:
        raise ValueError(
            f"{path} is not watertight, so its volume is undefined; the "
            "tessellation is not a closed solid")
    return float(mesh.volume) * (scale_to_m ** 3)


def mesh_from_density_field(density, spacing_m, threshold: float = 0.5):
    """Topology-optimization result to a surface mesh. NOT IMPLEMENTED.

    The intended route is marching cubes on the density field, then smoothing
    and decimation, then STL. What it will NOT produce is a clean parametric
    STEP: recovering analytic faces from a voxel field is surface
    reconstruction, and doing it badly yields geometry that looks
    manufacturable and is not.

    Left unimplemented on purpose until Phase 10 or later, rather than shipping
    something that quietly degrades the "always STEP" guarantee.
    """
    raise NotImplementedError(
        "Meshing a topology-optimized density field is not implemented. "
        "Parametric designs export clean STEP via geometry.cad_export."
        "hollow_rect.export_step; organic geometry needs surface "
        "reconstruction, which is a later phase. See the module docstring."
    )
