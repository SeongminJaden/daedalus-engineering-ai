"""Turning a topology density field into a smooth, closed surface.

The existing voxel export in `optimization.topology.export` is exact and
deliberately blocky: it emits the faces of retained voxels, so its volume is
known exactly and its appearance is honestly that of a voxel model. This is
the other option, and it trades that exactness for a surface a person would
accept on a part.

WHAT THE VOLUMES MEAN, because three different numbers get called "the
volume" and only one of them is what an isosurface encloses:

* `field_integral_m3` is the integral of the density, sum(rho) times the cell
  volume. For a SIMP result with grey in it this is the MASS-like quantity the
  optimiser was constraining.
* `thresholded_volume_m3` is the volume of the region above the level, which
  is what the voxel exporter emits.
* `enclosed_volume_m3` is what the isosurface actually bounds.

For a clean binary field all three converge as the grid is refined. For a grey
field they do NOT agree, and the difference is not an error: an isosurface has
to decide where the boundary is, and a half dense cell has no unambiguous
boundary. A report carries all three rather than picking one and calling it
the volume.

SMOOTHING, and a rationale that measurement destroyed. The textbook position
is that Laplacian smoothing shrinks a shape, so Taubin's alternating positive
and negative step should preserve volume better. That was the reason Taubin
was written here first. Measured on a thin plate, which is the worst case
because it has the most surface per unit volume, it is the wrong way round:

        passes      Taubin(0.5, -0.53)      Laplacian(0.5, -0.5)
             5                   +0.33%                   -0.02%
            25                   +1.58%                   -0.24%
           100                   +5.86%                   -1.54%
           200                  +11.90%                   -3.00%

The negative step overcorrects at these parameters and INFLATES the shape,
faster than plain Laplacian shrinks it, at every pass count tried and on every
shape tried. So the default is plain Laplacian, chosen on the measurement
rather than on the expectation. Taubin remains available by passing mu.

Volume change is the criterion measured here, and it is not the only thing
smoothing does; a claim about which produces a better SURFACE would need a
different measurement and is not made.

WHAT THIS DOES NOT DO. It does not check that the smoothed shape still carries
the load. Rounding a corner usually helps, by removing a stress concentration
the voxel boundary invented, but it can also thin a member below what it
needs. Re-running the structural check on the smoothed geometry is a separate
step and is not implied by anything here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The positive step. Larger relaxes faster per pass.
SMOOTHING_LAMBDA = 0.5

#: The negative step. Equal in magnitude to lambda, which is plain Laplacian
#: smoothing. A larger magnitude is Taubin's correction, and measurement here
#: shows it overcorrects and inflates rather than preserving volume, so it is
#: not the default. See the module docstring for the numbers.
SMOOTHING_MU = -0.5


@dataclass(frozen=True)
class SurfaceReport:
    """A surface and the three volumes that describe where it came from."""

    vertices: np.ndarray
    faces: np.ndarray
    enclosed_volume_m3: float
    thresholded_volume_m3: float
    field_integral_m3: float
    smoothing_passes: int
    volume_change_from_smoothing: float

    @property
    def triangle_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def enclosed_vs_thresholded(self) -> float:
        """Relative difference, which is a resolution measure on a binary
        field and a genuine ambiguity on a grey one."""
        if self.thresholded_volume_m3 == 0.0:
            return float("inf")
        return abs(self.enclosed_volume_m3 - self.thresholded_volume_m3) \
            / self.thresholded_volume_m3


def field_integral_m3(density: np.ndarray, spacing_m: float) -> float:
    """sum(rho) times the cell volume."""
    return float(np.sum(np.asarray(density, dtype=float))) * spacing_m ** 3


def thresholded_volume_m3(density: np.ndarray, spacing_m: float,
                          level: float = 0.5) -> float:
    """Volume of the region above the level, as the voxel exporter counts it."""
    return float(np.count_nonzero(np.asarray(density) > level)) * spacing_m ** 3


def isosurface(density: np.ndarray, spacing_m: float, level: float = 0.5):
    """Marching cubes on a density field, in metres.

    The field is padded with zeros first. Without that, material touching the
    edge of the grid produces an open surface there, and an open surface has
    no enclosed volume to speak of.
    """
    from skimage import measure

    field = np.asarray(density, dtype=float)
    if field.ndim != 3:
        raise ValueError(f"the density field must be 3D, got {field.ndim}D")
    if not np.any(field > level):
        raise ValueError(
            f"nothing in the field exceeds the level {level}, so there is no "
            f"surface to extract. The field's maximum is {field.max():.4g}")

    padded = np.pad(field, 1, mode="constant", constant_values=0.0)
    vertices, faces, _, _ = measure.marching_cubes(padded, level=level)
    # Undo the pad, then scale from cells to metres.
    return (vertices - 1.0) * spacing_m, faces


def enclosed_volume_m3(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume by the divergence theorem, summed over triangles.

    Uses the tetrahedron-to-origin formula, which needs a closed surface and
    consistent winding. Marching cubes gives both.
    """
    triangles = np.asarray(vertices)[np.asarray(faces)]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(abs(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0))


def _adjacency(vertices: np.ndarray, faces: np.ndarray):
    """Neighbour sums and counts, for one smoothing pass."""
    n = vertices.shape[0]
    sums = np.zeros_like(vertices)
    counts = np.zeros(n, dtype=np.int64)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        left, right = faces[:, i], faces[:, j]
        np.add.at(sums, left, vertices[right])
        np.add.at(sums, right, vertices[left])
        np.add.at(counts, left, 1)
        np.add.at(counts, right, 1)
    counts = np.maximum(counts, 1)[:, None]
    return sums / counts


def smooth(vertices: np.ndarray, faces: np.ndarray, passes: int = 10,
           lam: float = SMOOTHING_LAMBDA, mu: float = SMOOTHING_MU):
    """Alternate a positive and a negative Laplacian step.

    With mu equal to minus lambda this is plain Laplacian smoothing, which is
    the default because it held volume best in every case measured. Making mu
    more negative is Taubin's correction; it is available and it inflated the
    shape in every measurement taken here, so it is not chosen for you.

    Smoothing does not know that the part carries load. Whatever it does to
    the surface has to be re-checked structurally, which this module does not
    do and does not imply.
    """
    if passes < 0:
        raise ValueError("passes must not be negative")
    if lam <= 0.0 or mu >= 0.0:
        raise ValueError(
            "Taubin smoothing needs a positive lambda and a negative mu; "
            f"got lambda={lam}, mu={mu}")

    points = np.array(vertices, dtype=float, copy=True)
    faces = np.asarray(faces)
    for _ in range(passes):
        points += lam * (_adjacency(points, faces) - points)
        points += mu * (_adjacency(points, faces) - points)
    return points


def surface_from_density(density: np.ndarray, spacing_m: float,
                         level: float = 0.5,
                         smoothing_passes: int = 10) -> SurfaceReport:
    """The whole path: isosurface, smooth, and measure what changed."""
    vertices, faces = isosurface(density, spacing_m, level)
    before = enclosed_volume_m3(vertices, faces)

    if smoothing_passes:
        vertices = smooth(vertices, faces, passes=smoothing_passes)
    after = enclosed_volume_m3(vertices, faces)

    return SurfaceReport(
        vertices=vertices, faces=faces, enclosed_volume_m3=after,
        thresholded_volume_m3=thresholded_volume_m3(density, spacing_m, level),
        field_integral_m3=field_integral_m3(density, spacing_m),
        smoothing_passes=smoothing_passes,
        volume_change_from_smoothing=(0.0 if before == 0.0
                                      else (after - before) / before))
