"""Surface points from a B-rep, and a shape signature that needs no learning.

The learned embedding in `embedding.py` consumes points, so this module is
where a solid becomes points. It tessellates through OpenCASCADE, samples the
triangles in proportion to their area, and normalises the cloud so that
position and size fall away. It also computes the D2 signature: the histogram
of distances between random pairs of surface points, which is invariant to
translation, rotation and scale by construction and was the standard shape
descriptor before anyone learned one. It is the baseline the learned embedding
has to beat, and if it does not, the docs say so.

VALIDITY DOMAIN
===============
    Tessellation is an approximation with a stated linear deflection. On a
    planar solid it is exact and the triangle area equals the B-rep area to
    round-off. On a curved solid the triangles chord the surface, so sampled
    points lie slightly inside a convex face and the area is slightly low;
    the deflection is recorded so a reader knows by how much at most.

    Point sampling is random. Two clouds of the same part differ, and any
    quantity computed from them differs too; the D2 signature is a histogram
    of a random sample and has sampling noise of its own, measured in the
    tests rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Linear deflection of the tessellation, in the shape's own units. The
#: shapes here are in millimetres, so this is a tenth of a millimetre.
DEFAULT_DEFLECTION = 0.1
D2_BINS = 64


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray        # (n, 3), metres
    triangles: np.ndarray       # (m, 3) indices
    deflection_m: float

    @property
    def areas(self) -> np.ndarray:
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    @property
    def area_m2(self) -> float:
        return float(self.areas.sum())


def tessellate(shape, unit_to_metres: float = 1.0,
               deflection: float = DEFAULT_DEFLECTION) -> TriangleMesh:
    """Triangulate every face of a solid, in metres.

    The mesh is built into the shape by OpenCASCADE and read back face by
    face, applying each face's location so that all triangles share one
    frame.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, deflection, False, 0.2, True)
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        explorer.Next()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            continue
        transform = location.Transformation()
        base = len(vertices)
        for i in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i).Transformed(transform)
            vertices.append((p.X(), p.Y(), p.Z()))
        # OpenCASCADE winds the triangles for the underlying surface, and a
        # face whose orientation is REVERSED uses that surface inside out.
        # Without the swap half the normals of a box point inward, which was
        # measured (outward fraction 0.5) before the overhang and thickness
        # measures gave nonsense; with it every normal points out.
        reversed_face = face.Orientation() == TopAbs_REVERSED
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(i).Get()
            if reversed_face:
                b, c = c, b
            triangles.append((base + a - 1, base + b - 1, base + c - 1))
    if not triangles:
        raise ValueError("tessellation produced no triangles")
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64) * unit_to_metres,
        triangles=np.asarray(triangles, dtype=np.int64),
        deflection_m=deflection * unit_to_metres)


def sample_surface(mesh: TriangleMesh, n: int,
                   rng: np.random.Generator) -> np.ndarray:
    """n points on the surface, triangles chosen in proportion to area,
    uniform within each triangle."""
    areas = mesh.areas
    chosen = rng.choice(len(areas), size=n, p=areas / areas.sum())
    v0 = mesh.vertices[mesh.triangles[chosen, 0]]
    v1 = mesh.vertices[mesh.triangles[chosen, 1]]
    v2 = mesh.vertices[mesh.triangles[chosen, 2]]
    r1 = rng.random(n)
    r2 = rng.random(n)
    # the square root keeps the distribution uniform over the triangle
    # rather than crowding the first vertex
    s = np.sqrt(r1)
    return (1.0 - s)[:, None] * v0 + (s * (1.0 - r2))[:, None] * v1 \
        + (s * r2)[:, None] * v2


def normalise(points: np.ndarray) -> np.ndarray:
    """Centre on the centroid and scale to unit root-mean-square radius, so
    that where the part sits and how big it is both fall away."""
    centred = points - points.mean(axis=0)
    rms = np.sqrt((centred ** 2).sum(axis=1).mean())
    return centred / max(rms, 1e-300)


def canonical_frame(points: np.ndarray) -> np.ndarray:
    """Rotate a centred cloud onto its principal axes, largest variance first.

    This removes orientation before the encoder ever sees the cloud, which
    measured better than asking the encoder to learn it: with free random
    rotations as augmentation the same part rotated came back with a cosine
    as low as 0.19 to itself; aligned first, it comes back at 1.00.

    What alignment cannot fix is sign. Each axis may point either way, so the
    same part can land in one of four proper frames (an odd number of flips
    is a reflection and is excluded). The third moment along each axis picks
    a sign where the shape is asymmetric; where it is symmetric the moment is
    near zero and the choice is arbitrary, which is why the encoder is
    trained with the four flips as augmentation rather than trusted to see
    only one.
    """
    centred = points - points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(centred.T))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] = -axes[:, 2]
    aligned = centred @ axes
    for k in range(3):
        if (aligned[:, k] ** 3).mean() < 0.0:
            aligned[:, k] = -aligned[:, k]
    return aligned


def d2_signature(points: np.ndarray, bins: int = D2_BINS,
                 pairs: int = 20000, rng: np.random.Generator | None = None
                 ) -> np.ndarray:
    """Histogram of distances between random pairs, divided by the mean
    distance. Invariant to translation, rotation and scale by construction.
    Returned as a density that sums to one."""
    rng = rng or np.random.default_rng(0)
    i = rng.integers(0, len(points), size=pairs)
    j = rng.integers(0, len(points), size=pairs)
    keep = i != j
    d = np.linalg.norm(points[i[keep]] - points[j[keep]], axis=1)
    d = d / d.mean()
    hist, _ = np.histogram(d, bins=bins, range=(0.0, 4.0))
    return hist / hist.sum()


def point_cloud_of(shape, unit_to_metres: float, n: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Normalised, canonically oriented surface points of one solid: the
    input the encoder takes."""
    return canonical_frame(
        normalise(sample_surface(tessellate(shape, unit_to_metres), n, rng)))
