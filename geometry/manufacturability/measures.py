"""Geometric measurements the process rules are judged on, from a triangle mesh.

Everything here is ray casting and face normals on the tessellated B-rep, in
metres. No library beyond numpy: trimesh's thickness needs rtree, which is
not installed, and a dependency for four hundred ray casts is not worth it.

WALL THICKNESS. Surface points are sampled in proportion to triangle area.
From each, a ray is cast INWARD along the negative face normal and the
distance to the first triangle it meets is the local thickness there. On a
plate of thickness t every sample that is not on an edge face reports t; on
a hollow section the wall. The minimum over samples is what a minimum wall
rule is judged on, and it is a sample minimum: a wall thinner than any sample
happened to land on is missed, which the report says.

OVERHANG. For a build direction, every downward facing triangle (normal
against the build axis) has an overhang angle measured from the vertical:
zero for a vertical wall, ninety for a horizontal underside. The area
fraction steeper than a limit is the overhang measure. Faces lying on the
build plate are excluded, since the plate supports them.

TOOL ACCESS. For three-axis milling in one setup the tool comes from one
direction; flipping the part gives the opposite direction, and a machinist
with six setups covers all axis-aligned directions. A triangle is reachable
from a direction when a ray leaving it toward the tool escapes the part and
the face does not face away from the tool; a pocket wall is reachable from
above because the cutter's side does the work. The area fraction reachable
from none of the six is the inaccessible fraction. Cavities that open only
through a small hole read as reachable here because the ray is a line and a
tool has a radius; that is a known optimism and is stated in the process
notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def _triangle_normals_and_areas(vertices: np.ndarray, triangles: np.ndarray):
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    doubled = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    ok = doubled > EPS
    normals[ok] = cross[ok] / doubled[ok, None]
    return normals, 0.5 * doubled


def ray_hits(origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray,
             triangles: np.ndarray, skip: int | None = None) -> np.ndarray:
    """Distances t > 0 along the ray to every triangle it crosses
    (Moller-Trumbore, vectorised over triangles). Sorted ascending."""
    v0 = vertices[triangles[:, 0]]
    e1 = vertices[triangles[:, 1]] - v0
    e2 = vertices[triangles[:, 2]] - v0
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > EPS
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    s = origin - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = np.einsum("j,ij->i", direction, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (t > 1e-9)
    if skip is not None:
        hit[skip] = False
    return np.sort(t[hit])


def wall_thickness_samples(vertices: np.ndarray, triangles: np.ndarray,
                           n_samples: int = 400,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """Local thickness at area-weighted surface samples, in metres.

    Samples whose inward ray meets nothing (an open or non-watertight mesh)
    are dropped; the count returned is therefore also a check on the mesh.
    """
    rng = rng or np.random.default_rng(0)
    normals, areas = _triangle_normals_and_areas(vertices, triangles)
    chosen = rng.choice(len(areas), size=n_samples, p=areas / areas.sum())
    r1, r2 = rng.random(n_samples), rng.random(n_samples)
    s = np.sqrt(r1)
    v0 = vertices[triangles[chosen, 0]]
    v1 = vertices[triangles[chosen, 1]]
    v2 = vertices[triangles[chosen, 2]]
    points = (1 - s)[:, None] * v0 + (s * (1 - r2))[:, None] * v1 + (s * r2)[:, None] * v2
    out = []
    for point, index in zip(points, chosen):
        inward = -normals[index]
        if not np.any(inward):
            continue
        hits = ray_hits(point, inward, vertices, triangles, skip=int(index))
        if len(hits):
            out.append(hits[0])
    return np.asarray(out, dtype=float)


def overhang_area_fraction(vertices: np.ndarray, triangles: np.ndarray,
                           build_axis: int = 1, max_angle_deg: float = 45.0,
                           plate_tolerance_m: float = 1e-6) -> tuple[float, float]:
    """(area fraction of downward faces steeper than the limit, worst angle).

    The angle is from the vertical: a vertical wall is 0, a horizontal
    underside is 90. Faces on the build plate are excluded.
    """
    normals, areas = _triangle_normals_and_areas(vertices, triangles)
    down = -normals[:, build_axis]
    facing_down = down > 1e-9
    centroids = vertices[triangles].mean(axis=1)
    on_plate = centroids[:, build_axis] <= vertices[:, build_axis].min() + plate_tolerance_m
    candidate = facing_down & ~on_plate
    if not candidate.any():
        return 0.0, 0.0
    angle = np.degrees(np.arcsin(np.clip(down[candidate], 0.0, 1.0)))
    steep = angle > max_angle_deg
    fraction = float(areas[candidate][steep].sum() / areas.sum())
    return fraction, float(angle.max())


def tool_access_area_fraction(vertices: np.ndarray, triangles: np.ndarray,
                              directions: np.ndarray | None = None
                              ) -> tuple[float, int]:
    """(area fraction reachable from NO direction, number of directions).

    A face is reachable from a tool direction d when a ray leaving the face
    AGAINST d escapes the part without meeting it again, and the face does
    not face away from the tool (n dot d <= 0). The second clause lets a
    vertical wall count as reachable from above, because an end mill cuts a
    wall with its side, not its tip; the first version required the ray to
    hit the face itself first and so called every pocket wall unreachable,
    which was measured and wrong. Default directions are the six axes.
    """
    if directions is None:
        directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                               [0, 0, 1], [0, 0, -1]], dtype=float)
    normals, areas = _triangle_normals_and_areas(vertices, triangles)
    centroids = vertices[triangles].mean(axis=1)
    extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    reachable = np.zeros(len(triangles), dtype=bool)
    for direction in directions:
        not_facing_away = np.einsum("ij,j->i", normals, direction) <= 1e-9
        for index in np.flatnonzero(not_facing_away & ~reachable):
            start = centroids[index] + normals[index] * (1e-6 * extent)
            hits = ray_hits(start, -direction, vertices, triangles, skip=int(index))
            if len(hits) == 0:
                reachable[index] = True
    return float(areas[~reachable].sum() / areas.sum()), len(directions)


@dataclass(frozen=True)
class MeshMeasures:
    """Everything the rules read, measured once."""

    n_triangles: int
    surface_area_m2: float
    min_wall_m: float | None
    p05_wall_m: float | None
    median_wall_m: float | None
    wall_spread: float | None            # (p95 - p05) / median, uniformity
    wall_samples: int
    overhang_fraction_45: float
    overhang_fraction_50: float
    worst_overhang_deg: float
    inaccessible_fraction_6_axis: float
    build_axis: int


def measure_mesh(vertices: np.ndarray, triangles: np.ndarray, build_axis: int = 1,
                 n_samples: int = 400, rng: np.random.Generator | None = None
                 ) -> MeshMeasures:
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    _, areas = _triangle_normals_and_areas(vertices, triangles)
    walls = wall_thickness_samples(vertices, triangles, n_samples, rng)
    if len(walls):
        p05, p95, med = np.percentile(walls, 5), np.percentile(walls, 95), np.median(walls)
        spread = float((p95 - p05) / med) if med > 0 else None
    else:
        p05 = p95 = med = None
        spread = None
    f45, worst = overhang_area_fraction(vertices, triangles, build_axis, 45.0)
    f50, _ = overhang_area_fraction(vertices, triangles, build_axis, 50.0)
    inaccessible, _ = tool_access_area_fraction(vertices, triangles)
    return MeshMeasures(
        n_triangles=int(len(triangles)), surface_area_m2=float(areas.sum()),
        min_wall_m=float(walls.min()) if len(walls) else None,
        p05_wall_m=None if p05 is None else float(p05),
        median_wall_m=None if med is None else float(med),
        wall_spread=spread, wall_samples=int(len(walls)),
        overhang_fraction_45=f45, overhang_fraction_50=f50,
        worst_overhang_deg=worst, inaccessible_fraction_6_axis=inaccessible,
        build_axis=build_axis)
