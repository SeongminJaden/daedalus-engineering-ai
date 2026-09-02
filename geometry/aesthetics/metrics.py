"""Measurable properties of a shape's form, kept off the evidence ladder.

Nothing here says a shape is beautiful. Each function reports a geometric
quantity that people who care about form tend to care about, and the caller
decides what to do with it. The distinction is not pedantry:

* A safety factor is a claim about the world that can be checked against a
  test piece. It belongs on the evidence ladder, where the top rung needs
  physical data.
* A compactness number is a claim about the shape only. It is exactly true and
  proves nothing about whether anyone will like the part.

Putting the second on the same ladder as the first would let a shape earn
confidence by being round, which is not a thing that should be possible. So
these are a separate axis with no rung on that ladder at all, and the guard
that matters is elsewhere: a preference can never overturn feasibility, which
`integration.multi_review` enforces by refusing to rank an inadmissible design
against admissible ones at all.

WHAT THESE ARE NOT. They are not a model of taste, they are not validated
against anyone's judgement, and no study here connects them to whether a part
looks good. They are reported as what they are: numbers about a surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Stated as a constant so it appears in any listing of this module and cannot
#: be mistaken for an oversight.
PREFERENCE_IS_NOT_EVIDENCE = (
    "Shape metrics are a preference axis. They carry no evidence level, they "
    "never raise confidence in a physical claim, and they cannot overturn a "
    "feasibility verdict.")

#: A sphere is the most compact closed surface there is, so the isoperimetric
#: quotient reaches exactly one there and nowhere else.
SPHERE_COMPACTNESS = 1.0

#: 36 pi V^2 / A^3 for a cube: V = a^3, A = 6 a^2, giving pi / 6. An exact
#: second anchor, and one with corners, so it checks the measure on something
#: that is not the trivial case.
CUBE_COMPACTNESS = math.pi / 6.0


@dataclass(frozen=True)
class ShapeMetrics:
    """Geometric properties of a closed surface. Not a score, and not a rank."""

    surface_area_m2: float
    volume_m3: float
    compactness: float
    dihedral_roughness_rad: float
    mirror_asymmetry_m: float

    @property
    def note(self) -> str:
        return PREFERENCE_IS_NOT_EVIDENCE


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def surface_area_m2(vertices: np.ndarray, faces: np.ndarray) -> float:
    return float(np.sum(_triangle_areas(np.asarray(vertices),
                                        np.asarray(faces))))


def compactness(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Isoperimetric quotient, 36 pi V^2 / A^3.

    One for a sphere, pi/6 for a cube, and smaller for anything more
    elaborate. It is a statement about how much surface a shape spends on its
    volume, which is a real property; whether a person prefers a compact shape
    is not something this knows.
    """
    from geometry.surfacing.organic import enclosed_volume_m3

    area = surface_area_m2(vertices, faces)
    if area <= 0.0:
        raise ValueError("a surface with no area has no compactness")
    volume = enclosed_volume_m3(vertices, faces)
    return float(36.0 * math.pi * volume ** 2 / area ** 3)


def dihedral_roughness_rad(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Spread of the angles between neighbouring faces.

    Zero on a flat sheet. Small and uniform on a well resolved smooth surface.
    Large where the surface is faceted, which is what a voxel boundary looks
    like before it is smoothed.

    Reported as a standard deviation rather than a mean, because a sphere has
    a non zero mean dihedral angle by construction and that is not roughness;
    what distinguishes faceted from smooth is how much the angle VARIES.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)

    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 0.0
    normals = normals[keep] / lengths[keep][:, None]
    kept_faces = faces[keep]

    edges: dict = {}
    for index, face in enumerate(kept_faces):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = (min(face[i], face[j]), max(face[i], face[j]))
            edges.setdefault(key, []).append(index)

    angles = []
    for owners in edges.values():
        if len(owners) != 2:
            continue
        dot = float(np.clip(np.dot(normals[owners[0]], normals[owners[1]]),
                            -1.0, 1.0))
        angles.append(math.acos(dot))
    if not angles:
        raise ValueError(
            "no interior edges were found, so there are no neighbouring faces "
            "to compare and roughness is undefined")
    return float(np.std(angles))


def mirror_asymmetry_m(vertices: np.ndarray, faces: np.ndarray,
                       axis: int = 0, plane: float | None = None) -> float:
    """How far the surface departs from mirror symmetry, in metres.

    Compares the area weighted centroid with the plane it should sit on, and
    the extent reached each way. Area weighted because the mean VERTEX
    position is not the centre of a shape: vertex density depends on how each
    cell happened to be triangulated.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    areas = _triangle_areas(vertices, faces)
    centres = vertices[faces].mean(axis=1)
    centroid = float((centres[:, axis] * areas).sum() / areas.sum())

    coordinate = vertices[:, axis]
    if plane is None:
        plane = 0.5 * (coordinate.min() + coordinate.max())
    reach = abs((coordinate.max() - plane) - (plane - coordinate.min()))
    return float(max(abs(centroid - plane), reach))


def measure_shape(vertices: np.ndarray, faces: np.ndarray,
                  symmetry_axis: int = 0) -> ShapeMetrics:
    """Every metric at once, with the label that they are preferences."""
    from geometry.surfacing.organic import enclosed_volume_m3

    return ShapeMetrics(
        surface_area_m2=surface_area_m2(vertices, faces),
        volume_m3=enclosed_volume_m3(vertices, faces),
        compactness=compactness(vertices, faces),
        dihedral_roughness_rad=dihedral_roughness_rad(vertices, faces),
        mirror_asymmetry_m=mirror_asymmetry_m(vertices, faces, symmetry_axis))


# Aliases kept short for callers, matching the exported names.
dihedral_roughness = dihedral_roughness_rad
mirror_asymmetry = mirror_asymmetry_m
