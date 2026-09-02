"""Whether a shape can be made, checked on the density field and the surface.

Two checks, chosen because each has an exact answer on shapes where the answer
is known, rather than because they are the two that were easiest to write.

WALL THICKNESS, from the density field. The Euclidean distance transform gives
each material voxel its distance to the nearest empty one, so twice that
distance is the local thickness there. On the ridge of that field, where the
distance stops increasing, twice the distance is the wall thickness. For a
plate four voxels thick the ridge sits at two voxels and the reported
thickness is four, exactly.

DRAFT, from the surface. A mould has to open, and for a TWO PART mould each
face is formed by one half or the other: a face pointing along the pull
releases upward, one pointing against it releases downward, and both are fine.
The faces that cannot release are the ones nearly PARALLEL to the pull, which
drag against the tool as it opens. So the quantity is the angle between a face
and the parting direction taken WITHOUT sign, and the first version of this
check got it wrong: it treated the underside of a box as an undercut when the
lower mould half forms it perfectly well.

WHAT THIS DOES NOT COVER. Internal voids that no tool can reach, minimum
radius for a cutter, and the difference between casting, moulding and milling
are all real and none of them are here. A shape passing both checks is not
therefore manufacturable; it has passed two specific checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WallReport:
    """Thickness measured on the density field."""

    minimum_wall_m: float
    median_wall_m: float
    thinnest_location: tuple
    voxels_below_floor: int

    def passes(self, floor_m: float) -> bool:
        return self.minimum_wall_m >= floor_m


@dataclass(frozen=True)
class DraftReport:
    """How the surface stands relative to a pull direction."""

    pull_axis: int
    minimum_draft_deg: float
    total_area_m2: float
    #: Per face angle and area, kept so any threshold can be asked for after
    #: the fact rather than baked in at measurement time.
    face_draft_deg: np.ndarray
    face_area_m2: np.ndarray

    def area_fraction_below(self, required_deg: float) -> float:
        """Fraction of the surface that fails a given draft requirement.

        THIS is the usable measure, not the minimum. On a surface extracted
        from voxels there are always individual facets lying exactly along an
        axis, so the minimum draft is zero for a plain box, a tapered box and
        a sphere alike, and separates none of them. The area fraction does:
        measured at one degree, 63 percent for a square box against 28 percent
        for a tapered one.

        A sphere reports zero minimum draft correctly, incidentally: its
        equator really is parallel to any pull direction.
        """
        failing = self.face_area_m2[self.face_draft_deg < required_deg].sum()
        return float(failing / self.face_area_m2.sum())

    def passes(self, required_deg: float,
               allowed_area_fraction: float = 0.0) -> bool:
        """Whether little enough of the surface fails the requirement.

        The allowance is explicit and defaults to zero, so a caller who wants
        to tolerate faceting has to say so rather than inherit a tolerance
        someone chose for them.
        """
        return self.area_fraction_below(required_deg) <= allowed_area_fraction


def wall_thickness(density: np.ndarray, spacing_m: float,
                   level: float = 0.5,
                   floor_m: float | None = None) -> WallReport:
    """Local wall thickness from the distance transform of the material.

    Reported at the RIDGE of the distance field rather than everywhere. Every
    material voxel near a surface has a small distance to the outside, and
    calling that the wall thickness would report every part as infinitely
    thin. The ridge is where the distance stops growing, which is the middle
    of the wall.
    """
    from scipy import ndimage

    field = np.asarray(density, dtype=float)
    solid = field > level
    if not solid.any():
        raise ValueError("there is no material in this field to measure")

    distance = ndimage.distance_transform_edt(solid) * spacing_m
    # The ridge: a voxel no closer to the surface than any of its neighbours.
    peak = ndimage.maximum_filter(distance, size=3)
    ridge = solid & (distance >= peak - 1e-12) & (distance > 0.0)
    if not ridge.any():
        raise ValueError(
            "no interior ridge was found, so the material is everywhere one "
            "voxel thick and the grid cannot resolve a wall")

    thickness = 2.0 * distance[ridge]
    flat_index = int(np.argmin(np.where(ridge, distance, np.inf)))
    return WallReport(
        minimum_wall_m=float(thickness.min()),
        median_wall_m=float(np.median(thickness)),
        thinnest_location=tuple(int(i) for i in
                                np.unravel_index(flat_index, field.shape)),
        voxels_below_floor=(0 if floor_m is None
                            else int(np.count_nonzero(thickness < floor_m))))


def draft(vertices: np.ndarray, faces: np.ndarray, pull_axis: int = 2,
          pull_positive: bool = True) -> DraftReport:
    """Smallest angle any face makes with the parting direction.

    For a two part mould a face pointing along the pull is formed by the upper
    half and one pointing against it by the lower half; both release. What
    cannot release is a face nearly PARALLEL to the pull, which drags. The
    angle is therefore taken without sign, and a vertical wall reports zero.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    cross = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    keep = areas > 0.0
    if not keep.any():
        raise ValueError("every face has zero area, so there is no surface")

    normals = cross[keep] / (2.0 * areas[keep])[:, None]
    areas = areas[keep]

    pull = np.zeros(3)
    pull[pull_axis] = 1.0 if pull_positive else -1.0
    along = normals @ pull                      # cosine of normal to pull

    # Unsigned: a face at +90 degrees releases upward and one at -90 releases
    # downward. A face at 0 is parallel to the pull and drags on both halves.
    angles = np.degrees(np.arcsin(np.clip(np.abs(along), 0.0, 1.0)))

    return DraftReport(
        pull_axis=pull_axis,
        minimum_draft_deg=float(angles.min()),
        total_area_m2=float(areas.sum()),
        face_draft_deg=angles,
        face_area_m2=areas)
