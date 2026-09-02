"""Shape descriptors: a fixed vector of scale-free numbers read off a B-rep.

Every number here is either a count from the topology, a ratio of two
geometric quantities, or a fraction of faces of one surface type. None of them
depends on the unit the file was written in or on how large the part is, so a
100 mm plate and a 1 m plate of the same proportions describe identically.
That is what makes the vector usable for telling KINDS of part apart, and
useless for telling whether a part is big enough, which is a job for the
solvers and not for this module.

These are the hand-built baseline for the learned embeddings that come later.
An embedding that cannot beat this vector at separating the five synthetic
families has learned nothing the topology did not already say.

VALIDITY DOMAIN
===============
    One solid. The Euler characteristic is V - E + F - L, where L is the
    number of inner loops (wires beyond the first on each face). The first
    version left L out and reported genus 0 for a hollow tube, because its
    two annular end faces are not disks; measured on the families, the
    corrected count gives 2 for box, l_bracket and stepped_shaft, 0 for
    hollow_rect and 2 - 2n for a plate with n holes. It is a genus measure
    only for a B-rep whose seam and degenerate edges are the ones
    OpenCASCADE produces for planes, cylinders and cones. Spheres and tori
    carry degenerate edges that shift the count; the value is still reported
    for them, and it is still a number, but it is not two minus twice the
    genus there.

    The recogniser's holes and fillets are what they are in
    `nodes.feature_recognizer`: concave full-turn cylinders and faces tangent
    to all their neighbours. A hole broken open by a slot is not a hole here
    either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SURFACE_KINDS = ("plane", "cylinder", "sphere", "torus", "cone", "other")

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "log_compactness",        # log of area^3 / (36 pi V^2); a sphere is 0
    "aspect_mid",             # middle bounding side over the longest
    "aspect_min",             # shortest bounding side over the longest
    "fill",                   # volume over bounding box volume
    "inertia_mid_ratio",      # middle principal moment over the largest
    "inertia_min_ratio",      # smallest principal moment over the largest
    "com_offset",             # centre of mass to box centre, over the longest side
    "faces",
    "edges",
    "vertices",
    "euler",                  # V - E + F - inner loops; 2 - 2 genus
    "frac_plane",
    "frac_cylinder",
    "frac_sphere",
    "frac_torus",
    "frac_cone",
    "frac_other",
    "hole_count",
    "fillet_count",
    "hole_diameter_ratio",    # mean hole diameter over the longest side
    "fillet_radius_ratio",    # mean fillet radius over the longest side
    "other_cylinders",        # cylinder faces that are neither holes nor fillets
)


@dataclass(frozen=True)
class ShapeDescriptor:
    values: dict[str, float]

    def vector(self) -> np.ndarray:
        return np.array([self.values[n] for n in DESCRIPTOR_NAMES], dtype=float)

    def __getitem__(self, name: str) -> float:
        return self.values[name]


def _surface_counts(shape) -> dict[str, int]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    from nodes.feature_recognizer import _surface_kind

    counts = {k: 0 for k in SURFACE_KINDS}
    seen = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        explorer.Next()
        if any(face.IsSame(f) for f in seen):
            continue
        seen.append(face)
        counts[_surface_kind(BRepAdaptor_Surface(face))] += 1
    return counts


def _principal_moments(shape) -> tuple[float, float, float]:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    i1, i2, i3 = props.PrincipalProperties().Moments()
    return tuple(sorted((float(i1), float(i2), float(i3)), reverse=True))


def describe_shape(shape, unit_to_metres: float = 1.0) -> ShapeDescriptor:
    """The descriptor vector for one solid, from the same readers the
    analyzer and the recogniser use."""
    from OCP.TopAbs import TopAbs_WIRE

    from nodes.feature_recognizer import recognise
    from nodes.step_analyzer import _count_unique, _geometry_of, _topology_of

    geometry = _geometry_of(shape, unit_to_metres)
    topology = _topology_of(shape)
    inner_loops = _count_unique(shape, TopAbs_WIRE) - topology.faces
    features = recognise(shape, unit_to_metres)
    surfaces = _surface_counts(shape)

    sides = sorted(geometry.bounding_box_m, reverse=True)
    longest = sides[0]
    volume, area = geometry.volume_m3, geometry.surface_area_m2
    moments = _principal_moments(shape)
    # the bounding box centre is not in the summary; recompute it from the
    # shape so the offset is measured against the same box
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    box.SetGap(0.0)
    x0, y0, z0, x1, y1, z1 = box.Get()
    box_centre = ((x0 + x1) / 2 * unit_to_metres, (y0 + y1) / 2 * unit_to_metres,
                  (z0 + z1) / 2 * unit_to_metres)
    offset = math.dist(box_centre, geometry.centre_of_mass_m) / longest

    n_faces = max(topology.faces, 1)
    cylindrical_fillets = sum(1 for f in features.fillets
                              if f.surface_kind == "cylinder")
    other_cylinders = max(0, surfaces["cylinder"] - features.hole_count
                          - cylindrical_fillets)

    values = {
        # logged because a thin tube's compactness runs to hundreds while a
        # box sits near six, and an unlogged value let one thin hollow part
        # sit 7.8 standardised units from every other, alone
        "log_compactness": math.log(area ** 3 / (36.0 * math.pi * volume ** 2)),
        "aspect_mid": sides[1] / longest,
        "aspect_min": sides[2] / longest,
        "fill": volume / (sides[0] * sides[1] * sides[2]),
        "inertia_mid_ratio": moments[1] / moments[0],
        "inertia_min_ratio": moments[2] / moments[0],
        "com_offset": offset,
        "faces": float(topology.faces),
        "edges": float(topology.edges),
        "vertices": float(topology.vertices),
        "euler": float(topology.vertices - topology.edges + topology.faces
                       - inner_loops),
        "hole_count": float(features.hole_count),
        "fillet_count": float(features.fillet_count),
        "hole_diameter_ratio": (float(np.mean(features.hole_diameters_m()))
                                / longest if features.holes else 0.0),
        "fillet_radius_ratio": (float(np.mean(features.fillet_radii_m()))
                                / longest if features.fillets else 0.0),
        "other_cylinders": float(other_cylinders),
    }
    for kind in SURFACE_KINDS:
        values[f"frac_{kind}"] = surfaces[kind] / n_faces
    return ShapeDescriptor(values=values)


def describe_step(path: str | Path) -> list[ShapeDescriptor]:
    """One descriptor per solid in a STEP file."""
    from nodes.step_analyzer import read_step

    contents = read_step(path)
    return [describe_shape(shape, contents.unit_to_metres)
            for shape in contents.shapes]
