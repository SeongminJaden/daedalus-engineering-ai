"""Rule based feature recognition on a B-rep, and the rules' limits.

Rules before learning: every rule here is written out, has a stated validity
domain, and is checked against parts whose answer was known beforehand. A
learned recogniser comes later, and this stays as the control it is compared
against.

THE RULES, AND WHY THE OBVIOUS VERSIONS ARE WRONG
=================================================
Each of these replaced a simpler rule that measurement refuted.

A hole is a CONCAVE cylindrical face.
    Not "a cylindrical face of small radius". A fillet is a cylinder too, and
    a part whose fillet radius equals its hole radius merges the two under any
    radius based rule. What separates them is which side the material is on,
    so concavity is the test and radius is only reported.

A fillet is a curved face TANGENT TO AT LEAST TWO NEIGHBOURS.
    Not "a convex cylinder", and not "a toroidal face". Filleting every edge
    of a box produces twelve cylinders and eight spheres and no torus at all;
    a torus appears only where a fillet follows a curved edge, and there the
    fillet radius is the MINOR radius. Worse, a plain cylindrical body wall is
    also a convex cylinder, so a convexity rule reports the part itself as a
    fillet.

    "Tangent to EVERY neighbour" was the next attempt and was also wrong. A
    corner fillet on a plate is tangent to the two walls it blends and meets
    the top and bottom faces at a right angle, so requiring every neighbour
    found none of them. Counting is what separates the cases, measured:

        plate corner fillet   convex cylinder   tangent to 2 of 4
        cylinder body wall    convex cylinder   tangent to 1 of 2
        rim fillet            torus             tangent to 2 of 2

    A blend joins at least two faces, so at least two of its neighbours must
    be tangent. A body wall runs into exactly one blend and stops squarely
    against its end cap. Note that the plate's flat side walls are also
    tangent to two neighbours, which is why the rule is restricted to curved
    faces: a plane is never a fillet.

VALIDITY DOMAIN
===============
Stated before implementing.

Recognises
    Holes as concave cylinders, with diameter, axis and a point on the axis.
    Fillets as tangent-blended cylinders, spheres and toruses, with radius.

Does not recognise
    Chamfers, pockets, ribs, bosses, patterns or symmetry. Those need face
    adjacency reasoning beyond one face and its immediate neighbours, and
    guessing at them from a single surface would produce confident nonsense.

Does not distinguish
    Through holes from blind ones, and reports no depth. Deciding that needs
    the hole's extent compared against the solid's boundary, which is not done
    here and is therefore not claimed.

    A hole from any other concave cylinder. A cylindrical pocket, a bore, a
    clearance hole and a bearing seat are the same geometry, and which one it
    is is not in the geometry. This module reports what the shape IS, never
    what it is FOR.

Assumes
    Faces analytic enough that OpenCASCADE reports a surface type. A spline
    face that happens to be cylindrical is not recognised as one, and is
    reported as neither a hole nor a fillet rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

FEATURE_NODE_NAME = "feature.recognizer"
FEATURE_CAPABILITY = "analysis.cad.features"

#: Two faces count as tangent when their normals agree to this. Chosen well
#: away from both extremes: a real blend measures 1.000000 and a square corner
#: 0.000000, so anything near the middle is neither and should not be forced.
TANGENT_TOLERANCE = 1e-3


def _occ():
    try:
        import OCP  # noqa: F401
    except ImportError:
        return None
    return True


def is_available() -> bool:
    return _occ() is not None


def feature_recognizer_descriptor(available: bool | None = None
                                  ) -> NodeDescriptor:
    present = is_available() if available is None else available
    return NodeDescriptor(
        name=FEATURE_NODE_NAME, transport=Transport.IN_PROCESS, address="OCP",
        available=present,
        unavailable_reason="" if present else
        "unavailable: OpenCASCADE bindings (OCP) are not installed")


def feature_recognizer_capability_method():
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=FEATURE_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Rule based hole and fillet recognition on a B-rep.",
        inputs=("solid",),
        outputs=("holes", "fillets"),
        fidelity=Fidelity.ANALYTICAL,
        cost=Cost.CHEAP,
        conditions=(
            Condition("the input is a CAD solid rather than parameters",
                      lambda c: c.require("has_cad_input")),
        ),
        implementation="nodes.feature_recognizer.recognise",
        evidence="SIMULATED",
        notes="Rules, not learning, and each one replaced a simpler version "
              "that measurement refuted. A hole is a CONCAVE cylinder, not a "
              "small one, because a fillet is a cylinder too. A fillet is a "
              "face tangent to every neighbour, not a convex cylinder, "
              "because a cylindrical body wall is convex and would report the "
              "part as its own fillet. It reports what a shape IS and never "
              "what it is FOR: a bore, a clearance hole and a bearing seat "
              "are one geometry. Chamfers, pockets, ribs and patterns are not "
              "recognised, and through versus blind is not decided.")


@dataclass(frozen=True)
class Hole:
    """A concave cylindrical face. What it is for is not recorded.

    The axis is a LINE, so its direction has no inherent sign: a hole does not
    point anywhere. OpenCASCADE returns whichever sense the surface was built
    with, and Fusion reported (0,0,1) for a hole this reads as (0,0,-1). Both
    describe the same line. The sign is therefore canonicalised here, first
    non-zero component positive, so that two readings of one hole compare
    equal instead of appearing opposed.
    """

    diameter_m: float
    axis: tuple[float, float, float]
    point_on_axis_m: tuple[float, float, float]


@dataclass(frozen=True)
class Fillet:
    """A face that blends tangentially into all of its neighbours."""

    radius_m: float
    surface_kind: str          # cylinder, sphere or torus


@dataclass(frozen=True)
class FeatureReport:
    holes: tuple[Hole, ...]
    fillets: tuple[Fillet, ...]
    unclassified_faces: int

    @property
    def hole_count(self) -> int:
        return len(self.holes)

    @property
    def fillet_count(self) -> int:
        return len(self.fillets)

    def hole_diameters_m(self) -> tuple[float, ...]:
        return tuple(sorted(h.diameter_m for h in self.holes))

    def fillet_radii_m(self) -> tuple[float, ...]:
        return tuple(sorted(f.radius_m for f in self.fillets))


def _surface_kind(adaptor) -> str:
    from OCP.GeomAbs import (GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane,
                             GeomAbs_Sphere, GeomAbs_Torus)

    return {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder",
            GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus",
            GeomAbs_Cone: "cone"}.get(adaptor.GetType(), "other")


def _outward_normal_at(face, point):
    """The face's normal at a 3D point, flipped into the outward direction."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.ShapeAnalysis import ShapeAnalysis_Surface
    from OCP.TopAbs import TopAbs_REVERSED

    uv = ShapeAnalysis_Surface(BRep_Tool.Surface_s(face)).ValueOfUV(point, 1e-6)
    props = BRepLProp_SLProps(BRepAdaptor_Surface(face), uv.X(), uv.Y(), 1,
                              1e-7)
    if not props.IsNormalDefined():
        return None
    normal = props.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        normal.Reverse()
    return np.array([normal.X(), normal.Y(), normal.Z()])


def _is_concave_cylinder(face, adaptor) -> bool:
    """Whether the material lies OUTSIDE the cylinder, making it a hole.

    Compares the outward normal against the direction pointing away from the
    axis. Pointing back toward the axis means the solid surrounds the surface.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    surface = BRepAdaptor_Surface(face)
    u = 0.5 * (surface.FirstUParameter() + surface.LastUParameter())
    v = 0.5 * (surface.FirstVParameter() + surface.LastVParameter())
    point = surface.Value(u, v)
    normal = _outward_normal_at(face, point)
    if normal is None:
        return False

    cylinder = adaptor.Cylinder()
    origin = cylinder.Axis().Location()
    direction = cylinder.Axis().Direction()
    axis = np.array([direction.X(), direction.Y(), direction.Z()])
    radial = np.array([point.X() - origin.X(), point.Y() - origin.Y(),
                       point.Z() - origin.Z()])
    radial = radial - np.dot(radial, axis) * axis
    if np.linalg.norm(radial) < 1e-12:
        return False
    return float(np.dot(normal, radial)) < 0.0


#: A drilled hole's wall wraps a full turn about its axis. Anything less is
#: a piece of a cylinder, not a bore. Generous, because the span is compared
#: against an exact 2 pi that the kernel reproduces to machine precision.
FULL_TURN_TOLERANCE_RAD = 1e-6


def _wraps_a_full_turn(adaptor) -> bool:
    """Whether a cylindrical face closes on itself about its axis.

    This is what separates a hole from a concave fillet, and concavity alone
    cannot do it: the fillet in a reentrant corner is concave too. It is a
    ninety degree sector of a cylinder, while a bore is the whole three
    hundred and sixty.
    """
    span = adaptor.LastUParameter() - adaptor.FirstUParameter()
    return abs(span - 2.0 * math.pi) <= FULL_TURN_TOLERANCE_RAD


#: A blend joins at least this many faces. One tangent neighbour is a body
#: wall running into a fillet, not a fillet itself.
MIN_TANGENT_NEIGHBOURS = 2


def _count_tangent_neighbours(face, edge_map) -> int:
    """How many neighbours this face blends smoothly into.

    Seam edges are skipped. A periodic surface closes on itself, so it appears
    as its own neighbour and is trivially tangent there; counting that would
    say nothing.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopoDS import TopoDS

    checked = 0
    for index in range(1, edge_map.Extent() + 1):
        faces = [TopoDS.Face_s(f) for f in edge_map.FindFromIndex(index)]
        if len(faces) != 2:
            continue
        if not any(f.IsSame(face) for f in faces):
            continue
        other = faces[1] if faces[0].IsSame(face) else faces[0]
        if other.IsSame(face):
            continue                       # a seam, not a neighbour

        curve = BRepAdaptor_Curve(TopoDS.Edge_s(edge_map.FindKey(index)))
        point = curve.Value(0.5 * (curve.FirstParameter()
                                   + curve.LastParameter()))
        here = _outward_normal_at(face, point)
        there = _outward_normal_at(other, point)
        if here is None or there is None:
            continue
        if abs(abs(float(np.dot(here, there))) - 1.0) <= TANGENT_TOLERANCE:
            checked += 1
    return checked


def recognise(shape, unit_to_metres: float = 1.0) -> FeatureReport:
    """Find the holes and fillets in one solid.

    `unit_to_metres` converts from the shape's own units, which for a shape
    read out of a STEP file is what the analyzer read from its declaration.
    """
    if not is_available():
        raise CapabilityUnavailable(
            FEATURE_CAPABILITY, FEATURE_NODE_NAME,
            "OpenCASCADE bindings (OCP) are not installed")
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_map)

    holes, fillets, unclassified = [], [], 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    seen = []
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        explorer.Next()
        if any(face.IsSame(f) for f in seen):
            continue
        seen.append(face)

        adaptor = BRepAdaptor_Surface(face)
        kind = _surface_kind(adaptor)

        if kind == "cylinder" and _is_concave_cylinder(face, adaptor) \
                and _wraps_a_full_turn(adaptor):
            cylinder = adaptor.Cylinder()
            axis = cylinder.Axis().Direction()
            origin = cylinder.Axis().Location()
            direction = [axis.X(), axis.Y(), axis.Z()]
            for component in direction:
                if abs(component) > 1e-12:
                    if component < 0.0:
                        direction = [-v for v in direction]
                    break
            holes.append(Hole(
                diameter_m=2.0 * cylinder.Radius() * unit_to_metres,
                axis=(direction[0], direction[1], direction[2]),
                point_on_axis_m=(origin.X() * unit_to_metres,
                                 origin.Y() * unit_to_metres,
                                 origin.Z() * unit_to_metres)))
            continue

        if kind in ("cylinder", "sphere", "torus") and \
                _count_tangent_neighbours(face, edge_map) \
                >= MIN_TANGENT_NEIGHBOURS:
            if kind == "cylinder":
                radius = adaptor.Cylinder().Radius()
            elif kind == "sphere":
                radius = adaptor.Sphere().Radius()
            else:
                # For a torus the blend radius is the MINOR one; the major
                # radius is the path the fillet runs along.
                radius = adaptor.Torus().MinorRadius()
            fillets.append(Fillet(radius_m=radius * unit_to_metres,
                                  surface_kind=kind))
            continue

        if kind != "plane":
            unclassified += 1

    return FeatureReport(holes=tuple(holes), fillets=tuple(fillets),
                         unclassified_faces=unclassified)
