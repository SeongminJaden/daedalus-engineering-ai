"""Export an Assembly as URDF, so external libraries can read this model.

Nothing in this project consumed URDF before. It exists here because the
multibody cross-check needs an independent library to read the SAME model,
and hand-transcribing a model into a second format is how a comparison ends
up validating a typo.

WHAT MAPS AND WHAT DOES NOT
===========================
Stated before implementing, per the standing discipline.

Maps exactly
    Link and joint names, the parent and child relation, revolute and
    prismatic joint types, the joint axis, the 4x4 joint origin (as URDF's
    xyz plus fixed-axis rpy), the joint position limits, link mass, the
    centre of mass position, and the inertia tensor about the centre of mass
    in link-local axes. These are the quantities the dynamics is built from,
    so the comparison rests on them.

Does not exist in this project, and is therefore NOT exported as fact
    Effort and velocity limits. URDF requires both attributes on a revolute
    or prismatic joint, and this project models neither: a motor's torque
    comes from drivetrain selection, which is not part of the Assembly. The
    exporter writes a large sentinel and says so in an XML comment inside the
    generated file, because a plausible-looking number here would be read as
    a design limit by whoever opened the file next.

Does not map at all
    Visual and collision geometry. The section is parametric and the export
    would be a second, independent implementation of the CAD path; anything
    written here could disagree with the STEP export without either being
    obviously wrong. Catalogue parts have no geometry in this project either.

Structural requirement
    URDF describes a TREE. The Assembly model already refuses to construct a
    closed loop, so that is where the guarantee actually lives and no loop can
    reach this module. The check below is kept anyway, because an export that
    silently dropped a joint would produce a model that runs and is wrong, and
    the cost of re-checking an invariant is one pass over the joint list.

Frame convention
    This project is y-up with gravity along -y (see frames.py). URDF itself
    carries no gravity direction, so the consumer must set it. Reading this
    file into a library that assumes z-up gravity gives a wrong but entirely
    plausible answer, which is exactly the failure this module's docstring
    exists to prevent.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from physics.dynamics.inertia import link_inertia

from .model import Assembly, JointType

#: URDF demands effort and velocity on a bounded joint. This project models
#: neither, so the value is deliberately absurd rather than plausible.
UNMODELLED_LIMIT = 1.0e9

_URDF_TYPE = {JointType.REVOLUTE: "revolute",
              JointType.PRISMATIC: "prismatic",
              JointType.FIXED: "fixed"}


def rotation_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """Fixed-axis roll, pitch, yaw, the convention URDF's rpy attribute uses.

    R = Rz(yaw) Ry(pitch) Rx(roll). At pitch = +/- 90 degrees roll and yaw are
    not separable; the gimbal case puts all of the rotation into yaw, which is
    a choice rather than a recovery of the original angles.
    """
    r = np.asarray(rotation, dtype=np.float64)
    sin_pitch = -r[2, 0]
    if abs(sin_pitch) > 1.0 - 1e-12:
        pitch = math.copysign(math.pi / 2.0, sin_pitch)
        return 0.0, pitch, math.atan2(-r[0, 1], r[1, 1])
    pitch = math.asin(sin_pitch)
    return (math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0]))


def _triplet(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def _check_is_a_tree(joints) -> None:
    """Re-check the tree invariant the Assembly model already enforces."""
    seen: set[str] = set()
    for joint in joints:
        if joint.child in seen:
            raise ValueError(
                f"link {joint.child!r} has more than one parent joint; URDF "
                f"describes a tree and cannot represent this assembly")
        seen.add(joint.child)


def _inertial(parent: ET.Element, link, density_kg_m3: float) -> None:
    inertia = link_inertia(link, density_kg_m3)
    element = ET.SubElement(parent, "inertial")
    ET.SubElement(element, "origin", xyz=_triplet(link.com_local()),
                  rpy="0 0 0")
    ET.SubElement(element, "mass", value=repr(link.mass_kg(density_kg_m3)))
    ET.SubElement(element, "inertia",
                  ixx=repr(float(inertia[0, 0])), ixy=repr(float(inertia[0, 1])),
                  ixz=repr(float(inertia[0, 2])), iyy=repr(float(inertia[1, 1])),
                  iyz=repr(float(inertia[1, 2])), izz=repr(float(inertia[2, 2])))


def assembly_to_urdf(assembly: Assembly, density_kg_m3: float,
                     base_link_name: str = "base",
                     envelopes: bool = False) -> str:
    """Serialise the assembly as a URDF document.

    `density_kg_m3` is required rather than defaulted: the inertia written
    into the file is only meaningful for the material the links are made of,
    and a default would let the wrong one travel silently.

    `envelopes` adds a visual and a collision BOX per link: the section's
    outer width and height by the link length, along +x from the joint. It
    is the bounding envelope of the hollow section and not the part; the
    docstring above explains why the part's own geometry is not written
    here. A simulator needs some collision shape to detect interference and
    to draw, and an envelope is the honest one: every interference it
    reports is at least an interference of the envelopes, and the generated
    file says so in a comment on every such element.
    """
    _check_is_a_tree(assembly.joints)
    robot = ET.Element("robot", name=assembly.name)
    robot.append(ET.Comment(
        " Generated from an Assembly. Effort and velocity limits are set to "
        f"{UNMODELLED_LIMIT:g} because this project does not model them; they "
        "are NOT design limits. This model is y-up with gravity along -y, "
        "which URDF cannot record, so the consumer must set it. "))

    base = ET.SubElement(robot, "link", name=base_link_name)
    if envelopes:
        # sdformat's URDF reader turns a massless root link into a frame and
        # then drops the joints hanging from it ("child joints ignored"),
        # which was measured: the converted model kept two frames and no
        # links. A simulator needs the base to be a body, so it gets a
        # nominal inertia that is NOT a property of any part; the base is
        # fixed to the world in every world this project writes, so the
        # number never enters a result.
        base.append(ET.Comment(" nominal inertia so the converter keeps the "
                               "tree; the base is fixed to the world "))
        inertial = ET.SubElement(base, "inertial")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia", ixx="1e-3", ixy="0", ixz="0",
                      iyy="1e-3", iyz="0", izz="1e-3")
    for link in assembly.links:
        element = ET.SubElement(robot, "link", name=link.name)
        _inertial(element, link, density_kg_m3)
        if envelopes:
            section = link.genome.section
            size = _triplet([link.length_m, section.outer_height_m,
                             section.outer_width_m])
            for tag in ("visual", "collision"):
                shape = ET.SubElement(element, tag, name=f"{link.name}_envelope")
                shape.append(ET.Comment(
                    " ENVELOPE: the outer box of the hollow section, not the "
                    "part. Interference against it is interference of "
                    "envelopes. "))
                ET.SubElement(shape, "origin",
                              xyz=_triplet([link.length_m / 2.0, 0.0, 0.0]),
                              rpy="0 0 0")
                geometry = ET.SubElement(shape, "geometry")
                ET.SubElement(geometry, "box", size=size)

    for joint in assembly.joints:
        origin = np.asarray(joint.origin_transform(), dtype=np.float64)
        element = ET.SubElement(robot, "joint", name=joint.name,
                                type=_URDF_TYPE[joint.type])
        ET.SubElement(element, "parent",
                      link=joint.parent if joint.parent else base_link_name)
        ET.SubElement(element, "child", link=joint.child)
        ET.SubElement(element, "origin", xyz=_triplet(origin[:3, 3]),
                      rpy=_triplet(rotation_to_rpy(origin[:3, :3])))
        if joint.type is not JointType.FIXED:
            axis = np.asarray(joint.axis, dtype=np.float64)
            ET.SubElement(element, "axis",
                          xyz=_triplet(axis / np.linalg.norm(axis)))
            ET.SubElement(element, "limit",
                          lower=repr(float(joint.lower_limit)),
                          upper=repr(float(joint.upper_limit)),
                          effort=repr(UNMODELLED_LIMIT),
                          velocity=repr(UNMODELLED_LIMIT))

    tip = assembly.tip_link()
    ET.SubElement(robot, "link", name="tool")
    tool_joint = ET.SubElement(robot, "joint", name="tool_fixed", type="fixed")
    ET.SubElement(tool_joint, "parent", link=tip.name)
    ET.SubElement(tool_joint, "child", link="tool")
    ET.SubElement(tool_joint, "origin",
                  xyz=_triplet([tip.length_m + assembly.tool_offset_m, 0.0, 0.0]),
                  rpy="0 0 0")

    raw = ET.tostring(robot, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")
