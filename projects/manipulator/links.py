"""Generating each link as a shape, not as a box with a section.

The first version of this arm sized a hollow rectangle per link. That is a
section, not a structure: it has no place to bolt an actuator, no hole for a
shaft, and no reason for its material to be where it is. This module runs the
free form path instead, with three things the earlier topology runs did not
have.

THE INTERFACES ARE PART OF THE PROBLEM
======================================
Each link gets a slab of passive solid at both ends, thick enough for the
fastening the features stage demands: a 6.4 mm counterbore has to fit and a
tapped hole in aluminium wants 1.5 diameters, so 9 mm. The optimiser cannot
carve those away and has to route load into them, which is the difference
between a shape and a part.

THE ACTUATOR IS A HOLE, WHERE ITS SIZE IS PUBLISHED
===================================================
A drive occupies space that the structure may not. Where the manufacturer
prints an outline, that volume becomes passive void. Where it does not, no
void can be placed, and the link is reported as ungeneratable rather than
generated around a guess.

THE LOADS ARE THE ARM'S, AND THERE IS MORE THAN ONE
===================================================
A single load case design is between 1.90 and 9.30 times worse under a case it
never saw, measured in docs/topology_design.md. Each link is optimised for the
weighted set of cases its own joint actually sees: the transverse load in both
directions and the torsion its drive applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.materials import get_material
from optimization.topology import SimpProblem
from optimization.topology.manufacturing import (support_projection_with_gradient,
                                                 unsupported_fraction)
from optimization.topology.multiload import (LoadCase, couple_force_vector,
                                             cross_evaluation, optimize_multiload)
from optimization.topology.smooth import marching_surface, write_stl
from optimization.topology.verify import elements_touching
from physics.fem.mesh import solid_box_mesh

from .interfaces import face_separation_m
from .spec import SPEC, ManipulatorSpec

#: Iso level for the extraction. 0.5 is the middle of the three the smoothing
#: study measured (+24.6, -7.2 and -36.0 percent volume error against the
#: density field at 0.3, 0.5 and 0.7), and the one whose error is smallest.
ISO_LEVEL = 0.5

#: Exported geometry is scaled from metres to millimetres. A CAD package that
#: reads an STL has no unit to read and assumes millimetres, so a body written
#: in metres arrives a thousand times too small.
EXPORT_SCALE = 1000.0

#: The process these parts would be made by, and why. They are organic bodies
#: from a density field: a mill cannot reach inside them, and casting needs a
#: draft this shape does not have. Laser powder bed fusion can make them, so
#: the optimisation carries ITS constraint, the support filter, rather than
#: scoring the result against a process it was not designed for.
PROCESS = "slm"


@dataclass
class LinkDesign:
    name: str
    generated: bool
    reason: str = ""
    mass_kg: float | None = None
    volume_m3: float | None = None
    compliance_j: float | None = None
    grey_fraction: float | None = None
    unsupported_fraction: float | None = None
    volume_error_vs_field: float | None = None
    watertight: bool | None = None
    triangles: int | None = None
    stl_path: str | None = None
    step_path: str | None = None
    cross_evaluation: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: What the drawings do not print, so the design could not resolve it.
    #: Kept apart from `notes` because these are gaps, not observations.
    unresolved: list[str] = field(default_factory=list)

    def row(self) -> dict:
        return {"link": self.name, "generated": self.generated,
                "mass_kg": self.mass_kg, "compliance_j": self.compliance_j,
                "grey": self.grey_fraction,
                "unsupported": self.unsupported_fraction,
                "volume_error_vs_field": self.volume_error_vs_field,
                "watertight": self.watertight, "triangles": self.triangles,
                "reason": self.reason}


def actuator_for(joint_name: str, drives: dict[str, str]):
    from drivetrain.sourced import sourced_motor

    part = drives.get(joint_name)
    if not part or "+" in str(part):
        return None
    try:
        return sourced_motor(part)
    except Exception:
        return None


def scaled_surface(surface, scale: float):
    """The same surface with its vertices in another unit.

    Only the geometry is scaled. Every quantity the design reports stays in
    SI, so the scale cannot leak into a mass or a compliance.
    """
    from dataclasses import replace

    return replace(surface, vertices=np.asarray(surface.vertices) * scale)



#: CHOSEN. A radial gap between the drive and the material around it. Zero
#: clearance passes an interference check while the surfaces touch, and a
#: powder bed part's as built surface is rough enough that touching is
#: contact. One millimetre on the radius.
ACTUATOR_RADIAL_CLEARANCE_M = 0.001


def local_axis(spec, link_index: int, axis) -> np.ndarray:
    """A joint's axis expressed in the LINK's own frame.

    The joint axes in the specification are in the arm's frame, where y is up
    and x runs along the reach. A link's own file has x along ITSELF, and for
    the base column that is the vertical. So the base yaw axis, which is
    (0, 1, 0) in the arm, is (1, 0, 0) in the column: it turns about the
    column's length, not across it. Reading the arm's components as if they
    were the link's made the base column's drive a cross axis cylinder when
    it is a coaxial one, and put a 98 mm diameter pocket sideways through a
    part the motor runs along.
    """
    joints = spec.joints()
    following = (joints[link_index + 1]
                 if link_index + 1 < len(joints) else None)
    if following is None or following.origin_x_m >= following.origin_y_m:
        return np.asarray(axis, dtype=float)      # the link runs along x
    # The link runs along y: x_local = y_arm, y_local = -x_arm, z unchanged.
    arm = np.asarray(axis, dtype=float)
    return np.array([arm[1], -arm[0], arm[2]])


def actuator_void(mesh, joint, actuator, at_x: float, height_m: float,
                  width_m: float, face: str, spec,
                  axis=None, axis_offset: float = 0.0
                  ) -> tuple[np.ndarray, str]:
    """The volume a drive occupies, placed against the joint it belongs to.

    Two things decide the placement and only one of them used to be here.

    The AXIS. The drive turns about the joint axis, so its cylinder is about
    the line through the joint origin in that direction. The first version
    put the pocket at a fixed distance along the link instead, which for a
    joint whose axis crosses the arm placed it nowhere in particular: 39.95
    mm along a link whose joint sits at zero.

    The POSITION ALONG THAT AXIS. A motor does not straddle the joint plane.
    It hangs off one side of it, and how far is set by where its mounting
    face sits inboard of its own end, which the AK80-64 drawing prints and
    the AK80-9 drawing does not. Where it is printed the body is placed
    against that face. Where it is not, the drive is centred on the joint and
    the note says so, because centring is a guess that is symmetric rather
    than a guess that is hidden.
    """
    from .interfaces import face_for

    centroids = mesh.element_centroids()
    axis = np.asarray(joint.axis if axis is None else axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    origin = np.array([at_x, 0.5 * height_m, 0.5 * width_m])
    if abs(float(axis[0])) <= 0.5:
        # The axis crosses the link, so the mounting face is a plane across
        # it and the axis line lies IN that plane rather than through the
        # link's middle.
        origin = origin - axis * float(np.dot(origin, axis)) + axis * axis_offset
    offset = centroids - origin
    along = offset @ axis
    radial = np.linalg.norm(offset - np.outer(along, axis), axis=1)
    radius = 0.5 * actuator.outer_diameter_m + ACTUATOR_RADIAL_CLEARANCE_M
    length = actuator.axial_length_m

    mounting = face_for(actuator.id if hasattr(actuator, "id") else "", face)
    inset = mounting.face_inset_m if mounting is not None else None
    if inset is None:
        low, high = -0.5 * length, 0.5 * length
        why = (f"{actuator.part_number} centred on the joint, "
               f"{length * 1000:.1f} mm long by "
               f"{actuator.outer_diameter_m * 1000:.0f} mm across. NO "
               f"MOUNTING FACE INSET IS PUBLISHED for its {face} face, so "
               f"where it sits along its own axis is a guess")
    else:
        low, high = -(length - inset), inset
        why = (f"{actuator.part_number} placed against its {face} face, which "
               f"the drawing puts {inset * 1000:.1f} mm inboard of its end, "
               f"so it reaches {high * 1000:.1f} mm into this link and "
               f"{-low * 1000:.1f} mm out of it")
    return ((along >= low) & (along <= high) & (radial <= radius)), why


def link_domain(spec: ManipulatorSpec, link_index: int, drives: dict[str, str],
                divisions=(28, 12, 12), sections: dict | None = None):
    """The box a link may occupy, with its interfaces and actuator pockets.

    Returns (mesh, passive_solid, passive_void, extents) or a reason it cannot
    be built.
    """
    joints = spec.joints()
    links = spec.links()
    link = links[link_index]
    joint = joints[link_index]
    following = joints[link_index + 1] if link_index + 1 < len(joints) else None

    # The span is whichever offset the next joint carries. The base column
    # runs UP, not along the arm, so its span is the next joint's y offset;
    # taking the x offset alone reported it as having no room to design in.
    if following is None:
        span = link.length_m
    else:
        span = max(following.origin_x_m, following.origin_y_m)
    if span <= 0.0:
        return None, (f"{link.name}: its joints are coincident in every "
                      f"direction, so there is no space to design in")

    actuator = actuator_for(joint.name, drives)
    # The domain is the rectangle the sizing loop arrived at, not a square.
    # Taking one number for both directions gave the tool flange a 32 by 32
    # box where its section is 32 by 40, which throws away 8 mm of width the
    # design was allowed to use.
    section = (sections or {}).get(link.name)
    height = section.outer_height_m if section is not None else spec.minimum_section_m
    width = section.outer_width_m if section is not None else spec.minimum_section_m
    if actuator is not None and actuator.outer_diameter_m:
        height = max(height, actuator.outer_diameter_m)
        width = max(width, actuator.outer_diameter_m)

    # WHERE THE JOINT SITS ACROSS THE LINK, AND HOW THICK THE LINK MAY BE.
    # A link bolts to the output face of the joint that drives it, so that
    # face is a boundary of the link and not a plane through its middle: the
    # drive is entirely on the far side of it. For a joint whose axis crosses
    # the arm this fixes where the axis line lies in the link's own frame,
    # at the face rather than at the centre.
    #
    # When the joint at the far end turns about the SAME axis, the link is
    # also bounded by that drive's housing face, and the two faces are the
    # actuator's own mounting face separation apart. Then the thickness is
    # not a choice at all. This is the shoulder: the upper arm lies between
    # two AK80-64s facing opposite ways, 42.7 mm apart, and any wider a
    # section would be in the same place as a motor.
    boxes = world_boxes(spec, drives, sections)
    mine = next(b for b in boxes if b["link"] == link.name)
    if not mine["placed"]:
        return None, mine["reason"]
    width = mine["width"]
    reach_low, joint_span = mine["reach_low"], mine["joint_span"]
    span = mine["span"]
    thickness_note = mine["basis"]

    # KEEP THE CELL SIZE, NOT THE CELL COUNT. A cranked link's domain is
    # nearly twice as wide as a straight one, and holding the division count
    # fixed would give it 15.8 mm cells where the others have 8.2, which is
    # coarser than the features being designed. The count follows the width
    # instead, so only the two cranked links pay for their size.
    nx, ny, nz = divisions
    reference = 0.098 / float(nz)
    nz = max(nz, int(round(width / reference)))
    mesh = solid_box_mesh(span, height, width, nx, ny, nz)

    centroids = mesh.element_centroids()
    flange = spec.flange_thickness_m
    # THE FLANGE SITS AT THE JOINT PLANE, not at the edge of the box. Once
    # the box reaches back past a crossing joint to hold that joint's disc,
    # the box edge and the joint plane are 49 mm apart, and a slab at the
    # edge is a slab in mid air. A crossing joint's own flange is the ring
    # held solid further down, in the plane its bolts are actually in.
    proximal = reach_low
    distal = reach_low + joint_span
    passive_solid = np.zeros(mesh.n_elements, dtype=bool)
    if abs(float(local_axis(spec, link_index, joint.axis)[0])) > 0.5:
        passive_solid |= ((centroids[:, 0] >= proximal)
                          & (centroids[:, 0] <= proximal + flange))
    if following is None or abs(float(
            local_axis(spec, link_index, following.axis)[0])) > 0.5:
        passive_solid |= ((centroids[:, 0] >= distal - flange)
                          & (centroids[:, 0] <= distal))

    passive_void = np.zeros(mesh.n_elements, dtype=bool)
    notes: list[str] = []
    notes.append(thickness_note)
    for role, at_x, other in (("its own drive", reach_low, joint),
                              ("the drive it carries",
                               reach_low + joint_span, following)):
        if other is None:
            notes.append("this link carries no further joint")
            continue
        carried = actuator_for(other.name, drives)
        if carried is None or not (carried.outer_diameter_m
                                   and carried.axial_length_m):
            notes.append(f"{other.name}: no outline is printed for "
                         f"{drives.get(other.name, 'its drive')}, so no pocket "
                         f"was cut for {role}")
            continue
        axis_local = local_axis(spec, link_index, other.axis)
        # WHICH SIDE OF THE MOUNTING FACE THE LINK IS ON. A link sits on the
        # far side of the face it bolts to, and the drive sits on the near
        # side. For a joint whose axis crosses the link that fixes where the
        # axis line lies in this frame: at the FACE, not through the middle.
        # Putting it through the middle is what let a motor and a link claim
        # 235 cubic centimetres of the same space at the shoulder.
        if abs(float(axis_local[0])) > 0.5:
            offset = 0.0                       # coaxial: the face is the end
        else:
            # EVERY OUTPUT FACE IS THE ARM'S z = 0 PLANE, and this frame's
            # zero is wherever the link's box starts. So the face lands at
            # minus the box's low corner in local coordinates, and putting
            # 0.0 here instead cut every crossing pocket at the bottom of the
            # link rather than at the drive. It is the difference between a
            # pocket in the right place and a pocket 140.7 mm away from one,
            # and an assembly measured the material left behind: 154,744
            # cubic millimetres of link inside a motor, on every joint whose
            # axis crosses the arm and on no other.
            offset = -float(mine["low"][2])
        cut, why = actuator_void(
            mesh, other, carried, at_x, height, width,
            "output" if role == "its own drive" else "housing", spec,
            axis=axis_local, axis_offset=offset)
        passive_void |= cut
        notes.append(f"{role} at x = {at_x * 1000:.0f} mm: {why}")
    void_note = "; ".join(notes)

    # BEYOND ITS OWN JOINT PLANE A LINK IS ONLY A DISC. The domain reaches
    # back past a crossing joint so the bolt circle in that joint's plane
    # fits, and that is ALL it is allowed to do there. Everything past the
    # joint plane that is not within the flange of the mounting plane and
    # within the disc's radius belongs to the neighbour. Without this the two
    # links at the elbow both claim a hundred millimetres of each other's
    # length, and holding one another's whole box empty then deletes both
    # their flanges.
    beyond = 0
    for other, plane_x, outward in ((joint, reach_low, -1.0),
                                    (following, reach_low + joint_span, +1.0)
                                    if following is not None else (None, 0.0, 0.0)):
        if other is None:
            continue
        axis = local_axis(spec, link_index, other.axis)
        if abs(float(axis[0])) > 0.5:
            continue                      # coaxial: nothing reaches past
        carried = actuator_for(other.name, drives)
        if carried is None or not carried.outer_diameter_m:
            continue
        origin = _drive_face(spec, link_index, other, plane_x, height, width,
                             mine)
        if outward > 0:
            separation = face_separation_m(str(drives.get(other.name, "")))
            origin = origin - axis * (separation or 0.0)
        offset = centroids - origin
        along = offset @ axis
        radial = np.linalg.norm(offset - np.outer(along, axis), axis=1)
        past = ((centroids[:, 0] - plane_x) * outward) > 0.0
        disc = (radial <= 0.5 * carried.outer_diameter_m) & (
            np.abs(along) <= spec.flange_thickness_m)
        cut = past & ~disc
        beyond += int(cut.sum())
        passive_void |= cut
    if beyond:
        void_note += (f"; {beyond} elements past its own joint planes that "
                      f"are not part of a mounting disc")

    # NOBODY ELSE'S SPACE. Taking the union of the centred box and the
    # offset box is what lets a link be cranked, and it also hands the base
    # column back the 117,649 cubic millimetres of the shoulder that the
    # offset had just taken off it. So every other link's domain is held
    # empty here. The rule is general rather than a patch for one joint: a
    # link may design in its own box and in nobody else's.
    world_low, world_high = mine["low"], mine["high"]
    along, other = mine["along"], (0 if mine["along"] == 1 else 1)
    world = np.zeros_like(centroids)
    world[:, along] = world_low[along] + centroids[:, 0]
    world[:, other] = world_low[other] + centroids[:, 1]
    world[:, 2] = world_low[2] + centroids[:, 2]
    taken = 0
    for box in boxes:
        if not box["placed"] or box["link"] == link.name:
            continue
        inside = np.all((world >= box["low"]) & (world <= box["high"]), axis=1)
        taken += int(inside.sum())
        passive_void |= inside
    if taken:
        void_note += (f"; {taken} elements belong to a neighbouring link's "
                      f"domain and are held empty")

    # KEEP THE WITHDRAWAL CORRIDOR CLEAR. A drive has to be able to come
    # out along its own axis, and a link that has material on BOTH sides of
    # it within its own radius traps it however well the two sides avoid the
    # motor itself. The cranked links do exactly that: the base column's
    # domain reaches from 140.7 mm below the shoulder plane to 49 above it,
    # so it closes around a motor that lives between -53.9 and +8, and an
    # independent sweep measured it as trapped in both directions.
    #
    # A link bolts to one face and the drive is on the far side of it, so
    # within that drive's swept cylinder the link may exist on ONE side of
    # the face and not the other. That is not an extra rule: it is the same
    # rule that places the link, applied to the volume the drive sweeps
    # rather than only to the volume it occupies.
    #: CHOSEN, and stronger than assembly needs. A drive does not have to
    #: pass through a link to be fitted: the column is stood up, the motor
    #: bolted to it, and the upper arm bolted to the motor, and nothing goes
    #: through anything. Requiring that a single link never traps its own
    #: drive is a SERVICE condition, not an assembly one: it is what lets a
    #: motor be changed without taking the arm apart, and industrial arms are
    #: built that way. If mass becomes the binding problem this is the first
    #: condition to relax.
    corridors = 0
    for other, at_x, driven in ((joint, reach_low, True),
                                (following, reach_low + joint_span, False)
                                if following is not None
                                else (None, 0.0, False)):
        if other is None:
            continue
        carried = actuator_for(other.name, drives)
        if carried is None or not carried.outer_diameter_m:
            continue
        axis = local_axis(spec, link_index, other.axis)
        if abs(float(axis[0])) > 0.5:
            continue                  # coaxial: it leaves along the link
        origin = _drive_face(spec, link_index, other, at_x, height, width, mine)
        offset = centroids - origin
        along = offset @ axis
        radial = np.linalg.norm(offset - np.outer(along, axis), axis=1)
        radius = (0.5 * carried.outer_diameter_m
                  + ACTUATOR_RADIAL_CLEARANCE_M)
        if driven:
            far = along <= 0.0        # it bolts above the output face
        else:
            separation = face_separation_m(str(drives.get(other.name, "")))
            far = along >= -(separation or 0.0)
        corridor = (radial <= radius) & far
        corridors += int(corridor.sum())
        passive_void |= corridor
    if corridors:
        void_note += (f"; {corridors} elements held empty as the corridor the "
                      f"drives come out through")

    # HOLD THE BOLT RING SOLID. Cutting the boss and the tail out leaves a
    # thin annulus at each mounting face, and that annulus is what the bolts
    # bear on: 8 mm wide on the AK80-64's output face, with eight 3.4 mm
    # holes through it leaving 2.3 mm each side. Nothing else in this problem
    # protects it. A volume constraint would happily spend that material
    # somewhere stiffer and leave the holes standing in air, and the bolt
    # access check would still pass, because it looks for room to put a key
    # in rather than for something to tighten against.
    from .interfaces import drive_profile, face_for

    rings = 0
    ring_mask = np.zeros(mesh.n_elements, dtype=bool)
    for other, at_x, side in ((joint, reach_low, +1.0),
                              (following, reach_low + joint_span, -1.0)
                              if following is not None
                              else (None, 0.0, 0.0)):
        if other is None:
            continue
        carried = actuator_for(other.name, drives)
        profile = drive_profile(str(drives.get(other.name, "")))
        face = face_for(str(drives.get(other.name, "")), "output")
        if carried is None or profile is None or face is None:
            continue
        axis = local_axis(spec, link_index, other.axis)
        origin = _drive_face(spec, link_index, other, at_x, height, width, mine)
        offset = centroids - origin
        along = offset @ axis
        radial = np.linalg.norm(offset - np.outer(along, axis), axis=1)
        if side > 0:                       # driven: the ring is above zero
            low, high = 0.0, flange
            inner = profile[-1][2]
        else:                              # carrying: it is below the tail
            separation = face_separation_m(str(drives.get(other.name, "")))
            low, high = -(separation or 0.0) - flange, -(separation or 0.0)
            inner = profile[0][2]
        outer = 0.5 * carried.outer_diameter_m
        ring = ((along >= low) & (along <= high)
                & (radial >= inner) & (radial <= outer))
        rings += int(ring.sum())
        ring_mask |= ring
        passive_solid = passive_solid | ring
    # THE RING WINS OVER A NEIGHBOUR'S BOX. Two links both reach around the
    # shoulder now, because each needs the disc its own bolt circle sits in,
    # and holding every neighbour's box empty would delete one of the two
    # discs. They do not actually collide: the column bolts to the housing
    # face and the upper arm to the output face, and those planes are 42.7 mm
    # apart. The boxes overlap, the rings do not.
    passive_void = passive_void & ~ring_mask
    if rings:
        void_note += (f"; {rings} elements held solid as the bolt rings at "
                      f"the mounting faces")

    # THE DRIVE WINS OVER THE FLANGE where they meet. The AK80-64's output
    # boss stands 8 mm proud of its mounting face, which is 8 mm into a 9 mm
    # flange, so the flange has to be relieved for it. Holding both would ask
    # the optimiser for material in a place a solid steel part already
    # occupies. The relief this produces is the spigot register the wrist
    # joints need anyway.
    overlap = int((passive_solid & passive_void).sum())
    passive_solid = passive_solid & ~passive_void
    if overlap:
        void_note += (f"; {overlap} elements of flange are relieved where the "
                      f"drive stands proud of its mounting face")

    free = ~(passive_solid | passive_void)
    if free.sum() < 0.15 * mesh.n_elements:
        return None, (f"{link.name}: the interfaces and the actuator pocket "
                      f"leave {free.sum()} of {mesh.n_elements} elements free, "
                      f"which is no room to design in. The drive fills the link")
    return (mesh, passive_solid, passive_void, span, height, width,
            void_note), ""


def link_load_cases(mesh, torque_nm: float, transverse_n: float) -> list[LoadCase]:
    """The cases this link's joint actually sees, weighted equally.

    Two transverse directions and the torsion its own drive applies. A single
    case design is between 1.90 and 9.30 times worse under a case it never
    saw, which is why there are three.
    """
    tip = mesh.nodes_at_x(float(mesh.nx * mesh.dx))
    return [
        LoadCase("bending in the load plane", tip, total_load_n=-transverse_n,
                 load_direction=1),
        LoadCase("bending across it", tip, total_load_n=-transverse_n,
                 load_direction=2),
        LoadCase("torsion from the drive", tip,
                 force_vector=couple_force_vector(mesh, tip, torque_nm)),
    ]


def _drive_centre(spec, link_index, joint, actuator, at_x, height_m, width_m,
                  box):
    """The middle of a drive's body, in the link's own frame."""
    from .interfaces import face_for

    axis = local_axis(spec, link_index, joint.axis)
    centre = np.array([at_x, 0.5 * height_m, 0.5 * width_m], dtype=float)
    if abs(float(axis[0])) <= 0.5:
        centre = centre - axis * float(np.dot(centre, axis))
        centre = centre + axis * (-float(box["low"][2]))
    face = face_for(str(actuator.id), "output")
    inset = (face.face_inset_m if face is not None
             and face.face_inset_m is not None else 0.5 * actuator.axial_length_m)
    return centre + axis * (inset - 0.5 * actuator.axial_length_m)


def _near_axis(mesh, axis, at_x, height_m, width_m, box, radius_m):
    """Elements within `radius_m` of a drive's axis line."""
    centroids = mesh.element_centroids()
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    origin = np.array([at_x, 0.5 * height_m, 0.5 * width_m], dtype=float)
    if abs(float(axis[0])) <= 0.5:
        origin = origin - axis * float(np.dot(origin, axis))
        origin = origin + axis * (-float(box["low"][2]))
    offset = centroids - origin
    along = offset @ axis
    radial = np.linalg.norm(offset - np.outer(along, axis), axis=1)
    return radial <= radius_m


def generate_link(spec: ManipulatorSpec, link_index: int,
                  drives: dict[str, str], torques: dict[str, float],
                  out_dir: Path, iterations: int = 60,
                  volume_fraction: float = 0.3,
                  sections: dict | None = None) -> LinkDesign:
    """One link, from a design domain to a watertight body."""
    links = spec.links()
    joints = spec.joints()
    link = links[link_index]
    joint = joints[link_index]

    built, reason = link_domain(spec, link_index, drives, sections=sections)
    if built is None:
        return LinkDesign(name=link.name, generated=False, reason=reason)
    (mesh, passive_solid, passive_void, span, height, width,
     void_note) = built
    # The link's own box, which the boolean pass needs to place anything in
    # the arm's frame. `link_domain` has it and does not return it, and
    # reaching for a name that only exists inside that function is what cost
    # a full six link run: every worker died on it, an hour and a half after
    # the run started, because nothing evaluates that line until the very end
    # of a link.
    mine = next(box for box in world_boxes(spec, drives, sections)
                if box["link"] == link.name)
    joints = spec.joints()
    joint = joints[link_index]
    following = joints[link_index + 1] if link_index + 1 < len(joints) else None

    material = get_material(spec.materials["link"])
    torque = abs(torques.get(joint.name, 1.0)) or 1.0
    transverse = max(torque / max(span, 1e-6), 10.0)

    projection, vjp = support_projection_with_gradient(mesh, build_axis=1)
    problem = SimpProblem(
        mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio,
        fixed_nodes=mesh.nodes_at_x(0.0),
        load_nodes=mesh.nodes_at_x(float(mesh.nx * mesh.dx)),
        total_load_n=-transverse, load_direction=1,
        volume_fraction=volume_fraction, volume_fraction_of="free",
        filter_radius_elements=2.0,
        passive_solid=passive_solid, passive_void=passive_void,
        density_projection=projection, projection_vjp=vjp)

    cases = link_load_cases(mesh, torque, transverse)
    result = optimize_multiload(problem, cases, max_iterations=iterations)

    from optimization.topology.export import largest_connected_component

    kept = largest_connected_component(mesh, result.density, ISO_LEVEL)
    # DID THE EXTRACTION THROW AWAY AN INTERFACE? The flange slabs are held
    # solid because bolts go through them, but keeping only the largest
    # connected component will drop one of them without a word if the
    # optimiser hollowed out the middle. That is what shortened the tool
    # flange by 23 mm: its far slab survived the optimisation, became its own
    # island, and was silently discarded. A link that loses an interface has
    # not been designed, it has been truncated.
    lost = int((passive_solid & (kept < ISO_LEVEL)).sum())
    if lost:
        return LinkDesign(
            name=link.name, generated=False,
            reason=(f"{link.name}: the extraction dropped {lost} of "
                    f"{int(passive_solid.sum())} elements that were held "
                    f"solid for its interfaces. Keeping the largest connected "
                    f"component removed a flange the bolts go through, so "
                    f"this body is not the part that was specified"))
    surface = marching_surface(mesh, kept, ISO_LEVEL, smoothing_iterations=10)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Written in MILLIMETRES. Neither STL nor a faceted STEP carries a unit a
    # CAD package can read back reliably, so a file in metres imports as a
    # part a thousand times too small. Everything computed here stays in SI;
    # only the exported copy is scaled, and the scale is recorded.
    import trimesh

    exported = scaled_surface(surface, EXPORT_SCALE)
    body = trimesh.Trimesh(vertices=np.asarray(exported.vertices),
                           faces=np.asarray(exported.triangles), process=False)
    body, clipped = clip_to_domain(body, span, height, width, EXPORT_SCALE)
    # Add the exact interfaces BEFORE the holes, so the holes are drilled
    # through them, and before the drive envelopes, so a ring that reaches
    # into a motor is trimmed by the motor rather than left there.
    body, added = add_solids(body, interface_solids(
        spec, link_index, drives, span, height, width, mine), EXPORT_SCALE)
    before_mm3 = float(abs(body.volume))
    # CUT THE DRIVES OUT, do not merely ask the optimiser to avoid them.
    # Holding elements empty works on element CENTRES, and the iso surface
    # then runs between a void element and its solid neighbour, so material
    # stands up to half a cell into the pocket. Half a cell here is 4.1 mm
    # against a radial clearance of 1.0, and an assembly measured what that
    # leaves: 104 cubic centimetres of link inside a motor even with every
    # pocket in the right place. A motor is a solid steel part and its space
    # is not a preference, so it is subtracted rather than requested.
    envelopes = drive_envelopes(spec, link_index, drives, span, height, width,
                                mine)
    holes, unresolved = mounting_holes(spec, link_index, drives, span,
                                       height_m=height, width_m=width,
                                       box=mine)
    body, hole_report = cut_holes(body, holes + envelopes, height, width,
                                  scale=EXPORT_SCALE)
    after_mm3 = float(abs(body.volume))
    body.export(str(out_dir / f"{link.name}.stl"))
    stl = out_dir / f"{link.name}.stl"
    # THE POSITION CHECK THAT PAIRS WITH EVERY SIZE CHECK ABOVE. Three
    # defects in a row got past this module because everything it measured
    # was a magnitude: a body half an element low, a body inside out, and a
    # pocket 140.7 mm from the drive it was for. Volume is invariant under
    # translation, abs() is invariant under inversion, and an element count
    # is invariant under both. So the centroid of what was held empty for a
    # drive is checked against where that drive is, which is not.
    void_offsets = []
    for other, at_x in ((joint, mine["reach_low"]),
                        (following, mine["reach_low"] + mine["joint_span"])
                        if following is not None else (None, 0.0)):
        if other is None:
            continue
        carried = actuator_for(other.name, drives)
        if carried is None or not carried.axial_length_m:
            continue
        axis = local_axis(spec, link_index, other.axis)
        picked = passive_void & _near_axis(mesh, axis, at_x, height, width,
                                           mine, 0.5 * carried.outer_diameter_m
                                           + ACTUATOR_RADIAL_CLEARANCE_M)
        if not picked.any():
            void_offsets.append(f"{other.name}: NOTHING was held empty for it")
            continue
        # HOW FAR OUTSIDE THE DRIVE ANY OF IT LIES, not how far its centroid
        # is from the drive's centre. Comparing centroids compared the middle
        # of the VISIBLE pocket with the middle of the WHOLE drive, and most
        # of a drive is outside the link it is bolted to: the tool flange
        # read 21.4 mm and nothing was wrong, because its motor lies from
        # -41.5 to +1.5 while the link starts at zero. That is a false alarm
        # in the one check whose job is to catch a pocket in the wrong place.
        from .interfaces import face_for

        axis_face = face_for(str(drives.get(other.name, "")), "output")
        inset = (axis_face.face_inset_m if axis_face is not None
                 and axis_face.face_inset_m is not None
                 else 0.5 * carried.axial_length_m)
        origin = _drive_face(spec, link_index, other, at_x, height, width, mine)
        along = (mesh.element_centroids()[picked] - origin) @ axis
        low, high = inset - carried.axial_length_m, inset
        outside = float(np.max(np.maximum(low - along, along - high)))
        void_offsets.append(
            f"{other.name}: everything held empty for it is inside its own "
            f"span" if outside <= 0.0 else
            f"{other.name}: something held empty for it is "
            f"{outside * 1000:.1f} mm OUTSIDE the drive, so the pocket is "
            f"not where the drive is")

    volume_m3 = after_mm3 / EXPORT_SCALE ** 3
    design = LinkDesign(
        name=link.name, generated=True,
        mass_kg=volume_m3 * material.density_kg_m3,
        volume_m3=volume_m3,
        compliance_j=float(result.final_compliance),
        grey_fraction=float(np.mean((result.density > 0.1)
                                    & (result.density < 0.9))),
        unsupported_fraction=unsupported_fraction(mesh, result.density),
        volume_error_vs_field=surface.volume_error_vs_field,
        watertight=bool(body.is_watertight),
        triangles=int(body.faces.shape[0]),
        stl_path=str(stl))
    design.notes.extend(hole_report)
    design.notes.append(clipped)
    design.notes.extend(added)
    design.notes.extend(void_offsets)
    design.notes.append(
        f"the holes removed {100.0 * (1.0 - after_mm3 / before_mm3):.2f} "
        f"percent of the extracted volume")
    design.unresolved = unresolved
    design.notes.append(void_note)
    free = int((~(passive_solid | passive_void)).sum())
    design.notes.append(
        f"{100.0 * free / mesh.n_elements:.0f} percent of this domain is free "
        f"for the optimiser to decide; the rest is interfaces, drives and "
        f"corridors"
        + ("" if free > 0.4 * mesh.n_elements else
           ". THE SHAPE OF THIS PART IS SET BY ITS INTERFACES rather than by "
           "the optimisation, and it should be read that way"))
    design.notes.append(
        f"the volume fraction is a fraction of that free region, not of the "
        f"whole domain, so it means the same thing on every link")
    design.notes.append(
        f"design domain {span * 1000:.0f} by {height * 1000:.0f} by "
        f"{width * 1000:.0f} mm, {mesh.n_elements} elements, "
        f"{int(passive_solid.sum())} held solid for the interfaces and "
        f"{int(passive_void.sum())} held empty for the drive")
    design.notes.append(
        f"the exported STL and STEP are in millimetres; x = 0 is the start "
        f"joint plane and the joint axis line is at y = {height * 500:.1f}, "
        f"z = {width * 500:.1f} mm in the file's own frame")
    design.cross_evaluation = cross_evaluation(problem, {link.name: result.density},
                                               cases)
    return design


def faceted_step(stl_path: Path, step_path: Path) -> tuple[Path, int]:
    """Sew a triangle body into a solid and write it as STEP.

    THIS IS NOT A CLEAN B-REP, and geometry/cad_export/mesh_fallback.py says
    why at length: a STEP file made of thousands of planar facets is an STL
    wearing another extension. It is written anyway because a CAD system can
    stand it up and measure it, which is what it is for, and the face count is
    returned so nobody mistakes it for a surfaced model.

    build123d's `import_stl` returns ONE face carrying a triangulation, which
    sews into nothing; the facets have to be built as real planar faces first.
    """
    import numpy as np
    import trimesh
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_MakeSolid,
                                    BRepBuilderAPI_Sewing)
    from OCP.gp import gp_Pnt
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS

    mesh = trimesh.load(str(stl_path))
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.faces, dtype=int)

    sewing = BRepBuilderAPI_Sewing(1e-6)
    for a, b_, c in triangles:
        polygon = BRepBuilderAPI_MakePolygon(
            gp_Pnt(*vertices[a]), gp_Pnt(*vertices[b_]), gp_Pnt(*vertices[c]),
            True)
        face = BRepBuilderAPI_MakeFace(polygon.Wire())
        if face.IsDone():
            sewing.Add(face.Face())
    sewing.Perform()
    sewn = sewing.SewedShape()
    shape = sewn
    try:
        maker = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(sewn))
        if maker.IsDone():
            shape = maker.Solid()
    except Exception:
        pass

    step_path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(str(step_path))
    return step_path, int(triangles.shape[0])


def size_link_by_volume_fraction(spec: ManipulatorSpec, link_index: int,
                                 drives: dict[str, str], torques: dict[str, float],
                                 limit_m: float, out_dir: Path,
                                 fractions=(0.10, 0.15, 0.20, 0.25, 0.30),
                                 iterations: int = 30) -> list[dict]:
    """Sweep the volume fraction and report where the deflection limit is met.

    Topology takes a volume fraction, and the requirement is a deflection, so
    the two are joined by running the optimisation at several fractions and
    solving the EXTRACTED body each time. It is a sweep rather than a
    bisection because each point is a full optimisation and the sweep says
    what the curve looks like, which a bisection does not.
    """
    from optimization.topology.verify import (DisconnectedAtThreshold,
                                              tip_displacement_of_extracted)

    rows: list[dict] = []
    for fraction in fractions:
        design = generate_link(spec, link_index, drives, torques, out_dir,
                               iterations=iterations, volume_fraction=fraction)
        row = {"volume_fraction": fraction, "generated": design.generated,
               "mass_kg": design.mass_kg, "limit_m": limit_m}
        if design.generated:
            row["compliance_j"] = design.compliance_j
            row["meets_limit"] = None
            row["note"] = ("the deflection of the extracted body is measured "
                           "by the caller; this sweep reports mass and "
                           "compliance per fraction")
        else:
            row["reason"] = design.reason
        rows.append(row)
    return rows


#: CHOSEN. The narrowest a bolt ring may be. An M3 socket head is 5.5 mm
#: across and the ring has to leave material beyond it on both sides, and
#: 1.25 mm a side is the least that is worth calling a seat, so 8.0 mm.
#:
#: It exists because of how the last defect here was caught. The AK60-6's
#: housing flange came out one millimetre wide IN THE NEGATIVE, and a
#: negative number announces itself. Half a millimetre would not have: it
#: would have built, passed every check, and arrived as a bolt seat too
#: narrow to tighten against. A floor turns a class of silent failures into
#: a refusal, and the sign of a number is not a substitute for one.
MINIMUM_RING_WIDTH_M = 0.008

#: WHERE THE NARROWEST RING'S 8.5 mm COMES FROM, because it decomposes into
#: one number that is real and one that is a placeholder. Its inner edge is
#: 41.0, which is the AK80-64's measured 40.0 boss plus a millimetre of
#: clearance; its outer edge is 49.5, which is the 44.5 bolt circle plus 5 mm
#: of edge. So the millimetre of boss clearance comes straight out of the
#: ring's width.
#:
#: That millimetre is there because no drawing prints a tolerance for the
#: boss, which is already on the list of things this design cannot resolve.
#: Sized as a located fit it would be nearer 0.05 and the ring would be 9.4
#: mm. The 8.5 is therefore a pessimism produced by a missing tolerance, not
#: a limit of the design, and the two entries move together.
RING_WIDTH_DEPENDS_ON = (
    "the narrowest ring is 8.5 mm because the boss carries 1.0 mm of "
    "clearance for want of a printed tolerance. As a located fit it would be "
    "about 0.05 and the ring would be 9.4 mm")

#: CHOSEN. The cable has to leave the joint somewhere, and the only through
#: bore any of these drawings prints is the AK80-64 output's 21 mm. Using that
#: one diameter on every face means one cable size fits the whole arm, and it
#: is a choice, not a value read off a drawing.
CABLE_BORE_M = 0.021


def clip_to_domain(body, length_m: float, height_m: float, width_m: float,
                   scale: float = 1.0):
    """Cut the body back to the box it was designed in.

    Marching cubes puts the surface where the field crosses the iso level and
    smoothing moves it again, so an extracted body overshoots its domain by
    up to about 1.7 mm. That does not matter for one part and it matters for
    six in a row: a link whose domain is exactly the joint spacing grows past
    both its joint planes and interferes with its neighbours at both ends,
    which a Fusion assembly measured. Clipping also makes the two mounting
    faces flat, which is what they bolt against.
    """
    import trimesh

    before = float(abs(body.volume))
    box = trimesh.creation.box(
        extents=(length_m * scale, height_m * scale, width_m * scale),
        transform=trimesh.transformations.translation_matrix(
            [0.5 * length_m * scale, 0.5 * height_m * scale,
             0.5 * width_m * scale]))
    try:
        cut = body.intersection(box)
    except Exception as exc:                      # a boolean that will not run
        return body, (f"the body was NOT clipped to its domain: {exc}. It may "
                      f"overshoot its joint planes by about 1.7 mm at each end")
    if cut.is_empty or not cut.is_watertight:
        return body, ("the body was not clipped to its domain because the "
                      "result was not a closed solid")
    after = float(abs(cut.volume))
    return cut, (f"clipped to the {length_m * 1000:.1f} by "
                 f"{height_m * 1000:.0f} by {width_m * 1000:.0f} mm design "
                 f"domain, which removed "
                 f"{100.0 * (1.0 - after / before):.2f} percent of the "
                 f"extracted volume as marching cubes overshoot")


def _across(axis):
    """Two unit directions perpendicular to `axis`, right handed with it."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    seed = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    first = seed - axis * float(np.dot(seed, axis))
    first = first / np.linalg.norm(first)
    return first, np.cross(axis, first)


def mounting_holes(spec: ManipulatorSpec, link_index: int,
                   drives: dict[str, str], span_m: float,
                   clock_deg: float = 0.0, height_m: float = 0.098,
                   width_m: float = 0.098, box=None
                   ) -> tuple[list[dict], list[str]]:
    """Every hole this link's two end faces need, and what could not be cut.

    A link is bolted to the OUTPUT of the joint that drives it and carries
    the HOUSING of the joint after it, so its two ends take two different
    patterns. The last link has no next joint; its far end is a tool plate
    this arm has not specified, and no pattern is invented for it.

    EVERY HOLE FOLLOWS ITS OWN JOINT'S AXIS. That sounds obvious and it was
    not what happened: the axis was hardcoded along the link, which is right
    for a roll joint and wrong for every joint whose axis crosses the arm.
    On the shoulder it drilled the bolt circle sideways through the link and
    centred it on the section rather than on the drive, so three of the six
    links had their fastening in a place no bolt could reach. It is the same
    error as the pocket that sat 140.7 mm from its motor, in a third place.
    """
    from .interfaces import (bolt_holes, dowel_holes, face_for,
                             unresolved_features)

    joints = spec.joints()
    joint = joints[link_index]
    following = joints[link_index + 1] if link_index + 1 < len(joints) else None
    flange = spec.flange_thickness_m

    holes: list[dict] = []
    unresolved: list[str] = []
    ends = [("proximal", face_for(drives.get(joint.name, ""), "output"),
             joint, joint.name, +1.0)]
    if following is not None:
        ends.append(("distal", face_for(drives.get(following.name, ""), "housing"),
                     following, following.name, -1.0))
    else:
        unresolved.append(
            f"{spec.links()[link_index].name}: its far end is the tool plate "
            f"and this arm specifies no tool, so no pattern was cut there")

    for end, face, other, joint_name, side in ends:
        if face is None:
            unresolved.append(
                f"{joint_name}: no drawing was read for "
                f"{drives.get(joint_name, 'its drive')}, so its face has no "
                f"pattern and the {end} end of this link cannot be fastened")
            continue
        at_x = (box["reach_low"] if end == "proximal"
                else box["reach_low"] + box["joint_span"]) if box else (
                    0.0 if end == "proximal" else span_m)
        axis = local_axis(spec, link_index, other.axis)
        centre = _drive_face(spec, link_index, other, at_x, height_m, width_m,
                             box) if box is not None else np.array(
                                 [at_x, 0.5 * height_m, 0.5 * width_m])
        if end == "distal" and abs(float(axis[0])) <= 0.5:
            separation = face_separation_m(str(drives.get(joint_name, "")))
            centre = centre - axis * (separation or 0.0)
        # Two directions across the axis, to lay the bolt circle out in the
        # plane the face actually is. For a roll joint they come out as y and
        # z, which is what the hardcoded version assumed for every joint.
        first, second = _across(axis)
        low, high = (-0.002, flange + 0.002) if side > 0 else (
            -flange - 0.002, 0.002)

        def _record(kind, thread, diameter, u, v, deep=None, plane=0.0):
            # THE PLANE THIS CIRCLE IS IN. An output face has two: the outer
            # circle lies in the mounting face and the inner one on the END
            # OF THE BOSS, which stands proud of it by the face inset. One
            # plane for both put the inner holes inside the motor.
            base = centre + axis * side * plane
            start = base + axis * side * (low if deep is None else -deep)
            end_at = base + axis * side * (high if deep is None else 0.002)
            offset = first * u + second * v
            return {"end": end, "kind": kind, "face": face.face,
                    "thread": thread, "diameter_m": diameter,
                    "axis": [float(c) for c in axis],
                    "start_m": [float(c) for c in (start + offset)],
                    "end_m": [float(c) for c in (end_at + offset)],
                    "y_m": float(u), "z_m": float(v),
                    "x0_m": 0.0, "x1_m": 0.0}

        for hole in bolt_holes(face, clock_deg):
            holes.append(_record("clearance", hole["thread"],
                                 hole["diameter_m"], hole["y_m"], hole["z_m"],
                                 plane=hole.get("plane_offset_m", 0.0)))
        for dowel in dowel_holes(face):
            holes.append(_record("dowel", "", dowel["diameter_m"],
                                 dowel["y_m"], dowel["z_m"],
                                 deep=dowel["depth_m"],
                                 plane=dowel.get("plane_offset_m", 0.0)))
        bore = face.central_bore_m or CABLE_BORE_M
        holes.append(_record("bore", "", bore, 0.0, 0.0))
        unresolved.extend(unresolved_features(face))
    return holes, unresolved


def _drive_face(spec, link_index, joint, at_x, height_m, width_m, box):
    """The drive's OUTPUT mounting face, as a point in the link's frame."""
    axis = local_axis(spec, link_index, joint.axis)
    centre = np.array([at_x, 0.5 * height_m, 0.5 * width_m], dtype=float)
    if abs(float(axis[0])) <= 0.5:
        centre = centre - axis * float(np.dot(centre, axis))
        centre = centre + axis * (-float(box["low"][2]))
    return centre


def drive_envelopes(spec, link_index, drives, span_m, height_m, width_m, box
                    ) -> list[dict]:
    """The drives' own volumes, as cutters in the hole format.

    Same shape the pockets use and the same placement, so a body cut with
    these cannot be inside a motor whatever the mesh resolution was.
    """
    joints = spec.joints()
    following = (joints[link_index + 1]
                 if link_index + 1 < len(joints) else None)
    cutters = []
    for other, at_x in ((joints[link_index], 0.0), (following, span_m)):
        if other is None:
            continue
        actuator = actuator_for(other.name, drives)
        if actuator is None or not (actuator.outer_diameter_m
                                    and actuator.axial_length_m):
            continue
        from .interfaces import drive_profile

        profile = drive_profile(str(drives.get(other.name, "")))
        if profile is None:
            continue
        axis = local_axis(spec, link_index, other.axis)
        face_plane = _drive_face(spec, link_index, other, at_x, height_m,
                                 width_m, box)
        for low, high, radius in profile:
            start = face_plane + axis * low
            end = face_plane + axis * high
            cutters.append({
                "end": "drive", "kind": "envelope", "face": other.name,
                "thread": "", "diameter_m": 2.0 * radius,
                "axis": [float(v) for v in axis],
                "start_m": [float(v) for v in start],
                "end_m": [float(v) for v in end],
                "note": (f"{actuator.part_number} from {low * 1000:.1f} to "
                         f"{high * 1000:.1f} mm of its own axis at radius "
                         f"{radius * 1000:.1f}"),
            "y_m": 0.0, "z_m": 0.0, "x0_m": 0.0, "x1_m": 0.0})
    return cutters


def interface_solids(spec, link_index, drives, span_m, height_m, width_m,
                     box) -> list[dict]:
    """The bolt rings, as exact annuli to be added rather than voxels.

    THE INTERFACE IS SMALLER THAN THE GRID. The ring the bolts bear on is 8
    mm wide, the flange is 9 mm thick, and a clearance hole leaves 2.3 mm to
    the ring's edge, against a cell of 8.2 mm. A density field cannot hold
    any of those: holding the elements solid still leaves the iso surface to
    draw the boundary with half a cell of error, so a ring specified at 8 mm
    comes out somewhere between 4 and 12 and the number is the grid's rather
    than the drawing's.

    So the optimisation still holds those elements solid, which is what pulls
    the load path into them, and the finished dimensions come from a boolean
    instead. The rule generalises: a feature smaller than two or three cells
    belongs in the boolean pass. The density field decides WHERE material
    goes; an interface dimension came off a drawing and has no reason to
    negotiate with a mesh.
    """
    from .interfaces import drive_profile, face_for

    joints = spec.joints()
    following = (joints[link_index + 1]
                 if link_index + 1 < len(joints) else None)
    flange = spec.flange_thickness_m
    solids = []
    ends = [(joints[link_index], box["reach_low"], +1.0)]
    if following is not None:
        ends.append((following, box["reach_low"] + box["joint_span"], -1.0))
    for other, at_x, side in ends:
        actuator = actuator_for(other.name, drives)
        profile = drive_profile(str(drives.get(other.name, "")))
        face = face_for(str(drives.get(other.name, "")), "output")
        if actuator is None or profile is None or face is None:
            continue
        axis = local_axis(spec, link_index, other.axis)
        origin = _drive_face(spec, link_index, other, at_x, height_m, width_m,
                             box)
        # THE INNER EDGE IS WHATEVER THE DRIVE STILL OCCUPIES THERE, and
        # sometimes it occupies nothing. Past the AK80-64's output face the
        # drive is a 40 mm boss, so the flange is an annulus around it. Past
        # the AK60-6's housing face the drive stops altogether, so the flange
        # is a full disc and its bolt circle at 34 mm radius sits inside what
        # would otherwise have been called the drive's radius. Taking the
        # nearest profile segment either way gave that face a ring one
        # millimetre WIDE IN THE NEGATIVE, which is how it was noticed.
        separation = face_separation_m(str(drives.get(other.name, "")))
        if side > 0:
            low = 0.0
            beyond = [seg for seg in profile if seg[0] >= -1e-9]
        else:
            low = -(separation or 0.0) - flange
            beyond = [seg for seg in profile
                      if seg[1] <= -(separation or 0.0) + 1e-9]
        inner = max((seg[2] for seg in beyond), default=0.0)
        # ONE SEAT PER PLANE. The outer circle gets an annulus round the
        # boss at the mounting face; the inner circle needs its own disc ON
        # the boss end, a face inset further out, and that disc is what its
        # bolts pull against. There was no material there at all before,
        # which is why the wrist pitch body's inner ring reported nothing at
        # 64 of 64 points and reported it correctly.
        for pattern in face.patterns:
            if not pattern.plane_offset_m or side <= 0:
                continue
            seat = 0.5 * pattern.bolt_circle_m + 0.005
            base = origin + axis * pattern.plane_offset_m
            solids.append({
                "kind": "seat", "face": other.name,
                "outer_diameter_m": 2.0 * seat, "inner_diameter_m": 0.0,
                "axis": [float(v) for v in axis],
                "start_m": [float(v) for v in base],
                "end_m": [float(v) for v in (base + axis * flange)],
                "note": (f"the seat {other.name}'s inner {pattern.count} by "
                         f"{pattern.thread} circle pulls against, on the boss "
                         f"end {pattern.plane_offset_m * 1000:.1f} mm proud "
                         f"of the mounting face")})

        outer = max(0.5 * actuator.outer_diameter_m,
                    0.5 * max((p.bolt_circle_m for p in face.patterns
                               if not p.plane_offset_m), default=0.0) + 0.005)
        width = outer - inner
        if width < MINIMUM_RING_WIDTH_M:
            solids.append({
                "kind": "ring", "face": other.name, "refused": True,
                "width_m": float(width),
                "note": (f"{other.name}: its bolt ring would be "
                         f"{width * 1000:.1f} mm wide against a floor of "
                         f"{MINIMUM_RING_WIDTH_M * 1000:.0f}, which is an M3 "
                         f"head and 1.25 mm of material each side. No ring "
                         f"was added and this joint cannot be bolted")})
            continue
        start = origin + axis * low
        solids.append({
            "kind": "ring", "face": other.name,
            "outer_diameter_m": 2.0 * outer, "inner_diameter_m": 2.0 * inner,
            "axis": [float(v) for v in axis],
            "start_m": [float(v) for v in start],
            "end_m": [float(v) for v in (start + axis * flange)],
            "note": (f"the ring {other.name}'s bolts bear on, "
                     f"{(outer - inner) * 1000:.1f} mm wide and "
                     f"{flange * 1000:.0f} thick, added exactly rather than "
                     f"rounded to a {0.098 / 12 * 1000:.1f} mm cell")})
    return solids


def add_solids(body, solids, scale: float = 1.0, sections: int = 48):
    """Union the exact interface annuli onto the extracted body."""
    import trimesh

    if not solids:
        return body, []
    added = []
    for solid in solids:
        if solid.get("refused"):
            continue
        start = np.asarray(solid["start_m"], dtype=float) * scale
        end = np.asarray(solid["end_m"], dtype=float) * scale
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            continue
        transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0],
                                                   direction / length)
        transform[:3, 3] = 0.5 * (start + end)
        outer = trimesh.creation.cylinder(
            radius=0.5 * solid["outer_diameter_m"] * scale, height=length,
            sections=sections, transform=transform)
        bore = trimesh.creation.cylinder(
            radius=0.5 * solid["inner_diameter_m"] * scale,
            height=length * 3.0, sections=sections, transform=transform)
        if solid["inner_diameter_m"] <= 1e-9:
            added.append(outer)
            continue
        added.append(outer.difference(bore))
    before = float(abs(body.volume))
    for piece in added:
        body = body.union(piece)
    widths = [(s["face"], 0.5 * (s["outer_diameter_m"] - s["inner_diameter_m"]))
              for s in solids if not s.get("refused")]
    report = [f"{len(added)} interface rings added exactly, "
              f"{100.0 * (abs(body.volume) / before - 1.0):+.2f} percent "
              f"of volume"]
    if widths:
        joint, narrowest = min(widths, key=lambda pair: pair[1])
        report.append(
            f"the narrowest bolt ring is {narrowest * 1000:.1f} mm at "
            f"{joint}, against a floor of "
            f"{MINIMUM_RING_WIDTH_M * 1000:.0f} mm")
    report.extend(s["note"] for s in solids if s.get("refused"))
    return body, report


def cut_holes(body, holes: list[dict], height_m: float, width_m: float,
              scale: float = 1.0, sections: int = 24):
    """Subtract the holes from the extracted body.

    The holes are cut AFTER the optimisation, not held void during it. A 3.4
    mm clearance hole is under half an element on this grid, so the density
    field cannot represent one: asking it to would not produce a hole, it
    would produce a smeared grey patch. Cutting afterwards is what a drawing
    office does with a load path too, and it is honest about which part of the
    geometry the optimiser decided and which part the interface did.
    """
    import trimesh

    if not holes:
        return body, []
    cutters = []
    for hole in holes:
        # EVERY cutter carries its own axis and endpoints now. Bolt holes
        # used to be built along the link and offset in y and z, which is
        # right for a roll joint and puts a shoulder's bolt circle sideways
        # through the part.
        start = np.asarray(hole["start_m"], dtype=float) * scale
        end = np.asarray(hole["end_m"], dtype=float) * scale
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            continue
        transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0],
                                                   direction / length)
        transform[:3, 3] = 0.5 * (start + end)
        cutters.append(trimesh.creation.cylinder(
            radius=0.5 * hole["diameter_m"] * scale, height=length,
            sections=sections, transform=transform))

    tool = trimesh.util.concatenate(cutters)
    cut = body.difference(tool)
    # DROP THE ZERO VOLUME SHELLS. Cutting twenty four holes out of a
    # marching cubes body leaves the occasional closed surface with no
    # thickness: it is not a second part, it is an artefact, and it made the
    # tool flange arrive as two components with one of them measuring 0.0
    # cubic millimetres. Some CAD importers refuse a body like that and
    # others import it as a stray face nobody can select.
    pieces = [piece for piece in cut.split(only_watertight=False)
              if abs(float(piece.volume)) > 1e-9]
    if len(pieces) == 1:
        cut = pieces[0]
    elif len(pieces) > 1:
        cut = max(pieces, key=lambda piece: abs(float(piece.volume)))
    report = [f"{len(holes)} holes cut: "
              + ", ".join(sorted({f"{h['end']} {h['kind']} "
                                  f"{h['diameter_m'] * 1000:.1f} mm"
                                  for h in holes}))]
    return cut, report


def domain_extent(spec: ManipulatorSpec, link_index: int,
                  drives: dict[str, str], sections: dict | None = None):
    """A link's span and section, and where its own frame sits in the arm.

    Split out of `link_domain` because the placement of one link now depends
    on the placement of its neighbours, and building a mesh to find that out
    would be circular. Nothing here touches a mesh.

    Returns (span, height, width, z_low, z_high, basis) in metres, where the
    two z values are the link's extent along the arm's z, or None with a
    reason.
    """
    joints = spec.joints()
    links = spec.links()
    link = links[link_index]
    joint = joints[link_index]
    following = joints[link_index + 1] if link_index + 1 < len(joints) else None

    if following is None:
        span = link.length_m
    else:
        span = max(following.origin_x_m, following.origin_y_m)
    if span <= 0.0:
        return None, (f"{link.name}: its joints are coincident in every "
                      f"direction, so there is no space to design in")

    # A FLANGE AROUND A CROSSING AXIS IS A DISC, NOT AN END FACE. Where a
    # joint's axis runs along the link, the link bolts to a face across its
    # own end and everything is inside the span. Where the axis CROSSES the
    # link, the bolt circle lies in a plane the link runs through, centred on
    # the drive, so half of it is behind that joint's plane: the shoulder's
    # 89 mm circle reaches 44.5 mm back past the upper arm's own start. The
    # span used to stop exactly at the joint, which put those holes outside
    # the part.
    reach_low = reach_high = 0.0
    for other, at_start in ((joint, True), (following, False)):
        if other is None:
            continue
        if abs(float(local_axis(spec, link_index, other.axis)[0])) > 0.5:
            continue
        carried = actuator_for(other.name, drives)
        if carried is None or not carried.outer_diameter_m:
            continue
        if at_start:
            reach_low = max(reach_low, 0.5 * carried.outer_diameter_m)
        else:
            reach_high = max(reach_high, 0.5 * carried.outer_diameter_m)

    actuator = actuator_for(joint.name, drives)
    section = (sections or {}).get(link.name)
    height = (section.outer_height_m if section is not None
              else spec.minimum_section_m)
    width = (section.outer_width_m if section is not None
             else spec.minimum_section_m)
    if actuator is not None and actuator.outer_diameter_m:
        height = max(height, actuator.outer_diameter_m)
        width = max(width, actuator.outer_diameter_m)

    axis_own = local_axis(spec, link_index, joint.axis)
    driven_across = abs(float(axis_own[0])) <= 0.5
    carries_across = (
        following is not None
        and abs(float(local_axis(spec, link_index, following.axis)[0])) <= 0.5)
    separation = (face_separation_m(str(drives.get(following.name, "")))
                  if carries_across and following is not None else None)

    # EVERY DRIVE'S OUTPUT FACE IS THE ARM'S z = 0 PLANE, and the links
    # reach around them. This is not a rule that had to be chosen: it falls
    # out of putting each link on the far side of the face it bolts to.
    #
    #   driven by a crossing axis  the link is above that output face
    #   carries a crossing axis    the link is below that housing face,
    #                              which is the drive's face separation down
    #   driven by a coaxial axis   the link is centred on that axis
    #
    # A link can be under two of those at once and then its domain is the
    # UNION, which is a C or a crank, and which is what a real shoulder
    # casting and a real wrist body are. The alternative was to face
    # consecutive drives opposite ways, which removes the union but makes the
    # arm climb in z at every pitch joint and never come back.
    boxes = []
    reasons = []
    if driven_across:
        boxes.append((0.0, width))
        reasons.append("driven across, so above that output face")
    if carries_across:
        drop = separation or 0.0
        boxes.append((-drop - width, -drop))
        reasons.append(f"carries a crossing axis, so below that housing face "
                       f"{drop * 1000:.1f} mm down")
    if not driven_across:
        boxes.append((-0.5 * width, 0.5 * width))
        reasons.append("driven coaxially, so centred on that axis")
    low = min(b[0] for b in boxes)
    high = max(b[1] for b in boxes)
    basis = " AND ".join(reasons)
    if len(boxes) > 1:
        basis += (f". Its domain is the UNION of those, {(high - low) * 1000:.1f}"
                  f" mm across, and the shape between them is the optimiser's")
    return (span, height, low, high, basis, reach_low, reach_high), ""


def world_boxes(spec: ManipulatorSpec, drives: dict[str, str],
                sections: dict | None = None) -> list[dict]:
    """Every link's design domain as a box in the arm's frame."""
    joints = spec.joints()
    rows = []
    position = np.zeros(3)
    for index, link in enumerate(spec.links()):
        joint = joints[index]
        position = position + np.array([joint.origin_x_m, joint.origin_y_m, 0.0])
        built, reason = domain_extent(spec, index, drives, sections)
        if built is None:
            rows.append({"link": link.name, "placed": False, "reason": reason})
            continue
        span, height, z_low, z_high, basis, reach_low, reach_high = built
        following = joints[index + 1] if index + 1 < len(joints) else None
        along = 1 if (following is not None
                      and following.origin_y_m > following.origin_x_m) else 0
        other = 0 if along == 1 else 1
        low = np.array(position, dtype=float)
        high = np.array(position, dtype=float)
        low[along] -= reach_low
        high[along] += span + reach_high
        low[other] -= 0.5 * height
        high[other] += 0.5 * height
        low[2], high[2] = z_low, z_high
        rows.append({"link": link.name, "placed": True, "low": low,
                     "high": high, "span": span + reach_low + reach_high,
                     "joint_span": span, "reach_low": reach_low,
                     "reach_high": reach_high, "height": height,
                     "width": z_high - z_low, "z_low": z_low, "z_high": z_high,
                     "along": along, "basis": basis})
    return rows


def link_placements(spec: ManipulatorSpec, drives: dict[str, str],
                    sections: dict | None = None) -> list[dict]:
    """Where each link's own frame sits in the arm's frame.

    A link's file has x along itself from its start joint. What decides the
    rest is which side of a mounting face the link is on: it is on the far
    side of the face it bolts to, and the drive is on the near side. For a
    joint whose axis crosses the arm that offsets the whole link along that
    axis, and it is the offset that stops two links claiming one space.

    The base column carries the shoulder rather than being driven by it, so
    it hangs below that drive's HOUSING face; the upper arm is driven by it
    and sits above its OUTPUT face. The 42.7 mm between those two faces is
    the motor, and neither link is in it.
    """
    joints = spec.joints()
    links = spec.links()
    rows = []
    position = np.zeros(3)
    for index, link in enumerate(links):
        joint = joints[index]
        position = position + np.array([joint.origin_x_m, joint.origin_y_m, 0.0])
        following = joints[index + 1] if index + 1 < len(joints) else None
        built, reason = link_domain(spec, index, drives, sections=sections)
        if built is None:
            rows.append({"link": link.name, "placed": False, "reason": reason})
            continue
        _mesh, _solid, _void, span, height, width, _note = built

        along = 1 if (following is not None
                      and following.origin_y_m > following.origin_x_m) else 0
        across = [i for i in range(3) if i != along]
        low = np.array(position, dtype=float)
        high = np.array(position, dtype=float)
        high[along] += span
        low[across[0]] -= 0.5 * height
        high[across[0]] += 0.5 * height

        # The offset along the joint axis. Local z maps to the arm's z.
        axis_own = local_axis(spec, index, joint.axis)
        driven_across = abs(float(axis_own[0])) <= 0.5
        carries_across = (
            following is not None
            and abs(float(local_axis(spec, index, following.axis)[0])) <= 0.5)
        conflict = ""
        if driven_across:
            low[2], high[2] = 0.0, width          # driven: above the output face
            basis = "driven by a crossing axis, so it sits above that output face"
            if carries_across:
                basis += (" and carries another, which is why those two drives "
                          "must face opposite ways")
        elif carries_across:
            separation = face_separation_m(str(drives.get(following.name, "")))
            top = -(separation or 0.0)
            low[2], high[2] = top - width, top
            basis = (f"carries a crossing axis, so it hangs below that "
                     f"housing face, {(separation or 0.0) * 1000:.1f} mm "
                     f"below the joint")
            # TWO REQUIREMENTS THAT DO NOT BOTH HOLD. A link driven by a
            # coaxial joint has to be centred on that axis, because it bolts
            # to a face perpendicular to it and turns about it. A link that
            # carries a crossing joint has to sit clear of that drive's
            # housing face. Where a link does both, as the wrist roll body
            # does, the two cannot both be satisfied by a box: the pitch
            # drive has to hang to one side of the roll axis, which is what a
            # real wrist does and is not what this domain says.
            conflict = (
                f"{link.name} is driven by a coaxial joint, which wants it "
                f"centred on that axis, AND carries a crossing joint, which "
                f"pushes it {abs(top) * 1000:.1f} mm to one side. A box "
                f"cannot do both. A real wrist hangs the pitch drive off the "
                f"roll axis, and this domain does not say how")
        else:
            low[2], high[2] = -0.5 * width, 0.5 * width
            basis = "both its joints are coaxial with it, so it is centred"
        rows.append({"link": link.name, "placed": True,
                     "low_mm": [float(v) * 1000.0 for v in low],
                     "high_mm": [float(v) * 1000.0 for v in high],
                     "basis": basis, "conflict": conflict})
    return rows


def domain_overlaps(spec: ManipulatorSpec, drives: dict[str, str],
                    sections: dict | None = None, samples: int = 12
                    ) -> list[dict]:
    """Do two links ever hold the SAME element solid?

    The invariant used to be that adjacent domain boxes share nothing, and
    that stopped being the right question. The boxes overlap on purpose now:
    a flange around a crossing axis is a disc centred on the drive, so half
    of it lies behind that joint's own plane and inside the neighbour's box.
    Demanding no shared box would forbid the disc.

    What must never happen is that both links claim the same MATERIAL. Two
    rules can each carve out an exception, a link holding its own bolt ring
    against a neighbour's box being one, and two exceptions can fire in the
    same place without either rule noticing. So the shared volume is sampled
    and each point is asked of both links.
    """
    boxes = [box for box in world_boxes(spec, drives, sections)
             if box.get("placed")]
    built = {}
    for index, link in enumerate(spec.links()):
        made, _reason = link_domain(spec, index, drives, sections=sections)
        if made is not None:
            built[link.name] = (made[0], made[1], made[2],
                                next(b for b in boxes if b["link"] == link.name))

    def _state(name, point):
        """solid, void or free at a world point, in that link's own mesh."""
        if name not in built:
            return None
        mesh, solid, void, box = built[name]
        along, other = box["along"], (0 if box["along"] == 1 else 1)
        local = np.array([point[along] - box["low"][along],
                          point[other] - box["low"][other],
                          point[2] - box["low"][2]])
        index = np.array([local[0] / mesh.dx, local[1] / mesh.dy,
                          local[2] / mesh.dz], dtype=int)
        if np.any(index < 0) or index[0] >= mesh.nx or index[1] >= mesh.ny \
                or index[2] >= mesh.nz:
            return None
        flat = (index[0] * mesh.ny + index[1]) * mesh.nz + index[2]
        if flat >= mesh.n_elements:
            return None
        return "solid" if solid[flat] else ("void" if void[flat] else "free")

    rows = []
    for first, second in zip(boxes, boxes[1:]):
        low = np.maximum(first["low"], second["low"])
        high = np.minimum(first["high"], second["high"])
        extent = np.clip(high - low, 0.0, None)
        shared = float(np.prod(extent)) * 1e9
        both, contested = 0, 0
        if shared > 0.0:
            # SAMPLE INSIDE THE SHARED BOX, not on its faces. A face of the
            # shared box is a plane that runs THROUGH elements of both
            # meshes, so a point on it belongs to one element of each and
            # says nothing about whether the two links agree. Sampling the
            # faces reported four contested points between the two wrist
            # bodies, and all four sat exactly on x = 390.0, which is where
            # one box begins. They were an artefact of measuring on the seam.
            inset = 0.02 * np.clip(high - low, 0.0, None)
            grid = [np.linspace(low[i] + inset[i], high[i] - inset[i], samples)
                    for i in range(3)]
            for x in grid[0]:
                for y in grid[1]:
                    for z in grid[2]:
                        point = np.array([x, y, z])
                        a = _state(first["link"], point)
                        b = _state(second["link"], point)
                        if a == "solid" and b == "solid":
                            both += 1
                        if a in ("solid", "free") and b in ("solid", "free"):
                            contested += 1
        total = samples ** 3 if shared > 0.0 else 0
        rows.append({
            "pair": f"{first['link']} and {second['link']}",
            "shared_mm3": shared,
            "shared_extent_mm": [float(e) * 1000.0 for e in extent],
            "both_solid_samples": both,
            "both_could_use_samples": contested,
            "samples": total,
            # An upper bound on how much material the two could put in the
            # same place, if every contested point turned solid on both
            # sides. It is not a measurement of interpenetration; it is the
            # size of the room left for it.
            "contested_bound_mm3": (shared * contested / total
                                    if total else 0.0),
            "note": ("their domains meet face to face" if shared <= 0.0 else
                     "their boxes overlap, which a crossing flange requires; "
                     "what matters is that neither holds the same material. "
                     "A contested point is one element straddling the seam "
                     "between the two boxes: its centroid is outside the "
                     "neighbour's box so it is free, while part of the "
                     "element is inside it. That is bounded by half a cell "
                     "and it is the same half cell the drive envelopes are "
                     "subtracted to close")})
    return rows


#: The build volume of the machine whose data sheet this design's material
#: comes from, as the manufacturer prints it. It is a constraint, not a
#: statistic: a part that does not fit cannot be made on it whatever else is
#: true of the design.
EOS_M290_BUILD_VOLUME_M = (0.250, 0.250, 0.325)
EOS_M290_BUILD_VOLUME_SOURCE = (
    "EOS M 290 technical data, construction volume 250 x 250 x 325 mm, "
    "printed with the note that the height includes the build platform and "
    "is application dependent, read 2026-09-04 from "
    "https://www.eos.info/metal-solutions/metal-printers/eos-m-290")


def fits_the_build_volume(extent_m, volume_m=EOS_M290_BUILD_VOLUME_M
                          ) -> tuple[bool, str]:
    """Does a part fit the machine, in its best orientation?

    The part may be turned any way up, so the test is between the sorted
    extents and the sorted volume. The tallest axis of the machine is also
    the one whose height includes the build plate, so a part that only fits
    by using it is reported as fitting on a condition rather than fitting.
    """
    part = sorted(float(e) for e in extent_m)
    machine = sorted(float(v) for v in volume_m)
    if any(p > m for p, m in zip(part, machine)):
        worst = max((p - m) for p, m in zip(part, machine))
        return False, (f"does NOT fit: it is over by {worst * 1000:.1f} mm on "
                       f"its worst axis against {EOS_M290_BUILD_VOLUME_SOURCE}")
    margin = min(m - p for p, m in zip(part, machine))
    tall = part[-1] > machine[1]
    return True, (
        f"fits with {margin * 1000:.1f} mm to spare on its tightest axis"
        + (", but only by standing it up the build height, which the sheet "
           "says includes the build platform and is application dependent"
           if tall else ""))


def mounting_plane_chain(spec: ManipulatorSpec, drives: dict[str, str]
                         ) -> list[dict]:
    """Follow the mounting planes along the whole chain, not pair by pair.

    A placement rule can be right for one pair of joints and wrong for the
    arm. Facing consecutive drives on a shared axis in opposite directions
    removes the need for any link to wrap around a drive, and every pair of
    joints checks out under it. Followed along the chain it puts each pitch
    joint's output face 140.7 mm beyond the last one and never brings it
    back, so three pitch joints would carry the wrist a third of a metre out
    of the plane of the arm. That rule was written, and this is the check
    that would have killed it before it was.

    Reports each joint's output face position along the arm's z and how far
    the last one has drifted from the first.
    """
    rows = []
    position = 0.0
    for joint in spec.joints():
        rows.append({"joint": joint.name, "axis": list(joint.axis),
                     "output_face_z_mm": position * 1000.0,
                     "drive": str(drives.get(joint.name, ""))})
        # Every output face lies in one plane, so nothing accumulates. Under
        # the alternating rule this line advanced by the drive's own face
        # separation at each joint whose axis crossed the arm.
        position += 0.0
    drift = rows[-1]["output_face_z_mm"] - rows[0]["output_face_z_mm"]
    for row in rows:
        row["drift_from_first_mm"] = row["output_face_z_mm"] - rows[0]["output_face_z_mm"]
    rows.append({"joint": "TOTAL DRIFT", "axis": [], "drive": "",
                 "output_face_z_mm": None, "drift_from_first_mm": drift})
    return rows
