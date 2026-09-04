"""The faces a link actually bolts to, read off the manufacturer's drawing.

Until this module the arm had no fastening. A link ended in a 9 mm slab whose
thickness came from an assumption that turned out to be wrong: that the link
would be tapped and a bolt would thread into it, so the slab needed 1.5
diameters of engagement. It does not. The threads are already in the actuator,
they are M3 and M4, and the link side is a CLEARANCE hole. The engagement is
set by the depth the drawing prints for the motor's own tapped hole, and the
slab thickness is set by bolt length and head clearance instead.

WHAT IS PRINTED AND WHAT IS NOT
===============================
Every number below was read from the drawing named in its source. The two
drawings do not print the same things. The AK80-64 prints a thread depth for
every hole, a toleranced central bore and a toleranced dowel; the AK80-9 V3.0
prints the same bolt circles and no depths at all, and prints no central bore.
Where a value is not printed the field is None and the design has to refuse
the feature rather than assume a number.

THE TWO FACES ARE DIFFERENT
===========================
An actuator has a housing face, which is the stator and does not turn, and an
output face, which does. A link is bolted to the output of the joint that
drives it and carries the housing of the joint after it, so BOTH ends of a
link are interfaces and they are not the same pattern. On the AK80-64 the
housing is 8-M3 on a 85 mm circle and the output is 8-M3 on an 89 mm circle,
which is a 4 mm difference that would put every bolt in the wrong place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: ISO 273 medium series clearance holes, the standard's own values, for the
#: two thread sizes these actuators use.
ISO_273_MEDIUM_M = {"M3": 0.0034, "M4": 0.0045}


@dataclass(frozen=True)
class DrawingSource:
    """The drawing a set of values was read from."""

    id: str
    title: str
    publisher: str
    url: str
    retrieved: str
    note: str = ""


AK80_64_DRAWING = DrawingSource(
    id="cubemars_ak80_64_2d",
    title="AK80-64 Robotic Actuator 2D drawing",
    publisher="CubeMars (Cubemars, T-Motor)",
    url=("https://www.cubemars.com/data/cms/202602/"
         "ak80-64-robotic-actuator-2d-drawing.pdf"),
    retrieved="2026-09-04",
    note=("one sheet, three views, dimensions in millimetres; no title block "
          "revision or date is printed on it"))

AK80_9_DRAWING = DrawingSource(
    id="cubemars_ak80_9_v3_2d",
    title="AK80-9 V3.0 Robotic Actuator 2D drawing",
    publisher="CubeMars (Cubemars, T-Motor)",
    url=("https://www.cubemars.com/data/cms/202602/"
         "ak80-9-v3-0-robotic-actuator-2d-drawing.pdf"),
    retrieved="2026-09-04",
    note=("one sheet, three views, dimensions in millimetres; thread depths "
          "are not printed on this drawing"))


@dataclass(frozen=True)
class BoltPattern:
    """A circle of identical threaded holes in the actuator.

    `clock_deg` is where the first hole sits, measured from the face's +y
    axis toward +z. A drawing that prints a circle and a count does NOT fix
    it, and the relative clock of the two faces is what decides whether an
    assembled arm's bolts line up, so where it is known the source says how.
    """

    thread: str
    count: int
    bolt_circle_m: float
    thread_depth_m: float | None
    as_printed: str
    clock_deg: float | None = None
    clock_source: str = ""
    clock_tolerance_deg: float | None = None

    @property
    def clearance_hole_m(self) -> float:
        """The hole in the LINK, which is a clearance hole, not a thread."""
        return ISO_273_MEDIUM_M[self.thread]


@dataclass(frozen=True)
class MountingFace:
    """One face of one actuator, and everything the link must match on it."""

    actuator: str
    face: str                      # "housing" or "output"
    source: DrawingSource
    outer_diameter_m: float
    patterns: tuple[BoltPattern, ...]
    #: A hole through the middle of the face. None where the drawing prints
    #: none, which is not the same as there being none.
    central_bore_m: float | None = None
    central_bore_depth_m: float | None = None
    #: The spigot the link registers on, largest step first.
    boss_diameters_m: tuple[float, ...] = ()
    #: How far this mounting face sits inboard from ITS end of the actuator,
    #: measured along the actuator's own axis. It is what places the body
    #: relative to the joint frame: a motor does not straddle the joint plane,
    #: it hangs off one side of it. None where no source gives it.
    face_inset_m: float | None = None
    dowel_diameter_m: float | None = None
    dowel_depth_m: float | None = None
    dowel_count: int | None = None
    #: Where the dowels sit. Empty where no source gives their position, and
    #: a dowel whose position is unknown is not a locating feature.
    dowel_bolt_circle_m: float | None = None
    dowel_angles_deg: tuple[float, ...] = ()
    dowel_source: str = ""
    notes: tuple[str, ...] = ()

    def largest_bolt_circle_m(self) -> float:
        return max(p.bolt_circle_m for p in self.patterns)

    def rows(self) -> list[dict]:
        return [{"actuator": self.actuator, "face": self.face,
                 "thread": p.thread, "count": p.count,
                 "bolt_circle_mm": p.bolt_circle_m * 1000.0,
                 "thread_depth_mm": (None if p.thread_depth_m is None
                                     else p.thread_depth_m * 1000.0),
                 "link_side_clearance_mm": p.clearance_hole_m * 1000.0,
                 "as_printed": p.as_printed, "source": self.source.id}
                for p in self.patterns]


#: The clock angles below were measured on the manufacturer's own 3D model,
#: not read off the 2D sheet, and they are graded that way. On the AK80-64
#: that model lands on exact integers and half degrees, so it is the original
#: geometry rather than a converted approximation. On the AK80-9 it does not,
#: and that difference is carried through to the tolerances recorded there.
MODEL_MEASURED = ("measured on the manufacturer distributed 3D model, "
                  "integer coordinates, 2026-09-04")
MODEL_APPROXIMATE = ("measured on the manufacturer distributed 3D model, "
                     "which is a converted model with non integer "
                     "coordinates, 2026-09-04")

AK80_64_HOUSING = MountingFace(
    actuator="cubemars_ak80_64_kv80", face="housing", source=AK80_64_DRAWING,
    outer_diameter_m=0.098,
    patterns=(BoltPattern("M3", 8, 0.085, 0.007, "8-M3 depth 7mm, PCD 85",
                          clock_deg=22.5, clock_source=MODEL_MEASURED),),
    boss_diameters_m=(0.071,), face_inset_m=0.0112,
    notes=("THE HOUSING PATTERN IS CLOCKED 22.5 DEGREES FROM THE OUTPUT "
           "PATTERN, which is half of the 45 degree pitch. An arm assembled "
           "with both ends on the same clock has every bolt on one end "
           "halfway between two holes",
           "the 2D drawing marks a 15 degree angular reference on this face "
           "and the model measures 22.5. The two do not agree and the model "
           "is used, because 22.5 is exactly half the hole pitch and 15 is "
           "not. What the 15 refers to has not been established"))

AK80_64_OUTPUT = MountingFace(
    actuator="cubemars_ak80_64_kv80", face="output", source=AK80_64_DRAWING,
    outer_diameter_m=0.098,
    patterns=(BoltPattern("M3", 8, 0.089, 0.010, "8-M3 depth 10mm, PCD 89",
                          clock_deg=0.0, clock_source=MODEL_MEASURED),
              BoltPattern("M4", 6, 0.028, 0.008, "6-M4 depth 8mm, PCD 28",
                          clock_deg=30.0, clock_source=MODEL_MEASURED)),
    central_bore_m=0.021, central_bore_depth_m=0.0045,
    boss_diameters_m=(0.080, 0.035), face_inset_m=0.008,
    dowel_diameter_m=0.003, dowel_depth_m=0.003, dowel_count=2,
    dowel_bolt_circle_m=0.028, dowel_angles_deg=(0.0, 180.0),
    dowel_source=MODEL_MEASURED,
    notes=("central bore printed 21.0 +0.02/0 depth 4.5",
           "dowels printed 2-diameter 3.0 +0.05/0 depth 3, and the model "
           "puts them on the 28 mm circle at 0 and 180 degrees, the same two "
           "angles as the outer M3 holes at a different radius. That is what "
           "makes them a locating feature: they fix the assembly's rotation",
           "the 6-M4 circle is clocked 30 degrees, so it is 30 degrees from "
           "the dowels and 30 from the outer bolts"))

AK80_9_HOUSING = MountingFace(
    actuator="cubemars_ak80_9_v3", face="housing", source=AK80_9_DRAWING,
    outer_diameter_m=0.098,
    patterns=(BoltPattern("M3", 8, 0.085, None, "8-M3, PCD 85",
                          clock_deg=22.5, clock_source=MODEL_APPROXIMATE,
                          clock_tolerance_deg=1.5),),
    boss_diameters_m=(0.071,), face_inset_m=0.011,
    notes=("no thread depth is printed on this drawing, so the bolt length "
           "cannot be chosen from it",))

AK80_9_OUTPUT = MountingFace(
    actuator="cubemars_ak80_9_v3", face="output", source=AK80_9_DRAWING,
    outer_diameter_m=0.098,
    patterns=(BoltPattern("M3", 8, 0.085, None, "8-M3, PCD 85",
                          clock_deg=22.5, clock_source=MODEL_APPROXIMATE,
                          clock_tolerance_deg=1.5),
              BoltPattern("M4", 6, 0.028, None, "6-M4, PCD 28",
                          clock_deg=15.75, clock_source=MODEL_APPROXIMATE,
                          clock_tolerance_deg=1.5)),
    central_bore_m=None, central_bore_depth_m=None,
    boss_diameters_m=(0.048, 0.037), face_inset_m=0.003,
    dowel_diameter_m=0.003, dowel_depth_m=0.003, dowel_count=None,
    notes=("no central bore is printed on this drawing; the 48 and 37 are "
           "boss steps, not a hole, so no through bore may be designed here",
           "the dowel is labelled diameter 3 depth 3 without a count prefix, "
           "so how many there are is not printed, and it was not found in the "
           "3D model either. No dowel hole is cut for this actuator",
           "THIS MODEL IS APPROXIMATE. Its 8-M3 circle measures near 86 mm "
           "against a printed 85, and its 6-M4 angles scatter from 58.2 to "
           "61.9 degrees apart where they should be 60. The PRINTED "
           "diameters are used and only the clock is taken from the model, "
           "with 1.5 degrees of stated uncertainty",
           "THE AXIAL DIMENSIONS OF THIS MODEL ARE TRUSTWORTHY even though "
           "its angles are not. Its two widest faces perpendicular to the "
           "axis measure 3.00 mm inboard of the output end and 11.00 mm "
           "inboard of the housing end, and the 2D drawing's section prints "
           "3 and 11. The approximate model and the drawing confirm each "
           "other on these, which is why the face insets are used and the "
           "clock angles are not trusted"))

AK60_6_DRAWING = DrawingSource(
    id="cubemars_ak60_6_v3_2d",
    title="AK60-6 V3.0 Robotic Actuator 2D drawing",
    publisher="CubeMars (Cubemars, T-Motor)",
    url=("https://www.cubemars.com/data/cms/202602/"
         "ak60-6-v3-0-robotic-actuator-2d-drawing.pdf"),
    retrieved="2026-09-04",
    note=("one sheet, three views, dimensions in millimetres; this drawing "
          "prints thread depths on BOTH faces, which the AK80-9's does not"))

#: The AK60-6's model, like the AK80-9's, is a converted one, so its angles
#: carry about a degree of scatter: its two dowels measure 181.93 degrees
#: apart where they are nominally 180. Its AXIAL dimensions are trustworthy
#: for the same reason the AK80-9's are, because the drawing's section prints
#: the same numbers the model measures.
MODEL_APPROXIMATE_CROSSCHECKED = (
    "measured on the manufacturer distributed 3D model, which is a converted "
    "model, with the face insets cross checked against the drawing's own "
    "section, 2026-09-04")

AK60_6_HOUSING = MountingFace(
    actuator="cubemars_ak60_6_v3", face="housing", source=AK60_6_DRAWING,
    outer_diameter_m=0.079,
    patterns=(BoltPattern("M3", 6, 0.068, 0.0035, "6xM3 depth 3.5, PCD 68",
                          clock_deg=0.0,
                          clock_source=MODEL_APPROXIMATE_CROSSCHECKED,
                          clock_tolerance_deg=1.0),),
    boss_diameters_m=(0.057,), face_inset_m=0.012,
    notes=("ITS HOUSING RING IS ALIGNED WITH ITS OUTPUT RING, both on 68 mm "
           "and both at the same clock. The AK80-64's housing is turned half "
           "a pitch from its output. So the relationship is a property of the "
           "part and not a rule, and it is stored per part",
           "the face inset is 12.0 mm, which the model measures and the "
           "drawing's section confirms as 43 minus 31",
           "the drawing marks a 30 degree angular reference on this face and "
           "which hole it is measured to is not stated; the clock here comes "
           "from the model instead"))

AK60_6_OUTPUT = MountingFace(
    actuator="cubemars_ak60_6_v3", face="output", source=AK60_6_DRAWING,
    outer_diameter_m=0.079,
    patterns=(BoltPattern("M3", 6, 0.068, 0.006, "6xM3 depth 6, PCD 68",
                          clock_deg=0.0,
                          clock_source=MODEL_APPROXIMATE_CROSSCHECKED,
                          clock_tolerance_deg=1.0),
              BoltPattern("M3", 6, 0.020, 0.006, "6xM3 depth 6, PCD 20",
                          clock_deg=8.81,
                          clock_source=MODEL_APPROXIMATE_CROSSCHECKED,
                          clock_tolerance_deg=1.0)),
    boss_diameters_m=(0.049, 0.025), face_inset_m=0.0015,
    dowel_diameter_m=0.003, dowel_depth_m=0.003, dowel_count=2,
    dowel_bolt_circle_m=0.020, dowel_angles_deg=(38.81, 220.74),
    dowel_source=MODEL_APPROXIMATE_CROSSCHECKED,
    notes=("dowels printed 2x diameter 3.0 +0.02/0 depth 3. The model puts "
           "them on the 20 mm circle at 38.81 and 220.74 degrees from the "
           "outer ring, which is 181.93 degrees apart against a nominal 180. "
           "That 1.93 is the model's own error, so the dowel angles need "
           "measuring on the part before they are drilled",
           "the face inset is 1.5 mm, which the model measures and the "
           "drawing's section prints. The 0.5 in the section is a thin step "
           "on the output boss and not a mounting face",
           "no central bore is printed. The 25 in the section is a step, not "
           "a stated hole",
           "the inner circle is 6xM3 on a 20 mm circle, which is a smaller "
           "circle than the AK80 family's 28 and takes M3 rather than M4"))

FACES: dict[tuple[str, str], MountingFace] = {
    (f.actuator, f.face): f for f in
    (AK80_64_HOUSING, AK80_64_OUTPUT, AK80_9_HOUSING, AK80_9_OUTPUT,
     AK60_6_HOUSING, AK60_6_OUTPUT)}


def face_for(actuator_id: str, face: str) -> MountingFace | None:
    """The drawing-backed face, or None where no drawing was read.

    None is the answer for a frameless motor with no published outline. The
    caller has to report the link as unfastenable rather than invent a
    pattern.
    """
    for key, value in FACES.items():
        if face == key[1] and key[0] in actuator_id:
            return value
    return None


def link_interfaces(link_names: list[str], joint_names: list[str],
                    drives: dict[str, str]) -> list[dict]:
    """Which face each end of each link bolts to.

    A link is driven by the joint at its start, so its PROXIMAL end is bolted
    to that joint's OUTPUT flange. The next joint's actuator is carried by the
    link, so its DISTAL end holds that actuator's HOUSING. The last link has
    no next joint and its distal end is the tool plate, which this arm has not
    specified.
    """
    rows = []
    for index, link in enumerate(link_names):
        joint = joint_names[index]
        following = (joint_names[index + 1]
                     if index + 1 < len(joint_names) else None)
        proximal = face_for(drives.get(joint, ""), "output")
        distal = (face_for(drives.get(following, ""), "housing")
                  if following else None)
        rows.append({
            "link": link,
            "proximal_joint": joint,
            "proximal_mates_to": "output flange",
            "proximal_actuator": drives.get(joint, ""),
            "proximal_face": proximal,
            "distal_joint": following or "",
            "distal_mates_to": "housing" if following else "tool plate",
            "distal_actuator": drives.get(following, "") if following else "",
            "distal_face": distal})
    return rows


def bolt_holes(face: MountingFace, clock_deg: float = 0.0) -> list[dict]:
    """Where the link's clearance holes go, in the link's own frame.

    The clock angle is a DESIGN CHOICE, not a value from the drawing. Only the
    AK80-64 housing face prints an angular reference; every other pattern is
    defined by the drawing up to a rotation, and the two ends of one link have
    to be clocked against each other by someone. That someone is this
    function's caller, and the angle it used is recorded.
    """
    import math

    holes = []
    for pattern in face.patterns:
        radius = 0.5 * pattern.bolt_circle_m
        start = pattern.clock_deg if pattern.clock_deg is not None else clock_deg
        for index in range(pattern.count):
            angle = math.radians(start + index * 360.0 / pattern.count)
            holes.append({"thread": pattern.thread,
                          "diameter_m": pattern.clearance_hole_m,
                          "y_m": radius * math.cos(angle),
                          "z_m": radius * math.sin(angle),
                          "bolt_circle_m": pattern.bolt_circle_m,
                          "clock_deg": start,
                          "clock_source": pattern.clock_source or "CHOSEN"})
    return holes


def dowel_holes(face: MountingFace) -> list[dict]:
    """The locating holes, where a source gives their position.

    A dowel is the feature that fixes which way round the parts go on. One
    whose angle nobody printed and nobody measured is not a locating feature,
    so this returns nothing for such a face rather than putting a hole at a
    plausible angle.
    """
    import math

    if not face.dowel_angles_deg or face.dowel_bolt_circle_m is None:
        return []
    radius = 0.5 * face.dowel_bolt_circle_m
    return [{"diameter_m": face.dowel_diameter_m,
             "depth_m": face.dowel_depth_m,
             "y_m": radius * math.cos(math.radians(angle)),
             "z_m": radius * math.sin(math.radians(angle)),
             "angle_deg": angle, "source": face.dowel_source}
            for angle in face.dowel_angles_deg]


def unresolved_features(face: MountingFace) -> list[str]:
    """What the drawing does not say, listed rather than filled in."""
    missing = []
    for pattern in face.patterns:
        if pattern.thread_depth_m is None:
            missing.append(
                f"{face.actuator} {face.face}: no thread depth is printed for "
                f"{pattern.as_printed}, so a bolt length cannot be chosen")
    if face.dowel_diameter_m is not None and not face.dowel_angles_deg:
        missing.append(
            f"{face.actuator} {face.face}: a dowel of diameter "
            f"{face.dowel_diameter_m * 1000:.0f} mm is printed and its "
            f"ANGULAR POSITION is not, and it was not found in the 3D model "
            f"either, so the mating hole cannot be located and none was cut")
    if face.dowel_count is None and face.dowel_diameter_m is not None:
        missing.append(f"{face.actuator} {face.face}: the dowel count is not "
                       f"printed either")
    for pattern in face.patterns:
        if pattern.clock_tolerance_deg:
            missing.append(
                f"{face.actuator} {face.face}: the {pattern.thread} circle's "
                f"clock is measured on an approximate model to within "
                f"{pattern.clock_tolerance_deg} degrees, which is "
                f"{math.radians(pattern.clock_tolerance_deg) * 0.5 * pattern.bolt_circle_m * 1000:.2f} "
                f"mm at the bolt. The clearance hole absorbs it and a dowel "
                f"would not")
    if face.central_bore_m is None:
        missing.append(
            f"{face.actuator} {face.face}: no central bore is printed, so no "
            f"through hole was designed on this face and there is no routed "
            f"path for the cable through it")
    return missing


def clock_uncertainty_check(face: MountingFace) -> list[dict]:
    """Does the clearance hole absorb the uncertainty in the clock angle?

    A clearance hole allows the bolt to move by half the difference between
    the hole and the thread: 0.2 mm for M3 in a 3.4 mm hole. An angular
    uncertainty of a degree and a half on an 85 mm circle moves the hole by
    1.11 mm, which is five times that. So a pattern measured on an
    approximate model does not simply have a looser tolerance. It has a
    tolerance the fastener cannot take up, and the parts will not go
    together.
    """
    rows = []
    for pattern in face.patterns:
        if not pattern.clock_tolerance_deg:
            continue
        offset = (math.radians(pattern.clock_tolerance_deg)
                  * 0.5 * pattern.bolt_circle_m)
        allowance = 0.5 * (pattern.clearance_hole_m
                           - float(pattern.thread[1:]) / 1000.0)
        rows.append({
            "actuator": face.actuator, "face": face.face,
            "thread": pattern.thread,
            "clock_offset_mm": offset * 1000.0,
            "clearance_allowance_mm": allowance * 1000.0,
            "verdict": "absorbed" if offset <= allowance else "NOT ABSORBED",
            "note": ("the hole takes up the uncertainty"
                     if offset <= allowance else
                     f"the bolt has {allowance * 1000:.2f} mm to move and the "
                     f"pattern may be {offset * 1000:.2f} mm out. The clock "
                     f"has to be measured on the real part, or the holes "
                     f"slotted, before this joint can be assembled")})
    return rows


def assembly_gaps(joint_names: list[str], drives: dict[str, str],
                  designed: set[str] | None = None) -> list[dict]:
    """Parts this arm needs and does not have.

    An arm is not six links. The first joint's HOUSING has to be held by
    something, and that something is not a link: it is a base mount bolted to
    the floor. The last joint's output has to hold a tool. Neither exists
    here, and neither is a detail: they are the two places the arm meets the
    world, and both carry the whole payload path.
    """
    designed = designed or set()
    gaps = [{
        "gap": "base mount",
        "where": f"{joint_names[0]} housing",
        "why": ("the base yaw drive's housing has to be fixed to the floor. "
                "The column is bolted to that drive's OUTPUT, so nothing in "
                "this design holds its stator. A mount on its 8-M3 85 mm "
                "circle, and the floor fixing under it, are not designed"),
        "carries": "the entire arm's weight and the base yaw reaction torque",
        "status": ("designed" if "base_mount" in designed else "MISSING"),
    }, {
        "gap": "tool plate",
        "where": f"{joint_names[-1]} output",
        "why": ("the arm ends at the tool flange and no tool interface is "
                "specified, so the 3 kg payload has nothing to attach to"),
        "carries": "the 3 kg payload",
        "status": ("designed" if "tool_plate" in designed else "MISSING"),
    }]
    for joint in joint_names:
        if face_for(drives.get(joint, ""), "housing") is None and \
                face_for(drives.get(joint, ""), "output") is None:
            gaps.append({
                "gap": "no published outline",
                "where": joint,
                "why": (f"{drives.get(joint, 'its drive')} publishes neither "
                        f"an outline nor a mounting pattern, so the parts on "
                        f"either side of this joint cannot be fastened to it"),
                "carries": "everything outboard of this joint",
                "status": "MISSING"})
    return gaps


def face_separation_m(actuator_id: str) -> float | None:
    """How far apart an actuator's two mounting faces are.

    This is what a link between two joints on the SAME axis is thick, and it
    is not a choice: the AK80-64 is 61.9 mm long with its output face 8.0 mm
    inboard of one end and its housing face 11.2 inboard of the other, so
    anything bolted to both is 42.7 mm thick. Every number in that comes off
    the drawing.
    """
    output = face_for(actuator_id, "output")
    housing = face_for(actuator_id, "housing")
    if output is None or housing is None:
        return None
    if output.face_inset_m is None or housing.face_inset_m is None:
        return None
    from drivetrain.sourced import sourced_motor

    try:
        length = sourced_motor(actuator_id).axial_length_m
    except Exception:
        return None
    if not length:
        return None
    return float(length - output.face_inset_m - housing.face_inset_m)
