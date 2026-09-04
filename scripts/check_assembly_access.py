"""Can the arm be put together, not just be together.

Every check so far asks whether the finished assembly has parts in the same
place. None asks whether that state can be reached. A link that wraps a drive
by more than half a turn has no straight path to slide the drive in, and a
bolt whose hole is buried has nowhere for a key to go. Both parts pass every
interference test and neither can be built.

TWO SWEEPS, ON THE GENERATED BODIES
===================================
The drive is swept along its own axis, both ways, out past the link. If the
swept volume meets the link in one direction the drive cannot come out that
way; if it meets it in both, the joint cannot be assembled at all.

Each bolt is swept along its own axis from its head, out past the link, at
the diameter a key and a socket head need: 5.5 mm for the head of an M3 cap
screw and 2.5 across the flats for the key, so 5.5 is the binding one and it
has to be clear for the length of the key's short arm.

Both are the same operation with different cylinders, and both are run on the
bodies that were actually written rather than on the density field, because
the field does not know where the material ended up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

#: ISO 4762 socket head cap screw head diameters, and the short arm of the
#: key that drives them. The head is the binding dimension, not the key: an
#: M3 head is 5.5 mm across and its key is 2.5, so anywhere the head passes
#: the key follows.
#:
#: THAT ONLY HOLDS FOR SOCKET HEADS. A socket key moves along the bolt axis
#: and turns inside the space the head already needs. A hexagon head to ISO
#: 4014 is turned by a spanner that sweeps a circle AROUND the bolt, so it
#: needs clear space to the side that this check does not look for. If a
#: joint on this arm ever takes a hexagon head, this test has to be replaced
#: rather than reused.
HEAD_DIAMETER_M = {"M3": 0.0055, "M4": 0.0070, "M5": 0.0085, "M8": 0.0130}
KEY_REACH_M = 0.020

LINKS = Path("data/generated/manipulator_links")


def _cylinder(radius_m, length_m, centre_m, axis, scale=1000.0):
    import trimesh

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis)
    transform[:3, 3] = np.asarray(centre_m, dtype=float) * scale
    return trimesh.creation.cylinder(radius=radius_m * scale,
                                     height=length_m * scale, sections=24,
                                     transform=transform)


def _blocked(body, tool) -> float:
    """Volume of the link the swept tool would have to pass through."""
    try:
        hit = body.intersection(tool)
    except Exception:
        return float("nan")
    return 0.0 if hit.is_empty else float(abs(hit.volume))


def _sweep_line(published: dict, link_name: str, joint_name: str):
    """The drive's axis line in the link file's own frame, from the PUBLISHED
    summary rather than from the generator's helpers.

    This path shares no code with the thing it checks, and that is the whole
    point of it. The first version of this sweep called the same frame
    helpers the generator does, inherited the same missing offset, and put
    its cylinder 140.7 mm from the drive on every joint whose axis crosses
    the arm: the cylinder swept empty space, the intersection came out zero,
    and the check answered "comes out" while the pocket was in the wrong
    place. A check that shares a coordinate transform with its subject cannot
    catch an error in that transform.

    So the arithmetic here is done twice over, deliberately. The link's box
    and the joints' world origins come out of summary.json, which is what an
    outside reader consumes, and the world to local mapping is rebuilt from
    the box corner and the direction the two joint origins differ in.
    """
    links = {row["link"]: row for row in published.get("links", [])}
    joints = {row["tag"]: row for row in published.get("joints", [])}
    entry, joint = links.get(link_name), joints.get(joint_name)
    if entry is None or joint is None or "domain_box_world_mm" not in entry:
        return None, None
    low = np.asarray(entry["domain_box_world_mm"]["min"], dtype=float) / 1000.0
    high = np.asarray(entry["domain_box_world_mm"]["max"], dtype=float) / 1000.0

    # Which world axis the link runs along: the one its box is longest in,
    # among x and y. The generator decides this from the joint offsets; this
    # decides it from the published box, so the two disagree if either is
    # wrong.
    along = 0 if (high - low)[0] >= (high - low)[1] else 1
    other = 1 - along

    world_axis = np.asarray(joint["axis"], dtype=float)
    world_origin = np.asarray(joint["origin_mm"], dtype=float) / 1000.0
    # Every output face lies in the arm's z = 0 plane, which the summary
    # states per joint rather than this script assuming it.
    world_origin[2] = float(joint.get("output_face_world_z_mm", 0.0)) / 1000.0

    def to_local(point):
        return np.array([point[along] - low[along],
                         point[other] - low[other],
                         point[2] - low[2]], dtype=float)

    axis = to_local(world_origin + world_axis) - to_local(world_origin)
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        return None, None
    return axis / length, to_local(world_origin)


def check_link(index: int, spec, drives, sections, path: Path,
               published: dict) -> dict:
    import trimesh

    from projects.manipulator.interfaces import bolt_holes, face_for
    from projects.manipulator.links import (actuator_for, link_domain,
                                            local_axis, mounting_holes)

    link = spec.links()[index]
    row = {"link": link.name, "checked": False}
    if not path.exists():
        row["reason"] = f"{path.name} has not been generated"
        return row
    body = trimesh.load(str(path))
    built, reason = link_domain(spec, index, drives, sections=sections)
    if built is None:
        row["reason"] = reason
        return row
    _mesh, _solid, _void, span, height, width, _note = built
    joints = spec.joints()
    from projects.manipulator.links import world_boxes

    box = next((b for b in world_boxes(spec, drives, sections)
                if b["link"] == link.name and b.get("placed")), None)
    row.update({"checked": True, "drives": [], "bolts": []})

    for role, at_x, joint in (("its own drive", 0.0, joints[index]),
                              ("the drive it carries", span,
                               joints[index + 1] if index + 1 < len(joints)
                               else None)):
        if joint is None:
            continue
        actuator = actuator_for(joint.name, drives)
        if actuator is None or not actuator.outer_diameter_m:
            row["drives"].append({"joint": joint.name, "role": role,
                                  "verdict": "no outline, not checked"})
            continue
        axis, centre = _sweep_line(published, link.name, joint.name)
        if axis is None:
            row["drives"].append({"joint": joint.name, "role": role,
                                  "verdict": "not in the published summary, "
                                             "not checked"})
            continue
        reach = 2.0 * (span + height + width)
        out = {}
        for name, sign in (("forwards", 1.0), ("backwards", -1.0)):
            tool = _cylinder(0.5 * actuator.outer_diameter_m, reach,
                             centre + sign * axis * (0.5 * reach), axis)
            out[name] = _blocked(body, tool)
        clear = [name for name, volume in out.items() if volume == 0.0]
        row["drives"].append({
            "joint": joint.name, "role": role,
            "forwards_mm3": out["forwards"], "backwards_mm3": out["backwards"],
            "verdict": ("comes out " + " and ".join(clear) if clear else
                        "CANNOT BE WITHDRAWN either way along its axis")})

    # IS THE PART INSIDE A MOTOR? This is the check that three defects in a
    # row got past, because all three were positions and everything measured
    # here was a size. A pocket can be the right diameter, the right length
    # and inside the domain and still be cut 140.7 mm from the drive it is
    # for, and no count of elements or free fraction shows it. Intersecting
    # the body with the drive's own envelope does, and the answer has to be
    # zero.
    # IS THE PART INSIDE A MOTOR? This is the check three defects in a row
    # got past, because all three were positions and everything else measured
    # here is a size. It uses the SAME envelope the generator subtracts, so
    # the two cannot disagree about what the motor is: the volume between the
    # two mounting faces, where every drawing read here shows the drive at
    # its full diameter. Beyond those faces a drive steps down to a boss and
    # the material out there is the link's own, with the bolt circle running
    # through it.
    from projects.manipulator.links import drive_envelopes

    row["drive_overlap"] = []
    for envelope in drive_envelopes(spec, index, drives, span, height, width,
                                    box):
        start = np.asarray(envelope["start_m"], dtype=float)
        end = np.asarray(envelope["end_m"], dtype=float)
        axis = end - start
        length = float(np.linalg.norm(axis))
        if length <= 0.0:
            continue
        tool = _cylinder(0.5 * envelope["diameter_m"], length,
                         0.5 * (start + end), axis / length)
        row["drive_overlap"].append({
            "joint": envelope["face"], "role": envelope["note"],
            "overlap_mm3": _blocked(body, tool)})

    holes, _unresolved = mounting_holes(spec, index, drives, span)
    for hole in holes:
        if hole["kind"] != "clearance":
            continue
        head = HEAD_DIAMETER_M.get(hole["thread"])
        if head is None:
            continue
        axis = np.array([1.0, 0.0, 0.0])
        # The head sits on the outer face of the flange, so the key comes in
        # from outside the link along the hole's own axis.
        outward = -1.0 if hole["end"] == "proximal" else 1.0
        start = np.array([hole["x0_m"] if outward < 0 else hole["x1_m"],
                          0.5 * height + hole["y_m"],
                          0.5 * width + hole["z_m"]])
        tool = _cylinder(0.5 * head, KEY_REACH_M,
                         start + axis * outward * 0.5 * KEY_REACH_M, axis)
        row["bolts"].append({
            "end": hole["end"], "thread": hole["thread"],
            "blocked_mm3": _blocked(body, tool)})

    inside = [d for d in row.get("drive_overlap", [])
              if d["overlap_mm3"] > 1.0]
    row["inside_a_motor_mm3"] = sum(d["overlap_mm3"]
                                    for d in row.get("drive_overlap", []))
    # IS THERE ANYTHING LEFT TO TIGHTEN AGAINST? The ring the bolts bear on
    # is 8 mm wide with eight 3.4 mm holes through it, and nothing else in
    # the problem knows it matters. This walks the bolt circle itself at 64
    # points and asks how many are inside the body. It samples the CIRCLE
    # rather than the holes on purpose: hitting the holes would need the same
    # in-plane basis the generator used, and sharing that basis is how the
    # last three defects survived. A surviving ring is mostly solid, broken
    # only where the holes are, which is about a tenth of it.
    row["bolt_rings"] = []
    for envelope in drive_envelopes(spec, index, drives, span, height, width,
                                    box):
        face = face_for(str(drives.get(envelope["face"], "")), "output")
        if face is None:
            continue
        axis = np.asarray(envelope["axis"], dtype=float)
        seed = (np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9
                else np.array([1.0, 0.0, 0.0]))
        first = seed - axis * float(np.dot(seed, axis))
        first = first / np.linalg.norm(first)
        second = np.cross(axis, first)
        plane = np.asarray(envelope["end_m"], dtype=float)
        for pattern in face.patterns:
            radius = 0.5 * pattern.bolt_circle_m
            angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
            points = np.array([
                (plane + first * (radius * np.cos(a))
                 + second * (radius * np.sin(a))) * 1000.0
                for a in angles])
            try:
                inside = int(body.contains(points).sum())
            except Exception as exc:
                row["bolt_rings"].append({"joint": envelope["face"],
                                          "thread": pattern.thread,
                                          "error": str(exc)[:80]})
                continue
            row["bolt_rings"].append({
                "joint": envelope["face"], "thread": pattern.thread,
                "bolt_circle_mm": pattern.bolt_circle_m * 1000.0,
                "inside_of_64": inside,
                "verdict": ("ring is there" if inside >= 40 else
                            "RING IS GONE, the bolts have nothing to bear on")})
    lost_rings = [r for r in row["bolt_rings"]
                  if "GONE" in str(r.get("verdict"))]
    row["rings_lost"] = len(lost_rings)

    blocked = [b for b in row["bolts"] if b["blocked_mm3"] > 0.0]
    row["bolts_blocked"] = len(blocked)
    row["bolts_checked"] = len(row["bolts"])
    row["verdict"] = (
        "assembles" if not blocked and not inside and not lost_rings and all(
            "CANNOT" not in d.get("verdict", "") for d in row["drives"])
        else "DOES NOT ASSEMBLE")
    return row


def main(directory: str = str(LINKS)) -> int:
    from projects.manipulator.loop import run_loop
    from projects.manipulator.spec import SPEC

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    sections = loop.data["final_sections"]
    summary = Path(directory) / "summary.json"
    published = json.loads(summary.read_text()) if summary.exists() else {}
    rows = []
    for index, link in enumerate(SPEC.links()):
        row = check_link(index, SPEC, drives, sections,
                         Path(directory) / f"{link.name}.stl", published)
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
    out = Path(directory) / "assembly_access.json"
    out.write_text(json.dumps({"links": rows}, indent=1, default=str))
    bad = [r for r in rows if r.get("verdict") == "DOES NOT ASSEMBLE"]
    print(json.dumps({"checked": sum(1 for r in rows if r.get("checked")),
                      "cannot_assemble": [r["link"] for r in bad]}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=str(LINKS))
    args = parser.parse_args()
    raise SystemExit(main(args.directory))
