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


def check_link(index: int, spec, drives, sections, path: Path) -> dict:
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
        axis = local_axis(spec, index, joint.axis)
        centre = np.array([at_x, 0.5 * height, 0.5 * width], dtype=float)
        if abs(float(axis[0])) <= 0.5:
            centre = centre - axis * float(np.dot(centre, axis))
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

    blocked = [b for b in row["bolts"] if b["blocked_mm3"] > 0.0]
    row["bolts_blocked"] = len(blocked)
    row["bolts_checked"] = len(row["bolts"])
    row["verdict"] = (
        "assembles" if not blocked and all(
            "CANNOT" not in d.get("verdict", "") for d in row["drives"])
        else "DOES NOT ASSEMBLE")
    return row


def main(directory: str = str(LINKS)) -> int:
    from projects.manipulator.loop import run_loop
    from projects.manipulator.spec import SPEC

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    sections = loop.data["final_sections"]
    rows = []
    for index, link in enumerate(SPEC.links()):
        row = check_link(index, SPEC, drives, sections,
                         Path(directory) / f"{link.name}.stl")
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
