"""Generate the two parts that hold the arm to the world.

These were listed as gaps for several revisions: the base mount that holds
the base yaw drive's housing to the floor, and the tool plate that holds the
payload on the tool roll's output. A listed gap is honest and a gap that
stays listed is unfinished, so they are designed here from the same topology
path and the same drawing-backed interfaces the links use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.manipulator.arm import build_arm
from projects.manipulator.links import EXPORT_SCALE, faceted_step
from projects.manipulator.loop import run_loop
from projects.manipulator.mounts import (base_mount_loads, generate_mount,
                                         tool_plate_loads)
from projects.manipulator.spec import SPEC
from projects.manipulator.stages import dynamics_stage

OUT = Path("data/generated/manipulator_mounts")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(iterations: int = 60, volume_fraction: float = 0.45) -> int:
    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    sections = loop.data["final_sections"]
    arm = build_arm(sections, SPEC)
    dynamics = dynamics_stage(arm, SPEC, samples=60)
    peaks = {row["joint"]: max(row["peak_trapezoidal_nm"],
                               row["peak_s_curve_nm"])
             for row in dynamics.rows}
    spin = dynamics.rows[0]["payload_spin_nm"]

    joints = [joint.name for joint in SPEC.joints()]
    structure = loop.data["history"][-1].structure_mass_kg
    actuators = loop.data["history"][-1].actuator_mass_kg
    arm_mass = structure + actuators

    jobs = [
        # The base mount is 60 mm tall on a 140 mm square: tall enough for a
        # load path from an 85 mm bolt circle to a 120 mm one, and wide
        # enough that the floor bolts clear the drive.
        ("base_mount", drives[joints[0]], "housing", "floor",
         base_mount_loads(arm_mass, SPEC.payload_kg, peaks[joints[0]], SPEC),
         0.060, 0.140, 0.140),
        # The tool plate is 60 mm long on a 90 mm square. The square clears
        # the AK60-6's 68 mm bolt circle and its 79 mm body. The LENGTH was
        # 40 mm and had to grow: two 9 mm plates in a 40 mm part are 45
        # percent of the domain, which leaves the optimiser less volume than
        # the interfaces already use, and the extraction dropped a whole face.
        # At 60 mm the plates are 30 percent, the same share the base mount
        # has, and it generates.
        ("tool_plate", drives[joints[-1]], "output", "tool",
         tool_plate_loads(SPEC.payload_kg, spin, SPEC),
         0.060, 0.090, 0.090),
    ]

    rows = []
    for name, actuator, face, world, loads, length, height, width in jobs:
        started = time.perf_counter()
        design = generate_mount(name, actuator, face, world, loads, length,
                                height, width, OUT, SPEC,
                                iterations=iterations,
                                volume_fraction=volume_fraction)
        row = design.row()
        row["seconds"] = round(time.perf_counter() - started, 1)
        row["loads"] = design.loads
        row["notes"] = list(design.notes)
        row["unresolved"] = list(design.unresolved)
        row["envelope_mm"] = [length * 1000, height * 1000, width * 1000]
        if design.generated:
            stl = Path(design.stl_path)
            step, faces = faceted_step(stl, stl.with_suffix(".step"))
            row["stl"] = {"file": stl.name, "md5": md5(stl),
                          "bytes": stl.stat().st_size}
            row["step"] = {"file": step.name, "md5": md5(step),
                           "bytes": step.stat().st_size, "faces": faces}
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)

    made = [r for r in rows if r["generated"]]
    summary = {"units": "millimetre", "export_scale_from_si": EXPORT_SCALE,
               "volume_fraction": volume_fraction, "iterations": iterations,
               "arm_mass_kg": arm_mass, "parts": rows,
               "total_mass_kg": sum(r["mass_kg"] for r in made)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"total_mass_kg": summary["total_mass_kg"],
                      "generated": len(made)}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--volume-fraction", type=float, default=0.45)
    args = parser.parse_args()
    raise SystemExit(main(args.iterations, args.volume_fraction))
