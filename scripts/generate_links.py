"""Generating every link of the manipulator as a shape, reproducibly.

The first free form run was done from a throwaway script, which meant the
deliverable existed and the recipe for it did not. This is the recipe. It
writes the bodies, and it writes a summary that says what units they are in,
what design domain each came from, and what the six of them weigh together.

The files themselves stay out of git. Their path, their checksums and the
numbers below go in the document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.manipulator.arm import build_arm
from projects.manipulator.links import (EXPORT_SCALE, interfaces_are_reachable,
                                        world_boxes)
from projects.manipulator.loop import run_loop
from projects.manipulator.spec import SPEC
from projects.manipulator.stages import dynamics_stage

OUT = Path("data/generated/manipulator_links")

#: Threads each worker is allowed. Without this every worker's numpy and Warp
#: CPU backend try to use all sixteen and the workers fight each other, which
#: shows up as a throughput that gets WORSE as workers are added.
def _limit_threads(per_worker: int) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(per_worker)


def _one_link(payload):
    """One link in its own process. Everything is imported inside, because a
    fresh interpreter is what a process pool gives each worker."""
    (index, name, drives, torques, out_dir, iterations, fraction, threads,
     sections) = payload
    _limit_threads(threads)
    from projects.manipulator.links import faceted_step, generate_link
    from projects.manipulator.spec import SPEC

    started = time.perf_counter()
    design = generate_link(SPEC, index, drives, torques, Path(out_dir),
                           iterations=iterations, volume_fraction=fraction,
                           sections=sections)
    # The STEP conversion belongs in the worker. Left in the parent it is a
    # serial tail that grows with the link count and undoes the parallelism.
    files = {}
    if design.generated:
        stl = Path(design.stl_path)
        step, faces = faceted_step(stl, stl.with_suffix(".step"))
        files = {"stl": {"file": stl.name, "md5": md5(stl),
                         "bytes": stl.stat().st_size},
                 "step": {"file": step.name, "md5": md5(step),
                          "bytes": step.stat().st_size, "faces": faces}}
    return index, design, time.perf_counter() - started, files


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def searched_fractions(path: Path) -> dict:
    """Per link volume fractions from the search, where it found one.

    A link with no feasible fraction keeps the default, and the summary says
    which links those are. Using one number for six parts was never a design
    decision, it was a placeholder.
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {row["link"]: row["best_volume_fraction"]
            for row in data.get("rows", [])
            if row.get("best_volume_fraction") is not None}


def joint_rows(drives: dict) -> list[dict]:
    """Every joint's world origin, axis, drive and dowel clock.

    Enough for an assembly script to stand the arm up from the file instead
    of from a message. The origin accumulates the joint offsets, so it is the
    same arithmetic the arm model does rather than a second copy of it.
    """
    from projects.manipulator.interfaces import face_for

    origin = [0.0, 0.0, 0.0]
    rows = []
    for joint in SPEC.joints():
        origin = [origin[0] + joint.origin_x_m, origin[1] + joint.origin_y_m,
                  0.0]
        part = str(drives.get(joint.name, ""))
        face = face_for(part, "output")
        across = abs(float(joint.axis[0])) <= 0.5
        rows.append({
            "tag": joint.name,
            "origin_mm": [v * 1000.0 for v in origin],
            "axis": list(joint.axis),
            "actuator": part,
            # Every drive's output face is the arm's z = 0 plane and its body
            # is on the far side of it from the link it drives.
            "output_face_world_z_mm": 0.0,
            "output_faces": list(joint.axis),
            "dowel_angles_deg": (
                [90.0, 270.0] if across else [0.0, 180.0]),
            "dowel_basis": (
                "on the bending neutral axis, so the pins take shear and the "
                "bolts take the bending's axial part" if across else
                "no bending about this axis, so the angle is free"),
            "dowels_cut": bool(face is not None and face.dowel_angles_deg),
        })
    return rows


def undefined_names() -> list[str]:
    """Names this repository uses and does not have, found without running it.

    Two six link runs died ninety minutes in on exactly this: a variable that
    exists only inside another function, and two that were never bound in
    this one. Nothing evaluates the boolean pass until a link's optimisation
    has finished, so the cost of a misspelt name is a whole run. Pyflakes
    answers the question in a second and a half across the repository, needs
    nothing installed but itself, and starts no workers.

    It is a different question from the smoke test, which asks whether a
    stage actually runs. This one asks whether a name exists at all, and it
    is the cheaper of the two by four orders of magnitude.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    found = subprocess.run(
        [sys.executable, "-m", "pyflakes",
         *[str(root / part) for part in
           ("agent", "core", "drivetrain", "geometry", "nodes",
            "optimization", "physics", "projects", "scripts")]],
        capture_output=True, text=True)
    return [line for line in found.stdout.splitlines()
            if "undefined name" in line]


def main(iterations: int = 60, volume_fraction: float = 0.3,
         workers: int = 1, threads_per_worker: int = 4,
         search: str = "data/generated/manipulator_volume_search/search.json",
         only: set | None = None) -> int:
    blocking = undefined_names()
    if blocking:
        print(json.dumps({"refused": "undefined names in the repository",
                          "found": blocking}, indent=1))
        return 1

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    sections = loop.data["final_sections"]
    arm = build_arm(sections, SPEC)
    dynamics = dynamics_stage(arm, SPEC, samples=60)
    torques = {row["joint"]: max(row["peak_trapezoidal_nm"],
                                 row["peak_s_curve_nm"])
               for row in dynamics.rows}

    fractions = searched_fractions(Path(search))
    payloads = [(index, link.name, drives, torques, str(OUT), iterations,
                 fractions.get(link.name, volume_fraction),
                 threads_per_worker, sections)
                for index, link in enumerate(SPEC.links())
                if not only or link.name in only]
    if not payloads:
        print(json.dumps({"refused": "no link matched", "asked": sorted(only)}))
        return 1

    # CAN THE INTERFACES BE REACHED AT ALL? Fill each domain completely and
    # ask whether anything held solid is still cut off. It takes 0.2 seconds
    # for all six and it would have saved four hours and forty five minutes:
    # the upper arm and forearm were regenerated at a volume fraction of 0.45
    # to test the theory that they lacked material to join their flanges, and
    # the failure counts came back IDENTICAL TO THE DIGIT. They are severed by
    # what is held empty, and no volume fraction reaches them.
    unreachable = []
    for index, name, *_ in payloads:
        ok, why, counts = interfaces_are_reachable(SPEC, index, drives,
                                                   sections)
        if not ok:
            unreachable.append({"link": name, "why": why, **counts})
    if unreachable:
        print(json.dumps({
            "refused": "an interface cannot be reached inside its own domain",
            "note": ("this is a domain defect, not a volume fraction one. "
                     "Raising the volume fraction was tried and changed "
                     "nothing, at a cost of 4 h 45 min"),
            "links": unreachable}, indent=1))
        return 1
    if workers > 1:
        # SPAWN, not fork. The parent has already touched Warp to size the
        # arm, so it holds a CUDA context, and a forked child inherits that
        # context in a state CUDA refuses to use: every worker died with
        # "Warp CUDA error 3: initialization error". A spawned worker starts
        # a fresh interpreter and initialises its own context.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=context) as pool:
            # SUBMIT AND REPORT AS THEY LAND. Mapping and collecting the
            # whole list means a run of six links an hour long prints nothing
            # at all until it is over, and a run that is progressing looks
            # exactly like one that has hung.
            # If this script is killed, the SPAWNED WORKERS ARE NOT. They
            # keep running at full CPU and starve whatever is started next,
            # which once left two abandoned runs holding most of the machine
            # while a third crawled. Kill them by PID: they are children of
            # this process id and their command line says multiprocessing.
            futures = {pool.submit(_one_link, payload): payload[1]
                       for payload in payloads}
            done = []
            for future in as_completed(futures):
                result = future.result()
                done.append(result)
                print(json.dumps({"finished": futures[future],
                                  "seconds": round(result[2], 1),
                                  "of": len(payloads),
                                  "so_far": len(done)}), flush=True)
            done.sort(key=lambda r: r[0])
    else:
        done = [_one_link(payload) for payload in payloads]

    rows = []
    wanted = [link for link in SPEC.links() if not only or link.name in only]
    for (index, design, seconds, files), link in zip(done, wanted):
        row = {"link": link.name, "generated": design.generated,
               "mass_kg": design.mass_kg, "volume_m3": design.volume_m3,
               "compliance_j": design.compliance_j,
               "grey": design.grey_fraction,
               "unsupported": design.unsupported_fraction,
               "volume_error_vs_field": design.volume_error_vs_field,
               "watertight": design.watertight,
               "triangles": design.triangles,
               "notes": list(design.notes),
               "unresolved": list(design.unresolved),
               "volume_fraction": fractions.get(link.name, volume_fraction),
               "reason": design.reason, "seconds": round(seconds, 1)}
        row.update(files)
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)

    # WHERE EACH PART GOES, IN THE ARM'S FRAME, SO NOBODY HAS TO GUESS.
    # Standing these links up used to mean guessing the frame origin and
    # where the joint axis sits inside it, and half the placement errors so
    # far came from that transcription. The box is a number this generator
    # already has: it is the domain, and it is what the body is clipped to,
    # so a reader can fit the STEP's bounding box to it exactly. If the two
    # disagree the clip did not run, which makes this an independent check on
    # it rather than only a convenience.
    boxes = {box["link"]: box for box in world_boxes(SPEC, drives, sections)}
    for row in rows:
        box = boxes.get(row["link"])
        if box and box.get("placed"):
            row["domain_box_world_mm"] = {
                "min": [float(v) * 1000.0 for v in box["low"]],
                "max": [float(v) * 1000.0 for v in box["high"]],
                "basis": box["basis"]}

    generated = [r for r in rows if r["generated"]]
    summary = {
        "units": "millimetre",
        "export_scale_from_si": EXPORT_SCALE,
        "frame": ("each file is in its own link frame: x = 0 is the start "
                  "joint plane, x runs along the design domain, and the joint "
                  "axis line is at half the domain height and width in y "
                  "and z"),
        "volume_fraction": volume_fraction,
        "iterations": iterations,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "links": rows,
        "only": sorted(only) if only else None,
        "generated_count": len(generated),
        "total_mass_kg": sum(r["mass_kg"] for r in generated),
        "joints": joint_rows(drives),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"total_mass_kg": summary["total_mass_kg"],
                      "generated": summary["generated_count"]}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--volume-fraction", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=1,
                        help="links generated at once. Each worker holds its "
                             "own CUDA context, so this is bounded by video "
                             "memory as well as by cores")
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--links", default="",
                        help="regenerate only these links, comma separated")
    parser.add_argument(
        "--search",
        default="data/generated/manipulator_volume_search/search.json",
        help="per link volume fractions from scripts/size_links_by_volume.py")
    args = parser.parse_args()
    raise SystemExit(main(args.iterations, args.volume_fraction,
                          args.workers, args.threads_per_worker, args.search,
                          {name for name in args.links.split(",") if name}))
