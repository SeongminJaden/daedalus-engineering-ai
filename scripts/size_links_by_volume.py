"""Find the lightest volume fraction each link can meet its deflection at.

The links were generated at a volume fraction of 0.3 because 0.3 is a number
people use, and that is not a reason. This searches instead: for each link,
bisect the fraction and, at every step, solve the EXTRACTED body in CalculiX
and compare its deflection with the share of the tip budget that link owns.
The optimiser's own compliance is not used as the test, because the field's
compliance is not the part's.

Two outcomes are useful and one of them is negative. If a link meets its
limit far below 0.3 the arm gets lighter. If a link cannot meet it at any
fraction, or falls apart before it gets light, that is the answer and the
fixed 0.3 was hiding it.

The links are independent, so they run in parallel, one process each with its
own CUDA context and its thread count pinned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("data/generated/manipulator_volume_search")


def _limit_threads(per_worker: int) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(per_worker)


def _search_one(payload):
    index, drives, torques, sections, limit_m, steps, iterations, threads = payload
    _limit_threads(threads)

    import numpy as np

    from core.materials import get_material
    from optimization.topology import SimpProblem
    from optimization.topology.manufacturing import support_projection_with_gradient
    from optimization.topology.multiload import optimize_multiload
    from optimization.topology.verify import search_volume_fraction
    from projects.manipulator.links import link_domain, link_load_cases
    from projects.manipulator.spec import SPEC

    link = SPEC.links()[index]
    joint = SPEC.joints()[index]
    built, reason = link_domain(SPEC, index, drives, sections=sections)
    if built is None:
        return {"link": link.name, "searched": False, "reason": reason}
    mesh, passive_solid, passive_void, span, height, width, _ = built
    material = get_material(SPEC.materials["link"])
    torque = abs(torques.get(joint.name, 1.0)) or 1.0
    transverse = max(torque / max(span, 1e-6), 10.0)
    projection, vjp = support_projection_with_gradient(mesh, build_axis=1)
    cases = link_load_cases(mesh, torque, transverse)

    def build_problem(fraction: float) -> SimpProblem:
        return SimpProblem(
            mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
            poisson_ratio=material.poisson_ratio,
            fixed_nodes=mesh.nodes_at_x(0.0),
            load_nodes=mesh.nodes_at_x(float(mesh.nx * mesh.dx)),
            total_load_n=-transverse, load_direction=1,
            volume_fraction=fraction, filter_radius_elements=2.0,
            passive_solid=passive_solid, passive_void=passive_void,
            density_projection=projection, projection_vjp=vjp)

    def runner(problem, max_iterations):
        return optimize_multiload(problem, cases, max_iterations=max_iterations)

    started = time.perf_counter()
    search = search_volume_fraction(build_problem, runner, limit_m,
                                    material.density_kg_m3, low=0.05, high=0.4,
                                    steps=steps, iterations=iterations)
    best = search.best
    return {"link": link.name, "searched": True, "limit_m": limit_m,
            "span_m": span, "seconds": round(time.perf_counter() - started, 1),
            "best_volume_fraction": None if best is None else best.volume_fraction,
            "best_mass_kg": None if best is None else best.mass_kg,
            "best_displacement_m": None if best is None else best.tip_displacement_m,
            "steps": [{"volume_fraction": s.volume_fraction,
                       "mass_kg": None if s.mass_kg != s.mass_kg else s.mass_kg,
                       "tip_displacement_m": (None if s.tip_displacement_m
                                              != s.tip_displacement_m
                                              else s.tip_displacement_m),
                       "feasible": s.feasible, "note": s.note}
                      for s in search.steps]}


def main(steps: int = 5, iterations: int = 40, workers: int = 3,
         threads_per_worker: int = 4) -> int:
    from projects.manipulator.arm import build_arm
    from projects.manipulator.loop import run_loop
    from projects.manipulator.spec import SPEC
    from projects.manipulator.stages import dynamics_stage

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    sections = loop.data["final_sections"]
    arm = build_arm(sections, SPEC)
    dynamics = dynamics_stage(arm, SPEC, samples=60)
    torques = {row["joint"]: max(row["peak_trapezoidal_nm"],
                                 row["peak_s_curve_nm"])
               for row in dynamics.rows}

    # Each link owns the share of the tip deflection budget its own length is
    # of the reach. A link twice as long may bend twice as far, because the
    # tip sees the sum.
    reach = SPEC.reach_check_m()
    payloads = []
    for index, link in enumerate(SPEC.links()):
        share = link.length_m / reach
        payloads.append((index, drives, torques, sections,
                         max(SPEC.tip_deflection_limit_m * share, 1e-5),
                         steps, iterations, threads_per_worker))

    started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_search_one, payloads))
    else:
        rows = [_search_one(payload) for payload in payloads]
    for row in rows:
        print(json.dumps(row, default=str), flush=True)

    found = [r for r in rows if r.get("best_volume_fraction") is not None]
    summary = {"rows": rows, "workers": workers,
               "threads_per_worker": threads_per_worker,
               "seconds": round(time.perf_counter() - started, 1),
               "links_with_a_feasible_fraction": len(found),
               "total_best_mass_kg": sum(r["best_mass_kg"] for r in found)
               if found else None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "search.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    args = parser.parse_args()
    raise SystemExit(main(args.steps, args.iterations, args.workers,
                          args.threads_per_worker))
