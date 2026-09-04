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
import math
import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("data/generated/manipulator_volume_search")


def _limit_threads(per_worker: int) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(per_worker)


def _search_one(payload):
    (index, drives, torques, sections, limit_m, ladder, iterations,
     threads) = payload
    _limit_threads(threads)

    import numpy as np

    from core.materials import get_material
    from optimization.topology import SimpProblem
    from optimization.topology.manufacturing import support_projection_with_gradient
    from optimization.topology.multiload import optimize_multiload
    from optimization.topology.verify import tip_displacement_of_extracted
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
            volume_fraction=fraction, volume_fraction_of="free",
            filter_radius_elements=2.0,
            passive_solid=passive_solid, passive_void=passive_void,
            density_projection=projection, projection_vjp=vjp)

    def runner(problem, max_iterations):
        return optimize_multiload(problem, cases, max_iterations=max_iterations)

    # A LADDER, NOT A BISECTION. A bisection answers one question per link,
    # "what is the least volume fraction that meets THIS link's share of the
    # budget", and throws away the three evaluations it took to get there.
    # The share it is measured against is arbitrary anyway. What the four
    # evaluations actually build is a curve of mass against deflection, and
    # with six of those curves the budget can be spent where it is cheapest
    # rather than split by a rule.
    started = time.perf_counter()
    curve = []
    for fraction in ladder:
        problem = build_problem(fraction)
        result = runner(problem, max_iterations=iterations)
        point = {"volume_fraction": fraction}
        try:
            displacement = tip_displacement_of_extracted(problem,
                                                         result.density, 0.5)
        except Exception as exc:
            point.update({"feasible": False, "note": str(exc)[:120]})
            curve.append(point)
            continue
        kept = int((result.density >= 0.5).sum())
        point.update({
            "mass_kg": kept * problem.mesh.element_volume * material.density_kg_m3,
            "tip_displacement_m": float(displacement),
            "feasible": True})
        curve.append(point)

    return {"link": link.name, "searched": True, "span_m": span,
            "share_limit_m": limit_m,
            "seconds": round(time.perf_counter() - started, 1),
            "curve": curve}


def allocate(curves: list[dict], budget_m: float) -> dict:
    """Spend one deflection budget across six links, at least total mass.

    THE ASSUMPTION IS THAT THE DEFLECTIONS ADD. Each link's number is its own
    loaded face moving under its own load, and the tool's motion is taken as
    their sum. For small angles on a serial chain that is the usual first
    approximation and it is still an approximation: an upstream link's
    rotation carries everything outboard of it, so its deflection is worth
    more than a downstream one's, and this treats them as equal. It is
    checked afterwards by computing the whole arm once and comparing.

    The alternative it replaces was worse and less honest: each link was
    given the share of the budget its own length is of the reach, which has
    no argument behind it at all and which decides the final mass.
    """
    import itertools

    usable = [[point for point in row.get("curve", []) if point.get("feasible")]
              for row in curves]
    if any(not options for options in usable):
        return {"allocated": False,
                "reason": "at least one link has no feasible point on its "
                          "curve, so no allocation exists"}
    best = None
    for combination in itertools.product(*usable):
        total = sum(point["tip_displacement_m"] for point in combination)
        if total > budget_m:
            continue
        mass = sum(point["mass_kg"] for point in combination)
        if best is None or mass < best[0]:
            best = (mass, total, combination)
    if best is None:
        return {"allocated": False,
                "reason": "no combination meets the budget; the lightest "
                          "reachable deflection is above it"}
    mass, total, combination = best
    return {"allocated": True, "total_mass_kg": mass,
            "total_deflection_m": total, "budget_m": budget_m,
            "combinations": int(math.prod(len(o) for o in usable)),
            "per_link": [
                {"link": row["link"],
                 "volume_fraction": point["volume_fraction"],
                 "mass_kg": point["mass_kg"],
                 "tip_displacement_m": point["tip_displacement_m"],
                 "share_of_budget": point["tip_displacement_m"] / budget_m}
                for row, point in zip(curves, combination)]}


def by_length_share(curves: list[dict]) -> dict:
    """What the old rule would have chosen, for comparison.

    Each link takes the lightest point that meets the share of the budget its
    own length is of the reach. Reported beside the allocated answer so the
    cost of the rule is a number rather than a matter of opinion.
    """
    chosen, mass, total = [], 0.0, 0.0
    for row in curves:
        options = [p for p in row.get("curve", []) if p.get("feasible")
                   and p["tip_displacement_m"] <= row["share_limit_m"]]
        if not options:
            return {"allocated": False,
                    "reason": f"{row['link']} meets no point of its own share"}
        point = min(options, key=lambda p: p["mass_kg"])
        chosen.append({"link": row["link"],
                       "volume_fraction": point["volume_fraction"],
                       "mass_kg": point["mass_kg"],
                       "tip_displacement_m": point["tip_displacement_m"]})
        mass += point["mass_kg"]
        total += point["tip_displacement_m"]
    return {"allocated": True, "total_mass_kg": mass,
            "total_deflection_m": total, "per_link": chosen}


def main(ladder=(0.15, 0.25, 0.35, 0.45), iterations: int = 40,
         workers: int = 6, threads_per_worker: int = 2) -> int:
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
                         tuple(ladder), iterations, threads_per_worker))

    started = time.perf_counter()
    if workers > 1:
        # SPAWN, not fork. The parent has already touched Warp to size the
        # arm, so it holds a CUDA context, and a forked child inherits that
        # context in a state CUDA refuses to use: every worker died with
        # "Warp CUDA error 3: initialization error". A spawned worker starts
        # a fresh interpreter and initialises its own context.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=context) as pool:
            rows = list(pool.map(_search_one, payloads))
    else:
        rows = [_search_one(payload) for payload in payloads]
    for row in rows:
        print(json.dumps(row, default=str), flush=True)

    curves = [row for row in rows if row.get("searched")]
    budget = SPEC.tip_deflection_limit_m
    allocated = allocate(curves, budget)
    by_length = by_length_share(curves)
    summary = {"rows": rows, "workers": workers, "ladder": list(ladder),
               "threads_per_worker": threads_per_worker,
               "seconds": round(time.perf_counter() - started, 1),
               "budget_m": budget,
               "allocated": allocated,
               "by_length_share": by_length}
    summary["additivity_assumption"] = (
        "The tool's deflection is taken as the SUM of the six links'. Each "
        "number is that link's own loaded face moving under its own load, so "
        "adding them treats every link's bending as worth the same at the "
        "tool, and it is not: an upstream link rotates everything outboard "
        "of it, so a given angle at the shoulder moves the tool further than "
        "the same angle at the wrist. This is the usual small angle "
        "approximation for a serial chain and it is UNVERIFIED here. To "
        "check it, build the six allocated bodies, assemble them, solve the "
        "whole arm once at full reach, and compare that tip deflection with "
        "the sum. A large difference is a model problem rather than an "
        "allocation problem, and it would mean the allocation is spending a "
        "budget it has mismeasured.")
    if allocated.get("allocated") and by_length.get("allocated"):
        saved = by_length["total_mass_kg"] - allocated["total_mass_kg"]
        summary["allocation_saves_kg"] = saved
        summary["allocation_note"] = (
            f"splitting the deflection budget by each link's share of the "
            f"reach costs {saved:.3f} kg against spending it where it is "
            f"cheapest. The split was a rule with no argument behind it and "
            f"it decided the mass")
        summary["budget_used"] = {
            "allocated": allocated["total_deflection_m"] / budget,
            "by_length_share": by_length["total_deflection_m"] / budget}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "search.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", default="0.15,0.25,0.35,0.45",
                        help="volume fractions to evaluate on every link")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    args = parser.parse_args()
    raise SystemExit(main(tuple(float(v) for v in args.ladder.split(",")),
                          args.iterations, args.workers,
                          args.threads_per_worker))
