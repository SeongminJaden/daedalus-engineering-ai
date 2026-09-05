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

import numpy as np

OUT = Path("data/generated/manipulator_volume_search")


def _limit_threads(per_worker: int) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = str(per_worker)


def _search_one(payload):
    (index, drives, torques, sections, limit_m, ladder, iterations, threads,
     lever_m) = payload
    _limit_threads(threads)

    import numpy as np

    from core.materials import get_material
    from optimization.topology import SimpProblem
    from optimization.topology.manufacturing import support_projection_with_gradient
    from optimization.topology.multiload import optimize_multiload
    from optimization.topology.export import largest_connected_component
    from optimization.topology.verify import face_motion_of_extracted
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

        # A FRACTION THAT CANNOT REACH ITS OWN INTERFACES IS OUTSIDE THE
        # DOMAIN, not a point on the curve. At 0.3 of the free region the
        # upper arm and the forearm left their bolt rings unconnected, and
        # keeping the largest connected component threw them away. A body
        # like that still deflects, and its deflection is cheap, so it would
        # sit on the curve as an attractive point and the allocator would
        # choose it. It is not a lighter design; it is a part with no bolt
        # seats.
        kept_field = largest_connected_component(problem.mesh, result.density,
                                                 0.5)
        if problem.passive_solid is not None:
            lost = int((np.asarray(problem.passive_solid)
                        & (kept_field < 0.5)).sum())
            if lost:
                point.update({
                    "feasible": False,
                    "note": (f"the extraction dropped {lost} of "
                             f"{int(np.sum(problem.passive_solid))} elements "
                             f"held solid for the interfaces, so this "
                             f"fraction cannot reach them and is outside the "
                             f"domain rather than a point on the curve")})
                curve.append(point)
                continue
        try:
            motion = face_motion_of_extracted(problem, result.density, 0.5)
        except Exception as exc:
            point.update({"feasible": False, "note": str(exc)[:120]})
            curve.append(point)
            continue
        kept = int((result.density >= 0.5).sum())
        # WHAT THIS LINK DOES TO THE TOOL, as a vector, from the solved
        # field. The face's translation moves everything outboard of it and
        # its rotation swings the reach still to come, and both come out of
        # a rigid body fit to the displacements rather than out of a
        # coefficient. `lever_m` is the vector from this face to the tool in
        # the arm's frame, so the cross product is the swing.
        lever = np.asarray(lever_m, dtype=float)
        rotation = np.asarray(motion["rotation_rad"], dtype=float)
        at_tool = np.asarray(motion["translation_m"]) + np.cross(rotation,
                                                                 lever)
        point.update({
            "mass_kg": kept * problem.mesh.element_volume * material.density_kg_m3,
            "tip_displacement_m": motion["load_direction_m"],
            "translation_m": motion["translation_m"],
            "rotation_rad": motion["rotation_rad"],
            "rigid_fit_residual_m": motion["residual_m"],
            "at_tool_m": [float(v) for v in at_tool],
            "at_tool_magnitude_m": float(np.linalg.norm(at_tool)),
            "feasible": True})
        curve.append(point)

    return {"link": link.name, "searched": True, "span_m": span,
            "share_limit_m": limit_m,
            "seconds": round(time.perf_counter() - started, 1),
            "curve": curve}


#: CHOSEN. How much of the deflection limit the allocation may spend. The
#: limit itself is 1 mm, which the specification records as a CHOSEN number
#: with no safety factor in it, because the task states a tip deflection
#: constraint without giving one. So there is nothing to draw margin from and
#: it has to be taken here, explicitly, once.
#:
#: Spending 98 percent of an unqualified limit is not a design; it is a
#: rounding error away from failing. Eighty leaves a fifth.
DESIGN_MARGIN = 0.80


def tool_levers(spec) -> dict:
    """The vector from each link's distal face to the tool, in the arm frame.

    This replaces the amplification coefficient entirely. A link's face
    rotation crossed with this vector IS its contribution to the tool's
    motion, with no assumption about the load shape: the 1.5 that stood here
    was the tip loaded cantilever relation, and it is 1.333 under a
    distributed load and 2.0 under a pure end moment, which is what an
    upstream link of an arm mostly sees. Choosing among them would have
    understated exactly the links that matter most.
    """
    joints = spec.joints()
    tool = np.array([spec.reach_check_m(), 0.0, 0.0])
    levers = {}
    along = 0.0
    for index, link in enumerate(spec.links()):
        following = joints[index + 1] if index + 1 < len(joints) else None
        along += (following.origin_x_m if following is not None
                  else link.length_m)
        levers[link.name] = [float(v) for v in
                             (tool - np.array([along, 0.0, 0.0]))]
    return levers


def amplification(spec) -> dict:
    """How much each link's own deflection is worth AT THE TOOL.

    This is the correction the additivity assumption was missing, and it is
    not small. A link's measured number is its own loaded face moving under
    its own load. In the arm, that link also ROTATES everything outboard of
    it: a cantilever whose tip moves by d has an end slope of about 1.5 d
    over its length, and that slope swings the remaining reach.

        contribution at the tool = d + (1.5 d / L) * R = d (1 + 1.5 R / L)

    where L is the link's own length and R is the reach still outboard of
    it. For the base column, 150 mm long with the whole 600 mm arm above it,
    that factor is 7. For the tool flange it is 1.

    So a plain sum of the six numbers UNDERSTATES the tool's deflection,
    and by a different amount for every link. The length share rule was
    spending only a third of the budget, which was waste and was also
    accidentally covering this; an allocation that spends the budget removes
    the cover. The 1.5 is the cantilever relation between tip deflection and
    end slope, and it is an approximation of the same kind as the sum it is
    correcting.
    """
    # The distance along the REACH, which is not the sum of the link
    # lengths: the base column stands up rather than out, so it adds 150 mm
    # of length and no reach at all while rotating the entire arm. Adding
    # every link's length put the tool 750 mm out on a 600 mm arm and gave
    # the base column a factor of 5.5 where it should be 7.
    joints = spec.joints()
    reach = spec.reach_check_m()
    factors = {}
    along = 0.0
    for index, link in enumerate(spec.links()):
        following = joints[index + 1] if index + 1 < len(joints) else None
        # The last link ends at the tool, not at another joint.
        along += (following.origin_x_m if following is not None
                  else link.length_m)
        remaining = max(reach - along, 0.0)
        factors[link.name] = 1.0 + 1.5 * remaining / max(link.length_m, 1e-9)
    return factors


def allocate(curves: list[dict], budget_m: float,
             factors: dict | None = None) -> dict:
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

    factors = factors or {}

    def at_tool(row, point):
        """This link's contribution to the tool's motion, as a VECTOR.

        Adding scalars was a second error underneath the first. A rotation
        is a vector and its effect is a cross product, so two links can move
        the tool in different directions and partly cancel, or in the same
        direction and add. This arm's pitch axes all lie along z so most of
        it does line up, but the roll joints do not, and a scalar sum
        pretends they do.
        """
        vector = point.get("at_tool_m")
        if vector is not None:
            return np.asarray(vector, dtype=float)
        # Nothing measured: fall back on the coefficient, along the load
        # direction only, and the caller can see which happened.
        estimate = point["tip_displacement_m"] * factors.get(row["link"], 1.0)
        return np.array([0.0, -estimate, 0.0])

    usable = [[point for point in row.get("curve", []) if point.get("feasible")]
              for row in curves]
    if any(not options for options in usable):
        empty = [row["link"] for row, options in zip(curves, usable)
                 if not options]
        return {"allocated": False,
                "reason": f"no allocation exists: {', '.join(empty)} has no "
                          f"feasible point on its curve, so every fraction "
                          f"tried either failed to reach its interfaces or "
                          f"failed to solve"}
    best = None
    for combination in itertools.product(*usable):
        vector = sum((at_tool(row, point)
                      for row, point in zip(curves, combination)),
                     np.zeros(3))
        total = float(np.linalg.norm(vector))
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
    vector = sum((at_tool(row, point)
                  for row, point in zip(curves, combination)), np.zeros(3))
    return {"allocated": True, "total_mass_kg": mass,
            "total_vector_m": [float(v) for v in vector],
            "total_deflection_m": total, "budget_m": budget_m,
            "combinations": int(math.prod(len(o) for o in usable)),
            "raw_scalar_sum_m": sum(point["tip_displacement_m"]
                                    for point in combination),
            "per_link": [
                {"link": row["link"],
                 "volume_fraction": point["volume_fraction"],
                 "mass_kg": point["mass_kg"],
                 "own_displacement_m": point["tip_displacement_m"],
                 "rotation_rad": point.get("rotation_rad"),
                 "rigid_fit_residual_m": point.get("rigid_fit_residual_m"),
                 "coefficient_estimate": factors.get(row["link"], 1.0),
                 "at_the_tool_m": [float(v) for v in at_tool(row, point)],
                 "at_the_tool_magnitude_m": float(
                     np.linalg.norm(at_tool(row, point))),
                 "share_of_budget": float(
                     np.linalg.norm(at_tool(row, point))) / budget_m}
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
        # Judged on the link's OWN face motion, which is what the old rule
        # did, so the comparison shows what that rule chose rather than what
        # it would have chosen knowing better.
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
    levers = tool_levers(SPEC)
    payloads = []
    for index, link in enumerate(SPEC.links()):
        share = link.length_m / reach
        payloads.append((index, drives, torques, sections,
                         max(SPEC.tip_deflection_limit_m * share, 1e-5),
                         tuple(ladder), iterations, threads_per_worker,
                         levers[link.name]))

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
    factors = amplification(SPEC)
    budget = SPEC.tip_deflection_limit_m * DESIGN_MARGIN
    allocated = allocate(curves, budget, factors)
    by_length = by_length_share(curves)
    summary = {"rows": rows, "workers": workers, "ladder": list(ladder),
               "threads_per_worker": threads_per_worker,
               "seconds": round(time.perf_counter() - started, 1),
               "limit_m": SPEC.tip_deflection_limit_m,
               "design_margin": DESIGN_MARGIN,
               "budget_m": budget,
               "amplification": factors,
               "ladder_rungs": len(ladder),
               "limit_carries_no_safety_factor": (
                   "the 1 mm limit is recorded in the specification as a "
                   "CHOSEN number, 1/600 of the reach, because the task "
                   "states a tip deflection constraint without giving one. "
                   "There is no factor inside it to draw margin from, so the "
                   "margin is taken here, once, and it is "
                   f"{DESIGN_MARGIN:.0%} of the limit"),
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
            f"cheapest, on a {len(ladder)} rung ladder. The split was a rule "
            f"with no argument behind it and it decided the mass. HOW MUCH "
            f"OF THIS IS THE LADDER: a length share has to take the first "
            f"rung that clears its own share, so a link needing 0.16 spends "
            f"0.25, and that waste is independent per link where the "
            f"allocation can cancel it across six. A finer ladder narrows "
            f"the gap without closing it")
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
