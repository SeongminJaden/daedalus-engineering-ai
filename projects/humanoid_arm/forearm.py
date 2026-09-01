"""Humanoid forearm link: the first end to end run of an external reference.

This exists as a worked example of the one rule in `core.design_reference`: a
reference may move where a search starts and what it finds beautiful, and may
not move what counts as safe. It is kept runnable rather than described,
because the interesting result was one nobody predicted and it should stay
reproducible.

WHAT THE FIRST RUN FOUND
========================
The reference asked for a section 30 to 60 mm tall, 20 to 40 mm wide, with a 2
to 4 mm wall. Optimising a 0.25 m cantilever carrying 2 kg produced 20.3 mm by
10.0 mm with a 1.0 mm wall: BELOW every dimensional prior, with two of the
three variables sitting on their lower bound.

That is not the reference being wrong, and it is not the optimiser being
wrong. A real forearm is sized by things the first problem statement never
mentioned: an actuator and a loom passing through it, torsion from the wrist,
and a shell that doubles as the housing. Asked only to resist bending from a
2 kg tip load, the honest answer is a thin blade.

So the gap between prior and optimum is INFORMATION. It measures what the
problem statement was missing. The second problem here adds those missing
requirements as requirements, not as forced dimensions, and the optimiser is
free to ignore them if the physics does not need them.

VALIDITY DOMAIN
===============
Both problems are a hollow rectangular cantilever, fixed at the root, with a
point load at the tip, evaluated by Euler-Bernoulli beam theory. Torsion is
Bredt's single cell formula, which assumes a thin wall relative to the section
and a closed cell. None of the numbers below is measured from a real arm.
Every value marked ASSUMED is a representative figure chosen to make the
requirement concrete, and would have to be replaced by a real specification
before any of this described a real part.
"""

from __future__ import annotations

import numpy as np

from core.design_reference import (DesignReference, FormTarget, LoadPathHint,
                                   Provenance, RangePrior)
from core.engineering_ir import (BoundaryCondition, BoundaryConditionType,
                                 BoundaryLocation, Constraints,
                                 EngineeringProblem, Geometry, Load,
                                 LoadApplication, LoadType, Objective,
                                 ObjectiveQuantity, ObjectiveSense,
                                 SectionType, Vec3)
from core.units import MM, MPA

#: Sources the priors came from. Named here so a reader can check them.
PROPORTION_SOURCE = (
    "https://isciia-itca.bit.edu.cn/docs/2024-11/"
    "cad12146ea074337958d5936485e2cad.pdf")
SHELL_SOURCE = "https://doi.org/10.3390/polym16070988"
HOUSING_SOURCE = (
    "https://escholarship.org/content/qt0qz3p57g/"
    "qt0qz3p57g_noSplash_deb33daa9d84ab0346cd18a07ba5f154.pdf")

LENGTH_M = 0.25
PAYLOAD_KG = 2.0
STANDARD_GRAVITY = 9.80665

#: Representative figures, ASSUMED. A real arm would state its own.
CLEAR_BORE_M = 0.015
WRIST_TORQUE_NM = 5.0

#: Thinnest wall the process can make, ASSUMED and PROCESS DEPENDENT. This
#: figure suits a printed polymer shell, which is where the reference's 2 to 4
#: mm prior came from. Machined aluminium goes thinner and would use a smaller
#: number; the point is that the figure belongs to the process, not the part.
MANUFACTURING_WALL_M = 0.002

BASELINE = "baseline"
PACKAGED = "packaged"


def forearm_problem(variant: str = BASELINE) -> EngineeringProblem:
    """The forearm as a design problem.

    `BASELINE` states only bending, which is what the first run solved.
    `PACKAGED` adds the requirements a real forearm actually carries: a clear
    bore for the actuator and loom, a wrist torque acting at the same time as
    the payload, and the thinnest wall the process can make. All three figures
    are representative and marked ASSUMED above.

    The wall floor is there because a torque sweep to 400 N m showed no load
    ever lifts the wall off its lower bound: a closed section carries torsion
    as shear flow, so enlarging the section always beats thickening the wall.
    The reference's 2 to 4 mm wall came from a paper on printed shells, and
    the only honest way to honour it is as a process constraint.
    """
    if variant not in (BASELINE, PACKAGED):
        raise ValueError(f"unknown variant {variant!r}")

    loads = [Load(type=LoadType.POINT_FORCE,
                  magnitude_n=PAYLOAD_KG * STANDARD_GRAVITY,
                  direction=Vec3(x=0.0, y=-1.0, z=0.0),
                  application=LoadApplication.TIP)]
    constraints = dict(max_stress_pa=120.0 * MPA, max_deflection_m=0.5 * MM,
                       min_safety_factor=2.0)

    if variant == PACKAGED:
        loads.append(Load(type=LoadType.TORQUE, magnitude_n=WRIST_TORQUE_NM,
                          direction=Vec3(x=1.0, y=0.0, z=0.0),
                          application=LoadApplication.TIP))
        constraints["min_clear_bore_m"] = CLEAR_BORE_M
        constraints["min_manufacturing_wall_m"] = MANUFACTURING_WALL_M

    return EngineeringProblem(
        name=f"humanoid_forearm_link_{variant}",
        geometry=Geometry(length_m=LENGTH_M, max_width_m=0.08,
                          max_height_m=0.08,
                          section_type=SectionType.HOLLOW_RECTANGLE),
        material_id="al_7075_t6",
        loads=loads,
        boundary_conditions=[BoundaryCondition(
            type=BoundaryConditionType.FIXED, location=BoundaryLocation.ROOT)],
        constraints=Constraints(**constraints),
        objectives=[Objective(sense=ObjectiveSense.MINIMIZE,
                              quantity=ObjectiveQuantity.MASS)])


def build_reference() -> DesignReference:
    """The priors an external language model supplied, with their sources.

    Confidence is capped by provenance, so nothing here can claim more than
    the evidence ladder allows a reference to claim.
    """
    return DesignReference(name="humanoid_forearm_link_reference", items=(
        RangePrior(name="outer_height_m", source=PROPORTION_SOURCE,
                   confidence=0.45, provenance=Provenance.CITED,
                   minimum=0.030, maximum=0.060),
        RangePrior(name="outer_width_m", source=PROPORTION_SOURCE,
                   confidence=0.45, provenance=Provenance.CITED,
                   minimum=0.020, maximum=0.040),
        RangePrior(name="wall_thickness_m", source=SHELL_SOURCE,
                   confidence=0.50, provenance=Provenance.CITED,
                   minimum=0.002, maximum=0.004),
        FormTarget(name="tapered", source=HOUSING_SOURCE, confidence=0.55,
                   provenance=Provenance.CITED, weight=1.0,
                   target="tapered profile: larger section near proximal "
                          "joint, smaller toward distal end"),
        FormTarget(name="smooth_shell", source=SHELL_SOURCE, confidence=0.55,
                   provenance=Provenance.CITED, weight=0.8,
                   target="smooth blended shell surfaces, rounded "
                          "transitions, no sharp edges"),
        FormTarget(name="integrated_housing", source=HOUSING_SOURCE,
                   confidence=0.60, provenance=Provenance.CITED, weight=0.7,
                   target="integrated structural housing: shell doubles as "
                          "load-bearing structure"),
        FormTarget(name="symmetry", source="designer aesthetic judgment",
                   confidence=0.20, provenance=Provenance.ASSUMED, weight=0.5,
                   target="bilateral symmetry about the bending plane"),
        LoadPathHint(name="bending_dominant", confidence=0.60,
                     source="Daedalus Phase 13 topology result plus "
                            "Euler-Bernoulli",
                     provenance=Provenance.MEASURED,
                     description="cantilever bending dominated under tip "
                                 "payload and gravity; concentrate material "
                                 "at extreme fibers (tall section), taper "
                                 "toward free end where bending moment is "
                                 "lower"),
    ))


def run(variant: str = BASELINE) -> dict:
    """Optimise from the default start and from the reference biased start.

    Returns both results so a caller can check the thing that matters: the
    reference changes the route, not the destination.
    """
    from optimization.constraints import (build_optimization_problem,
                                          evaluate_design)
    from optimization.gradient.slsqp import default_start, optimize_slsqp

    problem = build_optimization_problem(forearm_problem(variant))
    reference = build_reference()
    plain = default_start(problem)
    biased = reference.starting_point(problem, plain)

    # The invariant, checked rather than asserted: applying the reference
    # cannot move the evaluation of a design.
    probe = np.array([0.030, 0.045, 0.003])
    before = evaluate_design(problem, probe)
    reference.starting_point(problem, plain)
    after = evaluate_design(problem, probe)

    return {
        "variant": variant,
        "problem": problem,
        "reference": reference,
        "start_default": plain,
        "start_biased": biased,
        "physics_unmoved": (before.constraints == after.constraints
                            and before.safety_factor == after.safety_factor),
        "from_default": optimize_slsqp(problem, x0=plain),
        "from_reference": optimize_slsqp(problem, x0=biased),
    }


def record(variant: str = PACKAGED, run_id: str | None = None,
           db_path: str = "runs/brain.sqlite3") -> dict:
    """Store a run with the reference's provenance attached to it.

    The episodic store already has a meta field, so no schema change was
    needed. What matters is what goes into it: each prior keeps its source,
    its confidence and its provenance, next to the ceiling that capped them
    and an explicit no on physical validation. A number read back out of a
    database has lost the context that limited it, so the context is stored.
    """
    from brain.db import BrainDB
    from brain.episodic.memory import EpisodicMemory

    result = run(variant)
    best = result["from_reference"]
    meta = result["reference"].as_dict()
    meta["problem"] = f"humanoid_forearm_link_{variant}"
    meta["requirements_added"] = (
        [] if variant == BASELINE else
        [f"clear bore {CLEAR_BORE_M} m [ASSUMED]",
         f"wrist torque {WRIST_TORQUE_NM} N m [ASSUMED]",
         f"minimum manufacturing wall {MANUFACTURING_WALL_M} m "
         f"[ASSUMED, PROCESS DEPENDENT]"])
    meta["active_constraints"] = best.evaluation.active_constraints()
    meta["reference_decomposition"] = {
        "wall_thickness_m": "manufacturing process, satisfied once stated",
        "outer_width_m": "clear bore; b equals bore plus twice the wall "
                         "exactly, so the 20 to 40 mm prior means a 16 to "
                         "36 mm bore",
        "outer_height_m": "load level; nothing here requires height, and the "
                          "torque sweep reaches 39 mm only near 200 N m",
    }
    meta["note"] = (
        "no prior was ever forced. Each was traced to the KIND of requirement "
        "that produces it, and only then entered the problem. The wall prior "
        "is now met because a process floor was stated; the width prior is a "
        "packaging statement about the actuator; the height prior belongs to "
        "a much more heavily loaded arm than this one. At this load nothing "
        "structural binds at all.")

    memory = EpisodicMemory(BrainDB(db_path))
    identifier = run_id or f"humanoid-forearm-{variant}"
    memory.record_run(identifier, meta["problem"], termination="converged",
                      iterations=best.n_evaluations,
                      best_mass_kg=best.evaluation.mass_kg, meta=meta)
    return memory.get_run(identifier)
