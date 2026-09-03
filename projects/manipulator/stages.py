"""The design stages, each returning a table and its own limits.

Every stage is a function of the arm and the specification and returns rows a
document can print. A stage that cannot run says so in its rows rather than
raising, because half the value of this exercise is the list of things the
pipeline cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.assembly import Assembly
from core.assembly.statics import joint_torques
from core.materials import get_material
from physics.dynamics import inverse_dynamics, plan_move, torque_profile
from physics.dynamics.actuator import (DriveDemand, best_ratio,
                                       inertia_matched_ratio, motor_torque_nm,
                                       reflected_inertia_kg_m2)

from .arm import build_arm, payload_force_n, stretched_pose
from .spec import SPEC, ManipulatorSpec


@dataclass
class StageResult:
    """What one stage produced, and what it could not."""

    name: str
    rows: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    could_not: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------- 1. policy

def policy_stage(spec: ManipulatorSpec = SPEC) -> StageResult:
    """The goal sentence through the policy layer, per link.

    The policy reads one cantilever at a time, so the arm is put to it as six
    sub problems. What it cannot do is the decomposition itself: nothing here
    turns "a six axis arm" into six links, and the fact that a human wrote the
    six sentences is the gap this stage reports.
    """
    from agent.policy import InvalidProposal, RuleBasedPolicy

    policy = RuleBasedPolicy(default_material=spec.materials["link"])
    result = StageResult(name="policy")
    payload_n = spec.payload_kg * 9.80665
    for link in spec.links():
        sentence = (f"a {link.length_m * 1000:.0f} mm long {link.role} "
                    f"carrying {payload_n:.0f} N at the tip, deflection under "
                    f"{spec.tip_deflection_limit_m * 1000:.1f} mm, safety "
                    f"factor {spec.static_safety_factor_metal}, "
                    f"{link.outer_height_m * 1000:.0f} mm tall and "
                    f"{link.outer_width_m * 1000:.0f} mm wide")
        try:
            proposal = policy.propose_problem(sentence)
            available = {"parametric_section", "generative_cad",
                         "topology_compliance", "freeform_topology"}
            choice = policy.choose_strategy(proposal.problem, available)
            result.rows.append({
                "link": link.name, "accepted": True,
                "length_m": proposal.problem.geometry.length_m,
                "load_n": proposal.problem.loads[0].magnitude_n,
                "safety_factor": proposal.problem.constraints.min_safety_factor,
                "strategy": choice.method,
                "verified": proposal.verified})
        except InvalidProposal as exc:
            result.rows.append({"link": link.name, "accepted": False,
                                "reason": str(exc)[:160]})
    result.could_not.append(
        "The policy turns ONE sentence into ONE cantilever problem. It does "
        "not decompose an arm into links: the six sentences above were "
        "written by hand from the specification, and nothing in this "
        "repository infers a kinematic chain from a goal.")
    result.could_not.append(
        "Every sub problem carries the full payload at its own tip, which is "
        "conservative for the inboard links and wrong about the moment they "
        "actually carry. The dynamics stage computes the real joint torques; "
        "the policy layer has no way to express one.")
    return result


# ------------------------------------------------------------ 2. dynamics

def dynamics_stage(arm: Assembly, spec: ManipulatorSpec = SPEC,
                   samples: int = 120) -> StageResult:
    """Inverse dynamics on the stated move, both profiles, plus statics."""
    material = get_material(arm.material_id)
    density = material.density_kg_m3
    result = StageResult(name="dynamics")

    q0 = stretched_pose(spec)
    q1 = q0 + spec.move_angle_rad
    dof = spec.degrees_of_freedom
    # The move must take the stated time, so the limits are derived from it
    # rather than guessed: a trapezoid that ramps for a third of the move
    # covers the angle in the time when v = 1.5 theta / T.
    velocity = np.full(dof, 1.5 * spec.move_angle_rad / spec.move_time_s)
    acceleration = np.full(dof, 4.5 * spec.move_angle_rad / spec.move_time_s ** 2)
    jerk = np.full(dof, spec.jerk_limit_rad_s3)

    payload = payload_force_n(spec)
    profiles = {}
    for kind, extra in (("trapezoidal", {}), ("s_curve", {"jerk_limits": jerk})):
        trajectory = plan_move(q0, q1, velocity, acceleration, samples=samples,
                               kind=kind, **extra)
        profiles[kind] = torque_profile(arm, trajectory, density,
                                        tip_force_n=payload)

    static = joint_torques(arm, q0, density, tip_force_n=payload)
    names = [j.name for j in arm.actuated_joints()]
    for index, name in enumerate(names):
        trap = profiles["trapezoidal"]
        curve = profiles["s_curve"]
        peak = float(trap.peak_torque_nm[index])
        result.rows.append({
            "joint": name,
            "static_nm": float(static[index]),
            "peak_trapezoidal_nm": peak,
            "rms_trapezoidal_nm": float(trap.rms_torque_nm[index]),
            "peak_s_curve_nm": float(curve.peak_torque_nm[index]),
            "rms_s_curve_nm": float(curve.rms_torque_nm[index]),
            "peak_over_rms": float(trap.peak_to_rms[index]),
            "gravity_share": (abs(float(static[index])) / peak if peak else 0.0),
        })
    result.data["profiles"] = profiles
    result.data["duration_s"] = {k: v.trajectory.duration_s for k, v in profiles.items()}
    result.notes.append(
        f"the move is {np.degrees(spec.move_angle_rad):.0f} degrees on every "
        f"joint in {spec.move_time_s} s; the trapezoid takes "
        f"{profiles['trapezoidal'].trajectory.duration_s:.2f} s and the s "
        f"curve {profiles['s_curve'].trajectory.duration_s:.2f} s")
    result.notes.append(
        "friction is zero because no measured coefficients exist for these "
        "joints, so every torque here is a lower bound")
    return result


def pinocchio_cross_check(arm: Assembly, spec: ManipulatorSpec = SPEC) -> StageResult:
    """The same inverse dynamics in an independent engine."""
    result = StageResult(name="pinocchio")
    from nodes import pinocchio_node

    if not pinocchio_node.is_available():
        result.could_not.append("pinocchio is not installed")
        return result
    density = get_material(arm.material_id).density_kg_m3
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, spec.degrees_of_freedom)
        qd = rng.uniform(-1.0, 1.0, spec.degrees_of_freedom)
        qdd = rng.uniform(-2.0, 2.0, spec.degrees_of_freedom)
        comparison = pinocchio_node.compare(arm, q, qd, qdd, density)
        worst = max(worst, comparison.inverse_dynamics_error_nm)
        result.rows.append({
            "state": "random",
            "inverse_dynamics_error_nm": comparison.inverse_dynamics_error_nm,
            "mass_matrix_error": comparison.mass_matrix_error,
            "gravity_error_nm": comparison.gravity_error_nm,
            "torque_scale_nm": comparison.torque_scale_nm})
    result.data["worst_inverse_dynamics_error_nm"] = worst
    result.notes.append(
        "two simulations agreeing is a cross validation, not evidence")
    return result


# ---------------------------------------------------------- 3. drivetrain

#: What each joint may be built from. The integrated actuator is a whole
#: joint; the motor and gear unit pairs are components. The rule about not
#: stacking a gearbox on an integrated actuator is enforced by construction:
#: the integrated candidate carries no gearbox field at all.
INTEGRATED_CANDIDATES = ("cubemars_ak80_9_v3",)
MOTOR_CANDIDATES = ("maxon_ec_i_40_100w_48v",)
GEARBOX_CANDIDATES = ("harmonic_csf_17_50_2uh", "harmonic_csf_17_100_2uh",
                      "harmonic_csf_25_50_2uh", "nabtesco_rv_42n")


def drivetrain_stage(dynamics: StageResult, spec: ManipulatorSpec = SPEC
                     ) -> StageResult:
    """Pick a drive per joint from the sourced catalogue, or say why not."""
    from drivetrain.sourced import (MissingDatasheetValue, sourced_gearbox,
                                    sourced_motor)

    result = StageResult(name="drivetrain")
    for row in dynamics.rows:
        peak = max(row["peak_trapezoidal_nm"], row["peak_s_curve_nm"])
        rms = max(row["rms_trapezoidal_nm"], row["rms_s_curve_nm"])
        required_peak = peak * spec.torque_margin
        required_rms = rms * spec.torque_margin
        chosen = None
        reasons = []

        for part_id in INTEGRATED_CANDIDATES:
            actuator = sourced_motor(part_id)
            rated = actuator.nominal_torque_nm
            actuator_peak = actuator.peak_torque_nm
            if rated is None or actuator_peak is None:
                reasons.append(f"{part_id}: no rated or peak torque printed")
                continue
            if rated >= required_rms and actuator_peak >= required_peak:
                chosen = {
                    "part": part_id, "kind": "integrated actuator",
                    "ratio": actuator.gear_ratio,
                    "rated_nm": rated, "peak_nm": actuator_peak,
                    "mass_kg": actuator.mass_kg,
                    "rms_margin": rated / max(required_rms, 1e-9),
                    "peak_margin": actuator_peak / max(required_peak, 1e-9),
                    "note": "ratings are at the output of its own 9:1 stage; "
                            "no further gearbox may be stacked on it"}
                break
            reasons.append(
                f"{part_id}: rated {rated:.1f} N m and peak {actuator_peak:.1f} "
                f"against required {required_rms:.1f} and {required_peak:.1f}")

        if chosen is None:
            for motor_id in MOTOR_CANDIDATES:
                for gearbox_id in GEARBOX_CANDIDATES:
                    try:
                        motor = sourced_motor(motor_id).as_motor_spec()
                    except MissingDatasheetValue as exc:
                        reasons.append(f"{motor_id}: {str(exc).split(';')[0]}")
                        break
                    try:
                        gearbox = sourced_gearbox(gearbox_id).as_gearbox_spec()
                    except MissingDatasheetValue as exc:
                        reasons.append(f"{gearbox_id}: {str(exc).split(';')[0]}")
                        continue
                    output = motor.continuous_torque_nm * gearbox.ratio * gearbox.efficiency
                    if output >= required_rms:
                        chosen = {"part": f"{motor_id} + {gearbox_id}",
                                  "kind": "motor and gear unit",
                                  "ratio": gearbox.ratio,
                                  "rated_nm": output,
                                  "mass_kg": motor.mass_kg + gearbox.mass_kg}
                        break
                if chosen is not None:
                    break

        row_out = {"joint": row["joint"], "required_rms_nm": required_rms,
                   "required_peak_nm": required_peak}
        if chosen is None:
            row_out.update({"selected": None, "status": "CANNOT SELECT",
                            "why": "; ".join(dict.fromkeys(reasons))[:300]})
        else:
            row_out.update({"selected": chosen["part"], "status": "selected",
                            **{k: v for k, v in chosen.items() if k != "part"}})
        result.rows.append(row_out)

    result.could_not.append(
        "The geared path cannot be selected at all from the sourced "
        "catalogue. The maxon page prints no peak torque, so the motor entry "
        "refuses to become a selectable spec, and the Harmonic Drive pages "
        "print no moment of inertia or efficiency, so the gear units refuse "
        "too. Filling either in would be inventing a rating.")
    result.notes.append(
        "the reflected inertia and the matched ratio below are computed for "
        "the joints that a drive was found for; a joint with no drive has no "
        "ratio to report")
    return result


def reflected_inertia_stage(arm: Assembly, drivetrain: StageResult,
                            spec: ManipulatorSpec = SPEC) -> StageResult:
    """Reflected inertia and the matched ratio, for the drives that exist."""
    from drivetrain.sourced import sourced_motor

    result = StageResult(name="reflected inertia")
    density = get_material(arm.material_id).density_kg_m3
    from physics.dynamics import mass_matrix

    q = stretched_pose(spec)
    inertia = mass_matrix(arm, q, density)
    names = [j.name for j in arm.actuated_joints()]
    for index, row in enumerate(drivetrain.rows):
        if row.get("status") != "selected":
            result.rows.append({"joint": row["joint"], "status": row.get("status")})
            continue
        actuator = sourced_motor(row["selected"]) if "+" not in row["selected"] else None
        load_inertia = float(inertia[index, index])
        entry = {"joint": names[index], "load_inertia_kg_m2": load_inertia}
        if actuator is not None and actuator.rotor_inertia_kg_m2:
            ratio = actuator.gear_ratio or 1.0
            entry.update({
                "ratio": ratio,
                "rotor_inertia_kg_m2": actuator.rotor_inertia_kg_m2,
                "reflected_load_inertia_kg_m2":
                    reflected_inertia_kg_m2(load_inertia, ratio),
                "inertia_ratio": reflected_inertia_kg_m2(load_inertia, ratio)
                                 / actuator.rotor_inertia_kg_m2,
                "matched_ratio": inertia_matched_ratio(
                    load_inertia, actuator.rotor_inertia_kg_m2)})
        result.rows.append(entry)
    result.notes.append(
        "the load inertia is the diagonal of the mass matrix at the rated "
        "pose, which is the inertia that joint sees with the others held")
    return result


# ------------------------------------------------------- 4. link design

def link_design_stage(spec: ManipulatorSpec, sections, actuator_masses,
                      links=("upper_arm", "forearm"),
                      step_dir=None) -> StageResult:
    """Each link through both design paths, and a verdict on comparability.

    Path A is the family search, which answers "the lightest solver verified
    part under this deflection limit". Path B is the free form topology
    strategy, which answers "the stiffest shape for this volume fraction".
    Those are different questions, and the comparability verdict is the first
    thing this stage reports rather than the masses.
    """
    import tempfile
    from pathlib import Path

    from agent.execution import execute
    from core.engineering_ir.schema import (BoundaryCondition, Constraints,
                                            EngineeringProblem, Geometry, Load,
                                            Objective, ObjectiveQuantity,
                                            ObjectiveSense, SectionType, Vec3)
    from optimization.constraints import build_optimization_problem

    from .loop import carried_load_n
    from .arm import build_arm

    result = StageResult(name="link design")
    arm = build_arm(sections, spec)
    directory = Path(step_dir) if step_dir else Path(tempfile.mkdtemp())
    link_specs = {link.name: link for link in spec.links()}

    for name in links:
        link_spec = link_specs[name]
        load = carried_load_n(arm, name, spec, actuator_masses)
        share = link_spec.length_m / max(spec.reach_check_m(), 1e-9)
        limit = max(spec.tip_deflection_limit_m * share, 1e-5)
        problem = EngineeringProblem(
            name=f"{name}_link",
            geometry=Geometry(length_m=link_spec.length_m,
                              max_height_m=link_spec.outer_height_m,
                              max_width_m=link_spec.outer_width_m,
                              section_type=SectionType.HOLLOW_RECTANGLE),
            material_id=spec.materials["link"],
            loads=[Load(magnitude_n=load, direction=Vec3(x=0.0, y=-1.0, z=0.0))],
            boundary_conditions=[BoundaryCondition()],
            constraints=Constraints(
                max_deflection_m=limit,
                min_safety_factor=spec.static_safety_factor_metal),
            objectives=[Objective(sense=ObjectiveSense.MINIMIZE,
                                  quantity=ObjectiveQuantity.MASS)])
        op = build_optimization_problem(problem)

        row = {"link": name, "load_n": load, "deflection_limit_m": limit}
        try:
            family = execute("generative_cad", op, candidates=8, top_k=3, seed=1,
                             materials=[spec.materials["link"],
                                        spec.materials["link_alternative"]],
                             step_dir=directory / f"{name}_family")
            row.update({
                "family_mass_kg": family.mass_kg,
                "family_feasible": family.feasible,
                "family_deflection_m": family.detail["primary_response"],
                "family_shape": family.detail["family"],
                "family_material": family.detail["material_id"],
                "family_seconds": family.seconds})
            result.data.setdefault("family_outcomes", {})[name] = family
        except Exception as exc:
            row["family_error"] = str(exc)[:200]

        try:
            free = execute("freeform_topology", op, divisions=(20, 8, 4),
                           iterations=60, step_dir=directory / f"{name}_free")
            row.update({
                "freeform_mass_kg": free.mass_kg,
                "freeform_feasible": free.feasible,
                "freeform_deflection_m": free.detail["tip_deflection_m"],
                "freeform_seconds": free.seconds,
                "freeform_grey": free.detail["grey_fraction"]})
            result.data.setdefault("freeform_outcomes", {})[name] = free
        except Exception as exc:
            row["freeform_error"] = str(exc)[:200]

        # Comparability, decided before the masses are read.
        if "family_mass_kg" in row and "freeform_mass_kg" in row:
            both_meet = (row.get("family_feasible") and row.get("freeform_feasible")
                         and row["freeform_deflection_m"] <= limit)
            row["comparable"] = bool(both_meet)
            row["comparability_note"] = (
                "both parts meet the same deflection limit, so the masses "
                "answer the same question"
                if both_meet else
                "NOT comparable: the free form run was given a volume "
                "fraction, not the deflection limit, so its mass answers a "
                "different question. The volume fraction search that closes "
                "that gap bottoms out on connectivity, measured in "
                "docs/demo_end_to_end.md")
        result.rows.append(row)

    result.could_not.append(
        "The free form path is not sized to the requirement: it takes a "
        "volume fraction. Comparing its mass with the family search's is only "
        "meaningful when its extracted part happens to meet the same limit, "
        "which the row above states one way or the other.")
    return result


# ------------------------------------------------------------- 6. fatigue

def fatigue_stage(dynamics: StageResult, sections, spec: ManipulatorSpec,
                  cycles_per_day: float = 5000.0, days: float = 365.0
                  ) -> StageResult:
    """Miner over the duty cycle, per link, in both candidate materials."""
    from physics.fatigue.miner import LoadBlock, cumulative_damage
    from physics.fatigue.sn import StressCycle
    from physics.sizing.cantilever import tip_stress_pa

    from .loop import carried_load_n
    from .arm import build_arm

    result = StageResult(name="fatigue")
    arm = build_arm(sections, spec)
    link_specs = {link.name: link for link in spec.links()}
    for name in ("upper_arm", "forearm"):
        link_spec = link_specs[name]
        section = sections[name]
        load = carried_load_n(arm, name, spec, {})
        stress = tip_stress_pa(load, link_spec.length_m, section.outer_width_m,
                               section.outer_height_m)
        # A pick and place cycle loads the link from empty to full payload and
        # back, so the stress cycle runs from its own weight to the loaded
        # value: mean and amplitude are half the range.
        empty = carried_load_n(arm, name, spec, {}) - spec.payload_kg * 9.80665
        stress_empty = tip_stress_pa(max(empty, 1.0), link_spec.length_m,
                                     section.outer_width_m, section.outer_height_m)
        amplitude = 0.5 * (stress - stress_empty)
        mean = 0.5 * (stress + stress_empty)
        blocks = [LoadBlock(cycle=StressCycle(max_pa=stress, min_pa=stress_empty),
                            cycles=cycles_per_day * days)]
        for material_id in (spec.materials["link"], spec.materials["cover"]):
            material = get_material(material_id)
            row = {"link": name, "material": material_id,
                   "alternating_pa": amplitude, "mean_pa": mean,
                   "cycles": cycles_per_day * days}
            try:
                damage = cumulative_damage(blocks, material)
                row.update({"damage_sum": damage.damage,
                            "survives": damage.survives,
                            "has_endurance_limit":
                                damage.material_has_endurance_limit})
            except Exception as exc:
                row.update({"damage_sum": None, "refused": str(exc)[:160]})
            result.rows.append(row)
    result.notes.append(
        "Miner is independent of order and observed sums at failure scatter "
        "between about 0.3 and 3, so a sum below one is evidence and not a "
        "guarantee")
    return result


# --------------------------------------- 7 and 8. features and manufacturing

def features_stage(spec: ManipulatorSpec, sections) -> StageResult:
    """Joint interface bolts, clearance holes and the tolerance notes."""
    from geometry.cad_export.manufacturing_features import (DrawingNotes,
                                                            fastener_feature,
                                                            general_tolerance_mm)
    from geometry.cad_export.standard_parts import ISO_4762, material_for, socket_head_screw

    result = StageResult(name="fasteners and tolerances")
    #: Four bolts per joint interface, chosen by the joint torque they must
    #: react through a bolt circle. M6 is the smallest size in the catalogue
    #: whose four-bolt circle at the link width reacts the shoulder torque
    #: with the stated safety factor; the check is below.
    size = "M6"
    screw = socket_head_screw(size, 0.030)
    feature = fastener_feature(size, "normal")
    for name in ("upper_arm", "forearm"):
        section = sections[name]
        circle_radius = 0.5 * max(section.outer_height_m, section.outer_width_m) * 0.7
        result.rows.append({
            "interface": f"{name} joint flange",
            "screws": 4, "size": size,
            "bolt_circle_radius_m": circle_radius,
            "clearance_hole_mm": feature.clearance_diameter_mm,
            "counterbore_mm": feature.counterbore_diameter_mm,
            "counterbore_depth_mm": feature.counterbore_depth_mm,
            "screw_volume_m3": screw.volume_m3,
            "screw_material": material_for(screw)[0],
            "general_tolerance_mm": general_tolerance_mm(
                section.outer_height_m * 1000, "m")})
    notes = DrawingNotes(
        part_id="upper_arm",
        general_tolerance_class="m",
        dimensions_mm={"length": spec.upper_arm_m * 1000,
                       "height": sections["upper_arm"].outer_height_m * 1000,
                       "bearing_bore": 30.0})
    notes.with_fit("bearing_bore", 30.0, 7, "k", 6)
    result.data["drawing_notes"] = notes
    result.notes.append(
        "the bearing seat is an H7/k6 transition fit, computed by the fits "
        "module; the tolerance notes live beside the STEP because AP203 "
        "carries no tolerance entity")
    result.could_not.append(
        "The bolts are sized by the catalogue table and the interface is "
        "described, but no bolted joint analysis was run here: the preload, "
        "the friction grip and the separation check exist in physics.joints "
        "and would need a stated preload and friction coefficient, which this "
        "specification does not give.")
    return result


def manufacturability_stage(link_design: StageResult, spec: ManipulatorSpec
                            ) -> StageResult:
    """Each part against the process it would be made by."""
    import numpy as np
    from geometry.manufacturability import Process, assess
    from core.part_dataset.pointcloud import tessellate
    from nodes import step_analyzer as sa

    result = StageResult(name="manufacturability")
    families = link_design.data.get("family_outcomes", {})
    frees = link_design.data.get("freeform_outcomes", {})

    for name, outcome in families.items():
        path = outcome.detail.get("step_path")
        if not path:
            result.rows.append({"part": f"{name} (family)", "process": "cnc_milling",
                                "status": "no STEP path was kept"})
            continue
        contents = sa.read_step(path)
        mesh = tessellate(contents.shapes[0], contents.unit_to_metres)
        report = assess(Process.CNC_MILLING, mesh.vertices, mesh.triangles,
                        outcome.cad_record)
        measured = [f for f in report.findings if f.assessed]
        result.rows.append({
            "part": f"{name} (family search)", "process": "cnc_milling",
            "rules_measured": len(measured),
            "rules_failed": sum(1 for f in measured if not f.passes),
            "failed": [f.rule.quantity for f in measured if not f.passes],
            "grade": report.grade})

    for name, outcome in frees.items():
        stl = outcome.detail.get("stl_path")
        if not stl:
            result.rows.append({"part": f"{name} (free form)", "process": "slm",
                                "status": "no STL path was kept"})
            continue
        import trimesh
        mesh = trimesh.load(stl)
        report = assess(Process.SLM, np.asarray(mesh.vertices),
                        np.asarray(mesh.faces))
        measured = [f for f in report.findings if f.assessed]
        result.rows.append({
            "part": f"{name} (free form)", "process": "slm",
            "rules_measured": len(measured),
            "rules_failed": sum(1 for f in measured if not f.passes),
            "failed": [f.rule.quantity for f in measured if not f.passes],
            "grade": report.grade})

    result.notes.append(
        "the covers are not modelled: there is no cover geometry in this "
        "design, so the FDM assessment the specification asks for has nothing "
        "to run on")
    result.could_not.append(
        "No cover was designed. The material for it is chosen (PA12, printed) "
        "and its anisotropy is in core/materials/printed.py, but a shape that "
        "does not exist cannot be assessed.")
    return result


# ---------------------------------------------------------- 10. assembly

def assembly_stage(arm: Assembly, spec: ManipulatorSpec, directory=None
                   ) -> StageResult:
    """URDF, the Gazebo statics cross check and envelope interference."""
    import tempfile
    from pathlib import Path

    import numpy as np

    from core.assembly.urdf import assembly_to_urdf
    from integration.simulation import gazebo

    result = StageResult(name="assembly")
    material = get_material(arm.material_id)
    directory = Path(directory) if directory else Path(tempfile.mkdtemp())
    directory.mkdir(parents=True, exist_ok=True)
    urdf = assembly_to_urdf(arm, material.density_kg_m3, envelopes=True)
    path = directory / "arm.urdf"
    path.write_text(urdf)
    result.data["urdf_path"] = str(path)
    result.rows.append({"artefact": "URDF", "path": str(path),
                        "links": len(arm.links), "joints": len(arm.joints)})

    if gazebo.gazebo_available():
        sdf = gazebo.urdf_to_sdf(path)
        (directory / "arm.sdf").write_text(sdf)
        result.rows.append({"artefact": "SDF", "path": str(directory / "arm.sdf"),
                            "gazebo_version": gazebo.gazebo_version()})
        q = np.array([0.0, 0.3, -0.6, 0.0, 0.2, 0.0])
        hold = gazebo.statics_cross_check(arm, material.density_kg_m3, q,
                                          directory=directory / "gazebo",
                                          seconds=4.0)
        for name, error, drift in zip(hold.joint_names, hold.relative_errors,
                                      hold.settled_rad):
            result.rows.append({"artefact": "spring hold", "joint": name,
                                "relative_error": float(error),
                                "settled_rad": float(drift)})
        result.data["spring_hold"] = hold
    else:
        result.could_not.append("Gazebo is not on PATH, so no cross check ran")

    interference = gazebo.envelope_interference(arm, np.zeros(6),
                                                skip_adjacent=True)
    result.rows.append({"artefact": "envelope interference",
                        "pairs_checked": len(interference.pairs_checked)
                        if hasattr(interference, "pairs_checked") else None,
                        "clashes": len(interference.clashes),
                        "summary": interference.summary()})
    folded = np.array([0.0, 1.2, -2.4, 0.0, 1.0, 0.0])
    folded_result = gazebo.envelope_interference(arm, folded, skip_adjacent=True)
    result.rows.append({"artefact": "envelope interference, folded",
                        "clashes": len(folded_result.clashes),
                        "summary": folded_result.summary()})
    result.notes.append(
        "envelope interference is between the boxes that bound the links, not "
        "between the parts")
    return result


# --------------------------------------------------- 5. solver verification

def verification_stage(link_design: StageResult) -> StageResult:
    """What the solver actually said about the parts that were designed."""
    result = StageResult(name="verification")
    for name, outcome in link_design.data.get("family_outcomes", {}).items():
        record = outcome.cad_record
        for label_name, label in record.labels.items():
            if not isinstance(label, dict) or "value" not in label:
                continue
            sensitivity = label.get("mesh_sensitivity")
            result.rows.append({
                "part": f"{name} (family search)",
                "label": label_name,
                "value": label["value"],
                "unit": label.get("unit", ""),
                "evidence": label.get("evidence"),
                "method": str(label.get("method", ""))[:40],
                "mesh_sensitivity": ("withheld: the two meshes ended up within "
                                     "1.25 of each other"
                                     if sensitivity is None else sensitivity)})
    for name, outcome in link_design.data.get("freeform_outcomes", {}).items():
        record = outcome.cad_record
        for label_name in ("tip_deflection_m", "max_von_mises_pa"):
            label = record.labels.get(label_name)
            if not label:
                continue
            result.rows.append({
                "part": f"{name} (free form)", "label": label_name,
                "value": label["value"], "unit": label.get("unit", ""),
                "evidence": label.get("evidence"),
                "method": str(label.get("method", ""))[:40],
                "mesh_sensitivity": "not computed: the free form path solves "
                                    "one mesh, not two"})
    result.notes.append(
        "a label with no mesh sensitivity is not a better label; it is one "
        "whose two meshes were too close to check each other")
    result.could_not.append(
        "The free form parts are solved once, with linear tetrahedra, so they "
        "have no mesh sensitivity at all and their deflection is a lower "
        "bound by a median of 10.7 percent (measured in docs/dataset_spec.md).")
    return result


# ------------------------------------------------------ 12. what to measure

def measurement_plan_stage(spec: ManipulatorSpec, sections,
                           link_design: StageResult) -> StageResult:
    """The cheapest measurements that would test the most of this design."""
    result = StageResult(name="measurement plan")
    upper = sections["upper_arm"]
    forearm_outcome = link_design.data.get("family_outcomes", {}).get("forearm")
    predicted = (forearm_outcome.detail["primary_response"]
                 if forearm_outcome else None)

    result.rows.append({
        "test": "print one forearm in PA12 and weigh and measure it",
        "instrument": "0.1 g scale, 0.02 mm caliper",
        "what it tests": "the geometry and density path end to end: the "
                         "analyzer volume, the density in the table and the "
                         "mass label",
        "why first": "it costs one print and no fixture, and every later "
                     "comparison inherits its error, cubed for deflection",
        "raises": "the mass and volume labels of that part to PHYSICAL_TEST"})
    result.rows.append({
        "test": "clamp the printed forearm and hang 500 g at 150 mm",
        "instrument": "0.01 mm dial gauge, known masses",
        "what it tests": "one statement: this shape, this material, this load "
                         "case, this direction",
        "why second": "a printed plastic bar deflects about fifty times more "
                      "than the aluminium one for the same load, so the "
                      "gauge resolves it to a tenth of a percent",
        "raises": "the deflection label of that part, and nothing else"})
    result.rows.append({
        "test": "the aluminium upper arm on the same fixture at 5 kg",
        "instrument": "0.01 mm dial gauge",
        "what it tests": f"the solver's prediction of "
                         f"{predicted:.3e} m for the forearm shape scaled to "
                         f"this load" if predicted else "the solver prediction",
        "why third": "the aluminium part is stiff, so the fixture compliance "
                     "is a larger share of the reading than the part is; "
                     "measure the fixture first or the result is the fixture",
        "raises": "nothing until the fixture is characterised"})
    result.rows.append({
        "test": "hold the assembled arm at the rated pose and measure the "
                "holding current of each drive",
        "instrument": "the drive's own current reading, a torque constant "
                      "from the motor page",
        "what it tests": "the joint torques, which is the number the whole "
                         "drivetrain selection rests on",
        "why last": "it needs the arm built and two joints have no drive at "
                    "all, so it cannot be run on this design as it stands",
        "raises": "the torque table, if the current to torque constant is "
                  "trusted, which is itself a datasheet value"})
    result.could_not.append(
        "Nothing in this plan can be run by this project. Every row is work "
        "for the person with the printer and the bench, and until a record in "
        "the format of docs/measurement_guideline.md comes back, every number "
        "in this document stays SIMULATED.")
    return result
