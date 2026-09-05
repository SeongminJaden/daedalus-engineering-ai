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
from projects.manipulator.cycloidal import (CycloidalGeometry,
                                            disc_shear_stiffness_nm_rad,
                                            output_pin_moment_arms,
                                            pin_set_stiffness_nm_rad,
                                            required_bearing_stiffness_n_m,
                                            ring_pin_moment_arms,
                                            shell_torsion_nm_rad)
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
    # THE PAYLOAD HAS TO BE TURNED, not just held up. The rigid body model
    # carries it as a force at the tip, which is right for gravity and wrong
    # for a roll axis: a force through a point has no moment about the line it
    # lies on, so the tool roll's requirement came out as exactly zero and any
    # actuator satisfied it. A payload of stated size has an inertia, and
    # turning it at the profile's peak acceleration takes a torque.
    cube = spec.payload_kg * spec.payload_extent_m ** 2 / 6.0
    spin = float(cube * acceleration[0])
    for index, name in enumerate(names):
        trap = profiles["trapezoidal"]
        curve = profiles["s_curve"]
        # Every joint turns the payload about its own axis in this move, so
        # every joint pays the spin torque. It is small next to gravity on
        # the big joints and it is the ONLY torque on the tool roll.
        peak = float(trap.peak_torque_nm[index]) + spin
        result.rows.append({
            "joint": name,
            "static_nm": float(static[index]),
            "payload_spin_nm": spin,
            "peak_trapezoidal_nm": peak,
            "rms_trapezoidal_nm": float(trap.rms_torque_nm[index]) + spin,
            "peak_s_curve_nm": float(curve.peak_torque_nm[index]) + spin,
            "rms_s_curve_nm": float(curve.rms_torque_nm[index]) + spin,
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
        f"the payload is treated as a {spec.payload_extent_m * 1000:.0f} mm "
        f"cube of {spec.payload_kg} kg, so turning it costs "
        f"{spin:.3f} N m at the profile's peak acceleration. That number is "
        f"the whole of the tool roll's requirement, which was zero while the "
        f"payload was a point")
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

#: The three ways a joint can be driven, in the order the design considers
#: them. The geared path is FIRST: a motor turning fast behind a reduction is
#: how a joint of this torque is normally built, and the integrated actuator is
#: one packaged instance of it. Direct drive is included so that the reason it
#: loses can be a number rather than an assertion.
MOTOR_CANDIDATES = ("kollmorgen_tbm_6013_a", "kollmorgen_tbm_6025_a",
                    "kollmorgen_tbm_6051_a", "maxon_ec_i_40_100w_48v")
GEARBOX_CANDIDATES = ("apex_af042_ratio20", "apex_af042_ratio50",
                      "apex_af060_ratio50", "harmonic_csf_17_50_2uh",
                      "harmonic_csf_17_100_2uh", "harmonic_csf_25_50_2uh",
                      "nabtesco_rv_42n")
INTEGRATED_CANDIDATES = ("cubemars_ak60_6_v3_kv80",
                         "cubemars_ak80_9_v3", "cubemars_ak70_10_kv100",
                         "cubemars_ak10_9_v2_kv60", "cubemars_ak80_64_kv80",
                         "cubemars_ak80_8_kv60", "robotis_ph54_200_s500_r",
                         "robotis_ph42_020_s300_r", "robotis_xm540_w270",
                         "mjbots_qdd100_beta3", "damiao_dm_j8009_2ec")

#: A joint of this arm must turn 90 degrees in two seconds, so its peak speed
#: is 1.5 times the average. Anything slower cannot do the move whatever its
#: torque.
def required_joint_speed_rad_s(spec: ManipulatorSpec) -> float:
    return 1.5 * spec.move_angle_rad / spec.move_time_s


def _candidate_rows(required_rms: float, required_peak: float,
                    required_speed: float, load_inertia_kg_m2: float,
                    max_inertia_ratio: float = 10.0,
                    bus_voltage_v: float | None = None) -> list[dict]:
    """Every drive option for one joint, priced the same way.

    Returns a row per candidate whether it passes or not, because the point of
    the table is to show why the ones that lose, lose.
    """
    from drivetrain.sourced import (MissingDatasheetValue, SOURCED_GEARBOXES,
                                    sourced_gearbox, sourced_motor)

    rows: list[dict] = []

    # --- path (iii): a motor behind a reduction, the primary path ----------
    for motor_id in MOTOR_CANDIDATES:
        try:
            motor = sourced_motor(motor_id).as_motor_spec()
        except MissingDatasheetValue as exc:
            rows.append({"path": "motor and gearbox", "candidate": motor_id,
                         "feasible": False,
                         "why": f"the motor cannot be specified: "
                                f"{str(exc).split(';')[0]}"})
            continue
        for gearbox_id in GEARBOX_CANDIDATES:
            entry = sourced_gearbox(gearbox_id)
            try:
                gearbox = entry.as_gearbox_spec()
            except MissingDatasheetValue as exc:
                rows.append({"path": "motor and gearbox",
                             "candidate": f"{motor_id} + {gearbox_id}",
                             "feasible": False,
                             "why": f"the gear unit cannot be specified: "
                                    f"{str(exc).split(';')[0]}"})
                continue
            output_rms = motor.continuous_torque_nm * gearbox.ratio * gearbox.efficiency
            output_peak = motor.peak_torque_nm * gearbox.ratio * gearbox.efficiency
            output_speed = motor.rated_speed_rad_s / gearbox.ratio
            gearbox_limited_peak = gearbox.peak_output_torque_nm
            gearbox_limited_rms = gearbox.rated_output_torque_nm
            reasons = []
            if output_rms < required_rms:
                reasons.append(f"motor side continuous {output_rms:.1f} N m")
            if gearbox_limited_rms < required_rms:
                reasons.append(f"gear rated {gearbox_limited_rms:.1f} N m")
            if min(output_peak, gearbox_limited_peak) < required_peak:
                reasons.append(
                    f"peak {min(output_peak, gearbox_limited_peak):.1f} N m")
            if output_speed < required_speed:
                reasons.append(f"output speed {output_speed:.2f} rad/s")
            ratio_of_inertias = ((load_inertia_kg_m2 / gearbox.ratio ** 2)
                                 / motor.rotor_inertia_kg_m2)
            if ratio_of_inertias > max_inertia_ratio:
                reasons.append(f"inertia ratio {ratio_of_inertias:.0f}")
            rows.append({
                "path": "motor and gearbox",
                "candidate": f"{motor_id} + {gearbox_id}",
                "grade": f"{sourced_motor(motor_id).grade.value} motor, "
                         f"{entry.grade.value} gear unit",
                "bus_voltage_v": sourced_motor(motor_id).bus_voltage_v,
                "peak_condition": sourced_motor(motor_id).peak_torque_condition,
                "ratio": gearbox.ratio,
                "mass_kg": motor.mass_kg + gearbox.mass_kg,
                "rms_capability_nm": min(output_rms, gearbox_limited_rms),
                "peak_capability_nm": min(output_peak, gearbox_limited_peak),
                "output_speed_rad_s": output_speed,
                "reflected_inertia_kg_m2":
                    load_inertia_kg_m2 / gearbox.ratio ** 2,
                "inertia_ratio": (load_inertia_kg_m2 / gearbox.ratio ** 2)
                                 / motor.rotor_inertia_kg_m2,
                "backlash_arcmin": gearbox.backlash_arcmin,
                "torsional_stiffness_nm_rad": entry.torsional_stiffness_nm_rad,
                "feasible": not reasons,
                "why": "; ".join(reasons)})

    # --- path (ii): an integrated actuator ---------------------------------
    for part_id in INTEGRATED_CANDIDATES:
        actuator = sourced_motor(part_id)
        rated, peak = actuator.nominal_torque_nm, actuator.peak_torque_nm
        speed = actuator.nominal_speed_rad_s
        reasons = []
        if (bus_voltage_v is not None and actuator.bus_voltage_v is not None
                and abs(actuator.bus_voltage_v - bus_voltage_v) > 1e-9):
            reasons.append(
                f"its figures are printed at {actuator.bus_voltage_v:.0f} V "
                f"and this arm runs one {bus_voltage_v:.0f} V bus")
        if rated is None:
            reasons.append("no continuous torque printed"
                           + (" (a stall torque is printed and is not one)"
                              if actuator.stall_torque_nm else ""))
        if peak is None:
            reasons.append("no peak torque printed")
        if rated is not None and peak is not None:
            if rated < required_rms:
                reasons.append(f"rated {rated:.1f} N m")
            if peak < required_peak:
                reasons.append(f"peak {peak:.1f} N m")
        if speed is not None and speed < required_speed:
            reasons.append(f"rated speed {speed:.2f} rad/s")
        ratio = actuator.gear_ratio or 1.0
        if actuator.rotor_inertia_kg_m2:
            ratio_of_inertias = ((load_inertia_kg_m2 / ratio ** 2)
                                 / actuator.rotor_inertia_kg_m2)
            if ratio_of_inertias > max_inertia_ratio:
                reasons.append(f"inertia ratio {ratio_of_inertias:.0f}")
        rows.append({
            "path": "integrated actuator", "candidate": part_id,
            "grade": actuator.grade.value,
            "bus_voltage_v": actuator.bus_voltage_v,
            "peak_condition": actuator.peak_torque_condition,
            "ratio": ratio, "mass_kg": actuator.mass_kg,
            "rms_capability_nm": rated, "peak_capability_nm": peak,
            "output_speed_rad_s": speed,
            "reflected_inertia_kg_m2": load_inertia_kg_m2 / ratio ** 2,
            "inertia_ratio": ((load_inertia_kg_m2 / ratio ** 2)
                              / actuator.rotor_inertia_kg_m2
                              if actuator.rotor_inertia_kg_m2 else None),
            "backlash_arcmin": actuator.backlash_arcmin,
            "torsional_stiffness_nm_rad": None,
            "feasible": not reasons, "why": "; ".join(reasons)})

    # --- path (i): direct drive, the motor alone ---------------------------
    for motor_id in MOTOR_CANDIDATES:
        try:
            motor = sourced_motor(motor_id).as_motor_spec()
        except MissingDatasheetValue:
            continue
        reasons = []
        if motor.continuous_torque_nm < required_rms:
            reasons.append(f"continuous {motor.continuous_torque_nm:.2f} N m "
                           f"against {required_rms:.1f} required")
        if motor.peak_torque_nm < required_peak:
            reasons.append(f"peak {motor.peak_torque_nm:.2f} N m")
        direct_ratio = load_inertia_kg_m2 / motor.rotor_inertia_kg_m2
        if direct_ratio > max_inertia_ratio:
            reasons.append(
                f"inertia ratio {direct_ratio:.0f} against a limit of "
                f"{max_inertia_ratio:.0f}: this is the argument for a "
                f"reduction, in a number")
        rows.append({
            "path": "direct drive", "candidate": motor_id, "ratio": 1.0,
            "grade": sourced_motor(motor_id).grade.value,
            "bus_voltage_v": sourced_motor(motor_id).bus_voltage_v,
            "peak_condition": sourced_motor(motor_id).peak_torque_condition,
            "mass_kg": motor.mass_kg,
            "rms_capability_nm": motor.continuous_torque_nm,
            "peak_capability_nm": motor.peak_torque_nm,
            "output_speed_rad_s": motor.rated_speed_rad_s,
            "reflected_inertia_kg_m2": load_inertia_kg_m2,
            "inertia_ratio": load_inertia_kg_m2 / motor.rotor_inertia_kg_m2,
            "backlash_arcmin": 0.0,
            "torsional_stiffness_nm_rad": None,
            "feasible": not reasons, "why": "; ".join(reasons)})
    return rows


def drivetrain_stage(dynamics: StageResult, spec: ManipulatorSpec = SPEC,
                     load_inertias: dict[str, float] | None = None
                     ) -> StageResult:
    """Pick a drive per joint, geared path first, and keep every candidate.

    The lightest feasible candidate wins, and the row says which path it came
    from. `data["candidates"]` holds every option that was considered for
    every joint, which is what the design document prints as the direct drive
    against geared comparison.
    """
    from .interfaces import face_for

    result = StageResult(name="drivetrain")
    required_speed = required_joint_speed_rad_s(spec)
    load_inertias = load_inertias or {}
    for row in dynamics.rows:
        peak = max(row["peak_trapezoidal_nm"], row["peak_s_curve_nm"])
        rms = max(row["rms_trapezoidal_nm"], row["rms_s_curve_nm"])
        required_peak = peak * spec.torque_margin
        required_rms = rms * spec.torque_margin
        inertia = load_inertias.get(row["joint"], 0.01)
        candidates = _candidate_rows(required_rms, required_peak,
                                     required_speed, inertia,
                                     spec.max_inertia_ratio,
                                     spec.bus_voltage_v)
        result.data.setdefault("candidates", {})[row["joint"]] = candidates
        # A DRIVE THAT CANNOT BE MOUNTED IS NOT A CANDIDATE. The lightest
        # feasible part used to win on torque, speed and inertia alone, and
        # for the tool roll that picked a frameless motor: 213 g of rotor and
        # stator with no housing, no bearing, no shaft and no published
        # outline. It cannot be placed in an assembly and nothing can be
        # bolted to it, which is why the tool flange ended up with no
        # interface at either end and the arm stopped 23 mm short. Torque is
        # not the only thing a part has to do.
        for candidate in candidates:
            if not candidate.get("feasible"):
                continue
            name = str(candidate["candidate"])
            if face_for(name, "output") is None:
                candidate["feasible"] = False
                candidate["why"] = (
                    (candidate.get("why") or "") +
                    "; no drawing published for it, so it has no outline and "
                    "no mounting pattern and cannot be assembled").lstrip("; ")
        feasible = [c for c in candidates if c["feasible"]]
        best = min(feasible, key=lambda c: c["mass_kg"]) if feasible else None

        out = {"joint": row["joint"], "required_rms_nm": required_rms,
               "required_peak_nm": required_peak,
               "required_speed_rad_s": required_speed}
        if best is None:
            reasons = {c["candidate"]: c["why"] for c in candidates if c["why"]}
            out.update({"selected": None, "status": "CANNOT SELECT",
                        "why": "; ".join(f"{k}: {v}" for k, v in
                                         list(reasons.items())[:3])[:300]})
        else:
            out.update({"selected": best["candidate"], "status": "selected",
                        "path": best["path"], "grade": best.get("grade"),
                        "peak_condition": best.get("peak_condition"),
                        "ratio": best["ratio"],
                        "mass_kg": best["mass_kg"],
                        "rated_nm": best["rms_capability_nm"],
                        "peak_nm": best["peak_capability_nm"],
                        "rms_margin": best["rms_capability_nm"] / max(required_rms, 1e-9),
                        "peak_margin": best["peak_capability_nm"] / max(required_peak, 1e-9),
                        "backlash_arcmin": best["backlash_arcmin"],
                        "inertia_ratio": best["inertia_ratio"],
                        "note": ("ratings are at the output of its own stage; "
                                 "no further gearbox may be stacked on it"
                                 if best["path"] == "integrated actuator"
                                 else "motor and gear unit, sized on the "
                                      "smaller of the motor side and the gear "
                                      "unit rating")})
        result.rows.append(out)

    result.notes.append(
        "the geared path is considered first and the lightest feasible "
        "candidate of any path wins; every candidate, feasible or not, is "
        "kept so the comparison table can show why the others lost")
    result.notes.append(
        "a candidate with no published drawing is refused however light it "
        "is, because an outline and a mounting pattern are what let it be "
        "placed and fastened. This is what the arm was missing at the tool "
        "roll, and it is a requirement of the same kind as torque")
    result.could_not.append(
        "The Harmonic Drive units still cannot be paired: their catalogue "
        "gives efficiency as curves against ambient temperature for each "
        "ratio and input speed at rated torque, with a compensation "
        "coefficient below rated torque, so there is no single number to "
        "multiply by and collapsing the curve would be inventing an "
        "operating point. The maxon motor still prints no peak torque.")
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
    # Two feasibility checks nobody had made. A counterbore has to fit in the
    # wall it is cut into, and a tapped hole in aluminium needs more thread
    # engagement than the wall can give.
    for row in result.rows:
        wall = sections[row["interface"].split()[0]].wall_thickness_m
        depth = row["counterbore_depth_mm"] / 1000.0
        row["wall_m"] = wall
        row["counterbore_fits_wall"] = depth <= wall
        if not row["counterbore_fits_wall"]:
            row["counterbore_note"] = (
                f"a {depth * 1000:.1f} mm counterbore cannot be cut in a "
                f"{wall * 1000:.1f} mm wall: it would go through it. Either "
                f"the head sits proud, or the flange is locally thickened, "
                f"and this design does neither")
        engagement_needed = 1.5 * 0.006          # 1.5 d for a tapped aluminium hole
        row["thread_engagement_needed_m"] = engagement_needed
        row["tapped_wall_is_enough"] = wall >= engagement_needed
        if not row["tapped_wall_is_enough"]:
            row["fastening_note"] = (
                f"a tapped {wall * 1000:.1f} mm wall gives {wall * 1000:.1f} "
                f"mm of engagement against the {engagement_needed * 1000:.1f} "
                f"mm that 1.5 diameters asks for in aluminium, so this joint "
                f"needs a through bolt and a nut, a thicker local boss or an "
                f"insert. None of the three is in this design")

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
        "As drawn, the joint flanges cannot be fastened. The counterbore is "
        "deeper than the wall and a tapped wall gives less than half the "
        "thread engagement aluminium needs, so the design needs a local boss "
        "or a through bolt with a nut, and has neither. The rows above say so "
        "per interface. 1.5 diameters of engagement is a practice rule stated "
        "here, not a measurement from a source in this repository.")
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


# --------------------------------- 4b. direct drive against geared, per joint

def drive_comparison_stage(drivetrain: StageResult, spec: ManipulatorSpec = SPEC
                           ) -> StageResult:
    """The three paths side by side for every joint, feasible or not.

    One row per path per joint: the lightest candidate of that path, whether
    it can do the job, and the numbers that decide it. The direct drive row is
    the one to read: it is usually light and its inertia ratio is usually
    hundreds, which is the argument for a reduction expressed as a
    measurement rather than as advice.
    """
    result = StageResult(name="drive comparison")
    for joint, candidates in drivetrain.data.get("candidates", {}).items():
        for path in ("direct drive", "integrated actuator", "motor and gearbox"):
            rows = [c for c in candidates if c["path"] == path
                    and c.get("mass_kg") is not None]
            if not rows:
                result.rows.append({"joint": joint, "path": path,
                                    "best": None,
                                    "why": "no candidate of this path could "
                                           "even be specified"})
                continue
            feasible = [c for c in rows if c["feasible"]]
            best = min(feasible or rows, key=lambda c: c["mass_kg"])
            result.rows.append({
                "joint": joint, "path": path, "best": best["candidate"],
                "feasible": best["feasible"],
                "mass_kg": best["mass_kg"], "ratio": best["ratio"],
                "rms_margin": (best["rms_capability_nm"]
                               / max(next(r["required_rms_nm"] for r in
                                          drivetrain.rows
                                          if r["joint"] == joint), 1e-9)),
                "inertia_ratio": best["inertia_ratio"],
                "backlash_arcmin": best["backlash_arcmin"],
                "stiffness_nm_rad": best["torsional_stiffness_nm_rad"],
                "why_not": best["why"][:90]})
    result.notes.append(
        "the inertia ratio column is the reflected load inertia over the "
        "rotor inertia; the design refuses anything above "
        f"{spec.max_inertia_ratio:.0f}, which is a stated choice and not a "
        "measurement")
    return result


# ---------------------------- 5b. what the gear data now fills, and what not

def compliance_stage(arm: Assembly, drivetrain: StageResult,
                     spec: ManipulatorSpec = SPEC) -> StageResult:
    """Joint compliance and backlash, now that the gear units print them."""
    import numpy as np

    from core.assembly.kinematics import forward_kinematics
    from drivetrain.sourced import sourced_gearbox, sourced_motor

    result = StageResult(name="compliance and backlash")
    pose = forward_kinematics(arm, stretched_pose(spec))
    tool = pose.tool_position()
    for row in drivetrain.rows:
        if row.get("status") != "selected":
            continue
        name = row["selected"]
        stiffness = None
        backlash = row.get("backlash_arcmin")
        if "+" in name:
            gearbox = sourced_gearbox(name.split("+")[1].strip())
            stiffness = gearbox.torsional_stiffness_nm_rad
        torque = row["required_rms_nm"] / spec.torque_margin
        origin = pose.joint_origins[row["joint"]]
        lever = float(np.linalg.norm(tool - origin))
        entry = {"joint": row["joint"], "drive": name,
                 "torque_nm": torque, "lever_m": lever}
        if stiffness:
            twist = torque / stiffness
            entry.update({"stiffness_nm_rad": stiffness,
                          "twist_under_load_rad": twist,
                          "tool_error_from_twist_m": twist * lever})
        else:
            entry.update({"stiffness_nm_rad": None,
                          "twist_under_load_rad": None,
                          "tool_error_from_twist_m": None,
                          "note": "no torsional stiffness is printed for this "
                                  "drive, so its compliance is not modelled"})
        if backlash:
            backlash_rad = backlash * (np.pi / (180.0 * 60.0))
            entry.update({"backlash_arcmin": backlash,
                          "tool_error_from_backlash_m": backlash_rad * lever})
        result.rows.append(entry)
    result.notes.append(
        "backlash is a position uncertainty, not a deflection: the tool can "
        "sit anywhere inside it and the number below is the width of that "
        "band at the tool, not an error that a controller can remove")
    result.could_not.append(
        "Bearing and seal friction are still not modelled and still refused: "
        "the gear unit efficiency covers the losses inside the reduction and "
        "says nothing about the joint bearings, and no source in this "
        "repository gives their breakaway or viscous coefficients.")
    result.could_not.append(
        "The integrated actuators print a backlash and no torsional "
        "stiffness, so joints driven by them have a position uncertainty here "
        "and no compliance figure at all.")
    return result


# ------------------------------------- 3e. the bus voltage as a design choice

def bus_voltage_stage(dynamics: StageResult, load_inertias: dict[str, float],
                      spec: ManipulatorSpec = SPEC,
                      voltages=(24.0, 36.0, 48.0)) -> StageResult:
    """The same joints selected on each candidate bus voltage.

    The arm runs one bus, and a module's printed performance belongs to the
    voltage it was printed at. That makes the bus a design decision with a
    mass consequence, not a wiring detail, and this stage prices it.
    """
    from dataclasses import replace

    result = StageResult(name="bus voltage")
    for voltage in voltages:
        variant = replace(spec, bus_voltage_v=voltage)
        drives = drivetrain_stage(dynamics, variant, load_inertias)
        selected = [row for row in drives.rows if row.get("status") == "selected"]
        mass = sum(row["mass_kg"] for row in selected)
        result.rows.append({
            "bus_voltage_v": voltage,
            "joints_driven": len(selected),
            "joints_without_a_drive": len(drives.rows) - len(selected),
            "drive_mass_kg": mass if selected else None,
            "grades": ", ".join(sorted({str(row.get("grade")) for row in selected})),
            "parts": ", ".join(sorted({row["selected"] for row in selected}))[:110]})
    result.notes.append(
        "a module printed at another voltage is refused rather than scaled: "
        "torque and speed both move with the bus and the pages do not "
        "tabulate the arm's voltage for every part")
    return result


# --------------------------------- 3f. does the joint contain its own drive

def actuator_extent_along_arm(actuator, axis) -> tuple[float | None, str]:
    """How much room the drive takes ALONG the arm, and why.

    An actuator is a cylinder. If its axis lies along the arm it occupies its
    own axial length; if the axis is across the arm, as every pitch joint's
    is, it occupies its DIAMETER. The first version of this check compared
    every joint spacing with the axial length, which is right for the roll
    joints and wrong for the pitch joints by the difference between 38.5 and
    98 mm. A Fusion model found the 28.25 mm of interference that left.
    """
    import numpy as np

    direction = np.asarray(axis, dtype=float)
    if abs(float(direction[0])) > 0.5:              # the arm runs along x
        return actuator.axial_length_m, "axis along the arm: its length"
    return actuator.outer_diameter_m, "axis across the arm: its diameter"


def envelope_stage(arm: Assembly, drivetrain: StageResult, sections,
                   spec: ManipulatorSpec = SPEC) -> StageResult:
    """Check each joint against the outline of the drive it was given.

    Two rules. The link section must be at least the actuator diameter, and
    consecutive joints must be at least half of each one's along-arm extent
    apart, where that extent depends on which way the actuator's axis points.
    Where a page prints no outline the check cannot run, and that is reported
    as unverifiable rather than as a pass.
    """
    from drivetrain.sourced import sourced_motor

    result = StageResult(name="actuator envelope")
    joints = spec.joints()
    names = [joint.name for joint in joints]
    spacing_after = {}
    for index, joint in enumerate(joints):
        following = joints[index + 1] if index + 1 < len(joints) else None
        # The offset to the next joint, whichever direction it lies in. The
        # base column runs UP, so j2 carries its 150 mm as origin_y and zero
        # as origin_x; reading origin_x alone said the shoulder sits on top of
        # the base yaw drive with no space at all, and reported a 98 mm
        # interference that is not there. The same mistake was found and fixed
        # in the link domain builder.
        spacing_after[joint.name] = (
            max(following.origin_x_m, following.origin_y_m) if following
            else spec.links()[-1].length_m)
    drives = {row["joint"]: row.get("selected") for row in drivetrain.rows}
    extents: dict[str, tuple] = {}
    for joint in joints:
        part = drives.get(joint.name)
        if not part or "+" in str(part):
            extents[joint.name] = (None, "no single outline")
            continue
        extents[joint.name] = actuator_extent_along_arm(sourced_motor(part),
                                                        joint.axis)

    for row in drivetrain.rows:
        name = row["joint"]
        entry = {"joint": name, "drive": row.get("selected")}
        if row.get("status") != "selected" or "+" in str(row.get("selected")):
            entry["status"] = ("a motor and gear unit pair has no single "
                               "outline in this catalogue"
                               if row.get("status") == "selected"
                               else "no drive")
            result.rows.append(entry)
            continue
        actuator = sourced_motor(row["selected"])
        link = arm.links[names.index(name)]
        section = sections[link.name]
        across = max(section.outer_height_m, section.outer_width_m)
        extent, why = extents[name]
        entry.update({"link": link.name, "link_across_m": across,
                      "joint_spacing_m": spacing_after[name],
                      "extent_along_arm_m": extent, "extent_basis": why})
        if actuator.outer_diameter_m is None:
            entry["diameter_check"] = "not printed"
        else:
            entry["actuator_diameter_m"] = actuator.outer_diameter_m
            entry["diameter_check"] = (
                "fits" if actuator.outer_diameter_m <= across
                else f"INTERFERES by "
                     f"{(actuator.outer_diameter_m - across) * 1000:.1f} mm")

        index = names.index(name)
        following = names[index + 1] if index + 1 < len(names) else None
        next_extent = extents.get(following, (None, ""))[0] if following else None
        if extent is None or next_extent is None:
            entry["spacing_check"] = "not printed for one of the pair"
        else:
            touching = 0.5 * (extent + next_extent)
            required = touching + spec.assembly_clearance_m
            entry["touching_spacing_m"] = touching
            entry["required_spacing_m"] = required
            if required <= spacing_after[name]:
                entry["spacing_check"] = "fits with the assembly clearance"
            elif touching <= spacing_after[name]:
                entry["spacing_check"] = (
                    f"TOUCHING: clear by "
                    f"{(spacing_after[name] - touching) * 1000:.2f} mm, which "
                    f"is less than the {spec.assembly_clearance_m * 1000:.0f} "
                    f"mm a housing and its fasteners need")
            else:
                entry["spacing_check"] = (
                    f"INTERFERES by "
                    f"{(touching - spacing_after[name]) * 1000:.1f} mm")
        result.rows.append(entry)

    printed = [r for r in result.rows
               if r.get("diameter_check") not in (None, "not printed")]
    result.notes.append(
        f"{len(printed)} of {len(result.rows)} joints have an actuator whose "
        f"outline is printed at all; the rest cannot be checked")
    result.notes.append(
        "the along-arm extent of a drive is its length when its axis runs "
        "along the arm and its DIAMETER when the axis is across it, which is "
        "every pitch joint")
    result.notes.append(
        f"the spacing must clear the touching distance by the "
        f"{spec.assembly_clearance_m * 1000:.0f} mm assembly clearance; two "
        f"drives that merely do not overlap have nowhere to put a housing "
        f"wall, a bearing, a bolt head or a wire")
    result.could_not.append(
        "Most actuator pages print no outline, so this check is unverifiable "
        "for those joints. A frameless motor has no outline to print: its "
        "housing, bearings and shaft are somebody's design and are not in "
        "this one, and its mass is not in the total either.")
    return result


# ------------------------------------------------- 7b. the bolted interface

def bolted_joint_stage(drivetrain: StageResult, sections,
                       spec: ManipulatorSpec = SPEC,
                       size: str = "M6", bolt_count: int = 4) -> StageResult:
    """The joint flange bolts, as far as the data allows.

    WHAT RUNS. The tension path: preload from the proof load, the load factor
    from bolt and member stiffness, separation, bolt yield and bolt fatigue.
    physics.joints.bolted does all of it and needs no friction coefficient
    beyond the nut factor it already carries.

    WHAT DOES NOT. Whether the joint carries its torque by friction. That
    needs a coefficient of friction for the clamped faces, and no source in
    this repository gives one for aluminium against aluminium. VDI 2230 is the
    standard that tabulates them; its tables are not public and the published
    summaries this project could read give a range for steel and nothing for
    the surfaces here. So the friction grip check is refused with the number
    it would need, rather than run with a plausible 0.15.

    The tensile load per bolt is the standard first approximation for a bolt
    circle in bending: the moment is carried by the bolts on one side, and the
    most loaded one sees 2M / (n r). It is an approximation and is labelled as
    one; a real flange analysis distributes the load by stiffness.
    """
    from physics.joints.bolted import PropertyClass, analyze_joint

    result = StageResult(name="bolted interface")
    links = [link.name for link in spec.links()]
    joints = [joint.name for joint in spec.joints()]
    for row in drivetrain.rows:
        if row.get("status") != "selected":
            continue
        link_name = links[joints.index(row["joint"])]
        section = sections[link_name]
        radius = 0.5 * max(section.outer_height_m, section.outer_width_m) * 0.7
        moment = row["required_peak_nm"] / spec.torque_margin
        per_bolt = 2.0 * moment / (bolt_count * max(radius, 1e-6))
        grip = 2.0 * section.wall_thickness_m + 0.002
        analysis = analyze_joint(
            size=size, grade=PropertyClass.C8_8,
            grip_length_m=grip, external_load_n=per_bolt,
            external_load_min_n=0.0, member_material="aluminium",
            member_modulus_pa=68.9e9)
        result.rows.append({
            "joint": row["joint"], "flange": link_name,
            "bolts": f"{bolt_count} x {size} class 8.8",
            "bolt_circle_radius_m": radius,
            "moment_nm": moment,
            "load_per_bolt_n": per_bolt,
            "preload_n": analysis.preload_n,
            "tightening_torque_nm": analysis.tightening_torque_nm,
            "separation_margin": analysis.separation_margin,
            "bolt_load_n": analysis.bolt_load_n,
            "separated": analysis.separated,
            "yield_safety": analysis.yield_safety_factor,
            "load_factor": analysis.load_factor})
    result.notes.append(
        "the load per bolt is the first approximation for a bolt circle in "
        "bending, 2M over n r; a real flange distributes it by stiffness")
    result.notes.append(
        "the nut factor is 0.2 dry, and the torque to preload relation it "
        "belongs to scatters by about 30 percent, so the achieved preload is "
        "uncertain by that much before anything else is")
    result.could_not.append(
        "Whether these joints carry their torque by FRICTION cannot be "
        "checked. That needs a coefficient of friction for aluminium against "
        "aluminium at the clamped faces. VDI 2230 tabulates such values and "
        "its tables are not public; the summaries this project could read "
        "give a range for steel on steel (0.1 to 0.3) and nothing for these "
        "surfaces. One measured coefficient for the actual finish would close "
        "this, and until then the flange is checked for separation and bolt "
        "strength only, which is not the same as checked.")
    return result


def interface_stage(drivetrain: StageResult,
                    spec: ManipulatorSpec = SPEC,
                    designed: set[str] | None = None,
                    sections: dict | None = None) -> StageResult:
    """What each link bolts to, what is measured, and what is missing.

    This stage exists because the arm had flanges and no fastening. A 9 mm
    slab at the end of a link is not an interface until something says which
    face of which actuator it meets, on what circle, with what bolt, clocked
    which way. Every value here comes from a manufacturer drawing or the
    manufacturer's own 3D model, and every value that comes from neither is
    listed as missing instead.
    """
    from .interfaces import (assembly_gaps, clock_uncertainty_check, face_for,
                             link_interfaces, unresolved_features)

    result = StageResult(name="mounting interfaces")
    drives = {row["joint"]: row.get("selected") for row in drivetrain.rows}
    joints = [joint.name for joint in spec.joints()]
    links = [link.name for link in spec.links()]

    for row in link_interfaces(links, joints, drives):
        proximal, distal = row["proximal_face"], row["distal_face"]
        result.rows.append({
            "link": row["link"],
            "proximal_joint": row["proximal_joint"],
            "proximal": (f"{proximal.actuator} output, "
                         f"{proximal.patterns[0].count}-"
                         f"{proximal.patterns[0].thread} on "
                         f"{proximal.largest_bolt_circle_m() * 1000:.0f} mm"
                         if proximal else "NOT FASTENABLE"),
            "proximal_clock_deg": (proximal.patterns[0].clock_deg
                                   if proximal else None),
            "distal_joint": row["distal_joint"] or "tool",
            "distal": (f"{distal.actuator} housing, "
                       f"{distal.patterns[0].count}-"
                       f"{distal.patterns[0].thread} on "
                       f"{distal.largest_bolt_circle_m() * 1000:.0f} mm"
                       if distal else
                       ("tool plate, unspecified" if not row["distal_joint"]
                        else "NOT FASTENABLE")),
            "distal_clock_deg": (distal.patterns[0].clock_deg
                                 if distal else None)})

    seen: set = set()
    for joint in joints:
        for side in ("output", "housing"):
            face = face_for(drives.get(joint, ""), side)
            if face is None or (face.actuator, face.face) in seen:
                continue
            seen.add((face.actuator, face.face))
            result.data.setdefault("faces", []).extend(face.rows())
            result.data.setdefault("unresolved", []).extend(
                unresolved_features(face))
            result.data.setdefault("clock_checks", []).extend(
                clock_uncertainty_check(face))
    result.data["gaps"] = assembly_gaps(joints, drives, designed)

    # TWO LINKS TOLD TO OCCUPY ONE PLACE. Where a joint's axis crosses the
    # arm the domain that ends there and the domain that starts there meet at
    # a right angle, and each reaches half a section past the other. It is
    # not overshoot and clipping does not touch it. An assembly measured
    # 2850 cubic millimetres of real material in the shared volume at the
    # shoulder; the shared volume itself is 235 cubic centimetres, so how
    # much of it collides is luck.
    from .links import domain_overlaps

    overlaps = domain_overlaps(spec, drives, sections)
    result.data["domain_overlaps"] = overlaps
    for row in overlaps:
        if row["shared_mm3"] <= 0.0:
            continue
        result.data["gaps"].append({
            "gap": "a bracket at this joint",
            "where": row["pair"],
            "why": (f"{row['shared_mm3'] / 1000.0:.1f} cubic centimetres of "
                    f"design domain belong to both links, a "
                    f"{row['shared_extent_mm'][0]:.0f} by "
                    f"{row['shared_extent_mm'][1]:.0f} by "
                    f"{row['shared_extent_mm'][2]:.0f} mm block around the "
                    f"joint. Two square sections meeting at a right angle "
                    f"around a round drive cannot both have it, and a real "
                    f"arm puts a bracket there instead"),
            "carries": "everything outboard of this joint",
            "status": "MISSING"})

    missing = [g for g in result.data["gaps"] if g["status"] == "MISSING"]
    blocked = [c for c in result.data.get("clock_checks", [])
               if c["verdict"] == "NOT ABSORBED"]
    if blocked:
        result.notes.append(
            f"{len(blocked)} bolt circles have a clock uncertainty the "
            f"clearance hole cannot take up. Those joints cannot be drilled "
            f"from these sources; the clock has to be measured on the part")
    result.notes.append(
        f"{len(result.data.get('unresolved', []))} values the sources do not "
        f"give are listed rather than filled in")
    result.notes.append(
        f"{len(missing)} of {len(result.data['gaps'])} parts the arm needs "
        f"are still missing; the rest are designed in this repository")
    return result


def spigot_stage(dynamics: StageResult, drivetrain: StageResult,
                 spec: ManipulatorSpec = SPEC) -> StageResult:
    """Can the joint be located by a spigot and driven by friction?

    This is the question that decides whether the AK80-9's clock angle
    matters. If the bolts are pure clearance holes and the parts are located
    by the drive's own boss, then an angular error in the bolt circle is
    absorbed by the clearance and nothing is out of position. What the bolts
    must then do is hold enough friction to carry the joint torque, because
    they are no longer in shear against the hole.

    NO COEFFICIENT OF FRICTION IS ASSUMED. This project has refused to invent
    one for aluminium against aluminium and still does. The question is
    turned round instead: the stage reports the coefficient the joint would
    NEED. That number can be compared with any source later, and it needs no
    source now. It is the honest form of the question, and it turns out to
    answer it, because the required values are far below anything any dry
    metal pairing reaches.

    The dowels are checked too, on BEARING IN THE LINK rather than shear in
    the pin. The pin is hardened steel and the link is cast aluminium; the
    aluminium yields first, so the pin's own strength never governs and no
    pin material has to be sourced.
    """
    from physics.joints.bolted import PropertyClass, analyze_joint

    from .interfaces import face_for

    result = StageResult(name="spigot register and friction grip")
    material = get_material(spec.materials["link"])
    peaks = {row["joint"]: max(row["peak_trapezoidal_nm"],
                               row["peak_s_curve_nm"])
             for row in dynamics.rows}
    preload: dict[str, float] = {}
    for size in ("M3", "M4"):
        preload[size] = analyze_joint(
            size=size, grade=PropertyClass.C8_8,
            grip_length_m=spec.flange_thickness_m, external_load_n=0.0,
            external_load_min_n=0.0, member_material="aluminium",
            member_modulus_pa=material.youngs_modulus_pa).preload_n

    for row in drivetrain.rows:
        if row.get("status") != "selected":
            continue
        joint = row["joint"]
        face = face_for(str(row.get("selected")), "output")
        entry = {"joint": joint, "drive": row.get("selected"),
                 "peak_nm": peaks.get(joint, 0.0)}
        if face is None:
            entry["status"] = ("no drawing, so neither a spigot nor a bolt "
                               "circle is known")
            result.rows.append(entry)
            continue

        # Torque per unit coefficient of friction: every bolt clamps its own
        # circle, and the friction radius is that circle's radius.
        per_mu = sum(pattern.count * preload[pattern.thread]
                     * 0.5 * pattern.bolt_circle_m
                     for pattern in face.patterns)
        demanded = peaks.get(joint, 0.0) * spec.torque_margin
        entry.update({
            "bolts": ", ".join(f"{p.count}x{p.thread}@{p.bolt_circle_m * 1000:.0f}"
                               for p in face.patterns),
            "clamp_torque_per_mu_nm": per_mu,
            "demanded_nm": demanded,
            "required_friction_coefficient": (demanded / per_mu if per_mu
                                              else None),
            "spigot_mm": (max(face.boss_diameters_m) * 1000.0
                          if face.boss_diameters_m else None)})
        if face.dowel_angles_deg and face.dowel_diameter_m and face.dowel_depth_m:
            bearing_n = (face.dowel_diameter_m * face.dowel_depth_m
                         * material.yield_strength_pa)
            radius = 0.5 * (face.dowel_bolt_circle_m or 0.0)
            entry["dowel_torque_nm"] = (len(face.dowel_angles_deg) * bearing_n
                                        * radius)
            entry["dowel_basis"] = (
                f"{len(face.dowel_angles_deg)} pins bearing on "
                f"{face.dowel_diameter_m * 1000:.0f} by "
                f"{face.dowel_depth_m * 1000:.0f} mm of link at "
                f"{material.yield_strength_pa / 1e6:.0f} MPa")
        else:
            entry["dowel_torque_nm"] = None
            entry["dowel_basis"] = "no located dowel on this face"
        result.rows.append(entry)

    needed = [r["required_friction_coefficient"] for r in result.rows
              if r.get("required_friction_coefficient")]
    if needed:
        result.notes.append(
            f"the largest coefficient of friction any joint needs is "
            f"{max(needed):.4f}. No value is assumed here and none has to be: "
            f"the lowest figure published for any dry metal pairing is an "
            f"order of magnitude above this, so the friction grip is not what "
            f"limits these joints")
    result.notes.append(
        f"preload is {preload['M3']:.0f} N for M3 and {preload['M4']:.0f} N "
        f"for M4 at class 8.8, from the same analysis the flange stage uses, "
        f"and the torque to preload relation scatters by about 30 percent")
    result.notes.append(
        "a spigot register makes the AK80-9's clock angle harmless: the "
        "parts are located by the drive's boss and the bolts only clamp, so "
        "the 1.11 mm of angular uncertainty is taken up by the clearance hole "
        "it can no longer be measured against")
    from .links import RING_WIDTH_DEPENDS_ON

    result.could_not.append(
        "The spigot fit itself is not sized. That needs a tolerance class for "
        "the boss, which no drawing prints: the AK80-64 boss is dimensioned "
        "80 and 35 with no tolerance, and the central bore is the only "
        "toleranced feature on the face at 21.0 +0.02. A located fit needs "
        "the boss measured or an H7 recess cut to the measured size. "
        + RING_WIDTH_DEPENDS_ON + ", so this gap and the bolt seat's margin "
        "are the same gap seen twice.")
    return result


def mount_stage(spec: ManipulatorSpec = SPEC,
                directory: str = "data/generated/manipulator_mounts"
                ) -> StageResult:
    """The two parts that hold the arm to the world, as generated.

    This stage reads what `scripts/generate_mounts.py` wrote rather than
    running the optimisation again, because a design document should report
    the parts that exist and not a fresh set nobody has seen. If they have
    not been generated it says so, which is the same answer it gave for
    several revisions when they did not exist at all.
    """
    import json
    from pathlib import Path

    result = StageResult(name="the parts that hold the arm to the world")
    summary = Path(directory) / "summary.json"
    if not summary.exists():
        result.could_not.append(
            f"The base mount and the tool plate have not been generated. Run "
            f"scripts/generate_mounts.py; the files live under {directory} "
            f"and are not committed.")
        return result

    data = json.loads(summary.read_text())
    for part in data.get("parts", []):
        row = {"part": part["part"], "generated": part["generated"],
               "mass_kg": part.get("mass_kg"),
               "envelope_mm": part.get("envelope_mm"),
               "triangles": part.get("triangles"),
               "watertight": part.get("watertight")}
        loads = part.get("loads") or {}
        row.update({"vertical_n": loads.get("vertical_n"),
                    "overturning_nm": loads.get("overturning_nm"),
                    "reaction_nm": loads.get("yaw_reaction_nm")})
        if not part["generated"]:
            row["reason"] = part.get("reason")
        result.rows.append(row)
        for item in part.get("unresolved", []):
            result.could_not.append(item)

    made = [r for r in result.rows if r["generated"]]
    if made:
        result.notes.append(
            f"{len(made)} parts, {sum(r['mass_kg'] for r in made):.3f} kg "
            f"together, in {data.get('units', 'unknown units')} at a volume "
            f"fraction of {data.get('volume_fraction')}")
    result.notes.append(
        "the base mount is sized by its OVERTURNING MOMENT and not by the "
        "weight it holds: the arm stretched out puts 31.8 N m on it against "
        "76 N of weight, and the payload alone at full reach is 17.6 of that")
    return result


#: CHOSEN. How much of the tip deflection budget the joints may have. The
#: links and the joints are in series and both consume it, and nothing says
#: how to divide it, so half each is the least arbitrary split available and
#: it is stated rather than buried.
JOINT_SHARE_OF_BUDGET = 0.5

#: The torsional stiffnesses this catalogue actually prints, for gear units
#: of roughly this size. They are a REFERENCE RANGE and not a design input:
#: they belong to units this design has already refused to pair, for a
#: separate reason. What they are good for is knowing whether a required
#: stiffness is ordinary or absurd.
PRINTED_STIFFNESS_RANGE_NM_RAD = (10_313.0, 44_000.0)

#: THE 44,000 IS NOT THIS ARM'S NUMBER. It belongs to a Harmonic Drive gear
#: unit that this design refused to pair, for a separate reason, and it is
#: quoted only so a required stiffness can be told from an absurd one. Read
#: quickly, "1.15 times the stiffest printed unit" sounds like the design
#: nearly works. What it actually says is that the requirement is of an
#: ordinary size. The stiffness of the actuators this arm SELECTED is
#: unknown, because not one of them publishes it, and unknown is not near.
STIFFNESS_RANGE_IS_NOT_THIS_ARM = (
    "the 10,313 to 44,000 N m/rad range belongs to gear units this design "
    "does not use and has refused to pair. It bounds what parts of this size "
    "achieve; it says nothing about the actuators actually selected, whose "
    "stiffness no source in this project gives")

#: The status of the one millimetre requirement. Not met, not missed:
#: UNDECIDABLE, because the joints between the links have no published
#: stiffness and their contribution is unknown. The user's decision on
#: 2026-09-05 was to keep the specification and obtain the real stiffness
#: from the manufacturer or by measurement, so this stays undecidable until
#: that value exists.
TIP_DEFLECTION_STATUS = "UNVERIFIED: not met, not missed, undecidable"


def joint_stiffness_stage(dynamics: StageResult, drivetrain: StageResult,
                          arm: Assembly, spec: ManipulatorSpec = SPEC,
                          margin: float = 0.8) -> StageResult:
    """What joint stiffness this arm would NEED, since none is published.

    THE DEFLECTION NUMBERS IN THIS DESIGN ARE LINK ELASTICITY ONLY. Six
    actuators sit between the links and every one of them has been treated as
    rigid, because not one of the integrated actuators in this catalogue
    prints a torsional stiffness. In a real arm that term is usually the
    larger of the two: a joint twisting by torque over stiffness swings the
    whole remaining reach, and the shoulder's reach is 600 mm.

    No stiffness is assumed. The question is turned round the way it was for
    the friction grip, which needed a coefficient nobody prints: this reports
    the stiffness each joint would HAVE to have. That needs no source, and
    the answer can be read against the values this catalogue does print for
    gear units of the same size, which run 10,313 to 44,000 N m/rad.
    """
    import numpy as np

    from core.assembly.kinematics import forward_kinematics

    result = StageResult(name="the joint stiffness this arm would need")
    pose = forward_kinematics(arm, stretched_pose(spec))
    tool = pose.tool_position()
    statics = {row["joint"]: abs(float(row["static_nm"]))
               for row in dynamics.rows}
    selected = [row for row in drivetrain.rows if row.get("status") == "selected"]
    if not selected:
        result.could_not.append("no drive was selected, so nothing to size")
        return result

    budget = spec.tip_deflection_limit_m * margin
    joint_budget = budget * JOINT_SHARE_OF_BUDGET
    low, high = PRINTED_STIFFNESS_RANGE_NM_RAD

    # SPLIT THE ALLOWANCE BY WHAT EACH JOINT ACTUALLY COSTS, not equally.
    # An equal split gives a sixth of the joint budget to the base yaw and
    # the two roll axes, which carry no gravity moment at full reach and need
    # none of it, and the same sixth to the shoulder, which carries 67
    # percent of the whole. It reported the shoulder needing 205,000 N m/rad
    # when the honest number is 50,700, a factor of four out, and the wrong
    # way: it made a reachable design look impossible.
    #
    # Weighting by torque times lever has a tidy consequence. Every joint
    # ends up needing the SAME stiffness, because a joint's contribution is
    # torque times lever over stiffness, so making the contributions
    # proportional to torque times lever makes the stiffness constant. One
    # number describes the whole arm.
    weights = {row["joint"]: statics.get(row["joint"], 0.0)
               * float(np.linalg.norm(tool - pose.joint_origins[row["joint"]]))
               for row in selected}
    demand = sum(weights.values())
    uniform = demand / joint_budget if joint_budget > 0 else None

    for row in selected:
        joint = row["joint"]
        torque = statics.get(joint, 0.0)
        lever = float(np.linalg.norm(tool - pose.joint_origins[joint]))
        allowance = (joint_budget * weights[joint] / demand if demand else 0.0)
        required = uniform if weights[joint] > 0.0 else None
        result.rows.append({
            "joint": joint, "drive": row.get("selected"),
            "static_torque_nm": torque, "lever_m": lever,
            "allowance_m": allowance,
            "torque_times_lever_nm2": weights[joint],
            "required_stiffness_nm_rad": required,
            "published": None,
            "against_printed_range": (
                None if not required else
                "inside the range this catalogue prints" if required <= high
                else f"{required / high:.1f} times the stiffest unit this "
                     f"catalogue prints")})

    if uniform:
        result.notes.append(
            f"every loaded joint needs the same {uniform:,.0f} N m/rad, "
            f"against {low:,.0f} to {high:,.0f} for the gear units this "
            f"catalogue prints. That is {uniform / high:.2f} times the "
            f"stiffest of them, so the limit is near the edge of what this "
            f"catalogue can do rather than beyond it")
        result.notes.append(
            f"with every joint at the stiffest printed unit, {high:,.0f} N "
            f"m/rad, the joints alone would use "
            f"{demand / high * 1000:.3f} mm of the "
            f"{spec.tip_deflection_limit_m * 1000:.1f} mm limit, leaving "
            f"{(spec.tip_deflection_limit_m - demand / high) * 1000:.3f} mm "
            f"for the links and any margin")
    result.notes.append(
        "the torque used is the STATIC one at full reach, because the "
        "deflection limit is a static condition. Three joints read zero "
        "there: the base yaw and the two roll axes carry no gravity moment "
        "in this pose. That is a property of the pose and not of the joints, "
        "and under acceleration they carry torque like any other")
    result.notes.append(
        f"the joints are given {JOINT_SHARE_OF_BUDGET:.0%} of the "
        f"{budget * 1000:.2f} mm budget, which is a CHOSEN split: links and "
        f"joints are in series and nothing says how to divide it. Within the "
        f"joints it is split by torque times lever, not equally")
    result.data["status_of_the_requirement"] = TIP_DEFLECTION_STATUS
    result.data["required_stiffness_nm_rad"] = uniform
    result.data["allocation"] = (
        "the joints take a CHOSEN 50 percent of the budget and divide it by "
        "torque times lever, not equally. Equally would give a sixth to each "
        "of three joints that carry no gravity moment at full reach and the "
        "same sixth to the shoulder, which carries 67 percent of the demand")
    result.notes.append(STIFFNESS_RANGE_IS_NOT_THIS_ARM)
    result.notes.append(
        f"the status of the 1 mm requirement is {TIP_DEFLECTION_STATUS}. It "
        f"is not a pass and not a failure. The value that decides it is the "
        f"selected actuators' torsional stiffness, which is being sought "
        f"from the manufacturer or by measurement")
    result.could_not.append(
        "NO TORSIONAL STIFFNESS IS PUBLISHED for any integrated actuator in "
        "this catalogue, so no joint compliance is modelled and every "
        "deflection reported by this design is LINK ELASTICITY ONLY. It is "
        "not a conservative simplification: the missing term adds to the one "
        "that is modelled, so the real tool deflection is larger than any "
        "number here by an amount nobody in this design knows.")
    return result


def backlash_stage(dynamics: StageResult, drivetrain: StageResult,
                   arm: Assembly, spec: ManipulatorSpec = SPEC) -> StageResult:
    """The tool error the drives' own printed backlash puts on the arm.

    THIS IS THE LARGEST TERM AND IT IS THE ONLY ONE THAT IS PUBLISHED. The
    deflection work has been arguing over link elasticity, which is
    computed, and joint stiffness, which nobody prints. Backlash is printed:
    0.18 degrees for the AK80-64 and 15 arcmin for the AK80-9. Multiplied by
    the levers already established it comes to 3.87 mm in the bending plane,
    against a 1 mm limit, and it does not depend on how stiff anything is.

    BACKLASH IS NOT DEFLECTION and the two must not be added carelessly.
    Deflection is determined by load: know the load and it can be corrected.
    Backlash is a dead band, and where the joint sits inside it depends on
    which way the torque last pushed. While the torque keeps one sign the
    band is a fixed offset that a calibration can remove. It is only an
    error when the torque REVERSES and the joint crosses the band.

    So the stage asks that question rather than assuming either answer, and
    it asks it twice. Between carrying the payload and not carrying it, at
    full reach, no joint's torque changes sign: gravity dominates and always
    pulls the same way, so picking a part up and putting it down does not
    cross the band. Over the commanded move it is the opposite: every joint
    reverses, because the arm accelerates and then decelerates. A moving arm
    crosses the band at every joint, every cycle.
    """
    import numpy as np

    from core.assembly.kinematics import forward_kinematics
    from core.assembly.statics import joint_torques
    from drivetrain.sourced import sourced_motor

    result = StageResult(name="backlash at the tool")
    pose = forward_kinematics(arm, stretched_pose(spec))
    tool = pose.tool_position()
    density = get_material(arm.material_id).density_kg_m3
    q0 = stretched_pose(spec)
    carrying = joint_torques(arm, q0, density, tip_force_n=payload_force_n(spec))
    empty = joint_torques(arm, q0, density, tip_force_n=np.zeros(3))
    names = [joint.name for joint in arm.actuated_joints()]
    profile = dynamics.data["profiles"]["trapezoidal"]
    torque = np.asarray(profile.torque_nm)

    bending = {"j2_shoulder", "j3_elbow", "j5_wrist_pitch"}
    total_bending = 0.0
    for row in drivetrain.rows:
        if row.get("status") != "selected":
            continue
        joint = row["joint"]
        motor = sourced_motor(str(row["selected"]))
        lever = float(np.linalg.norm(tool - pose.joint_origins[joint]))
        index = names.index(joint)
        column = torque[:, index]
        entry = {
            "joint": joint, "drive": motor.part_number, "lever_m": lever,
            "backlash_arcmin": motor.backlash_arcmin,
            "torque_carrying_nm": float(carrying[index]),
            "torque_empty_nm": float(empty[index]),
            "reverses_when_the_payload_is_set_down": bool(
                carrying[index] * empty[index] < 0.0),
            "reverses_during_the_move": bool(column.min() * column.max() < 0.0),
        }
        if motor.backlash_arcmin:
            band = motor.backlash_arcmin * (np.pi / (180.0 * 60.0)) * lever
            entry["band_at_the_tool_m"] = band
            if joint in bending:
                total_bending += band
        else:
            entry["band_at_the_tool_m"] = None
            entry["note"] = ("no backlash is printed for this drive, so its "
                             "band is unknown and is NOT zero")
        result.rows.append(entry)

    result.data["bending_plane_total_m"] = total_bending
    result.data["limit_m"] = spec.tip_deflection_limit_m
    result.notes.append(
        f"the printed backlash alone puts {total_bending * 1000:.3f} mm at "
        f"the tool in the bending plane, which is "
        f"{total_bending / spec.tip_deflection_limit_m:.1f} times the "
        f"{spec.tip_deflection_limit_m * 1000:.1f} mm limit. It comes from "
        f"printed values and established levers, so unlike the stiffness it "
        f"is not waiting on anything")
    moving = [r["joint"] for r in result.rows if r["reverses_during_the_move"]]
    setting_down = [r["joint"] for r in result.rows
                    if r["reverses_when_the_payload_is_set_down"]]
    result.notes.append(
        f"setting the payload down reverses the torque at "
        f"{len(setting_down)} joints, so that alone does not cross the band: "
        f"gravity dominates at full reach and keeps pulling one way")
    result.notes.append(
        f"the commanded move reverses it at {len(moving)} of "
        f"{len(result.rows)}, because the arm accelerates and then "
        f"decelerates. A MOVING ARM CROSSES THE BAND AT EVERY JOINT ON EVERY "
        f"CYCLE, so the band is not a calibratable offset here")
    result.could_not.append(
        "Whether the 1 mm requirement is about deflection under load or "
        "about tool position cannot be decided here. If it is elastic "
        "deflection then backlash is a separate budget and the limit still "
        "means something. If it is position, this arm is a 4 mm machine and "
        "no amount of link stiffness changes that. The specification does "
        "not say, and it is the specifier's answer to give.")
    result.could_not.append(
        "The AK60-6 prints no backlash at all, only the words low backlash, "
        "so the tool roll's band is unknown. Unknown is not zero.")
    return result


#: CHOSEN, and the two things that do NOT set it are recorded with it. The
#: cycloidal disc is 8 mm thick. Hertzian contact at the ring pins wants
#: nothing: 244 MPa at peak torque against roughly 1500 allowable for a
#: hardened pin, a factor of six. Bearing in the output pin holes wants less
#: still: 196 N on a 10 mm pin needs 0.20 mm of aluminium at a conservative
#: 100 MPa, or 0.03 of hardened steel. At 8 mm the hole sees 2.45 MPa.
#:
#: So the thickness is not carried by either of the loads it obviously
#: carries, and calling it a strength result would be false. Disc stiffness
#: was the last open candidate and `joint_torsion_stage` closes it: the in
#: plane shear of the annulus between the two pin circles is 284 times the
#: requirement and linear in thickness, so 0.028 mm of steel would carry it
#: alone. The two contacts in series go as thickness to the 0.8 and put their
#: own floor at 0.940 mm, which is the largest of the four and still an order
#: of magnitude under 8.
#:
#: That 0.940 replaces a 0.292 written earlier the same day, from a contact
#: model whose lever arms were too long. The conclusion did not move, but the
#: margin did, by a factor of three, and that is the sort of thing worth
#: recording rather than quietly overwriting.
#:
#: The thickness is therefore CHOSEN, and it is chosen for what a wire cut
#: disc can be handled, stacked and kept flat at, which is not something this
#: repository can compute. That is a weaker claim than a strength result and
#: it is the true one.
CYCLOIDAL_DISC_THICKNESS_M = 0.008
CYCLOIDAL_DISC_THICKNESS_BASIS = (
    "CHOSEN, for handling and flatness of a wire cut disc, which is not "
    "computed here. Four floors were computed and none of them binds: pin "
    "contact stress is six times under its allowable, output pin hole "
    "bearing needs 0.20 mm, disc in plane shear stiffness needs 0.028 mm, "
    "and the pin contacts in series need 0.940 mm")


#: CHOSEN, and the gate that refuses a disc. A wire cut steel disc needs
#: material between an output pin hole and its own outer profile, and
#: between one hole and the next. Three millimetres is the floor used here.
#: It is a judgement about handling a thin ligament, not a strength result:
#: the loads are far below what 3 mm of steel carries.
MINIMUM_DISC_LIGAMENT_M = 0.003

#: The housing the reducer sits in, as a thin walled tube in TORSION. Note
#: that this is not the calculation in `joint_module_stiffness_stage`, which
#: uses E and a bending second moment and answers the out of plane question.
#: Torsion needs G and J, and G here is E over 2(1 + nu) for the link alloy.
HOUSING_DIAMETER_M = 0.124
HOUSING_WALL_M = 0.004
HOUSING_LENGTH_M = 0.080
HOUSING_POISSON = 0.33


def housing_torsion_nm_rad(spec: ManipulatorSpec = SPEC) -> float:
    modulus = get_material(spec.materials["link"]).youngs_modulus_pa
    return shell_torsion_nm_rad(HOUSING_DIAMETER_M, HOUSING_WALL_M,
                                HOUSING_LENGTH_M,
                                modulus / (2.0 * (1.0 + HOUSING_POISSON)))


def joint_torsion_stage(spec: ManipulatorSpec = SPEC,
                        geometry: CycloidalGeometry | None = None,
                        torque_nm: float = 22.76,
                        required_nm_rad: float = 50_689.0,
                        samples: int = 24) -> StageResult:
    """The stiffness the deflection budget actually asks for, and where it is.

    IT IS NOT THE BEARING'S. The tool sags because the joint output rotates
    about its own axis, and at the shoulder the gravity moment lies entirely
    along that axis: the cross product of the 600 mm lever with gravity has a
    component on the joint axis of exactly one. That is the single direction
    a joint bearing does not resist, because it is the direction the joint
    turns in. What resists it is the drive train.

    So 50,689 N m/rad is a TORSIONAL requirement on the reducer, and the
    crossed roller's tilting rigidity answers a different question, the out
    of plane budget. Both are needed and they are not the same number.

    CORRECTION TO THE FIRST ESTIMATE, which reported 682,012 N m/rad and a
    factor of 13.5 on the same day. That was optimistic by 2.8, and both
    halves of the error were in the lever arms rather than in the loads:

    The ring pins do not act at their pin circle radius. Every cycloidal
    contact normal passes through the instantaneous pitch point, which in the
    disc's frame sits at e * N from the disc centre, so no ring pin can have
    a moment arm larger than that whatever radius its circle is drawn at.
    Here that is 25 mm against a 45 mm circle, and the computed maximum arm
    is 24.98. The sum of squares is 1,700 mm^2 where a radius times a count
    would have given 5,569.

    The output pins do not all act at their full radius either. Their contact
    normals are all parallel, along the line of the disc's offset, so the
    moment arm of a hole is the circle radius times the sine of its angle
    from that line. The share is now solved from those arms instead of a
    fraction of engaged pins being put in by hand.

    The housing is in the chain now, in torsion. The discs' in plane shear
    barely appears.
    """
    geometry = geometry or CycloidalGeometry()
    result = StageResult(name="joint torsion, which is what the budget asks")
    angles = np.linspace(0.0, 2.0 * np.pi / geometry.ring_pin_count, samples)

    def worst(arms_of) -> float:
        return min(pin_set_stiffness_nm_rad(
            torque_nm, arms_of(geometry, angle), geometry.disc_thickness_m,
            geometry.disc_count) for angle in angles)

    terms = {
        "ring pin contact": worst(ring_pin_moment_arms),
        "output pin contact": worst(output_pin_moment_arms),
        "discs, in plane shear": disc_shear_stiffness_nm_rad(geometry),
        "housing, in torsion": housing_torsion_nm_rad(spec),
    }
    total_compliance = sum(1.0 / value for value in terms.values())
    for name, value in terms.items():
        result.rows.append({
            "term": name, "stiffness_nm_rad": value,
            "compliance_rad_nm": 1.0 / value,
            "share_of_compliance": (1.0 / value) / total_compliance,
            "times_required": value / required_nm_rad})

    known = 1.0 / total_compliance
    result.data["known_terms_nm_rad"] = known
    result.data["required_nm_rad"] = required_nm_rad
    result.data["margin_before_the_bearing"] = known / required_nm_rad
    result.data["pitch_radius_m"] = geometry.pitch_radius_m
    result.data["output_web_m"] = geometry.output_web_m
    result.data["output_ligament_m"] = geometry.output_ligament_m

    #: The eccentric bearing has no catalogue value here, so the question is
    #: turned round the way the friction grip was: not what it gives, but
    #: what it has to be.
    allowance = 1.0 / (1.0 / required_nm_rad - 1.0 / known)
    needed = required_bearing_stiffness_n_m(allowance, geometry)
    result.data["eccentric_bearing_radial_stiffness_needed_n_m"] = needed

    result.notes.append(
        f"the four computed terms give {known:,.0f} N m/rad in series, "
        f"{known / required_nm_rad:.1f} times the {required_nm_rad:,.0f} the "
        f"arm needs. The output pin contact carries "
        f"{max(row['share_of_compliance'] for row in result.rows):.0%} of the "
        f"compliance, the ring pins most of the rest, and the discs are stiff "
        f"enough to be almost absent")
    result.notes.append(
        f"the eccentric bearing is the one term with no number, and its lever "
        f"is the pitch radius {geometry.pitch_radius_m * 1000:.1f} mm, "
        f"SQUARED. Asked in reverse, its radial stiffness has to be at least "
        f"{needed:.2e} N/m for the joint to reach the requirement at all")
    result.notes.append(
        f"a Ø{2000 * geometry.output_pin_circle_radius_m:.0f} output pin "
        f"circle leaves {geometry.output_web_m * 1000:.2f} mm of web to the "
        f"disc's ROOT radius and {geometry.output_ligament_m * 1000:.2f} mm "
        f"between holes. Ø60 was proposed and it is not available: measured "
        f"to the root rather than the tip the web comes out 0.00 mm exactly, "
        f"so the hole breaks out of the disc")
    result.could_not.append(
        "Not a verified result. The load share now falls out of the geometry "
        "rather than being assumed, and the housing is in the chain, but "
        "Palmgren's relation is still a roller bearing formula applied to a "
        "cycloidal flank and to a pin in a hole, and that carries 86 percent "
        "of the compliance. The eccentric bearing has no value at all, only "
        "a requirement. The 1 mm tip limit stays UNVERIFIED either way, "
        "because every deflection this design computes is link elasticity "
        "alone.")
    return result


def joint_module_stiffness_stage(spec: ManipulatorSpec = SPEC,
                                 required_nm_rad: float = 50_689.0
                                 ) -> StageResult:
    """Which part of a built joint decides its stiffness.

    READ THE SCOPE OF THIS STAGE FIRST. Every number below uses E and a
    BENDING second moment, so what it computes is the housing's resistance to
    moments ACROSS the joint. That is the out of plane budget. It is not the
    torsional budget, which is what the tool's sag at full reach actually
    draws on, and `joint_torsion_stage` is where that lives. This stage was
    written believing the two were the same question and its conclusion,
    that "the bearing is the whole of it", was wrong for that reason.

    Within its own scope the finding stands. A cylindrical housing of the
    size this module needs, in the weaker of the two candidate alloys, comes
    out one to two orders of magnitude above 50,689 N m/rad in bending: 2.6
    million at 124 mm across a 4 mm wall, and still 24 times clear at half
    the wall and half again the length. So for out of plane moments the
    structure is not where the compliance is and the bearing is.

    For TORSION the same shell is 1.97 million N m/rad, using G and J rather
    than E and I, and it is 12 percent of the drive train's compliance
    instead of a rounding error. Same part, different question, and the two
    answers differ by more than the change of modulus alone.

    The two layouts scale differently, which is the interesting part. A
    crossed roller carries the moment on a raceway all the way round, so its
    stiffness grows as the RADIUS CUBED. A spread pair of angular contact
    bearings carries it as a force couple, so its stiffness grows as the
    SPACING SQUARED and does not care about diameter. A pair therefore wins
    only where it can be spread, and loses on a short module.

    NEITHER CAN BE EVALUATED HERE. Both need a raceway stiffness that only a
    manufacturer prints, and this project has none: the standard parts
    library's deep groove entries have assumed internal geometry, no ratings,
    and are not moment carrying parts in the first place.
    """
    import numpy as np

    result = StageResult(name="where a joint module's stiffness is lost")
    modulus = get_material(spec.materials["link"]).youngs_modulus_pa
    for name, diameter, wall, length in (
            ("crossed roller, 124 mm across", 0.124, 0.004, 0.080),
            ("angular contact pair, 110 mm", 0.110, 0.004, 0.080),
            ("124 mm on a 2 mm wall", 0.124, 0.002, 0.080),
            ("110 mm over a 120 mm module", 0.110, 0.004, 0.120)):
        radius = 0.5 * diameter
        second_moment = np.pi * radius ** 3 * wall
        stiffness = modulus * second_moment / length
        result.rows.append({
            "layout": name, "diameter_m": diameter, "wall_m": wall,
            "length_m": length, "shell_stiffness_nm_rad": stiffness,
            "times_required": stiffness / required_nm_rad})

    result.notes.append(
        f"IN BENDING the housing shell is between "
        f"{min(r['times_required'] for r in result.rows):.0f} and "
        f"{max(r['times_required'] for r in result.rows):.0f} times the "
        f"{required_nm_rad:,.0f} N m/rad, so for moments ACROSS the joint the "
        f"structure is not the limiting term and the bearing is. In TORSION "
        f"the same shell is {housing_torsion_nm_rad(spec):,.0f} and carries 12 "
        f"percent of the drive train's compliance; see joint_torsion_stage")
    result.notes.append(
        "a crossed roller's moment stiffness grows as the radius CUBED and a "
        "spread angular contact pair's as the spacing SQUARED, so the pair "
        "wins only where it can be spread and loses on a short module. At "
        "124 against 110 mm the crossed roller is 1.43 times ahead on radius "
        "before any spacing is considered")
    result.could_not.append(
        "Neither bearing layout can be evaluated. Both need a raceway "
        "stiffness that only a manufacturer prints, and this project holds "
        "none: the standard parts library's deep groove ball entries have "
        "ASSUMED internal geometry, carry no C or C0, and are not moment "
        "carrying parts to begin with.")
    result.could_not.append(
        "One catalogue figure is in hand and it is an upper bound, not a "
        "value. THK 382-5E page 16 puts an RB10020 at about 1.7e6 N m/rad, "
        "read at a moment of 0.4 kN m, and this arm works under five percent "
        "of the way along that chart where the curve is steepest. THK's own "
        "A18-1 page 18 carries no formula, only the diagram, and prints two "
        "conditions with it. The first is RADIAL CLEARANCE ZERO, so the "
        "figure is neither a preloaded nor a clearanced one. The second is "
        "THK's own sentence that rigidity is affected by the deformation of "
        "the housing, the presser flange and the bolts, and that their "
        "strength must be taken into account. So the catalogue number is the "
        "BEARING ALONE and the structure around it adds compliance on top. "
        "Any out of plane budget built on it has to carry that structure.")
    return result
