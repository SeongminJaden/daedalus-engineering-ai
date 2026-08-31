"""The capstone: design and verify a driven revolute joint end to end.

Every method the project has, applied to one assembly, with a single
conjunctive verdict over all of them. The point is not any individual check,
all of which exist already, but that a design has to survive ALL of them at
once and that the gaps are named rather than passed over.

VALIDITY OF THE PIPELINE, before the code:

* **Sizing is sequential, not simultaneous.** The link is sized, then the
  drivetrain is selected for the resulting inertia, then the shaft and bearings
  are checked against the drivetrain's torque. A heavier link needs more
  torque, which needs a bigger motor, which is more mass at the joint. That
  loop is not closed here: one forward pass is made. So the result is a
  CONSISTENT design, not a jointly optimal one, and a real design iterates.

* **The load path is idealised at every handoff.** The link's tip load becomes
  a shaft bending moment through an assumed overhang, and the mount reaction
  becomes a bolt load through an assumed lever arm. Those geometric
  assumptions are inputs, not derivations, and the checks downstream are only
  as good as them.

* **The checks are independent by assumption.** Nothing here models a bearing
  whose wear changes the shaft alignment that changes the gear load
  distribution. Real assemblies couple that way and this treats each mode as
  isolated.

* **Every idealisation multiplies.** Each check reports a factor computed under
  its own optimistic assumptions, and the assembly inherits all of them
  simultaneously. The true margin is below the reported minimum, by an amount
  nothing here estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.materials import get_material
from core.registry import DEFAULT_REGISTRY, Category, ProblemContext

from .checks import (AssemblyVerdict, CheckResult, CheckStatus,
                     satisfies)

# Failure modes this project knows to name but has no method for. Listed so the
# coverage report can be explicit about them. The modes nobody has thought of
# are, by construction, not on this list, which is the limit of what a coverage
# claim can mean.
KNOWN_UNIMPLEMENTED_MODES: dict[str, tuple[str, ...]] = {
    "link": ("local_buckling_of_thin_walls", "impact_and_shock"),
    "shaft": ("keyway_stress_concentration_measured", "torsional_vibration"),
    "bearing": ("contact_fatigue_detail", "lubrication_and_contamination"),
    "gearbox": ("planetary_internal_load_sharing", "backlash_wear"),
    "mount": ("bolt_group_eccentricity", "self_loosening_under_vibration"),
    "assembly": ("manufacturing_variation", "assembly_tolerance_stackup",
                 "corrosion_and_fretting"),
}


@dataclass(frozen=True)
class JointSpec:
    """What the joint has to do. The only thing a caller must supply."""

    name: str = "knee"
    payload_kg: float = 5.0
    link_length_m: float = 0.35
    max_speed_rad_s: float = 3.0
    duty_cycle_fraction: float = 0.5
    ambient_c: float = 40.0
    required_bearing_life_h: float = 20000.0
    min_safety_factor: float = 2.0
    material_id: str = "al_7075_t6"
    shaft_material_id: str = "steel_scm440"
    # Load path geometry. Assumptions, not derivations: see the module note.
    bearing_span_m: float = 0.060
    shaft_overhang_m: float = 0.040
    shaft_diameter_m: float = 0.016
    mount_bolt_size: str = "M6"
    mount_lever_arm_m: float = 0.050
    mount_bolt_count: int = 4
    grip_length_m: float = 0.015
    # The gearbox's internal mesh, if it is known. None means it is NOT known,
    # and the tooth check is then reported as unassessed rather than run
    # against an invented geometry. Supply (module_m, pinion_teeth,
    # gear_teeth, face_width_m) to have it checked.
    gear_mesh: tuple[float, int, int, float] | None = None

    @property
    def static_torque_nm(self) -> float:
        """Gravity torque with the payload at full extension.

        The worst static case for a horizontal link, which is what a joint is
        sized against.
        """
        return self.payload_kg * 9.80665 * self.link_length_m

    @property
    def tip_load_n(self) -> float:
        return self.payload_kg * 9.80665


@dataclass
class CapstoneResult:
    """The design, the verdict, and everything needed to argue with both."""

    spec: JointSpec
    verdict: AssemblyVerdict
    link_design: dict = field(default_factory=dict)
    drivetrain: dict = field(default_factory=dict)
    load_path: dict = field(default_factory=dict)
    selected_bearing: str | None = None
    total_mass_kg: float = 0.0
    material_cost_usd: float = 0.0

    @property
    def passes(self) -> bool:
        return self.verdict.passes


def _problem_context(spec: JointSpec) -> ProblemContext:
    """What the registry needs in order to route this assembly.

    Every feature is stated. An unstated one excludes methods rather than
    admitting them, so silence here shows up as a coverage gap rather than as
    an unearned pass.
    """
    return ProblemContext(
        geometry="assembly",
        representations=("assembly", "prismatic_beam", "voxel_domain"),
        slenderness=spec.link_length_m / 0.05,
        material_class="isotropic",
        objective="mass",
        has_stress_constraint=True,
        needs_stress_field=False,
        needs_gradients=True,
        has_cyclic_load=True,
        has_compressive_load=False,
        transmits_torque=True,
        has_rotating_support=True,
        has_duty_cycle=True,
        has_temperature_change=False,
        has_bolted_joint=True,
        has_gear_mesh=True,
        n_objectives=2,
        has_discrete_variables=False,
        has_layup=False,
    )


def routed_methods(spec: JointSpec) -> dict[str, list[str]]:
    """Which registered methods apply to this problem, by category.

    The coverage claim rests on this: a mode is assessed because a method was
    routed to it, not because someone remembered to call one.
    """
    context = _problem_context(spec)
    return {category.value: list(DEFAULT_REGISTRY.query(context, category).names())
            for category in Category}


def build_link_problem(spec: JointSpec):
    """The structural problem for the link, from the joint spec."""
    from core.engineering_ir import (BoundaryCondition, BoundaryConditionType,
                                     BoundaryLocation, Constraints,
                                     EngineeringProblem, Geometry, Load,
                                     LoadApplication, LoadType, Objective,
                                     ObjectiveQuantity, ObjectiveSense,
                                     SectionType, Vec3)

    return EngineeringProblem(
        name=f"{spec.name}_link",
        geometry=Geometry(length_m=spec.link_length_m, max_width_m=0.08,
                          max_height_m=0.08,
                          section_type=SectionType.HOLLOW_RECTANGLE),
        material_id=spec.material_id,
        loads=[Load(type=LoadType.POINT_FORCE, magnitude_n=spec.tip_load_n,
                    direction=Vec3(x=0.0, y=-1.0, z=0.0),
                    application=LoadApplication.TIP)],
        boundary_conditions=[BoundaryCondition(
            type=BoundaryConditionType.FIXED,
            location=BoundaryLocation.ROOT)],
        constraints=Constraints(max_deflection_m=1.5e-3,
                                min_safety_factor=spec.min_safety_factor),
        objectives=[Objective(sense=ObjectiveSense.MINIMIZE,
                              quantity=ObjectiveQuantity.MASS)],
    )


def _link_checks(spec: JointSpec, verdict: AssemblyVerdict) -> dict:
    """Size the link, then check every structural mode the duty allows."""
    from optimization.constraints import build_optimization_problem, evaluate_design
    from optimization.gradient import default_start, optimize_slsqp
    from physics.fatigue import (MeanStressCriterion, StressCycle,
                                 fatigue_safety_factor)

    material = get_material(spec.material_id)
    op = build_optimization_problem(build_link_problem(spec))
    solved = optimize_slsqp(op, x0=default_start(op), max_iter=200)
    evaluation = evaluate_design(op, solved.x)

    stress_safety = (material.yield_strength_pa
                     / evaluation.max_bending_stress_pa)
    verdict.add(CheckResult(
        component="link", failure_mode="static_stress",
        status=(CheckStatus.PASSED
                if satisfies(stress_safety / spec.min_safety_factor)
                else CheckStatus.FAILED),
        method="beam_timoshenko", safety_factor=stress_safety / spec.min_safety_factor,
        detail=f"{evaluation.max_bending_stress_pa / 1e6:.1f} MPa against "
               f"{material.yield_strength_pa / 1e6:.0f} MPa yield, required "
               f"factor {spec.min_safety_factor:g}",
        optimistic_assumption="beam theory with no stress concentration at the "
                              "root fillet, where a real link is worst"))

    limit = 1.5e-3
    deflection_margin = limit / evaluation.tip_deflection_m
    verdict.add(CheckResult(
        component="link", failure_mode="deflection",
        status=(CheckStatus.PASSED if satisfies(deflection_margin)
                else CheckStatus.FAILED),
        method="beam_timoshenko", safety_factor=deflection_margin,
        detail=f"{evaluation.tip_deflection_m * 1e3:.3f} mm against a "
               f"{limit * 1e3:.1f} mm limit",
        optimistic_assumption="a rigid root: real mounting compliance adds "
                              "deflection this does not include"))

    # The joint reverses under gravity as the link swings, so the bending is
    # fully reversed rather than steady.
    cycle = StressCycle.fully_reversed(evaluation.max_bending_stress_pa)
    fatigue = fatigue_safety_factor(cycle, material, MeanStressCriterion.GOODMAN)
    verdict.add(CheckResult(
        component="link", failure_mode="fatigue",
        status=(CheckStatus.PASSED if satisfies(fatigue.safety_factor)
                else CheckStatus.FAILED),
        method="fatigue_sn", safety_factor=fatigue.safety_factor,
        detail=f"fully reversed {cycle.alternating_pa / 1e6:.1f} MPa against "
               f"{fatigue.endurance_pa / 1e6:.0f} MPa. {fatigue.notes}",
        optimistic_assumption="no notch, surface, size or temperature factor, "
                              "each of which lowers a real endurance limit"))

    verdict.add(CheckResult(
        component="link", failure_mode="buckling",
        status=CheckStatus.NOT_APPLICABLE, method="buckling_euler",
        detail="the link carries bending and no axial compression, and a "
               "member not in compression cannot buckle"))

    return {"design_vector": solved.x.tolist(), "mass_kg": evaluation.mass_kg,
            "stress_pa": evaluation.max_bending_stress_pa,
            "deflection_m": evaluation.tip_deflection_m,
            "natural_frequency_hz": evaluation.first_natural_frequency_hz,
            "converged": bool(solved.success)}


def _drivetrain_checks(spec: JointSpec, link: dict,
                       verdict: AssemblyVerdict) -> dict:
    """Select a motor and gearbox, then check the winding temperature."""
    from drivetrain.selection.select import Requirement, select_drivetrain
    from physics.thermal import DutySegment, check_motor_thermal

    # Payload plus the link itself, as a point mass at the tip for inertia.
    inertia = (spec.payload_kg + link["mass_kg"]) * spec.link_length_m ** 2
    requirement = Requirement(
        joint=spec.name, continuous_torque_nm=spec.static_torque_nm,
        peak_torque_nm=2.0 * spec.static_torque_nm,
        max_speed_rad_s=spec.max_speed_rad_s, load_inertia_kg_m2=inertia)
    best, alternatives = select_drivetrain(requirement)
    if best is None:
        verdict.add(CheckResult(
            component="drivetrain", failure_mode="torque_and_speed",
            status=CheckStatus.FAILED, method="drivetrain_selection",
            safety_factor=0.0,
            detail=f"no motor and gearbox pairing meets "
                   f"{requirement.continuous_torque_nm:.1f} N m at "
                   f"{requirement.max_speed_rad_s:.1f} rad/s",
            optimistic_assumption="the catalogue holds illustrative "
                                  "archetypes, not vendor parts"))
        return {}

    verdict.add(CheckResult(
        component="drivetrain", failure_mode="torque_and_speed",
        status=CheckStatus.PASSED, method="drivetrain_selection",
        safety_factor=best.limiting_check.margin + 1.0,
        detail=f"{best.motor.name} with {best.gearbox.id}; the limiting check "
               f"is {best.limiting_check.name}",
        optimistic_assumption="the catalogue holds illustrative archetypes, "
                              "so every rating here must be replaced with a "
                              "datasheet before anything is ordered"))

    duty = [DutySegment(best.motor.continuous_torque_nm,
                        spec.max_speed_rad_s * best.gearbox.ratio,
                        spec.duty_cycle_fraction)]
    if spec.duty_cycle_fraction < 1.0:
        duty.append(DutySegment(0.0, 0.0, 1.0 - spec.duty_cycle_fraction))
    thermal = check_motor_thermal(best.motor, duty, ambient_c=spec.ambient_c)
    verdict.add(CheckResult(
        component="drivetrain", failure_mode="winding_temperature",
        status=(CheckStatus.PASSED if thermal.passes else CheckStatus.FAILED),
        method="motor_thermal",
        safety_factor=(thermal.limit_c - spec.ambient_c)
        / max(thermal.temperature_rise_k, 1e-9),
        detail=f"winding {thermal.winding_c:.1f} C against a "
               f"{thermal.limit_c:.0f} C class "
               f"{best.motor.insulation_class.value} limit at "
               f"{spec.ambient_c:.0f} C ambient",
        optimistic_assumption="steady state with a single lumped thermal "
                              "resistance that the mounting dominates, so it "
                              "can be out by a factor of two either way"))
    return {"motor": best.motor.id, "gearbox": best.gearbox.id,
            "ratio": best.gearbox.ratio, "mass_kg": best.total_mass_kg,
            "candidate": best, "alternatives": len(alternatives)}


def _shaft_and_bearing_checks(spec: JointSpec, drivetrain: dict,
                              verdict: AssemblyVerdict) -> tuple[dict, str | None]:
    """Carry the drivetrain torque out through a shaft on two bearings."""
    from drivetrain.bearings import all_bearings, rate_bearing
    from drivetrain.loadpath import ShaftLayout, trace
    from physics.shaft import (analyze_shaft, de_goodman_diameter_m,
                               first_critical_speed_rad_s)

    candidate = drivetrain.get("candidate")
    if candidate is None:
        return {}, None

    layout = ShaftLayout(bearing_span_m=spec.bearing_span_m,
                         overhang_m=spec.shaft_overhang_m,
                         radial_load_n=spec.tip_load_n)
    path = trace(candidate, layout)
    steel = get_material(spec.shaft_material_id)
    loads = path.shaft_loads()

    shaft = analyze_shaft(loads, steel, spec.shaft_diameter_m)
    verdict.add(CheckResult(
        component="shaft", failure_mode="static_stress",
        status=(CheckStatus.PASSED if satisfies(shaft.static_safety_factor)
                else CheckStatus.FAILED),
        method="shaft_combined", safety_factor=shaft.static_safety_factor,
        detail=f"von Mises {shaft.von_mises_pa / 1e6:.1f} MPa at "
               f"{spec.shaft_diameter_m * 1e3:.1f} mm diameter",
        optimistic_assumption="a solid round section with no keyway modelled "
                              "beyond an assumed concentration factor"))
    verdict.add(CheckResult(
        component="shaft", failure_mode="fatigue",
        status=(CheckStatus.PASSED if satisfies(shaft.fatigue_safety_factor)
                else CheckStatus.FAILED),
        method="shaft_combined", safety_factor=shaft.fatigue_safety_factor,
        detail=f"DE-Goodman with reversed bending {path.bending_moment_nm:.1f} "
               f"N m and steady torque {path.output_torque_nm:.1f} N m; the "
               f"diameter for a factor of 2 would be "
               f"{de_goodman_diameter_m(loads, steel, 2.0) * 1e3:.1f} mm",
        optimistic_assumption="assumed stress concentration factors rather "
                              "than measured ones for the actual shoulder and "
                              "keyway"))

    critical = first_critical_speed_rad_s(
        steel, spec.shaft_diameter_m, spec.bearing_span_m + spec.shaft_overhang_m)
    speed_margin = critical / max(path.speed_rad_s, 1e-9)
    verdict.add(CheckResult(
        component="shaft", failure_mode="critical_speed",
        status=(CheckStatus.PASSED if satisfies(speed_margin / 2.0)
                else CheckStatus.FAILED),
        method="shaft_combined", safety_factor=speed_margin / 2.0,
        detail=f"first critical {critical:.0f} rad/s against an operating "
               f"{path.speed_rad_s:.2f} rad/s, required margin 2x",
        optimistic_assumption="a bare uniform shaft with no attached gear or "
                              "rotor mass, which is what actually lowers the "
                              "critical speed"))

    # Pick the smallest bearing that meets the required life, which is what a
    # designer does rather than taking the first that fits.
    chosen = None
    chosen_result = None
    for bearing in all_bearings():
        if bearing.bore_m > spec.shaft_diameter_m * 1.5:
            continue
        try:
            rated = rate_bearing(bearing, path.near_bearing_load_n,
                                 path.speed_rad_s,
                                 required_hours=spec.required_bearing_life_h)
        except ValueError:
            continue
        if rated.passes and (chosen_result is None
                             or bearing.mass_kg < chosen.mass_kg):
            chosen, chosen_result = bearing, rated
    if chosen is None:
        smallest = min((b for b in all_bearings()
                        if b.bore_m <= spec.shaft_diameter_m * 1.5),
                       key=lambda b: b.mass_kg, default=None)
        if smallest is not None:
            chosen, chosen_result = smallest, rate_bearing(
                smallest, path.near_bearing_load_n, path.speed_rad_s,
                required_hours=spec.required_bearing_life_h)

    if chosen_result is not None:
        verdict.add(CheckResult(
            component="bearing", failure_mode="l10_life",
            status=(CheckStatus.PASSED
                    if satisfies(chosen_result.life_margin or 0.0)
                    else CheckStatus.FAILED),
            method="bearing_l10", safety_factor=chosen_result.life_margin,
            detail=f"{chosen.designation} reaches "
                   f"{chosen_result.l10_hours:,.0f} h against a "
                   f"{spec.required_bearing_life_h:,.0f} h requirement at "
                   f"{path.near_bearing_load_n:.0f} N",
            optimistic_assumption="L10 is a statistic where one bearing in ten "
                                  "fails sooner, and the ISO 281 reliability, "
                                  "lubrication and contamination factors are "
                                  "not applied, each able to move the life by "
                                  "more than an order of magnitude"))
        verdict.add(CheckResult(
            component="bearing", failure_mode="static_capacity",
            status=(CheckStatus.PASSED
                    if satisfies(chosen_result.static_safety_factor)
                    else CheckStatus.FAILED),
            method="bearing_l10",
            safety_factor=chosen_result.static_safety_factor,
            detail=f"C0/P = {chosen_result.static_safety_factor:.1f}",
            optimistic_assumption="the catalogue ratings are representative "
                                  "for the size class, not a manufacturer's"))

    return ({"output_torque_nm": path.output_torque_nm,
             "bending_moment_nm": path.bending_moment_nm,
             "near_bearing_load_n": path.near_bearing_load_n,
             "speed_rad_s": path.speed_rad_s},
            None if chosen is None else chosen.designation)


def _mount_and_gear_checks(spec: JointSpec, load_path: dict,
                           drivetrain: dict, verdict: AssemblyVerdict) -> None:
    """The bolted mount that holds the joint, and the gearbox teeth."""
    from physics.gears import GearMesh, analyze_mesh
    from physics.joints import PropertyClass, analyze_joint

    torque = load_path.get("output_torque_nm")
    if torque is None:
        return

    # The reaction torque is carried by the bolt pattern as a couple.
    bolt_load = torque / (spec.mount_lever_arm_m * spec.mount_bolt_count) * 2.0
    joint = analyze_joint(spec.mount_bolt_size, PropertyClass.C8_8,
                          spec.grip_length_m, bolt_load,
                          external_load_min_n=0.0)
    for mode, factor, note in (
            ("separation", joint.separation_margin,
             "the nut factor relating torque to preload scatters by about 30 "
             "percent, so the achieved preload is uncertain by that much"),
            ("bolt_yield", joint.yield_safety_factor,
             "a single axially loaded bolt, with no bolt-group eccentricity"),
            ("bolt_fatigue", joint.fatigue_safety_factor,
             "rolled threads assumed, which are substantially better in "
             "fatigue than cut ones")):
        if factor is None:
            continue
        verdict.add(CheckResult(
            component="mount", failure_mode=mode,
            status=CheckStatus.PASSED if satisfies(factor) else CheckStatus.FAILED,
            method="bolted_joint", safety_factor=factor,
            detail=f"{spec.mount_bolt_count} x {spec.mount_bolt_size} class "
                   f"8.8 at {bolt_load:.0f} N each, preload "
                   f"{joint.preload_n:.0f} N, load factor "
                   f"{joint.load_factor:.3f}",
            optimistic_assumption=note))

    ratio = drivetrain.get("ratio")
    if ratio is None:
        return

    if spec.gear_mesh is None:
        # The catalogue gives a gearbox as a ratio, a torque rating and a mass.
        # It does not give the tooth geometry, and inventing a representative
        # mesh would produce a verdict about a gear that does not exist. An
        # earlier version did exactly that and reported the gearbox as FAILED
        # on the strength of a mesh nobody had specified, which is a fabricated
        # result wearing the same clothes as a measured one.
        verdict.add(CheckResult(
            component="gearbox", failure_mode="tooth_bending_and_pitting",
            status=CheckStatus.NOT_ASSESSED,
            detail="the gearbox archetype states a ratio, a torque rating and "
                   "a mass, not its internal tooth geometry. A method exists "
                   "(gear_tooth) and cannot be applied without the mesh. "
                   "Supply JointSpec.gear_mesh to have it checked"))
        return

    module_m, pinion_teeth, gear_teeth, face_width_m = spec.gear_mesh
    stage_ratio = gear_teeth / pinion_teeth
    mesh = GearMesh(module_m=module_m, pinion_teeth=pinion_teeth,
                    gear_teeth=gear_teeth, face_width_m=face_width_m,
                    torque_nm=torque / stage_ratio)
    gear = analyze_mesh(mesh, 200e6, 700e6, bending_correction=1.8,
                        contact_correction=1.3)
    for mode, factor in (("tooth_bending", gear.bending_safety_factor),
                         ("tooth_pitting", gear.contact_safety_factor)):
        verdict.add(CheckResult(
            component="gearbox", failure_mode=mode,
            status=CheckStatus.PASSED if satisfies(factor) else CheckStatus.FAILED,
            method="gear_tooth", safety_factor=factor,
            detail=f"final stage, module {module_m * 1e3:g} mm, "
                   f"{pinion_teeth}/{gear_teeth} teeth, "
                   f"{face_width_m * 1e3:g} mm face, carrying "
                   f"{mesh.torque_nm:.2f} N m",
            optimistic_assumption="Lewis and elementary Hertz with assumed "
                                  "correction factors, not a full AGMA rating"))


def _coverage_gaps(verdict: AssemblyVerdict) -> None:
    """Record every failure mode this project can name and cannot evaluate.

    Silence here would be the worst outcome available: an unchecked mode that
    nobody lists reads exactly like a mode that passed.
    """
    for component, modes in KNOWN_UNIMPLEMENTED_MODES.items():
        for mode in modes:
            verdict.add(CheckResult(
                component=component, failure_mode=mode,
                status=CheckStatus.NOT_ASSESSED,
                detail="no registered method evaluates this mode; it is listed "
                       "so the gap is visible rather than silent"))


def design_joint(spec: JointSpec | None = None) -> CapstoneResult:
    """Design and verify the whole joint, conjunctively.

    One forward pass: see the module note on sequential sizing. The verdict
    covers every component and reports the governing check plus every gap.
    """
    spec = spec or JointSpec()
    verdict = AssemblyVerdict()

    link = _link_checks(spec, verdict)
    drivetrain = _drivetrain_checks(spec, link, verdict)
    load_path, bearing = _shaft_and_bearing_checks(spec, drivetrain, verdict)
    _mount_and_gear_checks(spec, load_path, drivetrain, verdict)
    _coverage_gaps(verdict)

    material = get_material(spec.material_id)
    mass = link.get("mass_kg", 0.0) + drivetrain.get("mass_kg", 0.0)
    return CapstoneResult(
        spec=spec, verdict=verdict, link_design=link, drivetrain=drivetrain,
        load_path=load_path, selected_bearing=bearing, total_mass_kg=mass,
        material_cost_usd=link.get("mass_kg", 0.0)
        * (material.price_per_kg_usd or 0.0))
