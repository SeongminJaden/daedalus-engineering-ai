"""The mass-torque loop: the part of this design that has to be iterated.

A lighter link needs less torque, a smaller drive weighs less, and a lighter
drive needs less torque again. The loop below runs that circle until it stops
moving, and reports how many turns it took and what changed on each.

WHAT IS FED BACK
================
Three things, and only three, because these are the ones this repository can
compute:

    the section of each link, sized against the moment that link actually
    carries at the rated pose, with the deflection limit and the safety
    factor from the specification;
    the joint torques, which follow from the structure mass and the payload;
    the drive selection, which follows from the torques and whose mass then
    changes them.

WHAT IS NOT FED BACK, AND WHY
=============================
The actuator masses are carried OUTSIDE the assembly model. That model gives
a link a section and a length and computes its mass from them; it has no point
mass at a joint. So the actuator contribution to the joint torques is computed
here, by putting each actuator's weight at its joint origin and using the same
position Jacobian the statics uses. It is stated rather than hidden because it
means the assembly's own mass property does not include the drives, and any
number taken from the assembly alone is the structure only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.assembly import Assembly
from core.assembly.kinematics import forward_kinematics, position_jacobian
from core.assembly.statics import gravity_vector, joint_torques
from core.materials import get_material
from physics.sizing.cantilever import size_rectangular_cantilever

from .arm import Section, build_arm, payload_force_n, starting_sections, stretched_pose
from .spec import SPEC, ManipulatorSpec
from .stages import StageResult, drivetrain_stage, dynamics_stage


def actuator_gravity_torque(arm: Assembly, q, masses_by_joint: dict[str, float]
                            ) -> np.ndarray:
    """Joint torques from actuator masses placed at their joint origins.

    The assembly model has no point mass, so this is computed here with the
    same generalized force construction the statics uses: a weight at a point
    contributes J_point^T w, and the actuator must hold the negative of it.
    """
    actuated = arm.actuated_joints()
    torque = np.zeros(len(actuated))
    if not masses_by_joint:
        return torque
    pose = forward_kinematics(arm, q)
    weight_direction = gravity_vector()
    for joint in actuated:
        mass = masses_by_joint.get(joint.name, 0.0)
        if mass <= 0.0:
            continue
        point = pose.joint_origins[joint.name]
        jacobian = position_jacobian(arm, q, point_world=point,
                                     link_name=joint.child)
        torque += jacobian.T @ (mass * weight_direction)
    return -torque


@dataclass
class Iteration:
    index: int
    structure_mass_kg: float
    actuator_mass_kg: float
    total_mass_kg: float
    shoulder_torque_nm: float
    sections: dict[str, Section]
    selected: dict[str, str | None]
    unselected: list[str] = field(default_factory=list)

    def row(self) -> dict:
        return {"iteration": self.index,
                "structure_mass_kg": self.structure_mass_kg,
                "actuator_mass_kg": self.actuator_mass_kg,
                "total_mass_kg": self.total_mass_kg,
                "shoulder_peak_nm": self.shoulder_torque_nm,
                "joints_without_a_drive": len(self.unselected),
                "upper_arm_height_mm": self.sections["upper_arm"].outer_height_m * 1e3,
                "forearm_height_mm": self.sections["forearm"].outer_height_m * 1e3}


def carried_load_n(arm: Assembly, link_name: str, spec: ManipulatorSpec,
                   actuator_masses: dict[str, float]) -> float:
    """The weight a link carries beyond its own: everything outboard of it.

    A link's section is sized against the moment at its root, and that moment
    comes from the payload and from every link and actuator further out. This
    is the honest load for a cantilever sizing of that link.
    """
    material = get_material(arm.material_id)
    names = [link.name for link in arm.links]
    index = names.index(link_name)
    outboard_links = sum(link.mass_kg(material.density_kg_m3)
                         for link in arm.links[index + 1:])
    joints = [joint.name for joint in arm.actuated_joints()]
    outboard_actuators = sum(mass for joint, mass in actuator_masses.items()
                             if joints.index(joint) > index)
    return (spec.payload_kg + outboard_links + outboard_actuators) * 9.80665


def flange_mass_kg(spec: ManipulatorSpec, section: Section,
                   density_kg_m3: float, bolt_count: int = 4,
                   clearance_hole_m: float = 0.0066) -> float:
    """Mass of the two end flanges of one link.

    A flange is a plate of the section's outer size and the stated flange
    thickness, less the bolt holes. It exists because a 3 mm wall cannot take
    a 6.4 mm counterbore or hold 9 mm of thread, and it is counted because a
    structure that ignores its own joints is lighter than the thing it
    describes.

    This is an UPPER BOUND and a deliberate one. A real flange has a central
    bore for the actuator shaft and the wiring, and on the 98 mm wrist bodies
    that bore would remove most of the plate. Nothing here knows how big it
    is, because the actuator pages do not print a shaft or boss diameter, so
    the mass is counted as if the flange were solid and the note says which
    way the error runs.
    """
    import math

    plate = section.outer_height_m * section.outer_width_m
    holes = bolt_count * math.pi * (clearance_hole_m / 2.0) ** 2
    return 2.0 * max(plate - holes, 0.0) * spec.flange_thickness_m * density_kg_m3


def structure_mass_kg(arm: Assembly, spec: ManipulatorSpec,
                      sections: dict[str, Section]) -> tuple[float, float]:
    """(tube mass, flange mass) for the whole arm."""
    material = get_material(arm.material_id)
    tubes = sum(link.mass_kg(material.density_kg_m3) for link in arm.links)
    flanges = sum(flange_mass_kg(spec, sections[link.name],
                                 material.density_kg_m3) for link in arm.links)
    return tubes, flanges


def actuator_section_floor(spec: ManipulatorSpec,
                           drives: dict[str, str] | None = None
                           ) -> dict[str, float]:
    """The smallest section each link may have, from the drive it carries.

    A link has to contain its actuator. Where the actuator's outline is
    printed the floor is that outline; where it is not, the floor falls back
    to the joint interface, and the envelope stage reports which links are
    therefore unverified.
    """
    from drivetrain.sourced import sourced_motor

    floors = {link.name: spec.minimum_section_m for link in spec.links()}
    if not drives:
        return floors
    joints = [joint.name for joint in spec.joints()]
    links = [link.name for link in spec.links()]
    for joint_name, part in drives.items():
        if not part or "+" in part:
            continue
        try:
            actuator = sourced_motor(part)
        except Exception:
            continue
        if actuator.outer_diameter_m is None:
            continue
        link_name = links[joints.index(joint_name)]
        floors[link_name] = max(floors[link_name], actuator.outer_diameter_m)
    return floors


def size_sections(arm: Assembly, spec: ManipulatorSpec,
                  actuator_masses: dict[str, float],
                  minimum_wall_m: float = 0.003,
                  section_floors: dict[str, float] | None = None
                  ) -> dict[str, Section]:
    """One cantilever sizing per link, at the load that link carries.

    The width and the wall are held at the starting values and the height is
    what the sizing returns, because a hollow rectangle has three dimensions
    and the sizing routine solves for one. That is a limit of the routine and
    is reported as such.
    """
    material = get_material(arm.material_id)
    sections = {}
    for link_spec in spec.links():
        load = carried_load_n(arm, link_spec.name, spec, actuator_masses)
        # Each link gets the share of the deflection budget its own length is
        # of the reach, so the sum of the link deflections is the budget.
        share = link_spec.length_m / max(spec.reach_check_m(), 1e-9)
        sizing = size_rectangular_cantilever(
            load_n=load, length_m=link_spec.length_m,
            width_m=link_spec.outer_width_m, material=material,
            safety_factor=spec.static_safety_factor_metal,
            deflection_limit_m=max(spec.tip_deflection_limit_m * share, 1e-6),
            minimum_height_m=max(4.0 * minimum_wall_m, 0.02))
        floor = (section_floors or {}).get(link_spec.name, 0.0)
        sections[link_spec.name] = Section(
            outer_height_m=max(sizing.height_m, floor),
            outer_width_m=max(link_spec.outer_width_m, floor),
            wall_thickness_m=minimum_wall_m)
    return sections


def run_loop(spec: ManipulatorSpec = SPEC, max_iterations: int = 8,
             tolerance_kg: float = 1e-3, optimise_sections: bool = False
             ) -> StageResult:
    """Iterate sections, torques and drives until the mass stops moving.

    `optimise_sections` swaps the one dimensional sizing for the three
    dimensional optimiser under the same floors. It is off by default because
    it costs about a minute a pass and the design document reports both.
    """
    result = StageResult(name="mass torque loop")
    actuator_masses: dict[str, float] = {}
    sections = starting_sections(spec)
    history: list[Iteration] = []
    previous_total = None

    for index in range(max_iterations):
        arm = build_arm(sections, spec)
        material = get_material(arm.material_id)
        tubes, flanges = structure_mass_kg(arm, spec, sections)
        structure = tubes + flanges

        dynamics = dynamics_stage(arm, spec, samples=60)
        q = stretched_pose(spec)
        extra = actuator_gravity_torque(arm, q, actuator_masses)
        for row, added in zip(dynamics.rows, extra):
            row["peak_trapezoidal_nm"] = abs(row["peak_trapezoidal_nm"]) + abs(added)
            row["peak_s_curve_nm"] = abs(row["peak_s_curve_nm"]) + abs(added)
            row["rms_trapezoidal_nm"] = abs(row["rms_trapezoidal_nm"]) + abs(added)
            row["rms_s_curve_nm"] = abs(row["rms_s_curve_nm"]) + abs(added)

        from physics.dynamics import mass_matrix
        inertia_matrix = mass_matrix(arm, q, get_material(arm.material_id).density_kg_m3)
        load_inertias = {joint.name: float(inertia_matrix[index, index])
                         for index, joint in enumerate(arm.actuated_joints())}
        drives = drivetrain_stage(dynamics, spec, load_inertias)
        actuator_masses = {row["joint"]: row.get("mass_kg", 0.0)
                           for row in drives.rows
                           if row.get("status") == "selected"}
        unselected = [row["joint"] for row in drives.rows
                      if row.get("status") != "selected"]
        actuator_total = sum(actuator_masses.values())
        total = structure + actuator_total
        shoulder = next(row["peak_trapezoidal_nm"] for row in dynamics.rows
                        if row["joint"] == "j2_shoulder")

        result.data.setdefault("flange_mass_kg", []).append(flanges)
        history.append(Iteration(index=index, structure_mass_kg=structure,
                                 actuator_mass_kg=actuator_total,
                                 total_mass_kg=total,
                                 shoulder_torque_nm=shoulder,
                                 sections=dict(sections),
                                 selected={row["joint"]: row.get("selected")
                                           for row in drives.rows},
                                 unselected=unselected))
        if previous_total is not None and abs(total - previous_total) < tolerance_kg:
            result.notes.append(
                f"converged after {index + 1} iterations: the total mass moved "
                f"less than {tolerance_kg * 1000:.0f} g")
            break
        previous_total = total
        floors = actuator_section_floor(
            spec, {row["joint"]: row.get("selected") for row in drives.rows})
        sizer = (size_sections_optimised if optimise_sections
                 else size_sections)
        sections = sizer(arm, spec, actuator_masses, section_floors=floors)
    else:
        result.notes.append(
            f"did NOT converge in {max_iterations} iterations; the history "
            f"below is what it did instead")

    result.rows = [item.row() for item in history]
    result.data["history"] = history
    result.data["final_sections"] = sections
    result.data["actuator_masses"] = actuator_masses
    result.data["unselected"] = history[-1].unselected
    result.data["optimised_sections"] = optimise_sections
    result.notes.append(
        ("sections came from the three dimensional optimiser"
         if optimise_sections else
         "sections came from the one dimensional sizing; the three "
         "dimensional optimiser is reported separately"))
    result.notes.append(
        "the structure mass includes the end flanges, two per link, which "
        "exist because the wall cannot take the counterbore or the thread")
    result.notes.append(
        "the actuator masses are placed at their joint origins by this module, "
        "because the assembly model has no point mass; the assembly's own mass "
        "is the structure only")
    return result


def link_problem(link_spec, load_n: float, limit_m: float,
                 spec: ManipulatorSpec):
    """One link as an Engineering IR problem, for the optimiser.

    The envelope is the specification's starting section, and the FLOOR is
    the joint interface: a link narrower than four M6 counterbores cannot
    bolt to the joint it belongs to. The optimiser has no way to know that,
    and left to itself it returns a 10 mm wide tube.
    """
    from core.engineering_ir.schema import (BoundaryCondition, Constraints,
                                            EngineeringProblem, Geometry, Load,
                                            Objective, ObjectiveQuantity,
                                            ObjectiveSense, SectionType, Vec3)

    return EngineeringProblem(
        name=f"{link_spec.name}_section",
        geometry=Geometry(length_m=link_spec.length_m,
                          max_height_m=link_spec.outer_height_m,
                          max_width_m=link_spec.outer_width_m,
                          section_type=SectionType.HOLLOW_RECTANGLE),
        material_id=spec.materials["link"],
        loads=[Load(magnitude_n=load_n, direction=Vec3(x=0.0, y=-1.0, z=0.0))],
        boundary_conditions=[BoundaryCondition()],
        constraints=Constraints(
            max_deflection_m=limit_m,
            min_safety_factor=spec.static_safety_factor_metal),
        objectives=[Objective(sense=ObjectiveSense.MINIMIZE,
                              quantity=ObjectiveQuantity.MASS)])


#: Iterations for the section optimiser. MEASURED, not guessed: one
#: evaluation runs a finite element solve at 0.617 s, the default 200
#: iterations cost 186 s per link, and ten iterations reach the same point in
#: 15 s. Anything above ten is paying three minutes for no movement.
SECTION_OPTIMISER_ITERATIONS = 10


def size_sections_optimised(arm: Assembly, spec: ManipulatorSpec,
                            actuator_masses: dict[str, float],
                            minimum_wall_m: float = 0.003,
                            apply_interface_floor: bool = True,
                            section_floors: dict[str, float] | None = None,
                            max_iter: int = SECTION_OPTIMISER_ITERATIONS
                            ) -> dict[str, Section]:
    """All three section dimensions at once, by the existing SLSQP.

    The one dimensional routine solves for the height with the width and the
    wall held where the specification put them, which makes the starting
    guess part of the answer. This calls the optimiser the repository already
    has, on the same constraints, and lets it move all three.

    A link the optimiser cannot solve keeps the one dimensional result rather
    than failing the run, and `sizing_comparison` reports which ones those
    were.
    """
    from optimization.constraints import build_optimization_problem
    from optimization.gradient.slsqp import optimize_slsqp

    fallback = size_sections(arm, spec, actuator_masses, minimum_wall_m,
                             section_floors=section_floors)
    sections: dict[str, Section] = {}
    for link_spec in spec.links():
        load = carried_load_n(arm, link_spec.name, spec, actuator_masses)
        share = link_spec.length_m / max(spec.reach_check_m(), 1e-9)
        limit = max(spec.tip_deflection_limit_m * share, 1e-6)
        try:
            op = build_optimization_problem(
                link_problem(link_spec, load, limit, spec))
            result = optimize_slsqp(op, max_iter=max_iter)
            # SLSQP often stops with "positive directional derivative for
            # linesearch" at a point that satisfies every constraint. That is
            # a line search giving up, not an infeasible answer, so the test
            # is feasibility and an improvement in mass, not the success flag.
            if not result.evaluation.is_feasible():
                sections[link_spec.name] = fallback[link_spec.name]
                continue
            width, height, wall = result.x
            # The interface floor, applied after the optimiser rather than as
            # a bound, so the table can show what the unconstrained optimum
            # was and what the floor cost.
            if apply_interface_floor:
                floor = max(spec.minimum_section_m,
                            (section_floors or {}).get(link_spec.name, 0.0))
                width = max(width, floor)
                height = max(height, floor)
                wall = max(wall, spec.minimum_wall_m)
            if wall >= 0.5 * min(width, height):
                # The bounds allow a wall that leaves no cavity. Such a point
                # is not a hollow section at all and the genome refuses it.
                sections[link_spec.name] = fallback[link_spec.name]
                continue
            sections[link_spec.name] = Section(outer_height_m=float(height),
                                               outer_width_m=float(width),
                                               wall_thickness_m=float(wall))
        except Exception:
            sections[link_spec.name] = fallback[link_spec.name]

    # Keep whichever is lighter, link by link, so the comparison cannot make
    # a link heavier than the routine it replaces.
    material = get_material(arm.material_id)
    for link_spec in spec.links():
        name = link_spec.name
        chosen, previous = sections[name], fallback[name]
        if (chosen.genome(arm.material_id).section.mass(
                link_spec.length_m, material.density_kg_m3)
                > previous.genome(arm.material_id).section.mass(
                    link_spec.length_m, material.density_kg_m3)):
            sections[name] = previous
    return sections


def sizing_comparison(spec: ManipulatorSpec = SPEC,
                      actuator_masses: dict[str, float] | None = None,
                      section_floors: dict[str, float] | None = None
                      ) -> list[dict]:
    """One dimension against three, on the same links and the same limits.

    Three columns, because the difference between them is the finding. The one
    dimensional routine solves for the depth with the width and wall fixed.
    The three dimensional one moves all three under the same floors. The third
    column drops the floors, which is what the optimiser wants and what the
    joint cannot accept.
    """
    actuator_masses = actuator_masses or {}
    arm = build_arm(starting_sections(spec), spec)
    material = get_material(arm.material_id)
    one = size_sections(arm, spec, actuator_masses,
                        section_floors=section_floors)
    three = size_sections_optimised(arm, spec, actuator_masses,
                                    section_floors=section_floors)
    unconstrained = size_sections_optimised(arm, spec, actuator_masses,
                                            apply_interface_floor=False)
    rows = []
    for link_spec in spec.links():
        name = link_spec.name
        def mass(section: Section) -> float:
            return section.genome(arm.material_id).section.mass(
                link_spec.length_m, material.density_kg_m3)
        rows.append({
            "link": name,
            "one_dimension_mass_kg": mass(one[name]),
            "three_dimension_mass_kg": mass(three[name]),
            "saving": 1.0 - mass(three[name]) / max(mass(one[name]), 1e-12),
            "one_dimension_hwt_mm": [round(v * 1e3, 2) for v in
                                     (one[name].outer_height_m,
                                      one[name].outer_width_m,
                                      one[name].wall_thickness_m)],
            "three_dimension_hwt_mm": [round(v * 1e3, 2) for v in
                                       (three[name].outer_height_m,
                                        three[name].outer_width_m,
                                        three[name].wall_thickness_m)],
            "no_interface_floor_mass_kg": mass(unconstrained[name]),
            "no_interface_floor_hwt_mm": [round(v * 1e3, 2) for v in
                                          (unconstrained[name].outer_height_m,
                                           unconstrained[name].outer_width_m,
                                           unconstrained[name].wall_thickness_m)]})
    return rows
