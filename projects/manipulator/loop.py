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


def size_sections(arm: Assembly, spec: ManipulatorSpec,
                  actuator_masses: dict[str, float],
                  minimum_wall_m: float = 0.003) -> dict[str, Section]:
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
        sections[link_spec.name] = Section(
            outer_height_m=sizing.height_m,
            outer_width_m=link_spec.outer_width_m,
            wall_thickness_m=minimum_wall_m)
    return sections


def run_loop(spec: ManipulatorSpec = SPEC, max_iterations: int = 8,
             tolerance_kg: float = 1e-3) -> StageResult:
    """Iterate sections, torques and drives until the mass stops moving."""
    result = StageResult(name="mass torque loop")
    actuator_masses: dict[str, float] = {}
    sections = starting_sections(spec)
    history: list[Iteration] = []
    previous_total = None

    for index in range(max_iterations):
        arm = build_arm(sections, spec)
        material = get_material(arm.material_id)
        structure = sum(link.mass_kg(material.density_kg_m3) for link in arm.links)

        dynamics = dynamics_stage(arm, spec, samples=60)
        q = stretched_pose(spec)
        extra = actuator_gravity_torque(arm, q, actuator_masses)
        for row, added in zip(dynamics.rows, extra):
            row["peak_trapezoidal_nm"] = abs(row["peak_trapezoidal_nm"]) + abs(added)
            row["peak_s_curve_nm"] = abs(row["peak_s_curve_nm"]) + abs(added)
            row["rms_trapezoidal_nm"] = abs(row["rms_trapezoidal_nm"]) + abs(added)
            row["rms_s_curve_nm"] = abs(row["rms_s_curve_nm"]) + abs(added)

        drives = drivetrain_stage(dynamics, spec)
        actuator_masses = {row["joint"]: row.get("mass_kg", 0.0)
                           for row in drives.rows
                           if row.get("status") == "selected"}
        unselected = [row["joint"] for row in drives.rows
                      if row.get("status") != "selected"]
        actuator_total = sum(actuator_masses.values())
        total = structure + actuator_total
        shoulder = next(row["peak_trapezoidal_nm"] for row in dynamics.rows
                        if row["joint"] == "j2_shoulder")

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
        sections = size_sections(arm, spec, actuator_masses)
    else:
        result.notes.append(
            f"did NOT converge in {max_iterations} iterations; the history "
            f"below is what it did instead")

    result.rows = [item.row() for item in history]
    result.data["history"] = history
    result.data["final_sections"] = sections
    result.data["actuator_masses"] = actuator_masses
    result.data["unselected"] = history[-1].unselected
    result.notes.append(
        "the actuator masses are placed at their joint origins by this module, "
        "because the assembly model has no point mass; the assembly's own mass "
        "is the structure only")
    return result
