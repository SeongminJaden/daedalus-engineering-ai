"""projects.robotic_arm.arm: the two-link planar arm demonstration.

The capstone for Phase 10: define an assembly, pose it, work out what each link
carries, then run the existing structural stack on each link and export the
posed assembly to STEP.

Geometry convention matches the rest of the project: the arm works in the x-y
plane, joints rotate about +z, and gravity acts along -y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.assembly.frames import STANDARD_GRAVITY, translation
from core.assembly.kinematics import forward_kinematics
from core.assembly.model import Assembly, Joint, JointType, Link
from core.assembly.statics import joint_torques, link_load_cases, worst_gravity_pose
from core.design_genome import DesignGenome, HollowRectangleSection
from core.materials import get_material

MATERIAL_ID = "al_7075_t6"
PAYLOAD_KG = 2.0
LINK1_LENGTH_M = 0.30
LINK2_LENGTH_M = 0.25


def _link(name: str, length_m: float, b: float, h: float, t: float) -> Link:
    return Link(
        name=name,
        length_m=length_m,
        genome=DesignGenome(
            section=HollowRectangleSection(outer_width_m=b, outer_height_m=h,
                                           wall_thickness_m=t),
            material_id=MATERIAL_ID),
    )


def build_arm(
    link1_section: tuple[float, float, float] = (0.020, 0.040, 0.002),
    link2_section: tuple[float, float, float] = (0.016, 0.032, 0.002),
) -> Assembly:
    """Two revolute joints, both about +z, arm in the x-y plane."""
    return Assembly(
        name="two_link_planar_arm",
        material_id=MATERIAL_ID,
        links=[_link("link1", LINK1_LENGTH_M, *link1_section),
               _link("link2", LINK2_LENGTH_M, *link2_section)],
        joints=[
            Joint(name="shoulder", type=JointType.REVOLUTE, parent=None,
                  child="link1", axis=[0.0, 0.0, 1.0],
                  lower_limit=-np.pi, upper_limit=np.pi),
            Joint(name="elbow", type=JointType.REVOLUTE, parent="link1",
                  child="link2", axis=[0.0, 0.0, 1.0],
                  origin=translation(LINK1_LENGTH_M, 0.0, 0.0).tolist(),
                  lower_limit=-2.6, upper_limit=2.6),
        ],
    )


def payload_force_n(payload_kg: float = PAYLOAD_KG) -> np.ndarray:
    """A payload hanging under gravity: force along -y."""
    return np.array([0.0, -payload_kg * STANDARD_GRAVITY, 0.0])


@dataclass
class LinkVerdict:
    link: str
    equivalent_tip_load_n: float
    mass_kg: float
    max_bending_stress_pa: float
    tip_deflection_m: float
    safety_factor: float
    allowable_stress_pa: float
    passes: bool


def check_links(assembly: Assembly, q, payload_kg: float = PAYLOAD_KG,
                min_safety_factor: float = 2.0) -> list[LinkVerdict]:
    """Run the Phase 2 structural model on each link's own load case.

    Each link is treated as a cantilever carrying the equivalent tip load that
    reproduces its root bending moment. See `link_load_cases` for what that
    equivalence does and does not capture.
    """
    from physics.structural import BeamLoadCase, evaluate_beam_case

    material = get_material(assembly.material_id)
    allowable = material.yield_strength_pa / min_safety_factor
    loads = {lc.link: lc for lc in link_load_cases(
        assembly, q, material.density_kg_m3, tip_force_n=payload_force_n(payload_kg))}

    out = []
    for link in assembly.links:
        load = loads[link.name]
        case = BeamLoadCase(
            length_m=link.length_m,
            tip_load_n=abs(load.equivalent_tip_load_n),
            youngs_modulus_pa=material.axial_modulus_pa(),
            density_kg_m3=material.density_kg_m3,
            yield_strength_pa=material.yield_strength_pa,
            poisson_ratio=material.poisson_ratio,
        )
        section = link.genome.section
        metrics = evaluate_beam_case(
            np.array([section.outer_width_m]), np.array([section.outer_height_m]),
            np.array([section.wall_thickness_m]), case).candidate(0)
        out.append(LinkVerdict(
            link=link.name,
            equivalent_tip_load_n=abs(load.equivalent_tip_load_n),
            mass_kg=metrics["mass_kg"],
            max_bending_stress_pa=metrics["max_bending_stress_pa"],
            tip_deflection_m=metrics["tip_deflection_m"],
            safety_factor=metrics["safety_factor"],
            allowable_stress_pa=allowable,
            passes=metrics["max_bending_stress_pa"] <= allowable,
        ))
    return out


def analyse(assembly: Assembly | None = None, q=None,
            payload_kg: float = PAYLOAD_KG) -> dict:
    """Full pass: pose, torques, per-link loads and structural verdicts."""
    assembly = assembly or build_arm()
    material = get_material(assembly.material_id)
    if q is None:
        q, _ = worst_gravity_pose(assembly, material.density_kg_m3,
                                  payload_force_n(payload_kg))
    q = np.asarray(q, dtype=np.float64)

    pose = forward_kinematics(assembly, q)
    torques = joint_torques(assembly, q, material.density_kg_m3,
                            tip_force_n=payload_force_n(payload_kg))
    return {
        "assembly": assembly,
        "q": q,
        "tool_position_m": pose.tool_position(),
        "joint_torques_nm": torques,
        "link_loads": link_load_cases(assembly, q, material.density_kg_m3,
                                      tip_force_n=payload_force_n(payload_kg)),
        "verdicts": check_links(assembly, q, payload_kg),
        "total_mass_kg": assembly.total_mass_kg(material.density_kg_m3),
        "payload_kg": payload_kg,
    }
