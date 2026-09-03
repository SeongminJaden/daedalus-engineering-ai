"""Building the arm as an Assembly, from the specification and a set of sections.

The mass-torque loop changes sections and rebuilds, so this is a function of
the sections rather than a fixed object. Everything else in the pipeline reads
the Assembly this returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.assembly import Assembly, Joint, JointType, Link
from core.design_genome import DesignGenome, HollowRectangleSection

from .spec import SPEC, LinkSpec, ManipulatorSpec


@dataclass(frozen=True)
class Section:
    """The three numbers that describe one link's section here."""

    outer_height_m: float
    outer_width_m: float
    wall_thickness_m: float

    def genome(self, material_id: str) -> DesignGenome:
        return DesignGenome(
            section=HollowRectangleSection(
                outer_width_m=self.outer_width_m,
                outer_height_m=self.outer_height_m,
                wall_thickness_m=self.wall_thickness_m),
            material_id=material_id)


def starting_sections(spec: ManipulatorSpec = SPEC) -> dict[str, Section]:
    return {link.name: Section(link.outer_height_m, link.outer_width_m,
                               link.wall_thickness_m)
            for link in spec.links()}


def _origin(x_m: float = 0.0, y_m: float = 0.0) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[0, 3] = x_m
    matrix[1, 3] = y_m
    return matrix.tolist()


def build_arm(sections: dict[str, Section] | None = None,
              spec: ManipulatorSpec = SPEC,
              material_id: str | None = None) -> Assembly:
    """The six axis arm with the given sections.

    One material for the structure, because the assembly model carries a
    single density and mixing materials would make its mass wrong without
    saying so. The cover material in the specification is used by the
    manufacturability stage, not by the statics.
    """
    sections = sections or starting_sections(spec)
    material = material_id or spec.materials["link"]
    link_specs: dict[str, LinkSpec] = {link.name: link for link in spec.links()}
    joints = spec.joints()

    links = []
    for link_spec in spec.links():
        section = sections[link_spec.name]
        links.append(Link(name=link_spec.name, length_m=link_spec.length_m,
                          genome=section.genome(material)))

    built = []
    for index, joint in enumerate(joints):
        parent = None if index == 0 else links[index - 1].name
        built.append(Joint(name=joint.name, type=JointType.REVOLUTE,
                           parent=parent, child=links[index].name,
                           axis=list(joint.axis),
                           origin=_origin(joint.origin_x_m, joint.origin_y_m),
                           lower_limit=joint.lower_limit_rad,
                           upper_limit=joint.upper_limit_rad))
    return Assembly(name="daedalus_6dof", material_id=material, links=links,
                    joints=built)


def stretched_pose(spec: ManipulatorSpec = SPEC) -> np.ndarray:
    """The rated pose: the arm straight out horizontally at full reach.

    This is where the payload makes the largest moment about the shoulder,
    which is the pose the task rates the arm at.
    """
    return np.zeros(spec.degrees_of_freedom)


def payload_force_n(spec: ManipulatorSpec = SPEC) -> np.ndarray:
    # Gravity is -y here, so the payload weight is too.
    return np.array([0.0, -spec.payload_kg * 9.80665, 0.0])
