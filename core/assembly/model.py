"""core.assembly.model: links, joints and the assembly tree.

An open kinematic chain: one base, joints connecting a parent link to a child
link, no closed loops. Loops need constraint solving and are a later phase; the
model rejects them rather than silently producing a wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.design_genome import DesignGenome

from .frames import (
    identity, is_rigid_transform, rotation_about_axis, translation_along_axis,
)


class JointType(str, Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"


class Link(BaseModel):
    """A structural member: the Phase 1 genome plus where its mass acts."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    genome: DesignGenome
    length_m: float = Field(gt=0.0)
    # Centre of mass along the link's own x axis, as a fraction of length. A
    # uniform prismatic bar has it at mid-span, which is the default.
    com_fraction: float = Field(default=0.5, ge=0.0, le=1.0)

    def section_area_m2(self) -> float:
        return self.genome.section_properties().area_m2

    def mass_kg(self, density_kg_m3: float) -> float:
        return self.genome.section.mass(self.length_m, density_kg_m3)

    def com_local(self) -> np.ndarray:
        return np.array([self.com_fraction * self.length_m, 0.0, 0.0])


class Joint(BaseModel):
    """Connects a parent link to a child link.

    `origin` is the child frame's pose in the parent frame at zero joint value.
    The joint value then rotates about (revolute) or translates along
    (prismatic) `axis`, expressed in that origin frame.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    type: JointType
    parent: str | None = None          # None means attached to the base
    child: str = Field(min_length=1)
    axis: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    origin: list[list[float]] | None = None      # 4x4, identity when omitted
    lower_limit: float | None = None
    upper_limit: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> "Joint":
        if self.origin is not None:
            t = np.asarray(self.origin, dtype=np.float64)
            if not is_rigid_transform(t):
                raise ValueError(
                    f"joint {self.name}: origin is not a rigid transform")
        if self.type is not JointType.FIXED:
            if np.linalg.norm(np.asarray(self.axis, dtype=np.float64)) < 1e-12:
                raise ValueError(f"joint {self.name}: axis must be non-zero")
        if (self.lower_limit is not None and self.upper_limit is not None
                and self.lower_limit > self.upper_limit):
            raise ValueError(
                f"joint {self.name}: lower_limit exceeds upper_limit")
        return self

    def origin_transform(self) -> np.ndarray:
        if self.origin is None:
            return identity()
        return np.asarray(self.origin, dtype=np.float64)

    def motion_transform(self, value: float) -> np.ndarray:
        if self.type is JointType.FIXED:
            return identity()
        if self.type is JointType.REVOLUTE:
            return rotation_about_axis(self.axis, value)
        return translation_along_axis(self.axis, value)

    def is_actuated(self) -> bool:
        return self.type is not JointType.FIXED

    def clamp(self, value: float) -> float:
        if self.lower_limit is not None:
            value = max(value, self.lower_limit)
        if self.upper_limit is not None:
            value = min(value, self.upper_limit)
        return value

    def within_limits(self, value: float, tol: float = 1e-9) -> bool:
        if self.lower_limit is not None and value < self.lower_limit - tol:
            return False
        if self.upper_limit is not None and value > self.upper_limit + tol:
            return False
        return True


@dataclass(frozen=True)
class LimitViolation:
    """One actuated joint asked to go somewhere it cannot.

    Carries the limit it broke and by how much, because "infeasible" without a
    number tells a caller nothing about whether the pose was slightly outside
    or nowhere near.
    """

    joint: str
    value: float
    lower_limit: float | None
    upper_limit: float | None

    @property
    def excess(self) -> float:
        if self.lower_limit is not None and self.value < self.lower_limit:
            return self.lower_limit - self.value
        if self.upper_limit is not None and self.value > self.upper_limit:
            return self.value - self.upper_limit
        return 0.0

    def __str__(self) -> str:
        bound = (f"[{self.lower_limit}, {self.upper_limit}]")
        return (f"joint {self.joint!r} at {self.value:.6f} is outside {bound} "
                f"by {self.excess:.6f}")


class Assembly(BaseModel):
    """An open chain of links and joints, rooted at the base."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    links: list[Link] = Field(min_length=1)
    joints: list[Joint] = Field(min_length=1)
    material_id: str = Field(min_length=1)
    # Tool frame in the last link's frame; defaults to the link tip.
    tool_offset_m: float = 0.0

    @model_validator(mode="after")
    def _validate_tree(self) -> "Assembly":
        link_names = [link.name for link in self.links]
        if len(set(link_names)) != len(link_names):
            raise ValueError("duplicate link names")
        joint_names = [j.name for j in self.joints]
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("duplicate joint names")

        known = set(link_names)
        children = [j.child for j in self.joints]
        if len(set(children)) != len(children):
            raise ValueError(
                "a link has more than one parent joint; this model is an open "
                "chain and closed loops are not supported")
        for joint in self.joints:
            if joint.child not in known:
                raise ValueError(f"joint {joint.name}: unknown child {joint.child}")
            if joint.parent is not None and joint.parent not in known:
                raise ValueError(f"joint {joint.name}: unknown parent {joint.parent}")

        roots = [j for j in self.joints if j.parent is None]
        if len(roots) != 1:
            raise ValueError(
                f"expected exactly one joint attached to the base, found {len(roots)}")

        # Every link must be reachable from the base: a floating link would be
        # silently ignored by forward kinematics.
        reachable, frontier = set(), [roots[0].child]
        by_parent: dict[str, list[Joint]] = {}
        for joint in self.joints:
            by_parent.setdefault(joint.parent or "", []).append(joint)
        while frontier:
            current = frontier.pop()
            if current in reachable:
                raise ValueError("cycle detected in the joint tree")
            reachable.add(current)
            for joint in by_parent.get(current, []):
                frontier.append(joint.child)
        missing = set(link_names) - reachable
        if missing:
            raise ValueError(f"links not connected to the base: {sorted(missing)}")
        return self

    # --- accessors ------------------------------------------------------ #
    def link(self, name: str) -> Link:
        for link in self.links:
            if link.name == name:
                return link
        raise KeyError(f"unknown link {name}")

    def joint(self, name: str) -> Joint:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(f"unknown joint {name}")

    def ordered_joints(self) -> list[Joint]:
        """Joints from base outward, so a single pass computes the chain."""
        by_parent: dict[str | None, list[Joint]] = {}
        for joint in self.joints:
            by_parent.setdefault(joint.parent, []).append(joint)
        out, frontier = [], list(by_parent.get(None, []))
        while frontier:
            joint = frontier.pop(0)
            out.append(joint)
            frontier.extend(by_parent.get(joint.child, []))
        return out

    def limit_violations(self, q, tol: float = 1e-9
                         ) -> "tuple[LimitViolation, ...]":
        """Which actuated joints are outside their stated range at q.

        Joint limits were recorded and never checked anywhere but IK, which
        clamps silently. A pose that violates them is not a slightly worse
        design, it is a pose the mechanism cannot reach, and any torque or
        stress computed there describes something that cannot happen.

        Returns every violation rather than the first, so a caller sees the
        whole picture instead of fixing one and rediscovering the next.
        """
        import numpy as _np

        values = _np.asarray(q, dtype=float).reshape(-1)
        actuated = self.actuated_joints()
        if values.shape[0] != len(actuated):
            raise ValueError(
                f"expected {len(actuated)} joint values for {self.name}, "
                f"got {values.shape[0]}")
        return tuple(
            LimitViolation(joint=joint.name, value=float(value),
                           lower_limit=joint.lower_limit,
                           upper_limit=joint.upper_limit)
            for joint, value in zip(actuated, values)
            if not joint.within_limits(float(value), tol))

    def within_limits(self, q, tol: float = 1e-9) -> bool:
        """Whether every actuated joint is inside its stated range at q."""
        return not self.limit_violations(q, tol)

    def actuated_joints(self) -> list[Joint]:
        return [j for j in self.ordered_joints() if j.is_actuated()]

    @property
    def dof(self) -> int:
        return len(self.actuated_joints())

    def tip_link(self) -> Link:
        return self.link(self.ordered_joints()[-1].child)

    def total_mass_kg(self, density_kg_m3: float) -> float:
        return sum(link.mass_kg(density_kg_m3) for link in self.links)
