"""core.assembly.kinematics: forward kinematics, Jacobian, inverse kinematics.

Nothing here is specialised to a planar arm. The Jacobian is the standard
geometric one built from joint axes and origins in world coordinates, so it
stays correct for a spatial chain, and the planar closed-form results used in
the tests are an independent check on it rather than its source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import identity, inverse, position, rotation, transform_point
from .model import Assembly, Joint, JointType


@dataclass
class Pose:
    """World pose of every link and joint at a given configuration."""

    link_transforms: dict[str, np.ndarray] = field(default_factory=dict)
    joint_origins: dict[str, np.ndarray] = field(default_factory=dict)
    joint_axes: dict[str, np.ndarray] = field(default_factory=dict)
    tool: np.ndarray = field(default_factory=identity)

    def link_position(self, name: str) -> np.ndarray:
        return position(self.link_transforms[name])

    def tool_position(self) -> np.ndarray:
        return position(self.tool)


def _joint_values(assembly: Assembly, q) -> dict[str, float]:
    actuated = assembly.actuated_joints()
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != len(actuated):
        raise ValueError(
            f"expected {len(actuated)} joint values for {assembly.name}, "
            f"got {q.shape[0]}")
    return {joint.name: float(value) for joint, value in zip(actuated, q)}


def forward_kinematics(assembly: Assembly, q) -> Pose:
    """World pose of every link, plus the tool frame."""
    values = _joint_values(assembly, q)
    pose = Pose()
    world_of_link: dict[str | None, np.ndarray] = {None: identity()}

    for joint in assembly.ordered_joints():
        parent_world = world_of_link[joint.parent]
        origin_world = parent_world @ joint.origin_transform()
        value = values.get(joint.name, 0.0)
        child_world = origin_world @ joint.motion_transform(value)

        pose.joint_origins[joint.name] = position(origin_world)
        # The axis rotates with the parent, so it must be expressed in world.
        axis_local = np.asarray(joint.axis, dtype=np.float64)
        axis_local = axis_local / max(np.linalg.norm(axis_local), 1e-30)
        pose.joint_axes[joint.name] = rotation(origin_world) @ axis_local

        pose.link_transforms[joint.child] = child_world
        world_of_link[joint.child] = child_world

    tip = assembly.tip_link()
    tool_local = np.array([tip.length_m + assembly.tool_offset_m, 0.0, 0.0])
    tool = pose.link_transforms[tip.name].copy()
    tool[:3, 3] = transform_point(pose.link_transforms[tip.name], tool_local)
    pose.tool = tool
    return pose


def link_tip_position(assembly: Assembly, pose: Pose, link_name: str) -> np.ndarray:
    link = assembly.link(link_name)
    return transform_point(pose.link_transforms[link_name],
                           np.array([link.length_m, 0.0, 0.0]))


def supporting_joints(assembly: Assembly, link_name: str) -> list[str]:
    """Joints on the path from the base to a link, i.e. the ones that move it."""
    parent_joint = {j.child: j for j in assembly.joints}
    chain, current = [], link_name
    while current in parent_joint:
        joint = parent_joint[current]
        chain.append(joint.name)
        current = joint.parent
        if current is None:
            break
    return list(reversed(chain))


def geometric_jacobian(assembly: Assembly, q,
                       point_world: np.ndarray | None = None,
                       link_name: str | None = None) -> np.ndarray:
    """6 x n Jacobian at the tool, or at a point attached to `link_name`.

        revolute  : Jv = z x (p - p_joint),  Jw = z
        prismatic : Jv = z,                  Jw = 0

    A joint only contributes if the point actually moves with it. Without
    `link_name` the point is assumed to be on the tip link, which is right for
    the tool but WRONG for anything further up the chain: applying the formula
    to every joint would give an inboard link's centre of mass a sensitivity to
    an outboard joint that cannot move it. That error is small enough to look
    plausible in a torque result, which is exactly why it is guarded here.
    """
    pose = forward_kinematics(assembly, q)
    target = pose.tool_position() if point_world is None else np.asarray(
        point_world, dtype=np.float64)

    if link_name is None:
        link_name = assembly.tip_link().name
    contributing = set(supporting_joints(assembly, link_name))

    actuated = assembly.actuated_joints()
    jacobian = np.zeros((6, len(actuated)), dtype=np.float64)
    for i, joint in enumerate(actuated):
        if joint.name not in contributing:
            continue                      # this joint does not move the point
        axis = pose.joint_axes[joint.name]
        origin = pose.joint_origins[joint.name]
        if joint.type is JointType.REVOLUTE:
            jacobian[:3, i] = np.cross(axis, target - origin)
            jacobian[3:, i] = axis
        else:
            jacobian[:3, i] = axis
    return jacobian


def position_jacobian(assembly: Assembly, q,
                      point_world: np.ndarray | None = None,
                      link_name: str | None = None) -> np.ndarray:
    return geometric_jacobian(assembly, q, point_world, link_name)[:3, :]


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error_m: float
    hit_limits: list[str] = field(default_factory=list)


def inverse_kinematics(
    assembly: Assembly,
    target_position: np.ndarray,
    q0=None,
    tolerance_m: float = 1e-9,
    max_iterations: int = 500,
    damping: float = 1e-3,
    step_limit: float = 0.2,
) -> IKResult:
    """Position-only IK by damped least squares.

    Damping is what keeps this usable near a singularity: the pseudo-inverse
    blows up there, while (J J^T + lambda^2 I)^-1 stays bounded and simply
    stops making progress in the lost direction. The result reports whether it
    converged instead of returning a confident wrong answer.

    Joint limits are enforced by clamping each iterate, so the returned
    configuration is always reachable even when the target is not.
    """
    actuated = assembly.actuated_joints()
    target = np.asarray(target_position, dtype=np.float64).reshape(3)
    q = (np.zeros(len(actuated)) if q0 is None
         else np.asarray(q0, dtype=np.float64).reshape(-1).copy())
    if q.shape[0] != len(actuated):
        raise ValueError(f"q0 must have {len(actuated)} values")

    hit: list[str] = []
    error = np.inf
    for iteration in range(1, max_iterations + 1):
        pose = forward_kinematics(assembly, q)
        delta = target - pose.tool_position()
        error = float(np.linalg.norm(delta))
        if error < tolerance_m:
            return IKResult(q, True, iteration, error, sorted(set(hit)))

        j = position_jacobian(assembly, q)
        jjt = j @ j.T + (damping ** 2) * np.eye(3)
        step = j.T @ np.linalg.solve(jjt, delta)

        norm = np.linalg.norm(step)
        if norm > step_limit:
            step *= step_limit / norm
        q = q + step

        for i, joint in enumerate(actuated):
            clamped = joint.clamp(float(q[i]))
            if clamped != q[i]:
                hit.append(joint.name)
                q[i] = clamped

    return IKResult(q, False, max_iterations, error, sorted(set(hit)))
