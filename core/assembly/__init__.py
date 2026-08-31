"""core.assembly: multi-body assemblies, kinematics and statics.

Conventions are fixed in `frames`: right-handed, SI, y up, gravity along -y,
matching the beam model so load directions cannot silently disagree.
"""

from .frames import (
    GRAVITY_DIRECTION,
    STANDARD_GRAVITY,
    compose,
    identity,
    inverse,
    is_rigid_transform,
    rotation_about_axis,
    transform_point,
    translation,
    translation_along_axis,
)
from .kinematics import (
    IKResult,
    Pose,
    forward_kinematics,
    geometric_jacobian,
    inverse_kinematics,
    link_tip_position,
    position_jacobian,
    supporting_joints,
)
from .model import Assembly, Joint, JointType, Link
from .statics import (
    LinkLoad,
    gravity_vector,
    joint_torques,
    link_com_positions,
    link_load_cases,
    worst_gravity_pose,
)

__all__ = [
    "Assembly", "GRAVITY_DIRECTION", "IKResult", "Joint", "JointType", "Link",
    "LinkLoad", "Pose", "STANDARD_GRAVITY", "compose", "forward_kinematics",
    "geometric_jacobian", "gravity_vector", "identity", "inverse",
    "inverse_kinematics", "is_rigid_transform", "joint_torques",
    "link_com_positions", "link_load_cases", "link_tip_position",
    "position_jacobian", "rotation_about_axis", "supporting_joints",
    "transform_point", "translation", "translation_along_axis",
    "worst_gravity_pose",
]
