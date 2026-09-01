"""Joint limits are enforced, not merely recorded.

The limits were in the model from the start and consulted in exactly one
place: inverse kinematics, which clamps silently. Everything else accepted any
pose. This file covers the enforcement that closes that.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.assembly.model import Assembly, LimitViolation
from core.assembly.statics import joint_torques, worst_gravity_pose
from physics.dynamics.load_cases import LoadCase, evaluate_case
from projects.robotic_arm.arm import build_arm

DENSITY = 2810.0


def restricted(lower: float, upper: float) -> Assembly:
    """The standard arm with its shoulder confined to a band."""
    arm = build_arm()
    return Assembly(
        name="restricted", material_id=arm.material_id, links=arm.links,
        joints=[arm.joints[0].model_copy(update={"lower_limit": lower,
                                                 "upper_limit": upper}),
                arm.joints[1]])


# --------------------------------------------------------- reporting a breach

def test_a_reachable_pose_reports_no_violation():
    assert build_arm().within_limits([0.0, 0.0])
    assert build_arm().limit_violations([0.0, 0.0]) == ()


def test_every_violated_joint_is_reported_not_just_the_first():
    """Fixing one and rediscovering the next is a bad way to learn a model."""
    violations = build_arm().limit_violations([4.0, 3.0])
    assert {v.joint for v in violations} == {"shoulder", "elbow"}


def test_a_violation_carries_how_far_outside_it_is():
    """Infeasible without a number says nothing about how infeasible."""
    (violation,) = build_arm().limit_violations([0.0, 3.0])
    assert isinstance(violation, LimitViolation)
    assert violation.joint == "elbow"
    assert violation.excess == pytest.approx(0.4)
    assert "outside" in str(violation)


def test_a_pose_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="expected 2 joint values"):
        build_arm().limit_violations([0.0])


def test_the_boundary_itself_is_reachable():
    """A limit is a limit, not an exclusive bound."""
    arm = build_arm()
    assert arm.within_limits([np.pi, 2.6])
    assert not arm.within_limits([np.pi + 1e-6, 0.0])


# ------------------------------------- the worst case has to be a reachable one

def test_the_worst_pose_search_stays_inside_the_limits():
    """It used to sweep the whole circle whatever the joint allowed."""
    arm = restricted(-0.5, 0.5)
    q, _ = worst_gravity_pose(arm, DENSITY)
    assert arm.within_limits(q), f"returned an unreachable pose {q}"
    assert -0.5 - 1e-9 <= q[0] <= 0.5 + 1e-9


def test_restricting_the_joint_changes_the_sizing_case_not_just_the_pose():
    """The reason this matters, in numbers.

    Confined near vertical the arm carries far less gravity torque. The old
    search reported the horizontal value, which is a pose that joint cannot
    reach, and would have sized the actuator about fourteen times too high.
    """
    _, wide = worst_gravity_pose(restricted(-np.pi, np.pi), DENSITY)
    narrow_arm = restricted(1.5, 1.57)
    q, narrow = worst_gravity_pose(narrow_arm, DENSITY)
    assert narrow_arm.within_limits(q)
    assert narrow < wide / 10.0


def test_the_unrestricted_arm_is_unaffected():
    """The standard arm's shoulder already spans the search range."""
    arm = build_arm()
    q, torque = worst_gravity_pose(arm, DENSITY)
    assert arm.within_limits(q)
    assert torque == pytest.approx(0.7931, rel=1e-3)


def test_a_joint_with_no_reachable_angle_is_refused():
    arm = build_arm()
    impossible = Assembly(
        name="impossible", material_id=arm.material_id, links=arm.links,
        joints=[arm.joints[0].model_copy(update={"lower_limit": 4.0,
                                                 "upper_limit": 5.0}),
                arm.joints[1]])
    with pytest.raises(ValueError, match="no reachable angle"):
        worst_gravity_pose(impossible, DENSITY)


def test_held_joints_outside_their_limits_invalidate_the_whole_search():
    """The search holds the other joints at zero, so zero has to be legal."""
    arm = build_arm()
    offset = Assembly(
        name="offset", material_id=arm.material_id, links=arm.links,
        joints=[arm.joints[0],
                arm.joints[1].model_copy(update={"lower_limit": 1.0,
                                                 "upper_limit": 2.0})])
    with pytest.raises(ValueError, match="held joints"):
        worst_gravity_pose(offset, DENSITY)


# ------------------------------------------- a duty cycle of reachable poses

def test_an_unreachable_load_case_is_refused():
    """Sizing a motor from a pose the arm cannot enter is not conservative."""
    case = LoadCase(name="past the stop", description="elbow beyond its limit",
                    q=np.array([0.0, 3.0]), qd=np.zeros(2), qdd=np.zeros(2),
                    payload_kg=2.0, duty_fraction=0.1)
    with pytest.raises(ValueError, match="not reachable"):
        evaluate_case(build_arm(), case, DENSITY)


def test_studying_an_out_of_limit_pose_has_to_be_asked_for():
    """A legitimate thing to want, and it should not be the default."""
    case = LoadCase(name="past the stop", description="elbow beyond its limit",
                    q=np.array([0.0, 3.0]), qd=np.zeros(2), qdd=np.zeros(2),
                    payload_kg=2.0, duty_fraction=0.1)
    result = evaluate_case(build_arm(), case, DENSITY,
                           require_reachable=False)
    assert np.abs(result.torque_nm).max() > 0.0


def test_the_standard_duty_cycle_is_reachable_throughout():
    """The shipped load cases must not themselves break the new rule."""
    from physics.dynamics.load_cases import standard_load_cases

    arm = build_arm()
    for case in standard_load_cases(arm, DENSITY):
        assert arm.within_limits(case.q), f"{case.name} is unreachable"


def test_statics_still_computes_at_a_reachable_pose():
    arm = build_arm()
    torque = joint_torques(arm, np.array([np.pi / 4, 0.3]), DENSITY)
    assert np.isfinite(torque).all()
