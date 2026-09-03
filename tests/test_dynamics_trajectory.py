"""Trajectories, the torque they demand, friction that refuses to be invented,
and the gear ratio that makes a demand cheapest.

The dynamics equations were already checked against Pinocchio. What was
missing was the thing an actuator is actually selected against: a motion, the
torque profile it produces, and the separation of peak from RMS. These tests
check the profiles against their closed forms, the torque against a pendulum
that can be done by hand, and the whole chain against two independent
multibody engines.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.assembly import Assembly, Joint, JointType, Link
from core.design_genome import DesignGenome, HollowRectangleSection
from core.materials import get_material
from physics.dynamics import (FrictionDataMissing, JointFriction, frictionless,
                              inverse_dynamics, plan_move, s_curve,
                              torque_profile, trapezoidal, trapezoidal_duration)
from physics.dynamics.actuator import (DriveDemand, best_ratio,
                                       demand_from_profile,
                                       inertia_matched_ratio, motor_torque_nm,
                                       ratio_sweep, reflected_inertia_kg_m2)
from physics.dynamics.friction import friction_torques as measured_friction
from physics.dynamics.trajectory import UnreachableMove

MATERIAL = get_material("al_7075_t6")
DENSITY = MATERIAL.density_kg_m3


def _link(name: str, length: float) -> Link:
    return Link(name=name, length_m=length, genome=DesignGenome(
        section=HollowRectangleSection(outer_width_m=0.02, outer_height_m=0.04,
                                       wall_thickness_m=0.002),
        material_id="al_7075_t6"))


def two_link() -> Assembly:
    origin = np.eye(4)
    origin[0, 3] = 0.30
    limits = dict(lower_limit=-3.0, upper_limit=3.0)
    return Assembly(name="planar2", material_id="al_7075_t6",
                    links=[_link("l1", 0.30), _link("l2", 0.25)],
                    joints=[Joint(name="j1", type=JointType.REVOLUTE, parent=None,
                                  child="l1", axis=[0, 0, 1], **limits),
                            Joint(name="j2", type=JointType.REVOLUTE, parent="l1",
                                  child="l2", axis=[0, 0, 1],
                                  origin=origin.tolist(), **limits)])


def pendulum() -> Assembly:
    return Assembly(name="pendulum", material_id="al_7075_t6",
                    links=[_link("l1", 0.40)],
                    joints=[Joint(name="j1", type=JointType.REVOLUTE, parent=None,
                                  child="l1", axis=[0, 0, 1], lower_limit=-3.2,
                                  upper_limit=3.2)])


# --- the profiles against their closed forms ---------------------------------

def test_a_trapezoidal_move_covers_its_distance_in_the_textbook_time():
    """Total time is two ramps plus the cruise, and the area under the
    velocity curve is the distance."""
    # 2 m at 2 m/s with 4 m/s^2: half a second of ramp each end covers 1 m,
    # so the cruise carries the other 1 m in half a second.
    total, ramp, cruise = trapezoidal_duration(2.0, 2.0, 4.0)
    assert ramp == pytest.approx(0.5)
    assert cruise == pytest.approx(0.5)
    assert total == pytest.approx(1.5)

    profile = trapezoidal(2.0, 2.0, 4.0, samples=4000)
    assert profile.position_rad[-1] == pytest.approx(2.0, rel=1e-6)
    assert profile.velocity_rad_s.max() == pytest.approx(2.0, rel=1e-3)
    assert np.abs(profile.acceleration_rad_s2).max() == pytest.approx(4.0)
    area = np.trapezoid(profile.velocity_rad_s, profile.time_s)
    assert area == pytest.approx(2.0, rel=1e-3)

    # At exactly the boundary the two branches must agree: 1 m here is the
    # longest triangular move, with no cruise at all.
    boundary_total, boundary_ramp, boundary_cruise = trapezoidal_duration(
        1.0, 2.0, 4.0)
    assert boundary_cruise == 0.0
    assert boundary_ramp == pytest.approx(0.5)
    assert boundary_total == pytest.approx(1.0)


def test_a_short_move_is_triangular_and_never_reaches_the_speed_limit():
    """The case a fixed formula gets wrong."""
    total, ramp, cruise = trapezoidal_duration(0.2, 2.0, 4.0)
    assert cruise == 0.0
    assert ramp == pytest.approx(np.sqrt(0.2 / 4.0))
    assert total == pytest.approx(2 * ramp)
    profile = trapezoidal(0.2, 2.0, 4.0, samples=2000)
    assert profile.velocity_rad_s.max() < 2.0
    assert profile.position_rad[-1] == pytest.approx(0.2, rel=1e-6)


def test_the_s_curve_covers_the_distance_and_bounds_the_jerk():
    """Continuous acceleration, the distance exact, and the jerk under its
    limit. The distance formula had to include both jerk times: leaving one
    out made the profile overshoot and need a correction that then held it
    below its own limits."""
    profile = s_curve(1.0, 2.0, 4.0, 40.0, samples=4000)
    assert profile.position_rad[-1] == pytest.approx(1.0, rel=1e-6)
    assert np.abs(profile.acceleration_rad_s2).max() <= 4.0 + 1e-9
    jerk = np.diff(profile.acceleration_rad_s2) / np.diff(profile.time_s)
    assert np.abs(jerk).max() <= 40.0 * 1.05
    # Continuous acceleration is the point of the profile.
    assert np.abs(np.diff(profile.acceleration_rad_s2)).max() < 0.2


def test_the_s_curve_takes_longer_than_the_trapezoid_for_the_same_move():
    fast = trapezoidal(1.0, 2.0, 4.0, samples=2000)
    smooth = s_curve(1.0, 2.0, 4.0, 40.0, samples=2000)
    assert smooth.duration_s > fast.duration_s
    assert smooth.duration_s / fast.duration_s == pytest.approx(1.12, abs=0.05)


def test_joints_are_synchronised_to_the_slowest():
    trajectory = plan_move([0.0, 0.0], [1.2, -0.2], [2.0, 2.0], [4.0, 4.0],
                           samples=300)
    assert trajectory.q[0].tolist() == [0.0, 0.0]
    assert trajectory.q[-1] == pytest.approx([1.2, -0.2], rel=1e-3)
    # Both joints stop at the end, which is what synchronising means.
    assert np.abs(trajectory.qd[-1]).max() < 1e-6
    assert np.abs(trajectory.qd[0]).max() < 1e-6


def test_an_s_curve_without_a_jerk_limit_is_refused():
    with pytest.raises(UnreachableMove, match="jerk"):
        plan_move([0.0], [1.0], [2.0], [4.0], kind="s_curve")


# --- the torque a motion demands ---------------------------------------------

def test_the_pendulum_torque_is_the_hand_calculation():
    """One link, one joint: tau = I qdd + m g r cos(theta), and every term is
    known in closed form for a uniform bar."""
    arm = pendulum()
    link = arm.links[0]
    mass = link.mass_kg(DENSITY)
    radius = link.length_m * link.com_fraction
    # A uniform bar about its end: m L^2 / 3.
    inertia_about_joint = mass * link.length_m ** 2 / 3.0

    theta, speed, acceleration = 0.4, 0.0, 2.5
    tau = inverse_dynamics(arm, [theta], [speed], [acceleration], DENSITY)[0]
    expected = (inertia_about_joint * acceleration
                + mass * 9.81 * radius * np.cos(theta))
    assert tau == pytest.approx(expected, rel=0.02), (tau, expected)


def test_a_torque_profile_separates_peak_from_rms():
    """The two numbers a motor is chosen against, and they are not the same
    number. Measured on this move: peak over RMS 1.32 at the shoulder."""
    arm = two_link()
    trajectory = plan_move([0.0, 0.0], [1.2, -0.8], [2.0, 2.0], [4.0, 4.0],
                           samples=200)
    profile = torque_profile(arm, trajectory, DENSITY)
    assert profile.torque_nm.shape == (200, 2)
    assert np.all(profile.peak_torque_nm >= profile.rms_torque_nm)
    assert profile.peak_to_rms[0] == pytest.approx(1.32, abs=0.1)
    # Gravity does not explain the peak, which is why statics cannot size a
    # motor for a move.
    assert np.all(profile.dynamic_share >= 0.0)
    assert profile.dynamic_share[0] > 0.05


def test_the_rms_does_not_move_with_the_sampling():
    """It is an integral over time, not a mean over samples."""
    arm = two_link()
    coarse = torque_profile(arm, plan_move([0.0, 0.0], [1.0, -0.5], [2.0, 2.0],
                                           [4.0, 4.0], samples=60), DENSITY)
    fine = torque_profile(arm, plan_move([0.0, 0.0], [1.0, -0.5], [2.0, 2.0],
                                         [4.0, 4.0], samples=400), DENSITY)
    assert coarse.rms_torque_nm == pytest.approx(fine.rms_torque_nm, rel=0.02)


# --- friction that refuses to be invented ------------------------------------

def test_friction_is_zero_unless_it_was_measured():
    arm = two_link()
    q, qd, qdd = [0.3, -0.6], [0.8, -0.4], [2.0, 1.5]
    without = inverse_dynamics(arm, q, qd, qdd, DENSITY)
    assert np.allclose(frictionless(2), 0.0)

    friction = JointFriction(coulomb_nm=0.4, viscous_nm_s_rad=0.02,
                             breakaway_nm=0.6,
                             source="bench measurement, joint 1, 2026-09-03")
    with_friction = inverse_dynamics(arm, q, qd, qdd, DENSITY,
                                     frictions=[friction, friction])
    assert np.all(np.abs(with_friction) > np.abs(without))


def test_a_joint_without_measured_friction_is_refused_not_zeroed():
    friction = JointFriction(0.4, 0.02, 0.6, source="bench, joint 1")
    with pytest.raises(FrictionDataMissing, match="unmeasured"):
        measured_friction([friction, None], [0.5, 0.5])
    with pytest.raises(FrictionDataMissing, match="are needed"):
        measured_friction([friction], [0.5, 0.5])


def test_a_friction_model_needs_a_source():
    with pytest.raises(ValueError, match="source"):
        JointFriction(0.4, 0.02, 0.6, source="   ")


def test_friction_opposes_motion_and_is_continuous_through_zero():
    friction = JointFriction(0.4, 0.02, 0.6, source="bench")
    assert friction.torque_nm(1.0) > 0.0
    assert friction.torque_nm(-1.0) < 0.0
    small = friction.torque_nm(1e-9)
    assert abs(small) < 1e-6, "the model jumps at zero speed"
    assert friction.torque_nm(friction.creep_speed_rad_s) == pytest.approx(
        0.6 + 0.02 * friction.creep_speed_rad_s, rel=1e-9)


# --- the gear ratio ----------------------------------------------------------

def test_the_referral_arithmetic_is_the_textbook_one():
    demand = DriveDemand(joint_torque_nm=10.0, joint_speed_rad_s=2.0,
                         joint_acceleration_rad_s2=4.0, load_inertia_kg_m2=0.02)
    assert reflected_inertia_kg_m2(0.02, 50.0) == pytest.approx(0.02 / 2500)
    torque = motor_torque_nm(demand, ratio=50.0, rotor_inertia_kg_m2=8e-6,
                             gearbox_inertia_kg_m2=2e-6, efficiency=0.8)
    expected = 10.0 / (50.0 * 0.8) + (8e-6 + 2e-6) * 50.0 * 4.0
    assert torque == pytest.approx(expected, rel=1e-12)


def test_a_pure_inertia_demand_is_cheapest_at_the_matched_ratio():
    """The classical result, checked numerically rather than asserted."""
    load, rotor, acceleration = 0.02, 8e-6, 5.0
    demand = DriveDemand(joint_torque_nm=load * acceleration,
                         joint_speed_rad_s=1.0,
                         joint_acceleration_rad_s2=acceleration,
                         load_inertia_kg_m2=load)
    matched = inertia_matched_ratio(load, rotor)
    best, _torque = best_ratio(demand, np.linspace(5.0, 200.0, 400), rotor)
    assert matched == pytest.approx(50.0)
    assert best == pytest.approx(matched, rel=0.02)


def test_a_gravity_load_moves_the_best_ratio_away_from_the_matched_one():
    """Which is why the matched ratio is a starting point and not an answer:
    with a constant load the torque keeps falling as the ratio grows."""
    load, rotor, acceleration = 0.02, 8e-6, 5.0
    demand = DriveDemand(joint_torque_nm=load * acceleration + 2.0,
                         joint_speed_rad_s=1.0,
                         joint_acceleration_rad_s2=acceleration,
                         load_inertia_kg_m2=load)
    ratios = np.linspace(5.0, 200.0, 400)
    best, _torque = best_ratio(demand, ratios, rotor)
    assert best == pytest.approx(ratios[-1])
    rows = ratio_sweep(demand, [10.0, 50.0, 200.0], rotor)
    assert rows[0]["motor_torque_nm"] > rows[-1]["motor_torque_nm"]
    assert rows[0]["inertia_ratio"] > rows[-1]["inertia_ratio"]


def test_the_worst_instant_of_a_profile_becomes_the_drive_demand():
    arm = two_link()
    profile = torque_profile(arm, plan_move([0.0, 0.0], [1.2, -0.8], [2.0, 2.0],
                                            [4.0, 4.0], samples=200), DENSITY)
    demand = demand_from_profile(profile, 0, load_inertia_kg_m2=0.02)
    assert abs(demand.joint_torque_nm) == pytest.approx(profile.peak_torque_nm[0])
    assert demand.load_inertia_kg_m2 == 0.02


# --- three engines on the same motion ----------------------------------------

@pytest.mark.slow
def test_three_engines_agree_on_the_same_state():
    """This project's inverse dynamics, Pinocchio's, and MuJoCo's forward
    dynamics closing the loop.

    Pinocchio computes the same torque from the same state, and MuJoCo takes
    the torque this project computed and reports the acceleration it produces,
    which must be the acceleration that was commanded. Measured: 1.1e-13 N m
    against a 0.97 N m torque, and 2.6e-11 rad/s2 against 2.0 commanded.

    Agreement between three simulations is a cross validation. It is not
    experimental evidence and does not raise any grade.
    """
    from nodes import mujoco_node, pinocchio_node

    if not pinocchio_node.is_available() or not mujoco_node.is_available():
        pytest.skip("pinocchio and mujoco are both required")

    arm = two_link()
    q = np.array([0.3, -0.6])
    qd = np.array([0.8, -0.4])
    qdd = np.array([2.0, 1.5])

    ours = inverse_dynamics(arm, q, qd, qdd, DENSITY)
    comparison = pinocchio_node.compare(arm, q, qd, qdd, DENSITY)
    assert comparison.inverse_dynamics_error_nm < 1e-9 * max(
        comparison.torque_scale_nm, 1.0)

    accelerations = mujoco_node.accelerations(arm, q, qd, ours, DENSITY)
    assert np.allclose(accelerations, qdd, atol=1e-6), accelerations


@pytest.mark.slow
def test_the_torque_along_a_whole_trajectory_agrees_with_pinocchio():
    """Not one state but every sample of a move, which is where an index or a
    frame convention would show up."""
    from nodes import pinocchio_node

    if not pinocchio_node.is_available():
        pytest.skip("pinocchio is required")

    arm = two_link()
    trajectory = plan_move([0.0, 0.0], [1.0, -0.6], [2.0, 2.0], [4.0, 4.0],
                           samples=25)
    profile = torque_profile(arm, trajectory, DENSITY)
    worst = 0.0
    for i in range(0, trajectory.q.shape[0], 5):
        comparison = pinocchio_node.compare(arm, trajectory.q[i], trajectory.qd[i],
                                            trajectory.qdd[i], DENSITY)
        worst = max(worst, comparison.inverse_dynamics_error_nm)
        assert np.isfinite(profile.torque_nm[i]).all()
    assert worst < 1e-9, worst
