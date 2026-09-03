"""The six axis arm design, checked where it can be checked cheaply.

The design itself is a run of scripts/design_manipulator.py and takes about a
minute; these tests pin the things that would make that run wrong without
failing: the geometry the specification describes, the loop that has to
converge, the loads each link is sized against, and the honesty of the stages
that cannot run.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.assembly.kinematics import forward_kinematics
from core.assembly.statics import joint_torques
from core.materials import get_material
from projects.manipulator.arm import (build_arm, payload_force_n,
                                      starting_sections, stretched_pose)
from projects.manipulator.loop import (actuator_gravity_torque, carried_load_n,
                                       run_loop, size_sections)
from projects.manipulator.spec import SPEC
from projects.manipulator.stages import (drivetrain_stage, dynamics_stage,
                                         fatigue_stage, policy_stage)


# --- the arm is the arm the specification describes ---------------------------

def test_the_tool_sits_at_the_stated_reach():
    """Joint origins carry the geometry; a link's own length moves nothing
    except at the tip. Getting that wrong put the tool 55 mm short on the
    first build and 80 mm long on the one before it."""
    arm = build_arm()
    pose = forward_kinematics(arm, stretched_pose())
    tool = pose.tool_position()
    assert tool[0] == pytest.approx(SPEC.reach_m, abs=1e-9)
    assert tool[1] == pytest.approx(SPEC.base_height_m, abs=1e-9)
    assert SPEC.reach_check_m() == pytest.approx(SPEC.reach_m)


def test_the_axes_are_a_six_revolute_arm_with_the_column_along_gravity():
    arm = build_arm()
    assert len(arm.actuated_joints()) == SPEC.degrees_of_freedom
    joints = {j.name: np.asarray(j.axis, dtype=float) for j in arm.joints}
    # This project's gravity is -y, so the base yaw axis must be y.
    assert np.allclose(joints["j1_base_yaw"], [0.0, 1.0, 0.0])
    assert np.allclose(joints["j2_shoulder"], [0.0, 0.0, 1.0])
    assert np.allclose(joints["j4_wrist_roll"], [1.0, 0.0, 0.0])


def test_the_payload_loads_the_shoulder_hardest():
    arm = build_arm()
    density = get_material(arm.material_id).density_kg_m3
    torque = joint_torques(arm, stretched_pose(), density,
                           tip_force_n=payload_force_n())
    names = [j.name for j in arm.actuated_joints()]
    worst = names[int(np.argmax(np.abs(torque)))]
    assert worst == "j2_shoulder"
    # The roll axes are along the reach, so gravity gives them nothing.
    assert torque[names.index("j4_wrist_roll")] == pytest.approx(0.0, abs=1e-9)
    assert torque[names.index("j6_tool_roll")] == pytest.approx(0.0, abs=1e-9)


# --- the loads each link is sized against -------------------------------------

def test_a_link_carries_everything_outboard_of_it():
    arm = build_arm()
    inboard = carried_load_n(arm, "upper_arm", SPEC, {})
    outboard = carried_load_n(arm, "forearm", SPEC, {})
    assert inboard > outboard > SPEC.payload_kg * 9.8
    tip = carried_load_n(arm, "tool_flange", SPEC, {})
    assert tip == pytest.approx(SPEC.payload_kg * 9.80665)


def test_actuator_masses_add_torque_where_they_sit():
    """The assembly model has no point mass, so this contribution is computed
    outside it; a test is the only thing that keeps the two consistent."""
    arm = build_arm()
    q = stretched_pose()
    none = actuator_gravity_torque(arm, q, {})
    assert np.allclose(none, 0.0)

    with_drive = actuator_gravity_torque(arm, q, {"j3_elbow": 1.0})
    names = [j.name for j in arm.actuated_joints()]
    # A kilogram at the elbow loads the shoulder and the elbow, nothing else.
    assert abs(with_drive[names.index("j2_shoulder")]) > 1.0
    assert abs(with_drive[names.index("j5_wrist_pitch")]) < 1e-9


# --- the loop ------------------------------------------------------------------

@pytest.mark.slow
def test_the_mass_torque_loop_converges_and_says_how_far_it_moved():
    result = run_loop(SPEC)
    assert result.rows, "the loop produced no history"
    assert any("converged" in note for note in result.notes), result.notes
    first, last = result.rows[0], result.rows[-1]
    # The starting sections are deliberately generous, so the loop must remove
    # mass rather than add it.
    assert last["structure_mass_kg"] < first["structure_mass_kg"]
    # And the torque must RISE, because the drives it selects have mass.
    assert last["shoulder_peak_nm"] > first["shoulder_peak_nm"]
    assert len(result.rows) <= 8


@pytest.mark.slow
def test_the_sizing_solves_for_one_dimension_and_the_limit_says_so():
    arm = build_arm()
    sections = size_sections(arm, SPEC, {})
    for name, section in sections.items():
        start = starting_sections(SPEC)[name]
        assert section.outer_width_m == start.outer_width_m
        assert section.outer_height_m > 0.0


# --- the stages that cannot run say so ------------------------------------------

def test_the_policy_cannot_decompose_an_arm_and_reports_that():
    stage = policy_stage(SPEC)
    assert all(row.get("accepted") for row in stage.rows)
    assert any("does not decompose" in gap for gap in stage.could_not)
    assert any("full payload at its own tip" in gap for gap in stage.could_not)


@pytest.mark.slow
def test_every_joint_is_driven_once_the_catalogue_is_wide_enough():
    """With one actuator in the catalogue the shoulder and the elbow could not
    be driven at all. Four sourced actuators later every joint has one, and
    the selection is still only from printed values: the geared path remains
    unbuildable because no efficiency is printed for the gear units."""
    arm = build_arm()
    dynamics = dynamics_stage(arm, SPEC, samples=60)
    drives = drivetrain_stage(dynamics, SPEC)
    unselected = [row["joint"] for row in drives.rows
                  if row.get("status") != "selected"]
    assert unselected == [], unselected
    for row in drives.rows:
        assert row["path"] in ("integrated actuator", "motor and gearbox",
                               "direct drive")
        if row["path"] == "integrated actuator":
            assert "no further gearbox" in row["note"]
        assert row["rated_nm"] >= row["required_rms_nm"]
        assert row["peak_nm"] >= row["required_peak_nm"]
    assert any("cannot be paired" in gap for gap in drives.could_not)


def test_the_selection_checks_speed_as_well_as_torque():
    """The 64:1 actuator has torque to spare and turns at 48 rpm. A joint that
    must turn faster cannot use it, and the selection says so rather than
    choosing on torque alone."""
    from drivetrain.sourced import sourced_motor
    from projects.manipulator.stages import StageResult, drivetrain_stage

    slow = sourced_motor("cubemars_ak80_64_kv80")
    assert slow.nominal_speed_rad_s < 6.0
    fast_move = SPEC.__class__(move_time_s=0.2)
    dynamics = StageResult(name="fake", rows=[{
        "joint": "j2_shoulder", "peak_trapezoidal_nm": 30.0,
        "peak_s_curve_nm": 30.0, "rms_trapezoidal_nm": 20.0,
        "rms_s_curve_nm": 20.0}])
    drives = drivetrain_stage(dynamics, fast_move, {"j2_shoulder": 0.2})
    candidates = drives.data["candidates"]["j2_shoulder"]
    slow_row = next(c for c in candidates
                    if c["candidate"] == "cubemars_ak80_64_kv80")
    assert not slow_row["feasible"]
    assert "rated speed" in slow_row["why"]


@pytest.mark.slow
def test_fatigue_refuses_the_printed_material_and_passes_the_aluminium():
    result = run_loop(SPEC)
    stage = fatigue_stage(None, result.data["final_sections"], SPEC)
    by_material = {}
    for row in stage.rows:
        by_material.setdefault(row["material"], []).append(row)
    for row in by_material["pa12"]:
        assert row["damage_sum"] is None
        assert "no sourced fatigue strength" in row["refused"]
    for row in by_material[SPEC.materials["link"]]:
        assert row["damage_sum"] is not None
        assert row["survives"]
