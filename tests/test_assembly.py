"""Phase 10 verification: assemblies, kinematics and statics.

Each check compares the general implementation against something derived
independently: the two-link closed form for FK, finite differences for the
Jacobian, hand-computed moment sums for the torques. The general code is
axis-and-origin based and knows nothing about planar arms, so the closed-form
results are a genuine outside check rather than a restatement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.assembly.frames import (  # noqa: E402
    STANDARD_GRAVITY, identity, inverse, is_rigid_transform,
    rotation_about_axis, translation,
)
from core.assembly.kinematics import (  # noqa: E402
    forward_kinematics, geometric_jacobian, inverse_kinematics,
    position_jacobian, supporting_joints,
)
from core.assembly.model import Assembly, Joint, JointType, Link  # noqa: E402
from core.assembly.statics import (  # noqa: E402
    joint_torques, link_load_cases, worst_gravity_pose,
)
from core.design_genome import DesignGenome, HollowRectangleSection  # noqa: E402
from core.materials import get_material  # noqa: E402
from geometry.cad_export import kernel_available  # noqa: E402
from projects.robotic_arm.arm import analyse, build_arm, payload_force_n  # noqa: E402

cad = pytest.mark.skipif(not kernel_available(),
                         reason="no CAD kernel installed (optional dependency)")

L1, L2 = 0.30, 0.25
RHO = get_material("al_7075_t6").density_kg_m3


def make_link(name, length, b=0.020, h=0.040, t=0.002):
    return Link(name=name, length_m=length, genome=DesignGenome(
        section=HollowRectangleSection(outer_width_m=b, outer_height_m=h,
                                       wall_thickness_m=t),
        material_id="al_7075_t6"))


@pytest.fixture(scope="module")
def arm():
    return Assembly(
        name="planar2", material_id="al_7075_t6",
        links=[make_link("link1", L1), make_link("link2", L2)],
        joints=[
            Joint(name="j1", type=JointType.REVOLUTE, parent=None,
                  child="link1", axis=[0, 0, 1]),
            Joint(name="j2", type=JointType.REVOLUTE, parent="link1",
                  child="link2", axis=[0, 0, 1],
                  origin=translation(L1, 0, 0).tolist()),
        ])


def closed_form_tip(q):
    t1, t2 = float(q[0]), float(q[1])
    return np.array([L1 * np.cos(t1) + L2 * np.cos(t1 + t2),
                     L1 * np.sin(t1) + L2 * np.sin(t1 + t2), 0.0])


# =========================================================================== #
# frames
# =========================================================================== #
def test_rotation_is_a_rigid_transform():
    t = rotation_about_axis([0, 0, 1], 0.7)
    assert is_rigid_transform(t)
    assert np.allclose(inverse(t) @ t, identity(), atol=1e-12)


def test_rotation_about_z_matches_the_2d_rotation_matrix():
    angle = 0.4
    t = rotation_about_axis([0, 0, 1], angle)
    assert t[0, 0] == pytest.approx(np.cos(angle))
    assert t[0, 1] == pytest.approx(-np.sin(angle))


def test_non_rigid_transform_is_rejected():
    bad = identity()
    bad[0, 0] = 2.0
    assert not is_rigid_transform(bad)


def test_zero_axis_is_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        rotation_about_axis([0, 0, 0], 1.0)


# =========================================================================== #
# 1. FK against the closed form
# =========================================================================== #
@pytest.mark.parametrize("q", [
    [0.0, 0.0], [np.pi / 2, 0.0], [0.3, -0.7], [np.pi / 4, np.pi / 4],
    [-1.2, 2.0], [np.pi, -np.pi / 3],
])
def test_forward_kinematics_matches_closed_form(arm, q):
    got = forward_kinematics(arm, q).tool_position()
    assert np.allclose(got, closed_form_tip(q), atol=1e-12), (
        f"q={q}: FK {got} vs closed form {closed_form_tip(q)}")


def test_fk_rejects_wrong_number_of_joint_values(arm):
    with pytest.raises(ValueError, match="expected 2 joint values"):
        forward_kinematics(arm, [0.0])


def test_link_poses_are_rigid(arm):
    pose = forward_kinematics(arm, [0.4, -0.6])
    for transform in pose.link_transforms.values():
        assert is_rigid_transform(transform)


# =========================================================================== #
# 2. Jacobian against finite differences, and tau = J^T F
# =========================================================================== #
@pytest.mark.parametrize("q", [[0.3, -0.7], [0.0, 0.0], [1.1, 0.9], [-0.5, 1.7]])
def test_jacobian_matches_finite_differences(arm, q):
    q = np.asarray(q, dtype=float)
    analytic = position_jacobian(arm, q)
    numeric = np.zeros((3, 2))
    h = 1e-7
    for i in range(2):
        up, down = q.copy(), q.copy()
        up[i] += h
        down[i] -= h
        numeric[:, i] = (forward_kinematics(arm, up).tool_position()
                         - forward_kinematics(arm, down).tool_position()) / (2 * h)
    assert np.abs(analytic - numeric).max() < 1e-6, (
        f"q={q}\nanalytic\n{analytic}\nnumeric\n{numeric}")


def test_jacobian_angular_rows_are_the_joint_axes(arm):
    jac = geometric_jacobian(arm, [0.3, 0.2])
    assert np.allclose(jac[3:, 0], [0, 0, 1])
    assert np.allclose(jac[3:, 1], [0, 0, 1])


def test_a_joint_does_not_move_a_link_upstream_of_it(arm):
    """The elbow cannot move link1's centre of mass.

    Applying the geometric Jacobian formula to every joint regardless would give
    a non-zero column here, and the resulting torque error is small enough to
    look plausible. It is guarded explicitly.
    """
    assert supporting_joints(arm, "link1") == ["j1"]
    assert supporting_joints(arm, "link2") == ["j1", "j2"]
    com1 = np.array([L1 / 2, 0.0, 0.0])
    jac = position_jacobian(arm, [0.0, 0.0], point_world=com1, link_name="link1")
    assert np.allclose(jac[:, 1], 0.0), "elbow column must be zero for link1"
    assert not np.allclose(jac[:, 0], 0.0)


def test_torque_equals_jacobian_transpose_times_force(arm):
    """tau = J^T F, with the actuator supplying the negative for equilibrium."""
    force = np.array([0.0, -100.0, 0.0])
    q = [0.4, -0.3]
    expected = -(position_jacobian(arm, q).T @ force)
    got = joint_torques(arm, q, RHO, tip_force_n=force, include_gravity=False)
    assert np.allclose(got, expected, atol=1e-12)


# =========================================================================== #
# 3. IK
# =========================================================================== #
@pytest.mark.parametrize("target", [
    [0.35, 0.20, 0.0], [0.50, 0.05, 0.0], [0.10, 0.40, 0.0], [-0.20, 0.30, 0.0],
])
def test_ik_round_trip(arm, target):
    """Feed the IK solution back through FK and the target must come back."""
    result = inverse_kinematics(arm, np.array(target), q0=[0.1, 0.3])
    assert result.converged, f"IK did not converge, error {result.position_error_m}"
    recovered = forward_kinematics(arm, result.q).tool_position()
    assert np.allclose(recovered, target, atol=1e-6)


def test_ik_reports_failure_for_an_unreachable_target(arm):
    """Beyond L1+L2 there is no solution, and the solver must say so rather
    than returning a confident wrong answer."""
    result = inverse_kinematics(arm, np.array([2.0, 0.0, 0.0]),
                                max_iterations=200)
    assert not result.converged
    assert result.position_error_m > 1.0


def test_ik_respects_joint_limits():
    """A limited elbow must not be driven past its stop, even when that means
    the target cannot be reached."""
    limited = Assembly(
        name="limited", material_id="al_7075_t6",
        links=[make_link("link1", L1), make_link("link2", L2)],
        joints=[
            Joint(name="j1", type=JointType.REVOLUTE, parent=None,
                  child="link1", axis=[0, 0, 1], lower_limit=-0.1,
                  upper_limit=0.1),
            Joint(name="j2", type=JointType.REVOLUTE, parent="link1",
                  child="link2", axis=[0, 0, 1],
                  origin=translation(L1, 0, 0).tolist(),
                  lower_limit=-0.2, upper_limit=0.2),
        ])
    result = inverse_kinematics(limited, np.array([0.0, 0.5, 0.0]),
                                max_iterations=300)
    for joint, value in zip(limited.actuated_joints(), result.q):
        assert joint.within_limits(float(value)), (
            f"{joint.name} left its limits at {value}")


def test_ik_survives_a_singular_start(arm):
    """Fully extended is a singular configuration. Damped least squares must
    not blow up there; the pseudo-inverse would."""
    result = inverse_kinematics(arm, np.array([0.40, 0.15, 0.0]),
                                q0=[0.0, 0.0])
    assert np.all(np.isfinite(result.q))
    assert result.converged


# =========================================================================== #
# 4. static torques, hand-checked
# =========================================================================== #
def test_static_torque_hand_check_massless(arm):
    """Horizontal arm, tip load P down, links weightless.

        tau1 = P * (L1 + L2)      moment arm to the shoulder
        tau2 = P * L2             moment arm to the elbow
    """
    p = 100.0
    tau = joint_torques(arm, [0.0, 0.0], RHO, tip_force_n=[0.0, -p, 0.0],
                        include_gravity=False)
    assert tau[0] == pytest.approx(p * (L1 + L2), rel=1e-12)
    assert tau[1] == pytest.approx(p * L2, rel=1e-12)


def test_static_torque_hand_check_with_self_weight(arm):
    """Adds each link's own weight at its mid-span.

        tau1 = P(L1+L2) + m1 g (L1/2) + m2 g (L1 + L2/2)
        tau2 = P L2     + m2 g (L2/2)
    """
    p, g = 100.0, STANDARD_GRAVITY
    m1 = arm.link("link1").mass_kg(RHO)
    m2 = arm.link("link2").mass_kg(RHO)
    tau = joint_torques(arm, [0.0, 0.0], RHO, tip_force_n=[0.0, -p, 0.0])
    assert tau[0] == pytest.approx(
        p * (L1 + L2) + m1 * g * (L1 / 2) + m2 * g * (L1 + L2 / 2), rel=1e-12)
    assert tau[1] == pytest.approx(p * L2 + m2 * g * (L2 / 2), rel=1e-12)


def test_vertical_arm_needs_no_holding_torque(arm):
    """Pointing straight up, gravity and a vertical payload act along the links,
    so the moment arms vanish."""
    tau = joint_torques(arm, [np.pi / 2, 0.0], RHO,
                        tip_force_n=[0.0, -100.0, 0.0])
    assert np.allclose(tau, 0.0, atol=1e-9)


def test_worst_gravity_pose_is_the_extended_horizontal_one(arm):
    q, peak = worst_gravity_pose(arm, RHO, payload_force_n(2.0))
    assert abs(np.sin(q[0])) < 1e-6, "worst pose should be horizontal"
    assert peak > 0


def test_link_load_cases_reproduce_the_joint_torques(arm):
    """A link's root bending moment is the torque its joint has to hold."""
    force = payload_force_n(2.0)
    q = [0.0, 0.0]
    tau = joint_torques(arm, q, RHO, tip_force_n=force)
    loads = link_load_cases(arm, q, RHO, tip_force_n=force)
    for torque, load in zip(tau, loads):
        assert load.root_bending_moment_nm == pytest.approx(abs(torque), rel=1e-9)


def test_equivalent_tip_load_reproduces_the_root_moment(arm):
    loads = link_load_cases(arm, [0.0, 0.0], RHO,
                            tip_force_n=payload_force_n(2.0))
    for load in loads:
        link = arm.link(load.link)
        assert load.equivalent_tip_load_n * link.length_m == pytest.approx(
            load.root_bending_moment_nm, rel=1e-12)


# =========================================================================== #
# 5. the single-link bridge: the assembly must reproduce earlier results
# =========================================================================== #
def test_single_link_assembly_reproduces_the_cantilever_load():
    """One link, one joint, tip load: the assembly path must give exactly the
    root moment the standalone cantilever model used, or the two halves of the
    project have drifted apart."""
    length, load = 0.5, 196.2
    single = Assembly(
        name="single", material_id="al_7075_t6",
        links=[make_link("link1", length, 0.010, 0.0816185, 0.001)],
        joints=[Joint(name="root", type=JointType.FIXED, parent=None,
                      child="link1")])
    assert single.dof == 0
    loads = link_load_cases(single, [], RHO, tip_force_n=[0.0, -load, 0.0],
                            include_gravity=False)
    assert len(loads) == 1
    assert loads[0].root_bending_moment_nm == pytest.approx(load * length,
                                                            rel=1e-12)
    assert loads[0].equivalent_tip_load_n == pytest.approx(load, rel=1e-12)


# =========================================================================== #
# 6. model validity
# =========================================================================== #
def test_floating_link_is_rejected():
    with pytest.raises(Exception, match="not connected"):
        Assembly(name="broken", material_id="al_7075_t6",
                 links=[make_link("a", 0.2), make_link("orphan", 0.2)],
                 joints=[Joint(name="j", type=JointType.FIXED, parent=None,
                               child="a")])


def test_two_parents_for_one_link_is_rejected():
    with pytest.raises(Exception, match="more than one parent"):
        Assembly(name="loop", material_id="al_7075_t6",
                 links=[make_link("a", 0.2), make_link("b", 0.2)],
                 joints=[
                     Joint(name="j1", type=JointType.FIXED, parent=None, child="a"),
                     Joint(name="j2", type=JointType.FIXED, parent="a", child="b"),
                     Joint(name="j3", type=JointType.FIXED, parent=None, child="b"),
                 ])


def test_multiple_base_joints_are_rejected():
    with pytest.raises(Exception, match="attached to the base"):
        Assembly(name="two_roots", material_id="al_7075_t6",
                 links=[make_link("a", 0.2), make_link("b", 0.2)],
                 joints=[
                     Joint(name="j1", type=JointType.FIXED, parent=None, child="a"),
                     Joint(name="j2", type=JointType.FIXED, parent=None, child="b"),
                 ])


def test_inverted_joint_limits_are_rejected():
    with pytest.raises(Exception, match="lower_limit"):
        Joint(name="j", type=JointType.REVOLUTE, child="a",
              lower_limit=1.0, upper_limit=-1.0)


def test_total_mass_is_the_sum_of_links(arm):
    assert arm.total_mass_kg(RHO) == pytest.approx(
        sum(link.mass_kg(RHO) for link in arm.links), rel=1e-12)


# =========================================================================== #
# the capstone demonstration
# =========================================================================== #
def test_two_link_arm_analysis_runs_end_to_end():
    result = analyse()
    assert result["q"].shape == (2,)
    assert result["joint_torques_nm"].shape == (2,)
    assert len(result["verdicts"]) == 2
    for verdict in result["verdicts"]:
        assert verdict.passes, f"{verdict.link} failed its structural check"
        assert verdict.safety_factor > 2.0


def test_heavier_payload_raises_torques_and_stress():
    light = analyse(payload_kg=1.0)
    heavy = analyse(payload_kg=5.0)
    assert (np.abs(heavy["joint_torques_nm"]).max()
            > np.abs(light["joint_torques_nm"]).max())
    assert (heavy["verdicts"][0].max_bending_stress_pa
            > light["verdicts"][0].max_bending_stress_pa)


def test_an_absurd_payload_fails_the_structural_check():
    """The check has to be able to fail, or it is not a check."""
    result = analyse(payload_kg=400.0)
    assert any(not v.passes for v in result["verdicts"])


# =========================================================================== #
# assembly STEP
# =========================================================================== #
@cad
def test_assembly_step_round_trip_and_mass(tmp_path):
    from geometry.cad_export import export_assembly_step, find_kernel, import_step
    kernel = find_kernel()
    assembly = build_arm()
    report = export_assembly_step(assembly, [0.4, -0.8], RHO,
                                  tmp_path / "arm.step")
    assert report.part_count == len(assembly.links)
    assert report.mass_relative_error < 1e-9
    assert report.total_mass_kg == pytest.approx(
        assembly.total_mass_kg(RHO), rel=1e-9)

    reimported = import_step(tmp_path / "arm.step", kernel)
    assert len(reimported.solids()) == len(assembly.links)
    assert reimported.volume / 1e9 == pytest.approx(report.total_volume_m3,
                                                    rel=1e-9)


@cad
def test_posing_the_assembly_moves_the_parts_without_changing_mass(tmp_path):
    from geometry.cad_export import export_assembly_step
    assembly = build_arm()
    folded = export_assembly_step(assembly, [0.0, 0.0], RHO, tmp_path / "a.step")
    posed = export_assembly_step(assembly, [0.7, -1.1], RHO, tmp_path / "b.step")
    assert posed.total_mass_kg == pytest.approx(folded.total_mass_kg, rel=1e-12)
    assert (tmp_path / "a.step").read_bytes() != (tmp_path / "b.step").read_bytes()
