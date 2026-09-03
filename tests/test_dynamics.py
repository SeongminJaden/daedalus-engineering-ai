"""Phase 11 verification: rigid-body dynamics.

Every quantity is checked against something derived independently: the CAD
kernel for the inertia tensor, the textbook two-link closed form for M, C and G,
and the Phase 10 statics for the zero-velocity limit. The implementation is
general (Jacobians and Christoffel symbols, no planar assumptions), so the
two-link formulas are an outside check rather than a restatement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.assembly import (  # noqa: E402
    Assembly, Joint, JointType, Link, STANDARD_GRAVITY, joint_torques,
    translation,
)
from core.design_genome import DesignGenome, HollowRectangleSection  # noqa: E402
from core.materials import get_material  # noqa: E402
from geometry.cad_export import kernel_available  # noqa: E402
from physics.dynamics import (  # noqa: E402
    box_inertia_tensor, coriolis_matrix, evaluate_duty_cycle, friction_torques,
    gravity_torques, hollow_rect_inertia, inverse_dynamics, is_valid_inertia,
    joint_power_w, kinetic_energy_j, link_inertia, mass_matrix,
    mass_matrix_derivative, parallel_axis, standard_load_cases,
)
from projects.robotic_arm.arm import build_arm  # noqa: E402

cad = pytest.mark.skipif(not kernel_available(),
                         reason="no CAD kernel installed (optional dependency)")

L1, L2 = 0.30, 0.25
RHO = get_material("al_7075_t6").density_kg_m3
G = STANDARD_GRAVITY


def make_link(name, length, b, h, t):
    return Link(name=name, length_m=length, genome=DesignGenome(
        section=HollowRectangleSection(outer_width_m=b, outer_height_m=h,
                                       wall_thickness_m=t),
        material_id="al_7075_t6"))


@pytest.fixture(scope="module")
def arm():
    return Assembly(
        name="p2", material_id="al_7075_t6",
        links=[make_link("link1", L1, 0.020, 0.040, 0.002),
               make_link("link2", L2, 0.016, 0.032, 0.002)],
        joints=[
            Joint(name="j1", type=JointType.REVOLUTE, parent=None,
                  child="link1", axis=[0, 0, 1]),
            Joint(name="j2", type=JointType.REVOLUTE, parent="link1",
                  child="link2", axis=[0, 0, 1],
                  origin=translation(L1, 0, 0).tolist()),
        ])


@pytest.fixture(scope="module")
def params(arm):
    m1 = arm.link("link1").mass_kg(RHO)
    m2 = arm.link("link2").mass_kg(RHO)
    return {
        "m1": m1, "m2": m2, "lc1": 0.5 * L1, "lc2": 0.5 * L2,
        "I1": link_inertia(arm.link("link1"), RHO)[2, 2],
        "I2": link_inertia(arm.link("link2"), RHO)[2, 2],
    }


# --- the independent closed form for a two-link planar arm ---------------- #
def closed_form_mass(q, p):
    c2 = np.cos(q[1])
    m11 = (p["m1"] * p["lc1"] ** 2 + p["I1"]
           + p["m2"] * (L1 ** 2 + p["lc2"] ** 2 + 2 * L1 * p["lc2"] * c2)
           + p["I2"])
    m12 = p["m2"] * (p["lc2"] ** 2 + L1 * p["lc2"] * c2) + p["I2"]
    return np.array([[m11, m12], [m12, p["m2"] * p["lc2"] ** 2 + p["I2"]]])


def closed_form_coriolis(q, qd, p):
    h = -p["m2"] * L1 * p["lc2"] * np.sin(q[1])
    return np.array([[h * qd[1], h * (qd[0] + qd[1])], [-h * qd[0], 0.0]])


def closed_form_gravity(q, p):
    q1, q2 = q
    return np.array([
        (p["m1"] * p["lc1"] + p["m2"] * L1) * G * np.cos(q1)
        + p["m2"] * p["lc2"] * G * np.cos(q1 + q2),
        p["m2"] * p["lc2"] * G * np.cos(q1 + q2),
    ])


# =========================================================================== #
# 1. inertia from geometry, checked against the CAD kernel
# =========================================================================== #
def test_solid_box_inertia_matches_the_textbook_formula():
    mass, lx, ly, lz = 2.0, 0.3, 0.2, 0.1
    inertia = box_inertia_tensor(mass, lx, ly, lz)
    assert inertia[0, 0] == pytest.approx(mass * (ly**2 + lz**2) / 12)
    assert inertia[1, 1] == pytest.approx(mass * (lx**2 + lz**2) / 12)
    assert inertia[2, 2] == pytest.approx(mass * (lx**2 + ly**2) / 12)


def test_hollow_inertia_is_less_than_the_solid_it_came_from():
    solid = box_inertia_tensor(0.30 * 0.04 * 0.02 * RHO, 0.30, 0.04, 0.02)
    hollow = hollow_rect_inertia(0.30, 0.020, 0.040, 0.002, RHO)
    assert np.all(np.diag(hollow) < np.diag(solid))


@pytest.mark.parametrize("dims", [
    (0.30, 0.020, 0.040, 0.002), (0.25, 0.016, 0.032, 0.002),
    (0.50, 0.010, 0.0816, 0.001),
])
def test_inertia_tensor_is_physically_realisable(dims):
    """Symmetric, positive definite, and obeying the triangle inequality on the
    principal moments. An arbitrary positive definite matrix is not necessarily
    a rigid body's inertia."""
    assert is_valid_inertia(hollow_rect_inertia(*dims, RHO))


@cad
@pytest.mark.parametrize("dims", [
    (0.30, 0.020, 0.040, 0.002), (0.25, 0.016, 0.032, 0.002),
])
def test_analytic_inertia_matches_the_cad_kernel(dims):
    """Independent check: OpenCascade integrates the actual B-rep.

    build123d reports the inertia matrix per unit density in mm^5, so it is
    scaled by 1e-15 (mm^5 to m^5) times the density to reach kg m^2.
    """
    from geometry.cad_export import build_solid, find_kernel
    length, b, h, t = dims
    kernel = find_kernel()
    solid = build_solid(length, b, h, t, kernel)
    cad_inertia = np.array(solid.matrix_of_inertia) * 1e-15 * RHO
    analytic = hollow_rect_inertia(length, b, h, t, RHO)

    rel = np.abs(np.diag(cad_inertia) - np.diag(analytic)) / np.diag(analytic)
    assert rel.max() < 1e-9, f"diagonal relative error {rel}"
    off_diagonal = np.abs(cad_inertia - np.diag(np.diag(cad_inertia))).max()
    assert off_diagonal < 1e-12 * np.abs(analytic).max()


def test_parallel_axis_shifts_correctly():
    inertia = box_inertia_tensor(1.0, 0.2, 0.2, 0.2)
    shifted = parallel_axis(inertia, 1.0, [0.5, 0.0, 0.0])
    assert shifted[0, 0] == pytest.approx(inertia[0, 0])          # along the axis
    assert shifted[1, 1] == pytest.approx(inertia[1, 1] + 1.0 * 0.25)
    assert shifted[2, 2] == pytest.approx(inertia[2, 2] + 1.0 * 0.25)


def test_impossible_geometry_is_rejected():
    with pytest.raises(ValueError, match="cavity"):
        hollow_rect_inertia(0.3, 0.02, 0.02, 0.02, RHO)


# =========================================================================== #
# 2. M, C and G against the independent closed form
# =========================================================================== #
QS = [[0.0, 0.0], [0.3, -0.7], [np.pi / 4, np.pi / 3], [-1.1, 2.0], [1.5, -0.4]]


@pytest.mark.parametrize("q", QS)
def test_mass_matrix_matches_closed_form(arm, params, q):
    got = mass_matrix(arm, q, RHO)
    want = closed_form_mass(q, params)
    assert np.abs(got - want).max() < 1e-12 * max(np.abs(want).max(), 1.0)


@pytest.mark.parametrize("q", QS)
def test_coriolis_matrix_matches_closed_form(arm, params, q):
    qd = np.array([0.8, -1.3])
    got = coriolis_matrix(arm, q, qd, RHO)
    want = closed_form_coriolis(q, qd, params)
    # C uses central differences of M, so the floor is the differencing error.
    assert np.abs(got - want).max() < 1e-8


@pytest.mark.parametrize("q", QS)
def test_gravity_torques_match_closed_form(arm, params, q):
    got = gravity_torques(arm, q, RHO)
    want = closed_form_gravity(q, params)
    assert np.abs(got - want).max() < 1e-12 * max(np.abs(want).max(), 1.0)


# =========================================================================== #
# 3. the statics bridge
# =========================================================================== #
@pytest.mark.parametrize("q", [[0.0, 0.0], [0.4, -0.9], [2.0, 0.7]])
def test_zero_motion_limit_equals_phase_10_statics(arm, q):
    """At rest the dynamics must reproduce the holding torque exactly.

    G reuses the statics routine on purpose, so the two cannot drift apart and
    this limit is exact rather than approximate.
    """
    dynamic = inverse_dynamics(arm, q, np.zeros(2), np.zeros(2), RHO)
    static = joint_torques(arm, q, RHO, tip_force_n=None, include_gravity=True)
    assert np.abs(dynamic - static).max() < 1e-10


def test_zero_motion_with_a_payload_equals_statics(arm):
    force = np.array([0.0, -2.0 * G, 0.0])
    dynamic = inverse_dynamics(arm, [0.3, 0.2], np.zeros(2), np.zeros(2), RHO,
                               tip_force_n=force)
    static = joint_torques(arm, [0.3, 0.2], RHO, tip_force_n=force)
    assert np.abs(dynamic - static).max() < 1e-10


# =========================================================================== #
# 4. energy properties
# =========================================================================== #
@pytest.mark.parametrize("q", QS)
def test_mass_matrix_is_symmetric_positive_definite(arm, q):
    m = mass_matrix(arm, q, RHO)
    assert np.allclose(m, m.T, atol=1e-15)
    assert np.linalg.eigvalsh(m).min() > 0.0


@pytest.mark.parametrize("q,qd", [([0.35, -0.8], [1.1, -0.6]),
                                  ([1.0, 0.5], [-0.4, 2.0])])
def test_passivity_m_dot_minus_two_c_is_skew_symmetric(arm, q, qd):
    """The property that separates a correct C from one that merely gives the
    right torque. Built from Christoffel symbols it holds identically."""
    qd = np.asarray(qd, dtype=float)
    dm = mass_matrix_derivative(arm, q, RHO)
    m_dot = sum(dm[:, :, k] * qd[k] for k in range(len(qd)))
    s = m_dot - 2.0 * coriolis_matrix(arm, q, qd, RHO)
    assert np.abs(s + s.T).max() < 1e-9


def test_kinetic_energy_is_non_negative(arm):
    assert kinetic_energy_j(arm, [0.3, 0.4], [1.0, -2.0], RHO) > 0
    assert kinetic_energy_j(arm, [0.3, 0.4], [0.0, 0.0], RHO) == 0.0


def test_friction_is_zero_unless_it_was_measured(arm):
    """The term must not quietly contribute a made up value to a torque that
    an actuator gets selected from. Without measured parameters it is zero and
    the docstring says that is optimistic; with them it is real, and a joint
    whose parameters are missing is refused rather than zeroed."""
    from physics.dynamics.friction import FrictionDataMissing, JointFriction

    torques = friction_torques(arm, [1.0, 1.0])
    assert torques.shape == (arm.dof,)
    assert np.all(torques == 0.0)
    assert np.all(friction_torques(arm, [0.0, 0.0]) == 0.0)
    assert "zero unless measured" in friction_torques.__doc__

    measured = JointFriction(coulomb_nm=0.3, viscous_nm_s_rad=0.01,
                             breakaway_nm=0.5, source="bench, 2026-09-03")
    real = friction_torques(arm, [1.0, -1.0], frictions=[measured, measured])
    assert real[0] > 0.0 > real[1]
    with pytest.raises(FrictionDataMissing):
        friction_torques(arm, [1.0, 1.0], frictions=[measured, None])


# =========================================================================== #
# 5. load cases and duty cycle
# =========================================================================== #
def test_acceleration_raises_the_required_torque(arm):
    """The point of Phase 11: statics alone under-states what a joint needs."""
    q = [np.pi / 4, 0.2]
    still = inverse_dynamics(arm, q, np.zeros(2), np.zeros(2), RHO)
    accelerating = inverse_dynamics(arm, q, np.zeros(2), np.full(2, 30.0), RHO)
    assert np.abs(accelerating[0]) > np.abs(still[0])


def test_power_is_torque_times_speed():
    torque = np.array([3.0, -2.0])
    speed = np.array([1.5, 4.0])
    assert np.allclose(joint_power_w(torque, speed), [4.5, -8.0])


def test_duty_cycle_separates_peak_from_continuous():
    """A motor has both ratings and they differ by a factor of a few. Sizing to
    one alone gives either an over-specified drive or one that overheats."""
    arm = build_arm()
    duty = evaluate_duty_cycle(arm, RHO)
    peak = duty.peak_torque_nm()
    continuous = duty.continuous_torque_nm()
    assert np.all(peak > 0)
    assert np.all(continuous > 0)
    assert np.all(peak >= continuous)
    assert np.all(duty.peak_to_continuous_ratio() > 1.5)


def test_zero_duty_case_does_not_raise_the_continuous_rating():
    """An extreme that happens for a moment must not set the thermal rating."""
    arm = build_arm()
    cases = standard_load_cases(arm, RHO)
    worst = [c for c in cases if c.name == "LC7_combined_worst"][0]
    assert worst.duty_fraction == 0.0
    duty = evaluate_duty_cycle(arm, RHO, cases)
    without_worst = evaluate_duty_cycle(
        arm, RHO, [c for c in cases if c is not worst])
    assert np.allclose(duty.continuous_torque_nm(),
                       without_worst.continuous_torque_nm(), rtol=1e-9)
    # but it must still drive the peak
    assert np.all(duty.peak_torque_nm() >= without_worst.peak_torque_nm())


def test_standard_cases_cover_the_required_conditions():
    arm = build_arm()
    names = {c.name for c in standard_load_cases(arm, RHO)}
    for required in ("LC1_nominal", "LC2_max_payload", "LC3_max_acceleration",
                     "LC4_worst_gravity", "LC6_holding", "LC7_combined_worst"):
        assert required in names


def test_heavier_payload_raises_the_holding_torque():
    arm = build_arm()
    light = evaluate_duty_cycle(arm, RHO, max_payload_kg=2.0)
    heavy = evaluate_duty_cycle(arm, RHO, max_payload_kg=10.0)
    assert heavy.peak_torque_nm()[0] > light.peak_torque_nm()[0]
