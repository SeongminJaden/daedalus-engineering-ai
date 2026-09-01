"""A two link chain checked three ways: closed form, Pinocchio and MuJoCo.

Every piece here was already verified on its own. What was NOT verified is
that they agree with each other on a MULTI JOINT chain, which is where an
error in frame composition, in a Jacobian column, or in a sign convention
actually shows up. A single link cannot expose any of those, because there is
nothing to compose.

The three parties are genuinely independent:

* the closed form for a two link planar arm, written out here from the
  geometry and owing nothing to any solver
* Pinocchio, a different implementation reached through an exported URDF
* MuJoCo, reached the other way round, by feeding this project's inverse
  dynamics torque into its forward dynamics and requiring the original
  acceleration back

Agreement between this project and one other implementation could be a shared
misunderstanding. Agreement with a hand calculation as well is much harder to
arrange by accident.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.assembly import (STANDARD_GRAVITY, Assembly, Joint, JointType, Link,
                           forward_kinematics, geometric_jacobian,
                           joint_torques, link_com_positions)
from core.design_genome import DesignGenome, HollowRectangleSection
from nodes import mujoco_node, pinocchio_node
from physics.dynamics import equations, forward

L1, L2 = 0.30, 0.25
DENSITY = 2810.0
CONFIGURATIONS = [(0.3, -0.4), (0.0, 0.0), (1.1, 0.9), (-0.7, 1.3)]


def _link(name: str, length: float) -> Link:
    return Link(name=name, length_m=length, genome=DesignGenome(
        section=HollowRectangleSection(outer_width_m=0.020,
                                       outer_height_m=0.040,
                                       wall_thickness_m=0.002),
        material_id="al_7075_t6"))


def _translation(x: float) -> list:
    matrix = np.eye(4)
    matrix[0, 3] = x
    return matrix.tolist()


@pytest.fixture(scope="module")
def arm() -> Assembly:
    limits = dict(lower_limit=-3.0, upper_limit=3.0)
    return Assembly(
        name="planar2", material_id="al_7075_t6",
        links=[_link("link1", L1), _link("link2", L2)],
        joints=[
            Joint(name="j1", type=JointType.REVOLUTE, parent=None,
                  child="link1", axis=[0, 0, 1], **limits),
            Joint(name="j2", type=JointType.REVOLUTE, parent="link1",
                  child="link2", axis=[0, 0, 1], origin=_translation(L1),
                  **limits),
        ])


def analytic_tip(q) -> np.ndarray:
    """x = L1 c1 + L2 c12, y = L1 s1 + L2 s12."""
    t1, t2 = float(q[0]), float(q[1])
    return np.array([L1 * math.cos(t1) + L2 * math.cos(t1 + t2),
                     L1 * math.sin(t1) + L2 * math.sin(t1 + t2), 0.0])


def analytic_jacobian(q) -> np.ndarray:
    """The 2x2 position Jacobian of a planar two link arm, by differentiation."""
    t1, t2 = float(q[0]), float(q[1])
    s1, c1 = math.sin(t1), math.cos(t1)
    s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [L1 * c1 + L2 * c12, L2 * c12]])


# ------------------------------------------------- against the closed form

@pytest.mark.parametrize("q", CONFIGURATIONS)
def test_the_tip_matches_the_closed_form(arm, q):
    got = forward_kinematics(arm, q).tool_position()
    assert np.allclose(got, analytic_tip(q), atol=1e-12)


@pytest.mark.parametrize("q", CONFIGURATIONS)
def test_the_jacobian_matches_the_closed_form(arm, q):
    """Both linear rows and both angular rows.

    The angular part is the easy half to get wrong quietly: two joints about
    the same axis must each contribute exactly one, and nothing else.
    """
    got = geometric_jacobian(arm, q)
    assert np.allclose(got[:2, :], analytic_jacobian(q), atol=1e-12)
    assert np.allclose(got[2, :], 0.0, atol=1e-12)
    assert np.allclose(got[3:5, :], 0.0, atol=1e-12)
    assert np.allclose(got[5, :], [1.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("q", CONFIGURATIONS)
def test_the_jacobian_is_the_derivative_of_the_kinematics(arm, q):
    """A second, independent check by finite difference.

    The closed form above and the Jacobian could in principle share a mistake
    if both were derived the same way. Differencing the FK cannot: it uses
    only the positions.
    """
    step = 1e-7
    numerical = np.zeros((3, 2))
    for i in range(2):
        high = list(q)
        low = list(q)
        high[i] += step
        low[i] -= step
        numerical[:, i] = (forward_kinematics(arm, high).tool_position()
                           - forward_kinematics(arm, low).tool_position()
                           ) / (2.0 * step)
    assert np.allclose(geometric_jacobian(arm, q)[:3, :], numerical, atol=1e-6)


@pytest.mark.parametrize("q", CONFIGURATIONS)
def test_the_gravity_hold_torque_matches_a_hand_calculation(arm, q):
    """Torque about each joint is g times the moment of the mass beyond it.

    Written from the centres of mass and the link masses directly, with no
    solver involved. Gravity acts along minus y here, so the moment arm about
    a z axis joint is the horizontal distance.
    """
    masses = [link.mass_kg(DENSITY) for link in arm.links]
    coms = link_com_positions(arm, q)
    joint2_x = L1 * math.cos(float(q[0]))

    expected = [
        STANDARD_GRAVITY * (masses[0] * coms["link1"][0]
                            + masses[1] * coms["link2"][0]),
        STANDARD_GRAVITY * masses[1] * (coms["link2"][0] - joint2_x),
    ]
    assert np.allclose(joint_torques(arm, q, DENSITY), expected, atol=1e-12)


# ------------------------------------------------------- against Pinocchio

pinocchio_available = pytest.mark.skipif(
    not pinocchio_node.is_available(), reason="Pinocchio is not installed")


@pinocchio_available
@pytest.mark.parametrize("q", CONFIGURATIONS)
def test_pinocchio_agrees_on_the_whole_chain(arm, q, tmp_path):
    """Every quantity at once, on a chain rather than a single joint.

    A frame composition error, a wrong Jacobian column or a sign convention
    mistake all survive a one joint test and none of them survive this.
    """
    qd = np.array([0.5, -0.2])
    qdd = np.array([1.2, 0.7])
    result = pinocchio_node.compare(arm, list(q), qd, qdd, DENSITY,
                                    tmp_path / "arm.urdf")

    assert result.forward_kinematics_error_m < 1e-12
    assert result.jacobian_error < 1e-12
    assert result.mass_matrix_error < 1e-12
    assert result.coriolis_error < 1e-10
    assert result.gravity_error_nm < 1e-12
    assert result.inverse_dynamics_error_nm < 1e-10
    assert result.torque_scale_nm > 0.0


# ---------------------------------------------------------- against MuJoCo

mujoco_available = pytest.mark.skipif(
    not mujoco_node.is_available(), reason="MuJoCo is not installed")


@mujoco_available
@pytest.mark.parametrize("q", CONFIGURATIONS)
def test_mujoco_returns_the_acceleration_the_torque_was_built_for(arm, q):
    """The round trip, and it runs the two solvers in opposite directions.

    This project computes the torque that produces a chosen acceleration;
    MuJoCo is asked what acceleration that torque produces. Recovering the
    original is a statement about both implementations at once, and it does
    not depend on either agreeing about how to express a mass matrix.
    """
    qd = np.array([0.5, -0.2])
    qdd = np.array([1.2, 0.7])
    tau = equations.inverse_dynamics(arm, list(q), qd, qdd, DENSITY)
    recovered = mujoco_node.accelerations(arm, list(q), qd, tau, DENSITY)
    assert np.allclose(recovered, qdd, atol=1e-8)


@mujoco_available
def test_a_zero_torque_chain_is_not_held_up(arm):
    """A sanity check on the round trip above.

    If the comparison passed for a torque of zero it would prove nothing, so
    this pins that zero torque gives a non zero acceleration: the arm falls.
    """
    q = [0.3, -0.4]
    zero = np.zeros(2)
    falling = mujoco_node.accelerations(arm, q, zero, zero, DENSITY)
    assert np.linalg.norm(falling) > 1.0


# ------------------------------------------------------------ conservation

#: Energy is sampled rather than evaluated at every step. Drift accumulates
#: steadily rather than in spikes, so a sample every few steps finds the same
#: maximum far more cheaply.
ENERGY_SAMPLE_EVERY = 5


def _relative_drift(assembly: Assembly, integrator: str,
                    duration_s: float = 0.3, dt_s: float = 1e-3) -> float:
    """Largest departure from the starting energy, as a fraction of it."""
    q0 = np.array([1.2, -0.5])
    qd0 = np.zeros(2)
    start = forward.total_energy_j(assembly, q0, qd0, DENSITY)
    trajectory = forward.simulate(assembly, q0, qd0, DENSITY,
                                  duration_s=duration_s, dt_s=dt_s,
                                  integrator=integrator)
    sampled = list(zip(trajectory.q, trajectory.qd))[::ENERGY_SAMPLE_EVERY]
    worst = max(abs(forward.total_energy_j(assembly, q, qd, DENSITY) - start)
                for q, qd in sampled)
    return worst / abs(start)

def test_free_fall_conserves_energy(arm):
    """No torque and no friction, so the total energy must not drift.

    This checks the integrator rather than the model: an unstable or badly
    scaled scheme shows up as energy climbing, and would otherwise be
    invisible in a trajectory that still looks plausible.
    """
    assert _relative_drift(arm, "rk4") < 1e-9


def test_the_energy_check_would_notice_a_drifting_integrator(arm):
    """The check above is only meaningful if it can fail.

    Semi implicit Euler on the same problem drifts far more, so a comparison
    against it shows the tolerance is doing work rather than being loose
    enough to pass anything. Measured here the separation is about eight
    orders of magnitude, so this is not a marginal distinction.
    """
    assert _relative_drift(arm, "rk4") < _relative_drift(
        arm, "semi_implicit_euler")
