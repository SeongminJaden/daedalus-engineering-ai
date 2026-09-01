"""Forward dynamics: what the mechanism does, given the torques applied.

General multibody, not robot specific. The checks are against an independent
library and against energy, which is the property an integrator is most likely
to quietly violate.
"""

from __future__ import annotations

import numpy as np
import pytest

from nodes import pinocchio_node as pn
from physics.dynamics.forward import (INTEGRATORS, forward_dynamics,
                                      potential_energy_j, simulate,
                                      total_energy_j)
from projects.robotic_arm.arm import build_arm

requires_pinocchio = pytest.mark.skipif(
    not pn.is_available(), reason="pinocchio is not installed")

DENSITY = 2810.0

STATES = [
    (np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])),
    (np.array([np.pi / 4, 0.3]), np.array([0.7, -0.4]), np.array([0.0, 0.0])),
    (np.array([-1.2, 2.0]), np.array([-2.5, 1.5]), np.array([0.5, -0.3])),
    (np.array([2.9, -2.5]), np.array([3.0, 3.0]), np.array([-1.0, 2.0])),
]


@requires_pinocchio
@pytest.mark.parametrize("index", range(len(STATES)))
def test_forward_dynamics_matches_pinocchio_aba(index):
    """Against an independently written articulated body algorithm."""
    import pinocchio

    arm = build_arm()
    q, qd, tau = STATES[index]
    ours = forward_dynamics(arm, q, qd, tau, DENSITY)
    model, data, _ = pn.load_model(arm, DENSITY)
    theirs = np.asarray(pinocchio.aba(model, data, q, qd, tau))
    # Relative to the size of the accelerations, which reach several hundred
    # rad/s^2 here. The worst state measured lands at 1.0e-12.
    scale = max(np.abs(ours).max(), 1.0)
    assert np.abs(ours - theirs).max() / scale < 1e-11


def test_forward_and_inverse_dynamics_are_inverses():
    """The cheapest check available, and it needs no second library."""
    from physics.dynamics.equations import inverse_dynamics

    arm = build_arm()
    q = np.array([0.4, -0.9])
    qd = np.array([1.1, -0.6])
    qdd = np.array([2.0, 1.5])
    tau = inverse_dynamics(arm, q, qd, qdd, DENSITY)
    assert forward_dynamics(arm, q, qd, tau, DENSITY) == pytest.approx(
        qdd, rel=1e-9)


def test_mismatched_state_lengths_are_refused():
    arm = build_arm()
    with pytest.raises(ValueError, match="agree in length"):
        forward_dynamics(arm, [0.0, 0.0], [0.0, 0.0], [0.0], DENSITY)


def test_an_unknown_integrator_is_refused():
    arm = build_arm()
    with pytest.raises(ValueError, match="unknown integrator"):
        simulate(arm, [0.1, 0.1], [0.0, 0.0], DENSITY, duration_s=0.01,
                 dt_s=0.001, integrator="verlet")
    assert set(INTEGRATORS) == {"rk4", "semi_implicit_euler"}


def test_potential_energy_follows_the_projects_gravity_direction():
    """y-up: lifting the arm must raise its potential energy."""
    arm = build_arm()
    down = potential_energy_j(arm, np.array([-np.pi / 2, 0.0]), DENSITY)
    up = potential_energy_j(arm, np.array([np.pi / 2, 0.0]), DENSITY)
    assert up > down


def test_an_unforced_mechanism_conserves_energy_to_round_off():
    """No applied torque and no friction, so the exact motion is conservative.

    At the small steps a trajectory actually uses, RK4's energy error is at
    round-off rather than at its truncation limit.
    """
    arm = build_arm()
    trajectory = simulate(arm, [0.6, -0.4], [0.0, 0.0], DENSITY,
                          duration_s=0.05, dt_s=0.001)
    assert trajectory.energy_drift() < 1e-9 * abs(trajectory.energy_j[0])


def test_rk4_energy_error_falls_at_fourth_order_at_coarse_steps():
    """Measured with a phase independent metric, and here is why.

    E(end) - E(0) depends on where in an oscillation the run stops, which
    makes it a noisy way to measure an order. The largest deviation over a
    fixed window does not.
    """
    arm = build_arm()
    deviations = []
    for step in (1e-2, 5e-3, 2.5e-3):
        trajectory = simulate(arm, [0.6, -0.4], [0.0, 0.0], DENSITY,
                              duration_s=0.4, dt_s=step)
        deviations.append(float(np.abs(trajectory.energy_j
                                       - trajectory.energy_j[0]).max()))
    # Two halvings, each predicted to cut the error by sixteen.
    assert 8.0 < deviations[0] / deviations[1] < 32.0
    assert 8.0 < deviations[1] / deviations[2] < 32.0


def test_the_symplectic_integrator_oscillates_rather_than_drifting():
    """The signature that makes it worth having despite being first order.

    This is evidence of oscillation over the window tested, NOT a proof that
    the band stays bounded forever, and the module does not claim the
    stronger thing.
    """
    arm = build_arm()
    trajectory = simulate(arm, [0.6, -0.4], [0.0, 0.0], DENSITY,
                          duration_s=2.0, dt_s=2e-3,
                          integrator="semi_implicit_euler")
    energy = trajectory.energy_j
    reversals = int(np.sum(np.diff(np.sign(np.diff(energy))) != 0))
    assert reversals > 4, "monotonic drift, not the symplectic signature"


def test_the_integrator_travels_with_the_trajectory():
    """Energy drift is a property of the method, not of the mechanism."""
    arm = build_arm()
    for name in INTEGRATORS:
        trajectory = simulate(arm, [0.3, 0.2], [0.0, 0.0], DENSITY,
                              duration_s=0.01, dt_s=0.001, integrator=name)
        assert trajectory.integrator == name


def test_an_applied_torque_actually_drives_the_mechanism():
    arm = build_arm()
    free = simulate(arm, [0.0, 0.0], [0.0, 0.0], DENSITY, duration_s=0.05,
                    dt_s=0.001)
    driven = simulate(arm, [0.0, 0.0], [0.0, 0.0], DENSITY, duration_s=0.05,
                      dt_s=0.001, torque=lambda t, q, qd: np.array([5.0, 0.0]))
    assert abs(driven.q[-1, 0] - free.q[-1, 0]) > 1e-3


def test_friction_is_zero_and_that_is_a_stated_limitation():
    """A simulated mechanism swings forever. A real one does not."""
    from physics.dynamics.equations import friction_torques

    assert np.all(friction_torques(build_arm(), np.array([5.0, -5.0])) == 0.0)
