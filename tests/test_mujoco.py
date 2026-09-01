"""Contact benchmarks with known answers, and the limits they expose.

Every case here has an analytical result, so the comparison is against
mathematics. Where MuJoCo disagrees, the disagreement is measured and
explained rather than absorbed into a loose tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.assembly.frames import STANDARD_GRAVITY
from core.registry import Category, ProblemContext
from nodes import mujoco_node as mj
from nodes.descriptor import CapabilityUnavailable
from nodes.roster import build_roster
from physics.dynamics.forward import forward_dynamics
from projects.robotic_arm.arm import build_arm

requires_mujoco = pytest.mark.skipif(
    not mj.is_available(), reason="mujoco is not installed")

DENSITY = 2810.0


# ------------------------------------------------------------------ the node

def test_availability_is_read_from_the_import_system():
    descriptor = mj.mujoco_descriptor()
    assert descriptor.available is mj.is_available()


def test_a_missing_library_raises_rather_than_returning_numbers(monkeypatch):
    monkeypatch.setattr(mj, "_mujoco", lambda: None)
    with pytest.raises(CapabilityUnavailable):
        mj.resting_contact_force_n()


def test_the_capability_is_general_and_not_robot_specific():
    """The registration must not narrow this to arms."""
    method = mj.mujoco_capability_method()
    assert "not robot specific" in method.notes
    registry = build_roster()
    assert mj.MUJOCO_CAPABILITY in registry
    if mj.is_available():
        assert mj.MUJOCO_CAPABILITY in registry.query(
            ProblemContext(has_contact=True), Category.ANALYSIS).names()
    # A problem with nothing touching must not route here, and an unstated
    # feature must fail closed.
    assert mj.MUJOCO_CAPABILITY not in registry.query(
        ProblemContext(), Category.ANALYSIS).names()


# ------------------------------------------- contactless: three implementations

@requires_mujoco
@pytest.mark.parametrize("q,qd", [
    ([0.0, 0.0], [0.0, 0.0]),
    ([np.pi / 4, 0.3], [0.7, -0.4]),
    ([-1.2, 2.0], [-2.5, 1.5]),
])
def test_without_contact_mujoco_agrees_with_this_project(q, qd):
    """The bridge to the Pinocchio work: same model, same answer."""
    arm = build_arm()
    q, qd = np.array(q), np.array(qd)
    tau = np.zeros(2)
    ours = forward_dynamics(arm, q, qd, tau, DENSITY)
    theirs = mj.accelerations(arm, q, qd, tau, DENSITY)
    assert np.abs(ours - theirs).max() < 1e-9 * max(np.abs(ours).max(), 1.0)


@requires_mujoco
def test_the_gravity_convention_is_carried_across():
    """y-up here, z-down in MuJoCo. Getting this wrong reads as plausible."""
    model, _, _ = mj.load_assembly(build_arm(), DENSITY)
    assert model.opt.gravity[1] == pytest.approx(-STANDARD_GRAVITY)
    assert abs(model.opt.gravity[2]) < 1e-12


@requires_mujoco
def test_joint_limits_are_switched_off_to_match_this_project():
    """This project records limits and does not enforce them as constraints."""
    model, _, _ = mj.load_assembly(build_arm(), DENSITY)
    assert not model.jnt_limited.any()


# ------------------------------------------------ contact, against known answers

@requires_mujoco
def test_a_resting_body_is_held_up_by_exactly_its_weight():
    force, weight = mj.resting_contact_force_n()
    assert force == pytest.approx(weight, rel=1e-6)


@requires_mujoco
@pytest.mark.parametrize("friction", [0.2, 0.4, 0.7])
def test_the_friction_limit_follows_atan_of_the_coefficient(friction):
    """Measured slightly LOW, consistently, and that is the model talking.

    Soft contact permits a little slip inside the friction cone, so the block
    lets go marginally early. The bias is one sided and small over this range,
    which is what makes it an explanation rather than an excuse.
    """
    measured = mj.critical_angle_deg(friction)
    exact = math.degrees(math.atan(friction))
    assert measured < exact                  # one sided, never late
    assert exact - measured < 0.5


@requires_mujoco
def test_the_friction_bias_grows_with_the_coefficient():
    """Stated so a future change that flattens it has to be noticed.

    At a coefficient of 1.0 the shortfall reaches about 1.3 degrees, far more
    than the 0.2 at a coefficient of 0.2. Anyone relying on this near a
    coefficient of 1 should know the error is several times larger there.
    """
    low = math.degrees(math.atan(0.2)) - mj.critical_angle_deg(0.2)
    high = math.degrees(math.atan(1.0)) - mj.critical_angle_deg(1.0)
    assert high > low
    assert high < 2.0


@requires_mujoco
def test_displacement_cannot_tell_sticking_from_sliding():
    """The control for the criterion this module had to change to.

    Well below the friction limit the block still moves, because soft contact
    creeps. A displacement threshold calls that sliding, which is how the
    first version of this benchmark reported a block sliding at 10 degrees
    against a limit of 21.8.
    """
    import mujoco

    model = mj._model_from_xml(mj.block_on_incline_xml(0.4, 10.0))
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    crept = abs(float(data.qpos[0]))
    assert crept > 1e-4                       # it moved
    assert not mj.block_slides(0.4, 10.0)     # and it is not sliding


# -------------------------------------- restitution is measured, never set

@requires_mujoco
def test_bounce_comes_from_the_contact_parameters_not_from_a_setting():
    """MuJoCo has no restitution coefficient, so this reads one back out."""
    lively = mj.drop_ball(damping_ratio=0.2)
    dead = mj.drop_ball(damping_ratio=1.0)
    # The raw maximum is the drop height in both cases and says nothing. What
    # differs is how high it comes back.
    assert lively.rebound_apex_m() > dead.rebound_apex_m() + 0.01
    assert not lively.energy_grew
    assert not dead.energy_grew


@requires_mujoco
def test_a_contact_run_never_gains_energy_at_a_sound_timestep():
    for ratio in (1.0, 0.5, 0.2):
        assert not mj.drop_ball(damping_ratio=ratio).energy_grew


@requires_mujoco
def test_too_large_a_timestep_inflates_energy_instead_of_failing():
    """The control that gives the energy check its teeth.

    A lightly damped contact at too coarse a step does not raise, warn, or
    look wrong. It runs, and the ball ends with a hundred times the energy it
    started with. This is why every contact result here carries an energy
    history rather than a final position.
    """
    unstable = mj.drop_ball(damping_ratio=0.05, timestep_s=2e-4)
    assert unstable.energy_grew
    assert unstable.energy_j.max() > 10.0 * unstable.energy_j[0]

    stable = mj.drop_ball(damping_ratio=0.05, timestep_s=2e-5)
    assert not stable.energy_grew


# --------------------------------------------------------------- the limit

def test_contact_agreement_is_not_a_measurement_of_a_real_surface():
    """Rigid bodies and one friction number. Real contact has more in it."""
    method = mj.mujoco_capability_method()
    assert method.evidence == "SIMULATED"
    assert "SOFT CONSTRAINT" in method.notes
