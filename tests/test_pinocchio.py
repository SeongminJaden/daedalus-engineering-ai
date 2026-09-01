"""Cross-validation of the multibody kinematics and dynamics against Pinocchio.

The model is exported to URDF and read back rather than transcribed, so these
tests compare two implementations of the same model rather than two models.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from core.assembly.frames import GRAVITY_DIRECTION, STANDARD_GRAVITY
from core.assembly.kinematics import geometric_jacobian
from core.assembly.model import Assembly, Joint, JointType, Link
from core.assembly.urdf import (UNMODELLED_LIMIT, _check_is_a_tree,
                                assembly_to_urdf)
from core.design_genome import DesignGenome, HollowRectangleSection
from core.registry import Category, ProblemContext
from nodes import pinocchio_node as pn
from nodes.descriptor import CapabilityUnavailable
from nodes.roster import build_roster
from physics.dynamics.equations import (CHRISTOFFEL_STEP, coriolis_matrix,
                                        gravity_torques,
                                        mass_matrix_derivative)
from projects.robotic_arm.arm import build_arm

requires_pinocchio = pytest.mark.skipif(
    not pn.is_available(), reason="pinocchio is not installed")

DENSITY = 2810.0
MATERIAL = "al_7075_t6"

# Five configurations rather than one. A single pose can agree by accident,
# for instance where a sine vanishes.
STATES = [
    (np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])),
    (np.array([np.pi / 4, 0.3]), np.array([0.7, -0.4]), np.array([1.1, 2.3])),
    (np.array([-1.2, 2.0]), np.array([-2.5, 1.5]), np.array([-3.0, 0.8])),
    (np.array([np.pi / 2, -np.pi / 3]), np.array([0.1, 0.2]),
     np.array([-1.0, -1.0])),
    (np.array([2.9, -2.5]), np.array([3.0, 3.0]), np.array([5.0, -5.0])),
]


def _link(name: str, length: float, b: float, h: float, t: float) -> Link:
    return Link(name=name, length_m=length,
                genome=DesignGenome(
                    section=HollowRectangleSection(
                        outer_width_m=b, outer_height_m=h, wall_thickness_m=t),
                    material_id=MATERIAL))


def three_link_arm() -> Assembly:
    """A third link, so the comparison is not tuned to two degrees of freedom.

    The wrist axis is +z like the others, keeping the chain planar, which
    makes the closed-form gravity torque still checkable by hand if needed.
    """
    lengths = (0.30, 0.25, 0.18)
    sections = ((0.020, 0.040, 0.002), (0.016, 0.032, 0.002),
                (0.012, 0.024, 0.0015))
    links = [_link(f"link{i + 1}", lengths[i], *sections[i]) for i in range(3)]
    origin = np.eye(4)
    joints = [Joint(name="shoulder", type=JointType.REVOLUTE, parent=None,
                    child="link1", axis=[0.0, 0.0, 1.0],
                    lower_limit=-np.pi, upper_limit=np.pi)]
    for i, (parent, child) in enumerate((("link1", "link2"),
                                         ("link2", "link3"))):
        shift = origin.copy()
        shift[0, 3] = lengths[i]
        joints.append(Joint(name=("elbow", "wrist")[i],
                            type=JointType.REVOLUTE, parent=parent,
                            child=child, axis=[0.0, 0.0, 1.0],
                            origin=shift.tolist(),
                            lower_limit=-2.6, upper_limit=2.6))
    return Assembly(name="three_link_planar_arm", material_id=MATERIAL,
                    links=links, joints=joints)


# ------------------------------------------------------------------ the node

def test_availability_is_read_from_the_import_system():
    descriptor = pn.pinocchio_descriptor()
    assert descriptor.available is pn.is_available()
    if not descriptor.available:
        assert "unavailable" in descriptor.unavailable_reason


def test_a_missing_library_raises_rather_than_returning_numbers(monkeypatch):
    monkeypatch.setattr(pn, "_pinocchio", lambda: None)
    with pytest.raises(CapabilityUnavailable):
        pn.compare(build_arm(), [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], DENSITY)


def test_the_capability_routes_only_for_a_mechanism():
    registry = build_roster()
    assert pn.PINOCCHIO_CAPABILITY in registry
    chain = ProblemContext(has_articulated_chain=True)
    if pn.is_available():
        assert pn.PINOCCHIO_CAPABILITY in registry.query(
            chain, Category.ANALYSIS).names()
    # A bracket has no mass matrix, and an unstated feature must fail closed.
    assert pn.PINOCCHIO_CAPABILITY not in registry.query(
        ProblemContext(), Category.ANALYSIS).names()


# ------------------------------------------------------------------- the URDF

def test_the_urdf_round_trips_the_model_rather_than_a_copy():
    """Pinocchio must read the same degrees of freedom this project has."""
    arm = build_arm()
    model, _, path = pn.load_model(arm, DENSITY)
    assert model.nv == arm.dof
    assert path.exists()
    for joint in arm.joints:
        assert model.getJointId(joint.name) < model.njoints


def test_the_urdf_does_not_claim_limits_this_project_does_not_model():
    """Effort and velocity are required by URDF and unknown here.

    The sentinel is deliberately absurd so nobody reads it as a design limit,
    and the file says so in a comment next to it.
    """
    text = assembly_to_urdf(build_arm(), DENSITY)
    assert f'effort="{UNMODELLED_LIMIT!r}"' in text
    assert "does not model them" in text
    assert "NOT design limits" in text


def test_a_closed_loop_cannot_reach_the_exporter():
    """The Assembly model refuses the loop first, which is where it belongs."""
    arm = build_arm()
    with pytest.raises(ValidationError, match="closed loops"):
        Assembly(name="looped", material_id=MATERIAL, links=arm.links,
                 joints=list(arm.joints) + [
                     Joint(name="closure", type=JointType.REVOLUTE,
                           parent="link2", child="link1", axis=[0.0, 0.0, 1.0],
                           lower_limit=-1.0, upper_limit=1.0)])


def test_the_exporter_re_checks_the_tree_invariant_anyway():
    """Exercised directly, since no Assembly can carry a loop to it.

    An export that dropped a joint would produce a model that runs and is
    wrong, which is worth one pass over the joint list to prevent.
    """
    arm = build_arm()
    duplicated = list(arm.joints) + [arm.joints[-1]]
    with pytest.raises(ValueError, match="tree"):
        _check_is_a_tree(duplicated)


# ------------------------------------------------- the six-way cross-check

@requires_pinocchio
@pytest.mark.parametrize("index", range(len(STATES)))
def test_the_two_link_arm_agrees_on_every_analytic_quantity(index):
    q, qd, qdd = STATES[index]
    result = pn.compare(build_arm(), q, qd, qdd, DENSITY)
    assert result.forward_kinematics_error_m < 1e-14
    assert result.jacobian_error < 1e-14
    assert result.mass_matrix_error < 1e-14
    assert result.gravity_error_nm < 1e-12
    assert result.inverse_dynamics_error_nm < 1e-9
    assert result.worst_analytic_error() < 1e-9


@requires_pinocchio
def test_the_three_link_arm_agrees_too(tmp_path):
    """Two degrees of freedom can hide an indexing error that three exposes."""
    q = np.array([0.4, -0.9, 1.3])
    qd = np.array([1.2, -0.6, 0.8])
    qdd = np.array([2.0, 1.0, -1.5])
    result = pn.compare(three_link_arm(), q, qd, qdd, DENSITY,
                        urdf_path=tmp_path / "three.urdf")
    assert result.worst_analytic_error() < 1e-9
    assert result.torque_scale_nm > 0.1


@requires_pinocchio
def test_the_coriolis_residual_is_differencing_error_not_a_bug():
    """Measured, not asserted: the error falls as the square of the step.

    This project builds C from a centrally differenced mass matrix while
    Pinocchio differentiates analytically, so a residual is expected. Second
    order convergence is what distinguishes truncation from a mistake.
    """
    import pinocchio

    arm = build_arm()
    q, qd = np.array([np.pi / 4, 0.3]), np.array([0.7, -0.4])
    model, data, _ = pn.load_model(arm, DENSITY)
    reference = np.asarray(pinocchio.computeCoriolisMatrix(model, data, q, qd))

    ratios = []
    for step in (1e-2, 1e-3, 1e-4):
        dm = mass_matrix_derivative(arm, q, DENSITY, step=step)
        n = arm.dof
        built = np.array([[0.5 * sum(
            (dm[i, j, k] + dm[i, k, j] - dm[j, k, i]) * qd[k] for k in range(n))
            for j in range(n)] for i in range(n)])
        ratios.append(float(np.abs(built - reference).max()) / step ** 2)

    # A constant ratio across two decades of step is second order.
    assert max(ratios) / min(ratios) < 1.05


@requires_pinocchio
def test_the_shipped_step_beats_the_steps_on_either_side_of_it():
    """The default is the measured optimum, not a round number.

    Truncation falls as h^2 and round-off grows as 1/h, so the best step is
    where they cross. Checking that the neighbours are both worse is what
    makes it an optimum rather than an assertion.
    """
    import pinocchio

    arm = build_arm()
    q, qd = np.array([np.pi / 4, 0.3]), np.array([0.7, -0.4])
    model, data, _ = pn.load_model(arm, DENSITY)
    reference = np.asarray(pinocchio.computeCoriolisMatrix(model, data, q, qd))
    n = arm.dof

    def error_at(step: float) -> float:
        dm = mass_matrix_derivative(arm, q, DENSITY, step=step)
        built = np.array([[0.5 * sum(
            (dm[i, j, k] + dm[i, k, j] - dm[j, k, i]) * qd[k] for k in range(n))
            for j in range(n)] for i in range(n)])
        return float(np.abs(built - reference).max())

    shipped = error_at(CHRISTOFFEL_STEP)
    assert shipped < error_at(CHRISTOFFEL_STEP * 10.0)   # truncation side
    assert shipped < error_at(CHRISTOFFEL_STEP / 10.0)   # round-off side


# ------------------------------------------- the control that makes it mean something

@requires_pinocchio
def test_the_comparison_would_catch_a_wrong_gravity_direction():
    """The control. This project is y-up; Pinocchio defaults to z-down.

    If the node forgot to set the gravity direction, the planar arm's gravity
    torque would come back as zero from Pinocchio and the disagreement would
    be total. Passing the real comparison therefore means the convention was
    actually carried across, not that gravity happened not to matter.
    """
    import pinocchio

    arm = build_arm()
    q = np.array([np.pi / 4, 0.3])
    model, data, _ = pn.load_model(arm, DENSITY)

    ours = gravity_torques(arm, q, DENSITY)
    correct = np.asarray(pinocchio.computeGeneralizedGravity(model, data, q))
    assert np.abs(ours - correct).max() < 1e-12

    model.gravity.linear = np.array([0.0, 0.0, -STANDARD_GRAVITY])
    wrong = np.asarray(
        pinocchio.computeGeneralizedGravity(model, model.createData(), q))
    assert np.abs(wrong).max() < 1e-12          # z gravity, planar arm, no torque
    assert np.abs(ours).max() > 0.1             # ours is not small
    assert np.abs(ours - wrong).max() > 0.1     # so the mistake is detectable


def test_the_project_gravity_convention_is_still_y_up():
    """If this ever changes, the node's explicit gravity must change with it."""
    assert np.allclose(GRAVITY_DIRECTION, [0.0, -1.0, 0.0])


# --------------------------------------------------------------- the limit

def test_agreeing_with_pinocchio_is_not_physical_validation():
    """Both idealise identically, so they can be wrong together.

    Rigid links, ideal frictionless joints, no backlash and no compliance.
    On a real mechanism those are what dominate the error, and neither
    implementation models any of them.
    """
    method = pn.pinocchio_capability_method()
    assert method.evidence == "SIMULATED"
    assert "not a measurement" in method.notes
    assert "Contact and friction dynamics are out of scope" in method.notes
