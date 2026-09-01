"""Pinocchio as an external multibody solver, for cross-checking the dynamics.

Until now the external checks covered only continua: CalculiX for structural
FEA and OpenFOAM for fluids. The multibody kinematics and dynamics had been
verified against a closed-form two-link solution and by hand, which catches a
great deal but shares an author with the thing it checks. Pinocchio is written
independently at INRIA, and it reads the SAME model rather than a
hand-transcribed copy: the assembly is exported to URDF first, so a typo in a
second description cannot be mistaken for agreement.

WHAT THIS IS NOT
================
Pinocchio idealises exactly as this project does: rigid bodies, ideal joints,
no friction, no flexibility, no backlash. That is why agreement is expected at
round-off rather than merely close. It also means agreement tests the
IMPLEMENTATION and says nothing about a real mechanism, where joint friction
and structural compliance are what actually limit accuracy. Contact and
friction dynamics are not covered here at all and belong to a separate node.

VALIDITY DOMAIN
===============
Stated before implementing, per the standing discipline.

Applies
    Open kinematic chains (a tree) of rigid bodies with revolute or prismatic
    joints, which is what URDF can express and what this project builds.

Does not apply
    Closed loops, flexible bodies, contact, friction, joint limits as
    constraints rather than as recorded numbers, and any actuator dynamics.

Expected agreement, and where it is looser on purpose
    Forward kinematics, the geometric Jacobian, M(q), G(q) and the inverse
    dynamics torque are analytic on both sides and agree to round-off. The
    Coriolis matrix does NOT: this project builds it from Christoffel symbols
    of a CENTRALLY DIFFERENCED mass matrix, while Pinocchio differentiates
    analytically. The residual is therefore truncation error and scales as the
    square of the differencing step, which the tests measure rather than
    assume.

Frame convention
    This project is y-up with gravity along -y. URDF cannot record a gravity
    direction, so :func:`load_model` sets it explicitly. Leaving Pinocchio's
    z-down default in place would produce a wrong but completely plausible
    gravity torque, and that is the single most likely way this comparison
    could silently pass while being meaningless.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.assembly.frames import GRAVITY_DIRECTION, STANDARD_GRAVITY
from core.assembly.kinematics import forward_kinematics, geometric_jacobian
from core.assembly.model import Assembly
from core.assembly.urdf import assembly_to_urdf
from physics.dynamics.equations import (coriolis_matrix, gravity_torques,
                                        inverse_dynamics, mass_matrix)

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

PINOCCHIO_NODE_NAME = "pinocchio.local"
PINOCCHIO_CAPABILITY = "analysis.multibody.pinocchio"


def _pinocchio():
    try:
        import pinocchio
    except ImportError:
        return None
    return pinocchio


def is_available() -> bool:
    return _pinocchio() is not None


def version() -> str | None:
    """The reported version, or None when Pinocchio is absent."""
    module = _pinocchio()
    return f"Pinocchio {module.__version__}" if module else None


def pinocchio_descriptor(available: bool | None = None) -> NodeDescriptor:
    """The node as the registry sees it, read from the import system."""
    present = is_available() if available is None else available
    return NodeDescriptor(
        name=PINOCCHIO_NODE_NAME, transport=Transport.STDIO,
        address="pinocchio", available=present,
        unavailable_reason="" if present else
        "unavailable: the pinocchio package is not installed")


def pinocchio_capability_method():
    """The capability declaration, in the registry's schema."""
    from core.registry import Category, Condition, Cost, Fidelity, Method

    return Method(
        name=PINOCCHIO_CAPABILITY,
        category=Category.ANALYSIS,
        summary="Independent rigid-body kinematics and dynamics in Pinocchio, "
                "for cross-validating this project's multibody equations.",
        inputs=("assembly", "configuration", "material"),
        outputs=("forward_kinematics", "jacobian", "mass_matrix",
                 "coriolis_matrix", "gravity_torques", "inverse_dynamics"),
        fidelity=Fidelity.ANALYTICAL,
        cost=Cost.CHEAP,
        conditions=(
            Condition("the mechanism is an open chain of rigid links",
                      lambda c: c.require("has_articulated_chain")),
        ),
        implementation="nodes.pinocchio_node.compare",
        evidence="SIMULATED",
        notes="A second independently written rigid-body library, not a "
              "measurement. It makes the same idealisations this project does, "
              "rigid links and ideal frictionless joints, which is why "
              "agreement is expected at round-off and why agreement says "
              "nothing about a real mechanism where friction and compliance "
              "dominate the error. The model is exported to URDF and read "
              "back rather than transcribed, so a typo in a second description "
              "cannot pass as agreement. URDF cannot carry a gravity "
              "direction, so this project's y-up convention is set explicitly; "
              "leaving the z-down default would give a plausible wrong answer. "
              "Contact and friction dynamics are out of scope here.")


def load_model(assembly: Assembly, density_kg_m3: float,
               urdf_path: Path | None = None):
    """Export the assembly to URDF and build a Pinocchio model from it.

    Returns the model, its data, and the path the URDF was written to, so a
    caller can inspect exactly what the external library was given.
    """
    pinocchio = _pinocchio()
    if pinocchio is None:
        raise CapabilityUnavailable(
            PINOCCHIO_CAPABILITY, PINOCCHIO_NODE_NAME,
            "the pinocchio package is not installed")
    if urdf_path is None:
        urdf_path = Path(tempfile.mkdtemp()) / f"{assembly.name}.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path.write_text(assembly_to_urdf(assembly, density_kg_m3))

    model = pinocchio.buildModelFromUrdf(str(urdf_path))
    # URDF carries no gravity direction. This project is y-up.
    model.gravity.linear = np.asarray(GRAVITY_DIRECTION,
                                      dtype=np.float64) * STANDARD_GRAVITY
    return model, model.createData(), urdf_path


@dataclass(frozen=True)
class MultibodyComparison:
    """Measured differences on all six quantities, with their scales.

    A raw difference means little without the magnitude it sits against, so
    each error is reported alongside the size of the quantity itself.
    """

    solver: str
    solver_version: str
    forward_kinematics_error_m: float
    jacobian_error: float
    mass_matrix_error: float
    coriolis_error: float
    gravity_error_nm: float
    inverse_dynamics_error_nm: float
    torque_scale_nm: float
    urdf_path: str

    def worst_analytic_error(self) -> float:
        """The worst of the five terms that are analytic on both sides.

        Coriolis is excluded deliberately: this project differences it
        numerically, so holding it to the same bar would be measuring the
        differencing step rather than the implementation.
        """
        return max(self.forward_kinematics_error_m, self.jacobian_error,
                   self.mass_matrix_error, self.gravity_error_nm,
                   self.inverse_dynamics_error_nm)


def compare(assembly: Assembly, q, qd, qdd, density_kg_m3: float,
            urdf_path: Path | None = None) -> MultibodyComparison:
    """Run all six comparisons against Pinocchio at one state."""
    pinocchio = _pinocchio()
    if pinocchio is None:
        raise CapabilityUnavailable(
            PINOCCHIO_CAPABILITY, PINOCCHIO_NODE_NAME,
            "the pinocchio package is not installed")
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    qdd = np.asarray(qdd, dtype=np.float64).reshape(-1)

    model, data, path = load_model(assembly, density_kg_m3, urdf_path)
    pinocchio.forwardKinematics(model, data, q, qd, qdd)
    pinocchio.updateFramePlacements(model, data)

    pose = forward_kinematics(assembly, q)
    fk_error = 0.0
    for joint in assembly.joints:
        theirs = data.oMi[model.getJointId(joint.name)]
        ours = pose.link_transforms[joint.child]
        fk_error = max(fk_error,
                       float(np.abs(ours[:3, 3] - theirs.translation).max()),
                       float(np.abs(ours[:3, :3] - theirs.rotation).max()))
    tool = model.getFrameId("tool")
    fk_error = max(fk_error, float(np.abs(
        pose.tool_position() - data.oMf[tool].translation).max()))

    their_jacobian = pinocchio.computeFrameJacobian(
        model, data, q, tool, pinocchio.LOCAL_WORLD_ALIGNED)
    jacobian_error = float(
        np.abs(geometric_jacobian(assembly, q) - their_jacobian).max())

    their_mass = np.asarray(pinocchio.crba(model, data, q))
    their_mass = np.triu(their_mass) + np.triu(their_mass, 1).T
    mass_error = float(
        np.abs(mass_matrix(assembly, q, density_kg_m3) - their_mass).max())

    their_gravity = np.asarray(
        pinocchio.computeGeneralizedGravity(model, data, q))
    gravity_error = float(
        np.abs(gravity_torques(assembly, q, density_kg_m3) - their_gravity).max())

    their_coriolis = np.asarray(
        pinocchio.computeCoriolisMatrix(model, data, q, qd))
    coriolis_error = float(np.abs(
        coriolis_matrix(assembly, q, qd, density_kg_m3) - their_coriolis).max())

    our_torque = inverse_dynamics(assembly, q, qd, qdd, density_kg_m3)
    their_torque = np.asarray(pinocchio.rnea(model, data, q, qd, qdd))
    torque_error = float(np.abs(our_torque - their_torque).max())

    return MultibodyComparison(
        solver="Pinocchio", solver_version=version() or "unknown",
        forward_kinematics_error_m=fk_error, jacobian_error=jacobian_error,
        mass_matrix_error=mass_error, coriolis_error=coriolis_error,
        gravity_error_nm=gravity_error,
        inverse_dynamics_error_nm=torque_error,
        torque_scale_nm=float(np.abs(our_torque).max()), urdf_path=str(path))
