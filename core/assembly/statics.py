"""core.assembly.statics: holding torques and per-link load cases.

STATICS ONLY. There is no inertia, no Coriolis term, no acceleration torque, no
friction, no backlash and no joint compliance here. Everything is a rigid body
on an ideal joint. That covers the "hold this pose against gravity and a
payload" question, which is what sizes a link, and it does NOT cover motor
selection or gearbox sizing, which need the dynamic terms. See the roadmap.

Sign convention: `joint_torques` returns the **actuator torque required to hold
the pose**. A horizontal arm carrying a downward tip load needs positive
(counter-clockwise, right-hand rule about +z) torque to hold it up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import GRAVITY_DIRECTION, STANDARD_GRAVITY, transform_point
from .kinematics import forward_kinematics, position_jacobian
from .model import Assembly, JointType


@dataclass
class LinkLoad:
    """Resultant transmitted at a link's root, and the beam load case for it."""

    link: str
    root_force_n: np.ndarray          # world frame
    root_moment_nm: np.ndarray        # world frame, about the root
    transverse_force_n: float         # magnitude perpendicular to the link axis
    axial_force_n: float
    root_bending_moment_nm: float
    equivalent_tip_load_n: float

    def as_dict(self) -> dict:
        return {
            "link": self.link,
            "root_force_n": self.root_force_n.tolist(),
            "root_moment_nm": self.root_moment_nm.tolist(),
            "transverse_force_n": self.transverse_force_n,
            "axial_force_n": self.axial_force_n,
            "root_bending_moment_nm": self.root_bending_moment_nm,
            "equivalent_tip_load_n": self.equivalent_tip_load_n,
        }


def gravity_vector(magnitude: float = STANDARD_GRAVITY) -> np.ndarray:
    return GRAVITY_DIRECTION * magnitude


def link_com_positions(assembly: Assembly, q) -> dict[str, np.ndarray]:
    pose = forward_kinematics(assembly, q)
    return {
        link.name: transform_point(pose.link_transforms[link.name],
                                   link.com_local())
        for link in assembly.links
    }


def joint_torques(
    assembly: Assembly,
    q,
    density_kg_m3: float,
    tip_force_n: np.ndarray | None = None,
    include_gravity: bool = True,
    gravity: float = STANDARD_GRAVITY,
) -> np.ndarray:
    """Actuator torque needed to hold the configuration, one per actuated joint.

    Built from generalized forces: a force f applied at a point contributes
    J_point^T f to the generalized force, and the actuator must supply the
    negative of the total to stay in equilibrium.
    """
    actuated = assembly.actuated_joints()
    generalized = np.zeros(len(actuated), dtype=np.float64)

    if tip_force_n is not None:
        f = np.asarray(tip_force_n, dtype=np.float64).reshape(3)
        generalized += position_jacobian(assembly, q).T @ f

    if include_gravity:
        g = gravity_vector(gravity)
        coms = link_com_positions(assembly, q)
        for link in assembly.links:
            weight = link.mass_kg(density_kg_m3) * g
            jac = position_jacobian(assembly, q, point_world=coms[link.name],
                                    link_name=link.name)
            generalized += jac.T @ weight

    return -generalized


def _distal_links(assembly: Assembly, joint_name: str) -> list[str]:
    """Links carried by (outboard of) a joint, including its own child."""
    by_parent: dict[str | None, list] = {}
    for joint in assembly.joints:
        by_parent.setdefault(joint.parent, []).append(joint)
    start = assembly.joint(joint_name)
    out, frontier = [], [start.child]
    while frontier:
        current = frontier.pop()
        out.append(current)
        for joint in by_parent.get(current, []):
            frontier.append(joint.child)
    return out


def link_load_cases(
    assembly: Assembly,
    q,
    density_kg_m3: float,
    tip_force_n: np.ndarray | None = None,
    include_gravity: bool = True,
    gravity: float = STANDARD_GRAVITY,
) -> list[LinkLoad]:
    """Force and moment transmitted at each link's root, and its beam load case.

    `equivalent_tip_load_n` is the transverse tip load that would produce the
    SAME ROOT BENDING MOMENT on a cantilever of that link's length. Root moment
    is what drives bending stress and it is where the structure is critical, so
    matching it is the right equivalence for sizing. It deliberately does not
    reproduce the distributed shape of self-weight along the span, so the
    deflection of a self-weight-dominated link is approximated by this
    substitution. For the payload-dominated cases here the tip load is the
    dominant term and the substitution is close; a link whose own weight
    dominates would need a distributed-load model.
    """
    pose = forward_kinematics(assembly, q)
    g = gravity_vector(gravity)
    coms = link_com_positions(assembly, q)

    forces: list[tuple[np.ndarray, np.ndarray]] = []      # (point, force)
    if tip_force_n is not None:
        forces.append((pose.tool_position(),
                       np.asarray(tip_force_n, dtype=np.float64).reshape(3)))
    if include_gravity:
        for link in assembly.links:
            forces.append((coms[link.name], link.mass_kg(density_kg_m3) * g))

    out: list[LinkLoad] = []
    for joint in assembly.ordered_joints():
        link = assembly.link(joint.child)
        root = pose.joint_origins[joint.name]
        carried = set(_distal_links(assembly, joint.name))

        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        for point, force in forces:
            # A force belongs to this link's load only if it acts on something
            # this link carries. The tool force always does.
            owner = None
            for name, com in coms.items():
                if np.allclose(point, com):
                    owner = name
                    break
            if owner is not None and owner not in carried:
                continue
            total_force += force
            total_moment += np.cross(point - root, force)

        axis = np.asarray(pose.link_transforms[link.name], dtype=np.float64)[:3, 0]
        axis = axis / max(np.linalg.norm(axis), 1e-30)
        axial = float(np.dot(total_force, axis))
        transverse = float(np.linalg.norm(total_force - axial * axis))
        bending = float(np.linalg.norm(total_moment))

        out.append(LinkLoad(
            link=link.name,
            root_force_n=total_force,
            root_moment_nm=total_moment,
            transverse_force_n=transverse,
            axial_force_n=axial,
            root_bending_moment_nm=bending,
            equivalent_tip_load_n=bending / link.length_m,
        ))
    return out


def worst_gravity_pose(assembly: Assembly, density_kg_m3: float,
                       tip_force_n: np.ndarray | None = None,
                       samples: int = 73) -> tuple[np.ndarray, float]:
    """The fully extended horizontal pose, and the peak joint torque there.

    For a serial arm the worst static case is the one with the longest moment
    arm, which is the chain stretched horizontally. This searches the base
    joint angle on a grid rather than asserting it, so the answer is measured.

    THE SEARCH IS RESTRICTED TO REACHABLE POSES. It used to sweep the whole
    circle regardless of what the joint allows, which sizes a motor for a pose
    the mechanism cannot get into: with the shoulder limited to plus or minus
    0.5 rad it still returned -3.1416. An unreachable worst case is not
    conservative, it is wrong in an unknown direction, because the true worst
    REACHABLE pose is a different pose with a different torque.

    Raises when the resting pose itself is unreachable, since the other joints
    are held at zero and a zero outside their limits makes every sample
    invalid rather than merely the extremes.
    """
    base = assembly.actuated_joints()[0]
    low = -np.pi if base.lower_limit is None else max(-np.pi, base.lower_limit)
    high = np.pi if base.upper_limit is None else min(np.pi, base.upper_limit)
    if low > high:
        raise ValueError(
            f"joint {base.name!r} has no reachable angle in [-pi, pi]")

    resting = np.zeros(assembly.dof)
    resting[0] = low
    violations = [v for v in assembly.limit_violations(resting)
                  if v.joint != base.name]
    if violations:
        raise ValueError(
            "the held joints are outside their limits at zero, so no sample "
            "in this search is reachable: "
            + "; ".join(str(v) for v in violations))

    best_q, best = None, -np.inf
    for angle in np.linspace(low, high, samples):
        q = np.zeros(assembly.dof)
        q[0] = angle
        torque = np.abs(joint_torques(assembly, q, density_kg_m3,
                                      tip_force_n)).max()
        if torque > best:
            best, best_q = float(torque), q.copy()
    return best_q, best
