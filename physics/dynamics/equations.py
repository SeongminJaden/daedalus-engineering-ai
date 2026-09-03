"""physics.dynamics.equations: the rigid-body equations of motion.

    tau = M(q) qdd + C(q, qd) qd + G(q) + F_friction

Friction is present as a term and is zero: modelling it needs joint-level data
(breakaway torque, viscous coefficient, gearbox efficiency) that this project
does not have, and inventing values would put made-up numbers into a torque a
motor gets sized from. The slot exists so adding real data later does not change
the interface.

M is built from the kinetic energy rather than by a recursive algorithm:

    M(q) = sum_k [ m_k Jv_k^T Jv_k + Jw_k^T R_k I_k R_k^T Jw_k ]

which reuses the Phase 10 Jacobians directly, including their ancestry
restriction. C comes from the Christoffel symbols of M.
"""

from __future__ import annotations

import numpy as np

from core.assembly.kinematics import forward_kinematics, geometric_jacobian
from core.assembly.model import Assembly
from core.assembly.statics import joint_torques, link_com_positions

# Step for the numerical derivatives of M used by the Christoffel symbols.
# Central differences, so truncation falls as h^2 while round-off grows as
# 1/h, and the best step is where the two cross. That crossing was MEASURED
# against Pinocchio's analytic Coriolis matrix rather than estimated: the
# error over h^2 is constant at 1.5985e-04 for h of 1e-2, 1e-3 and 1e-4,
# confirming pure truncation there, and the total error is 1.6e-12 at h=1e-6
# against 8.3e-14 at h=1e-5. The earlier 1e-6 sat past the crossing, in the
# round-off dominated region, and cost about seventeen times the error for
# nothing. See tests/test_pinocchio.py, which measures the exponent.
#
# The absolute size of either figure is meaningless beside any physical
# uncertainty in a real mechanism; this is chosen because there is no reason
# to prefer the worse of two free options, not because it matters physically.
CHRISTOFFEL_STEP = 1e-5


def link_world_inertia(assembly: Assembly, pose, link, density_kg_m3: float
                       ) -> np.ndarray:
    """Link inertia about its centre of mass, rotated into world axes."""
    from .inertia import link_inertia

    rotation = np.asarray(pose.link_transforms[link.name],
                          dtype=np.float64)[:3, :3]
    return rotation @ link_inertia(link, density_kg_m3) @ rotation.T


def mass_matrix(assembly: Assembly, q, density_kg_m3: float) -> np.ndarray:
    """Joint-space mass matrix M(q), symmetric positive definite."""
    pose = forward_kinematics(assembly, q)
    coms = link_com_positions(assembly, q)
    n = assembly.dof
    m = np.zeros((n, n), dtype=np.float64)

    for link in assembly.links:
        jac = geometric_jacobian(assembly, q, point_world=coms[link.name],
                                 link_name=link.name)
        jv, jw = jac[:3, :], jac[3:, :]
        mass = link.mass_kg(density_kg_m3)
        inertia = link_world_inertia(assembly, pose, link, density_kg_m3)
        m += mass * (jv.T @ jv) + jw.T @ inertia @ jw

    return 0.5 * (m + m.T)          # symmetrize against round-off


def mass_matrix_derivative(assembly: Assembly, q, density_kg_m3: float,
                           step: float = CHRISTOFFEL_STEP) -> np.ndarray:
    """dM/dq_k as an (n, n, n) array, by central differences."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = q.shape[0]
    out = np.zeros((n, n, n), dtype=np.float64)
    for k in range(n):
        up, down = q.copy(), q.copy()
        up[k] += step
        down[k] -= step
        out[:, :, k] = (mass_matrix(assembly, up, density_kg_m3)
                        - mass_matrix(assembly, down, density_kg_m3)) / (2 * step)
    return out


def coriolis_matrix(assembly: Assembly, q, qd, density_kg_m3: float
                    ) -> np.ndarray:
    """C(q, qd) from the Christoffel symbols of the first kind.

        c_ijk = 1/2 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
        C_ij  = sum_k c_ijk qd_k

    Built this way, M_dot - 2C is skew-symmetric, which is the passivity
    property the tests check. A C assembled some other way can reproduce the
    correct torque while failing that, so it is worth verifying.
    """
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    dm = mass_matrix_derivative(assembly, q, density_kg_m3)
    n = qd.shape[0]
    c = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            c[i, j] = 0.5 * sum(
                (dm[i, j, k] + dm[i, k, j] - dm[j, k, i]) * qd[k]
                for k in range(n))
    return c


def gravity_torques(assembly: Assembly, q, density_kg_m3: float) -> np.ndarray:
    """G(q). Identical to the Phase 10 holding torque, by construction.

    Reusing the statics routine is deliberate: it means the dynamics cannot
    drift away from the static result, and the q_dot = q_ddot = 0 limit is
    exact rather than approximate.
    """
    return joint_torques(assembly, q, density_kg_m3, tip_force_n=None,
                         include_gravity=True)


def friction_torques(assembly: Assembly, qd, frictions=None) -> np.ndarray:
    """Friction torque per joint, zero unless measured parameters are given.

    Real friction needs breakaway torque, a viscous coefficient and gearbox
    efficiency per joint. None of that ships with this repository, and
    inventing it would put fabricated numbers into a torque that a motor gets
    selected from. A caller who has measured them passes a JointFriction per
    joint (see physics.dynamics.friction) and gets a real term; a caller who
    has not gets zero, which is optimistic and is documented as such.
    """
    if frictions is None:
        return np.zeros(assembly.dof, dtype=np.float64)
    from .friction import friction_torques as measured_friction
    return measured_friction(frictions, qd)


def inverse_dynamics(
    assembly: Assembly,
    q,
    qd,
    qdd,
    density_kg_m3: float,
    tip_force_n=None,
    frictions=None,
) -> np.ndarray:
    """Joint torques for a commanded motion, including an external tip force.

    `frictions` is one JointFriction per joint when they have been measured.
    Without it the friction term is zero, which under-states the torque a
    motor must supply.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    qdd = np.asarray(qdd, dtype=np.float64).reshape(-1)

    tau = (mass_matrix(assembly, q, density_kg_m3) @ qdd
           + coriolis_matrix(assembly, q, qd, density_kg_m3) @ qd
           + gravity_torques(assembly, q, density_kg_m3)
           + friction_torques(assembly, qd, frictions))

    if tip_force_n is not None:
        from core.assembly.kinematics import position_jacobian
        force = np.asarray(tip_force_n, dtype=np.float64).reshape(3)
        tau = tau - position_jacobian(assembly, q).T @ force
    return tau


def joint_power_w(torque, qd) -> np.ndarray:
    """Mechanical power per joint, P = tau * omega."""
    return np.asarray(torque, dtype=np.float64) * np.asarray(qd, dtype=np.float64)


def kinetic_energy_j(assembly: Assembly, q, qd, density_kg_m3: float) -> float:
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    return float(0.5 * qd @ mass_matrix(assembly, q, density_kg_m3) @ qd)
