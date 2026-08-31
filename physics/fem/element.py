"""physics.fem.element - 8-node trilinear hexahedral element.

Isotropic linear elasticity, small strain. The mesh is a structured axis-aligned
grid, so every element has the same shape and **one element stiffness matrix is
shared by all of them**. That is what makes a matrix-free solve cheap: the
24x24 Ke is built once on the CPU and reused by every element on the GPU.

Node ordering (standard hex, natural coordinates xi,eta,zeta in [-1,1]):

        7---------6          zeta
       /|        /|           |
      4---------5 |           |  eta
      | |       | |           | /
      | 3-------|-2           |/
      |/        |/            +----- xi
      0---------1

DOF ordering within an element: node0_x, node0_y, node0_z, node1_x, ...
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Corner signs in (xi, eta, zeta) for the node ordering drawn above.
NODE_SIGNS = np.array([
    [-1, -1, -1],
    [+1, -1, -1],
    [+1, +1, -1],
    [-1, +1, -1],
    [-1, -1, +1],
    [+1, -1, +1],
    [+1, +1, +1],
    [-1, +1, +1],
], dtype=np.float64)

# 2-point Gauss-Legendre, exact for the trilinear element's stiffness.
GAUSS_2PT = np.array([-1.0 / np.sqrt(3.0), +1.0 / np.sqrt(3.0)])
GAUSS_2PT_WEIGHT = 1.0


def shape_derivatives(xi: float, eta: float, zeta: float) -> np.ndarray:
    """dN/d(xi,eta,zeta) for the 8 nodes. Shape (8, 3)."""
    s = NODE_SIGNS
    natural = np.array([xi, eta, zeta], dtype=np.float64)
    # N_i = 1/8 (1 + s_xi*xi)(1 + s_eta*eta)(1 + s_zeta*zeta)
    terms = 1.0 + s * natural            # (8, 3)
    out = np.empty((8, 3), dtype=np.float64)
    for d in range(3):
        others = [k for k in range(3) if k != d]
        out[:, d] = 0.125 * s[:, d] * terms[:, others[0]] * terms[:, others[1]]
    return out


def shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    s = NODE_SIGNS
    natural = np.array([xi, eta, zeta], dtype=np.float64)
    return 0.125 * np.prod(1.0 + s * natural, axis=1)


def elasticity_matrix(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    """Isotropic 3D constitutive matrix D (6x6), Voigt order
    [xx, yy, zz, xy, yz, zx]."""
    E, nu = float(youngs_modulus), float(poisson_ratio)
    if not 0.0 <= nu < 0.5:
        raise ValueError(f"poisson_ratio must be in [0, 0.5), got {nu}")
    if E <= 0.0:
        raise ValueError(f"youngs_modulus must be > 0, got {E}")

    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d = np.zeros((6, 6), dtype=np.float64)
    d[0, 0] = d[1, 1] = d[2, 2] = factor * (1.0 - nu)
    d[0, 1] = d[0, 2] = d[1, 0] = d[1, 2] = d[2, 0] = d[2, 1] = factor * nu
    shear = E / (2.0 * (1.0 + nu))
    d[3, 3] = d[4, 4] = d[5, 5] = shear
    return d


def strain_displacement(dn_dx: np.ndarray) -> np.ndarray:
    """B matrix (6 x 24) from dN/dx. Voigt order [xx, yy, zz, xy, yz, zx]."""
    b = np.zeros((6, 24), dtype=np.float64)
    for i in range(8):
        nx, ny, nz = dn_dx[i]
        c = 3 * i
        b[0, c + 0] = nx
        b[1, c + 1] = ny
        b[2, c + 2] = nz
        b[3, c + 0] = ny
        b[3, c + 1] = nx
        b[4, c + 1] = nz
        b[4, c + 2] = ny
        b[5, c + 0] = nz
        b[5, c + 2] = nx
    return b


def gauss_points():
    """The 8 integration points: (xi, eta, zeta, weight)."""
    for xi in GAUSS_2PT:
        for eta in GAUSS_2PT:
            for zeta in GAUSS_2PT:
                yield xi, eta, zeta, GAUSS_2PT_WEIGHT ** 3


def element_b_matrices(dx: float, dy: float, dz: float):
    """B and detJ at every Gauss point of an axis-aligned box element.

    For an axis-aligned box the Jacobian is constant and diagonal, so dN/dx is
    just dN/dxi scaled - no per-point inversion needed.
    """
    if min(dx, dy, dz) <= 0.0:
        raise ValueError(f"element sizes must be > 0, got {(dx, dy, dz)}")
    scale = np.array([2.0 / dx, 2.0 / dy, 2.0 / dz], dtype=np.float64)
    det_j = dx * dy * dz / 8.0

    out = []
    for xi, eta, zeta, weight in gauss_points():
        dn_dx = shape_derivatives(xi, eta, zeta) * scale
        out.append((strain_displacement(dn_dx), det_j, weight))
    return out


def incompatible_mode_b(xi: float, eta: float, zeta: float,
                        dx: float, dy: float, dz: float) -> np.ndarray:
    """B matrix (6 x 9) for the incompatible bubble modes.

    The three bubble shapes (1 - xi^2), (1 - eta^2), (1 - zeta^2) are added for
    each displacement component, giving 9 internal DOFs that are condensed out
    at element level.
    """
    scale = np.array([2.0 / dx, 2.0 / dy, 2.0 / dz], dtype=np.float64)
    # d/d(natural) of each bubble; each bubble varies in one direction only.
    grads = np.zeros((3, 3), dtype=np.float64)
    grads[0, 0] = -2.0 * xi
    grads[1, 1] = -2.0 * eta
    grads[2, 2] = -2.0 * zeta
    grads = grads * scale                       # -> d/dx

    b = np.zeros((6, 9), dtype=np.float64)
    for mode in range(3):
        gx, gy, gz = grads[mode]
        for comp in range(3):
            col = mode * 3 + comp
            if comp == 0:
                b[0, col] = gx
                b[3, col] = gy
                b[5, col] = gz
            elif comp == 1:
                b[1, col] = gy
                b[3, col] = gx
                b[4, col] = gz
            else:
                b[2, col] = gz
                b[4, col] = gy
                b[5, col] = gx
    return b


@lru_cache(maxsize=32)
def element_stiffness(dx: float, dy: float, dz: float,
                      youngs_modulus: float, poisson_ratio: float,
                      incompatible_modes: bool = True) -> np.ndarray:
    """Ke (24x24) for one axis-aligned box element.

    Cached: a structured grid has one element shape, so this is computed once
    per mesh no matter how many elements there are.

    **Incompatible modes are on by default, and they are not optional in
    practice.** A fully-integrated trilinear hex shear-locks in bending: with
    4 elements through the depth of a slender beam it returned only 71% of the
    Euler-Bernoulli tip deflection, i.e. it was ~40% too stiff, and refining
    the mesh fixed it only slowly. Adding the three bubble modes per direction
    and condensing them out (Wilson's incompatible-mode element) removes the
    spurious shear energy while keeping a plain 24x24 element matrix - so the
    matrix-free solve is unchanged.
    """
    d = elasticity_matrix(youngs_modulus, poisson_ratio)
    ke = np.zeros((24, 24), dtype=np.float64)

    if not incompatible_modes:
        for b, det_j, weight in element_b_matrices(dx, dy, dz):
            ke += weight * det_j * (b.T @ d @ b)
        return ke

    k_ua = np.zeros((24, 9), dtype=np.float64)
    k_aa = np.zeros((9, 9), dtype=np.float64)
    scale = np.array([2.0 / dx, 2.0 / dy, 2.0 / dz], dtype=np.float64)
    for xi, eta, zeta, weight in gauss_points():
        det_j = dx * dy * dz / 8.0
        bu = strain_displacement(shape_derivatives(xi, eta, zeta) * scale)
        ba = incompatible_mode_b(xi, eta, zeta, dx, dy, dz)
        w = weight * det_j
        ke += w * (bu.T @ d @ bu)
        k_ua += w * (bu.T @ d @ ba)
        k_aa += w * (ba.T @ d @ ba)

    # Static condensation: Ke <- Kuu - Kua Kaa^-1 Kau
    return ke - k_ua @ np.linalg.solve(k_aa, k_ua.T)


@lru_cache(maxsize=32)
def element_stress_operator(dx: float, dy: float, dz: float,
                            youngs_modulus: float,
                            poisson_ratio: float,
                            incompatible_modes: bool = True) -> np.ndarray:
    """D @ B at the element centre: maps element displacements to stress.

    The centre is used deliberately - it is the element's superconvergent point
    for this element type, and it keeps one stress value per element.

    With incompatible modes the internal DOFs contribute too. At the centre
    (xi = eta = zeta = 0) the bubble gradients vanish, so the enhanced strain
    is zero there and the compatible part is exactly the stress. That is a
    second reason to sample at the centre.
    """
    d = elasticity_matrix(youngs_modulus, poisson_ratio)
    scale = np.array([2.0 / dx, 2.0 / dy, 2.0 / dz], dtype=np.float64)
    dn_dx = shape_derivatives(0.0, 0.0, 0.0) * scale
    return d @ strain_displacement(dn_dx)


def von_mises(stress: np.ndarray) -> np.ndarray:
    """von Mises stress from Voigt stress [xx, yy, zz, xy, yz, zx]."""
    s = np.atleast_2d(np.asarray(stress, dtype=np.float64))
    sxx, syy, szz, sxy, syz, szx = (s[:, i] for i in range(6))
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    )
