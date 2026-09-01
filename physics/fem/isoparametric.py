"""physics.fem.isoparametric - trilinear hexahedra of arbitrary shape.

The structured element in `element.py` shares ONE stiffness matrix across the
whole mesh, which is exact and cheap, and is true only while every element is
the same axis aligned box. This module drops that assumption: each element
carries its own eight node positions and gets its own Ke.

What that costs is stated plainly, because it is easy to present a general
element as strictly better than a special one:

* The Jacobian is no longer constant inside the element, so it is evaluated at
  every integration point rather than once.
* Two point Gauss is NO LONGER EXACT. For a box the integrand of B^T C B det J
  is a polynomial and 2x2x2 integrates it exactly. For a distorted hex, B
  contains J^-1 whose entries are rational in the natural coordinates, so no
  fixed Gauss rule is exact. On distorted elements the quadrature is an
  approximation.
* The incompatible modes need the Taylor correction. Wilson's bubbles are
  derived for a rectangle; used unchanged on a distorted hex they fail the
  patch test, meaning the element no longer reproduces constant strain and
  converges to the wrong answer rather than merely a coarse one.

The structured path is not replaced. It remains faster and remains the default
whenever the mesh is a grid.

See docs/scoping_isoparametric.md for the validity domains, which were written
before this file existed.
"""

from __future__ import annotations

import numpy as np

from physics.fem.element import (
    NODE_SIGNS,
    elasticity_matrix,
    gauss_points,
    shape_derivatives,
    strain_displacement,
)

#: A Jacobian determinant at or below this is a folded element, not a thin one.
#: Expressed relative to the element's own scale so it does not become a
#: length unit test in disguise: a part modelled in metres has determinants a
#: billion times smaller than the same part in millimetres.
MIN_RELATIVE_JACOBIAN = 1e-12


class DegenerateElement(ValueError):
    """A hex whose Jacobian is not positive throughout.

    Raised rather than clamped. A folded element still produces numbers, and
    those numbers look like an answer.
    """


def box_nodes(dx: float, dy: float, dz: float) -> np.ndarray:
    """The eight corners of an axis aligned box, in this module's ordering.

    Provided so the general path can be compared against the structured one on
    geometry where they must agree exactly.
    """
    return (NODE_SIGNS + 1.0) * 0.5 * np.array([dx, dy, dz], dtype=np.float64)


def jacobian(nodes: np.ndarray, xi: float, eta: float, zeta: float):
    """(J, det J) at one natural coordinate.

    J[a][b] is d x_a / d xi_b.
    """
    dn = shape_derivatives(xi, eta, zeta)          # (8, 3)
    j = np.asarray(nodes, dtype=np.float64).T @ dn  # (3, 3)
    return j, float(np.linalg.det(j))


def _checked_jacobian(nodes, xi, eta, zeta, scale: float):
    j, det = jacobian(nodes, xi, eta, zeta)
    if det <= MIN_RELATIVE_JACOBIAN * scale:
        raise DegenerateElement(
            f"Jacobian determinant {det:.6g} at natural coordinate "
            f"({xi:.4f}, {eta:.4f}, {zeta:.4f}) is not positive relative to "
            f"the element scale {scale:.6g}. The hex is folded or inverted, "
            f"and no stiffness can be formed from it. Node ordering is a "
            f"common cause: this module expects the ordering drawn in "
            f"physics/fem/element.py")
    return j, det


def _element_scale(nodes: np.ndarray) -> float:
    """A volume-like number for the element, used to judge det J relatively."""
    span = np.ptp(np.asarray(nodes, dtype=np.float64), axis=0)
    return float(max(np.prod(span), np.finfo(float).tiny))


def shape_gradients(nodes: np.ndarray, xi: float, eta: float, zeta: float):
    """(dN/dx, det J), the physical gradients of the eight shape functions."""
    scale = _element_scale(nodes)
    j, det = _checked_jacobian(nodes, xi, eta, zeta, scale)
    return shape_derivatives(xi, eta, zeta) @ np.linalg.inv(j), det


def _incompatible_b(xi, eta, zeta, inv_j0):
    """B (6 x 9) for the bubble modes, with gradients taken from the CENTRE.

    Using the centre Jacobian, not the local one, is the Taylor correction.
    Without it the element fails the patch test on a distorted mesh.
    """
    grads = np.zeros((3, 3), dtype=np.float64)
    grads[0, 0] = -2.0 * xi
    grads[1, 1] = -2.0 * eta
    grads[2, 2] = -2.0 * zeta
    grads = grads @ inv_j0

    b = np.zeros((6, 9), dtype=np.float64)
    for mode in range(3):
        gx, gy, gz = grads[mode]
        for comp in range(3):
            col = mode * 3 + comp
            if comp == 0:
                b[0, col], b[3, col], b[5, col] = gx, gy, gz
            elif comp == 1:
                b[1, col], b[3, col], b[4, col] = gy, gx, gz
            else:
                b[2, col], b[4, col], b[5, col] = gz, gy, gx
    return b


def element_stiffness_from_c(nodes: np.ndarray, stiffness: np.ndarray,
                             incompatible_modes: bool = True) -> np.ndarray:
    """The 24x24 Ke for one hex of arbitrary shape.

    Raises DegenerateElement if the Jacobian is not positive at every
    integration point.
    """
    nodes = np.ascontiguousarray(np.asarray(nodes, dtype=np.float64))
    if nodes.shape != (8, 3):
        raise ValueError(f"nodes must be (8, 3), got {nodes.shape}")
    c = np.ascontiguousarray(np.asarray(stiffness, dtype=np.float64))
    if c.shape != (6, 6):
        raise ValueError(f"stiffness must be 6x6, got {c.shape}")

    scale = _element_scale(nodes)
    j0, det0 = _checked_jacobian(nodes, 0.0, 0.0, 0.0, scale)
    inv_j0 = np.linalg.inv(j0)

    ke = np.zeros((24, 24), dtype=np.float64)
    k_ua = np.zeros((24, 9), dtype=np.float64)
    k_aa = np.zeros((9, 9), dtype=np.float64)

    for xi, eta, zeta, weight in gauss_points():
        j, det = _checked_jacobian(nodes, xi, eta, zeta, scale)
        bu = strain_displacement(
            shape_derivatives(xi, eta, zeta) @ np.linalg.inv(j))
        w = weight * det
        ke += w * (bu.T @ c @ bu)
        if incompatible_modes:
            # The det0/det factor is the second half of the Taylor correction.
            ba = _incompatible_b(xi, eta, zeta, inv_j0) * (det0 / det)
            k_ua += w * (bu.T @ c @ ba)
            k_aa += w * (ba.T @ c @ ba)

    if incompatible_modes:
        ke = ke - k_ua @ np.linalg.solve(k_aa, k_ua.T)
    return ke


def element_stiffness(nodes: np.ndarray, youngs_modulus: float,
                      poisson_ratio: float,
                      incompatible_modes: bool = True) -> np.ndarray:
    """Isotropic wrapper, so the two paths cannot drift apart."""
    return element_stiffness_from_c(
        nodes, elasticity_matrix(youngs_modulus, poisson_ratio),
        incompatible_modes=incompatible_modes)
