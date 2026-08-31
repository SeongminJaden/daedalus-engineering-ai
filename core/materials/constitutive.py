"""core.materials.constitutive: material to 6x6 stiffness matrix.

One path for every material class. An isotropic entry expands to the nine
orthotropic constants first (exactly, by the definition of isotropy) and then
goes through the same compliance inversion as a composite. That is deliberate:
if isotropy took a separate shortcut, the two paths could quietly disagree, and
the isotropic results from Phase 7 would stop being a regression check on the
general code.

Voigt order is [xx, yy, zz, xy, yz, zx], matching physics.fem.element.

A stiffness matrix that is not symmetric positive definite describes a material
that can release energy from nothing. Both properties are asserted, and a
material that fails is rejected rather than quietly producing a solver that
diverges in a way nobody traces back to the data file.
"""

from __future__ import annotations

import numpy as np

# Voigt index of each shear pair, matching the element's B matrix ordering.
SHEAR_INDEX = {"xy": 3, "yz": 4, "zx": 5}


def compliance_matrix(constants: dict[str, float],
                      reciprocal: dict[str, float]) -> np.ndarray:
    """Orthotropic compliance S (6x6), symmetric by construction.

        e_xx = s_xx/E1 - nu21 s_yy/E2 - nu31 s_zz/E3
        ...
        gamma_xy = tau_xy / G12   (and yz -> G23, zx -> G13)
    """
    e1, e2, e3 = constants["E1"], constants["E2"], constants["E3"]
    nu12, nu13, nu23 = constants["nu12"], constants["nu13"], constants["nu23"]
    nu21, nu31, nu32 = (reciprocal["nu21"], reciprocal["nu31"],
                        reciprocal["nu32"])

    s = np.zeros((6, 6), dtype=np.float64)
    s[0, 0] = 1.0 / e1
    s[1, 1] = 1.0 / e2
    s[2, 2] = 1.0 / e3
    s[0, 1] = s[1, 0] = -nu21 / e2      # == -nu12 / e1 by reciprocity
    s[0, 2] = s[2, 0] = -nu31 / e3      # == -nu13 / e1
    s[1, 2] = s[2, 1] = -nu32 / e3      # == -nu23 / e2

    s[SHEAR_INDEX["xy"], SHEAR_INDEX["xy"]] = 1.0 / constants["G12"]
    s[SHEAR_INDEX["yz"], SHEAR_INDEX["yz"]] = 1.0 / constants["G23"]
    s[SHEAR_INDEX["zx"], SHEAR_INDEX["zx"]] = 1.0 / constants["G13"]
    return s


def check_stiffness(c: np.ndarray, label: str = "material") -> np.ndarray:
    """Reject a stiffness matrix that is not symmetric positive definite."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (6, 6):
        raise ValueError(f"{label}: stiffness must be 6x6, got {c.shape}")
    asymmetry = np.abs(c - c.T).max() / max(np.abs(c).max(), 1e-30)
    if asymmetry > 1e-10:
        raise ValueError(
            f"{label}: stiffness matrix is not symmetric (relative asymmetry "
            f"{asymmetry:.3e}); the elastic constants violate reciprocity")
    eigenvalues = np.linalg.eigvalsh(0.5 * (c + c.T))
    if eigenvalues.min() <= 0.0:
        raise ValueError(
            f"{label}: stiffness matrix is not positive definite (min "
            f"eigenvalue {eigenvalues.min():.4g}); these constants describe a "
            "material that would release energy under deformation, which is "
            "physically impossible")
    return c


def stiffness_from_constants(constants: dict[str, float],
                             reciprocal: dict[str, float],
                             label: str = "material") -> np.ndarray:
    s = compliance_matrix(constants, reciprocal)
    try:
        c = np.linalg.inv(s)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{label}: compliance matrix is singular") from exc
    return check_stiffness(0.5 * (c + c.T), label)


def stiffness_matrix(material) -> np.ndarray:
    """6x6 stiffness for a MaterialSpec, whatever its class."""
    return stiffness_from_constants(
        material.elastic_constants(), material.reciprocal_poisson(),
        label=material.id)


def isotropic_stiffness(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    """Convenience for callers that only have E and nu.

    Routed through the same compliance inversion as everything else so it is
    guaranteed to agree with a material of the same constants.
    """
    if youngs_modulus <= 0.0:
        raise ValueError(f"youngs_modulus must be > 0, got {youngs_modulus}")
    if not 0.0 < poisson_ratio < 0.5:
        raise ValueError(f"poisson_ratio must be in (0, 0.5), got {poisson_ratio}")
    g = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    constants = {"E1": youngs_modulus, "E2": youngs_modulus, "E3": youngs_modulus,
                 "G12": g, "G13": g, "G23": g,
                 "nu12": poisson_ratio, "nu13": poisson_ratio,
                 "nu23": poisson_ratio}
    reciprocal = {"nu21": poisson_ratio, "nu31": poisson_ratio,
                  "nu32": poisson_ratio}
    return stiffness_from_constants(constants, reciprocal, "isotropic")
