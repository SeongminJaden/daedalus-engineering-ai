"""Stress transformation, principal stresses and the maximum shear.

VALIDITY, before the implementation:

* **The two-dimensional formulas are PLANE STRESS.** They give the principal
  stresses in the plane and the largest shear WITHIN it.

* **The in-plane maximum shear is not always the absolute maximum shear, and
  this is the trap.** A plane stress state has a third principal stress of
  zero. When the two in-plane principals have the SAME SIGN, the absolute
  maximum shear involves that zero and is (sigma_max - 0)/2, which is larger
  than the in-plane value (sigma_1 - sigma_2)/2. Using the in-plane number for
  a biaxial tension state under-predicts the shear, by up to a factor of two in
  the equibiaxial case. Both are returned and named.

* **Principal directions are ambiguous when the state is hydrostatic in-plane.**
  If sigma_x equals sigma_y and there is no shear, every direction is
  principal and the reported angle is arbitrary rather than wrong.

* **This is a stress state, not a failure criterion.** Which combination breaks
  a material is a separate question with several answers, and von Mises and
  maximum shear disagree by up to 15%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrincipalStress2D:
    """Principal stresses of a plane stress state, in descending order."""

    sigma_1_pa: float
    sigma_2_pa: float
    principal_angle_deg: float
    in_plane_max_shear_pa: float
    absolute_max_shear_pa: float

    @property
    def sigma_3_pa(self) -> float:
        """Always zero: that is what plane stress means."""
        return 0.0

    @property
    def absolute_shear_is_out_of_plane(self) -> bool:
        """True when the zero third principal governs the shear.

        Which happens whenever both in-plane principals share a sign.
        """
        return self.absolute_max_shear_pa > self.in_plane_max_shear_pa * (
            1.0 + 1e-12)

    @property
    def von_mises_pa(self) -> float:
        return math.sqrt(self.sigma_1_pa ** 2
                         - self.sigma_1_pa * self.sigma_2_pa
                         + self.sigma_2_pa ** 2)


def transform_stress_2d(sigma_x_pa: float, sigma_y_pa: float,
                        tau_xy_pa: float, angle_deg: float
                        ) -> tuple[float, float, float]:
    """Rotate a plane stress state by an angle, giving (sx', sy', txy')."""
    theta = math.radians(angle_deg)
    c, s = math.cos(2.0 * theta), math.sin(2.0 * theta)
    average = 0.5 * (sigma_x_pa + sigma_y_pa)
    half_difference = 0.5 * (sigma_x_pa - sigma_y_pa)
    return (average + half_difference * c + tau_xy_pa * s,
            average - half_difference * c - tau_xy_pa * s,
            -half_difference * s + tau_xy_pa * c)


def principal_stress_2d(sigma_x_pa: float, sigma_y_pa: float,
                        tau_xy_pa: float) -> PrincipalStress2D:
    """Principal stresses and both maximum shears.

        sigma_1,2 = (sx + sy)/2 +- sqrt(((sx - sy)/2)^2 + txy^2)

    The absolute maximum shear is half the spread of ALL THREE principals,
    including the zero one that plane stress carries.
    """
    average = 0.5 * (sigma_x_pa + sigma_y_pa)
    radius = math.hypot(0.5 * (sigma_x_pa - sigma_y_pa), tau_xy_pa)
    s1, s2 = average + radius, average - radius
    ordered = sorted([s1, s2, 0.0], reverse=True)
    return PrincipalStress2D(
        sigma_1_pa=s1, sigma_2_pa=s2,
        principal_angle_deg=math.degrees(
            0.5 * math.atan2(2.0 * tau_xy_pa, sigma_x_pa - sigma_y_pa)),
        in_plane_max_shear_pa=radius,
        absolute_max_shear_pa=0.5 * (ordered[0] - ordered[2]))


def principal_stress_3d(tensor: np.ndarray) -> np.ndarray:
    """Principal stresses of a full 3x3 stress tensor, descending.

    The eigenvalues of a symmetric tensor, which is what principal stresses
    are. `eigvalsh` is used rather than `eigvals` because it exploits the
    symmetry and returns real values rather than complex ones with vanishing
    imaginary parts.
    """
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (3, 3):
        raise ValueError("a stress tensor is 3 by 3")
    if not np.allclose(tensor, tensor.T, rtol=1e-10, atol=1e-10):
        raise ValueError(
            "the stress tensor is not symmetric, which no equilibrium state is")
    return np.sort(np.linalg.eigvalsh(tensor))[::-1]


def von_mises_3d(tensor: np.ndarray) -> float:
    """Von Mises equivalent stress from the principal values."""
    s1, s2, s3 = principal_stress_3d(tensor)
    return math.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))
