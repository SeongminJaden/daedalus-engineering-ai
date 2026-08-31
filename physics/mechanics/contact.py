"""Hertzian contact and elastic stress concentration.

VALIDITY, before the implementation:

* **Hertz theory is frictionless, elastic, and for contacts small against the
  bodies' radii.** Once the contact patch approaches the size of the curvature
  that formed it, the half-space assumption behind the derivation fails.

* **The peak shear is BELOW the surface, not on it.** For a spherical contact
  it sits at about 0.48 times the contact radius deep, at roughly 0.31 of the
  peak pressure. That is why rolling contact fatigue starts as a subsurface
  crack and only later spalls the surface, and it is why surface hardness
  alone does not prevent pitting.

* **Concentration factors here are ELASTIC (Kt).** They are not what a fatigue
  calculation wants, which is Kf, the FATIGUE concentration factor, always
  smaller because materials are notch-insensitive to a degree that depends on
  the notch radius and the material. Using Kt directly in a fatigue check is
  conservative; using it in a STATIC check on a ductile material is very
  conservative, because local yielding redistributes the peak. Neither is
  simply wrong, but neither is the number the check actually needs.

* **The concentration expressions are published curve fits, not derivations.**
  They reproduce the standard charts to within a few percent over their stated
  range and are clamped outside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# For a spherical Hertzian contact: the depth of the maximum shear as a
# fraction of the contact radius, and its magnitude as a fraction of the peak
# pressure. Standard results for a Poisson ratio near 0.3.
SUBSURFACE_SHEAR_DEPTH_RATIO = 0.48
SUBSURFACE_SHEAR_PRESSURE_RATIO = 0.31


def effective_modulus_pa(e1_pa: float, nu1: float, e2_pa: float,
                         nu2: float) -> float:
    """E* from 1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2."""
    if min(e1_pa, e2_pa) <= 0.0:
        raise ValueError("moduli must be positive")
    return 1.0 / ((1.0 - nu1 ** 2) / e1_pa + (1.0 - nu2 ** 2) / e2_pa)


def effective_radius_m(r1_m: float, r2_m: float | None = None) -> float:
    """1/R = 1/R1 + 1/R2. A flat surface is R2 = None, meaning infinite."""
    if r1_m <= 0.0:
        raise ValueError("the first radius must be positive")
    if r2_m is None:
        return r1_m
    if r2_m <= 0.0:
        raise ValueError("use None for a flat surface, not zero or negative")
    return 1.0 / (1.0 / r1_m + 1.0 / r2_m)


@dataclass(frozen=True)
class HertzContact:
    contact_radius_m: float
    max_pressure_pa: float
    approach_m: float
    max_shear_pa: float
    max_shear_depth_m: float


def sphere_contact(force_n: float, r1_m: float, r2_m: float | None,
                   e1_pa: float, nu1: float, e2_pa: float,
                   nu2: float) -> HertzContact:
    """Point contact between two spheres, or a sphere on a flat.

        a = (3 F R / (4 E*))^(1/3)      p_max = 3F / (2 pi a^2)

    Note that the peak pressure grows only as the cube root of load, so
    doubling the force raises it by 26%. Contact stresses are remarkably
    insensitive to load, which is why they are usually governed by geometry.
    """
    if force_n < 0.0:
        raise ValueError("contact force is a magnitude")
    radius = effective_radius_m(r1_m, r2_m)
    modulus = effective_modulus_pa(e1_pa, nu1, e2_pa, nu2)
    a = (3.0 * force_n * radius / (4.0 * modulus)) ** (1.0 / 3.0)
    if a <= 0.0:
        return HertzContact(0.0, 0.0, 0.0, 0.0, 0.0)
    pressure = 3.0 * force_n / (2.0 * math.pi * a ** 2)
    return HertzContact(
        contact_radius_m=a, max_pressure_pa=pressure,
        approach_m=a ** 2 / radius,
        max_shear_pa=SUBSURFACE_SHEAR_PRESSURE_RATIO * pressure,
        max_shear_depth_m=SUBSURFACE_SHEAR_DEPTH_RATIO * a)


def line_contact_half_width_m(force_per_length_n_m: float, r1_m: float,
                              r2_m: float | None, e1_pa: float, nu1: float,
                              e2_pa: float, nu2: float) -> float:
    """Half-width of a cylindrical line contact: b = sqrt(4 F' R / (pi E*))."""
    if force_per_length_n_m < 0.0:
        raise ValueError("load per length is a magnitude")
    radius = effective_radius_m(r1_m, r2_m)
    modulus = effective_modulus_pa(e1_pa, nu1, e2_pa, nu2)
    return math.sqrt(4.0 * force_per_length_n_m * radius
                     / (math.pi * modulus))


def line_contact_pressure_pa(force_per_length_n_m: float, half_width_m: float
                             ) -> float:
    """p_max = 2 F' / (pi b)."""
    if half_width_m <= 0.0:
        raise ValueError("the contact half-width must be positive")
    return 2.0 * force_per_length_n_m / (math.pi * half_width_m)


def kt_plate_with_hole(hole_diameter_m: float, plate_width_m: float) -> float:
    """Kt for a central hole in a plate under tension, on the NET section.

    The Heywood fit: Kt = 3 - 3.14 x + 3.667 x^2 - 1.527 x^3 with x = d/w. It
    approaches 3.0 as the hole becomes small, which is the classical result for
    a circular hole in an infinite plate, and that limit is checked.
    """
    if hole_diameter_m <= 0.0 or plate_width_m <= 0.0:
        raise ValueError("hole and plate dimensions must be positive")
    ratio = hole_diameter_m / plate_width_m
    if ratio >= 1.0:
        raise ValueError("the hole cannot be as wide as the plate")
    return 3.0 - 3.14 * ratio + 3.667 * ratio ** 2 - 1.527 * ratio ** 3


def kt_shoulder_fillet_bending(fillet_radius_m: float, small_diameter_m: float,
                               large_diameter_m: float) -> float:
    """Kt for a shouldered round bar in bending, as a published power fit.

        Kt = A (r/d)^b,  with A and b depending on D/d

    A curve fit to the standard chart rather than a derivation, accurate to a
    few percent over 0.05 <= r/d <= 0.3 and clamped outside it. The dominant
    behaviour is that Kt rises steeply as the fillet radius shrinks, which is
    why a sharp shoulder is the classic place for a shaft to break.
    """
    if min(fillet_radius_m, small_diameter_m) <= 0.0:
        raise ValueError("radius and diameter must be positive")
    if large_diameter_m <= small_diameter_m:
        raise ValueError("the larger diameter must exceed the smaller one")
    ratio = min(max(fillet_radius_m / small_diameter_m, 0.03), 0.30)
    step = large_diameter_m / small_diameter_m
    # Coefficients for the common step ratios, interpolated in D/d.
    if step <= 1.2:
        a, b = 0.94836, -0.23757
    elif step <= 2.0:
        low = (0.94836, -0.23757)
        high = (0.93836, -0.25759)
        weight = (step - 1.2) / 0.8
        a = low[0] + weight * (high[0] - low[0])
        b = low[1] + weight * (high[1] - low[1])
    else:
        a, b = 0.93836, -0.25759
    return a * ratio ** b
