"""Torsion of non-circular sections, open and closed.

The circular shaft formula tau = Tr/J applies to circles and to nothing else.
A non-circular section WARPS out of plane when twisted, and the stress
distribution is not linear in radius.

VALIDITY, before the implementation:

* **Free warping is assumed.** These expressions are for a section free to warp
  along its length. Restraining the warping, by welding an end plate or
  building the section into a rigid support, adds axial stresses that can
  exceed the torsional ones and are not modelled at all. That restraint is
  common in real structures.

* **The open and closed cases differ by ORDERS OF MAGNITUDE and are not
  interchangeable.** A closed tube carries torque as a shear flow around the
  enclosed area; slitting it lengthwise removes that path entirely and leaves
  only the thin-strip mechanism, which is typically hundreds of times weaker in
  stiffness. Applying the closed formula to an open section, or forgetting that
  a slit or a bolted seam makes a section open, is a large and unsafe error.

* **The thin-walled expressions need thin walls.** Both assume the wall
  thickness is small against the section's overall size; the usual guidance is
  a ratio of ten or more, and neither degrades gracefully outside it.

* **The rectangular coefficients are a published table, interpolated.** They
  are not a formula, and outside the tabulated aspect ratios the values are
  clamped rather than extrapolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Timoshenko coefficients for a solid rectangle, by aspect ratio a/b (long over
# short). alpha sets the peak shear, beta the torsional stiffness. Both tend to
# 1/3 as the section becomes a thin strip, which is the open-section limit.
_RECTANGLE_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.208, 0.1406), (1.5, 0.231, 0.1961), (2.0, 0.246, 0.229),
    (2.5, 0.258, 0.249), (3.0, 0.267, 0.263), (4.0, 0.282, 0.281),
    (5.0, 0.291, 0.291), (10.0, 0.312, 0.312),
)
_THIN_STRIP_LIMIT = 1.0 / 3.0


def rectangle_coefficients(aspect_ratio: float) -> tuple[float, float]:
    """(alpha, beta) interpolated in the published table, clamped at the ends."""
    if aspect_ratio < 1.0:
        raise ValueError("the aspect ratio is the long side over the short one")
    if aspect_ratio >= _RECTANGLE_COEFFICIENTS[-1][0]:
        # Both coefficients approach 1/3 and are within 6% of it by a ratio of
        # ten, so clamping there is closer than extrapolating a curve that
        # flattens.
        return _THIN_STRIP_LIMIT, _THIN_STRIP_LIMIT
    for (low, a_low, b_low), (high, a_high, b_high) in zip(
            _RECTANGLE_COEFFICIENTS, _RECTANGLE_COEFFICIENTS[1:]):
        if low <= aspect_ratio <= high:
            weight = (aspect_ratio - low) / (high - low)
            return (a_low + weight * (a_high - a_low),
                    b_low + weight * (b_high - b_low))
    return _THIN_STRIP_LIMIT, _THIN_STRIP_LIMIT     # pragma: no cover


@dataclass(frozen=True)
class TorsionResult:
    max_shear_stress_pa: float
    torsion_constant_m4: float
    twist_rad_per_m: float
    section_kind: str


def solid_rectangle(torque_nm: float, long_side_m: float, short_side_m: float,
                    shear_modulus_pa: float) -> TorsionResult:
    """tau_max = T / (alpha a b^2), J = beta a b^3.

    The peak shear is at the middle of the LONG side, not at a corner. The
    corners of a twisted rectangle carry no shear at all, which is the opposite
    of the intuition carried over from circular shafts.
    """
    if short_side_m > long_side_m:
        long_side_m, short_side_m = short_side_m, long_side_m
    if short_side_m <= 0.0:
        raise ValueError("section dimensions must be positive")
    alpha, beta = rectangle_coefficients(long_side_m / short_side_m)
    constant = beta * long_side_m * short_side_m ** 3
    return TorsionResult(
        max_shear_stress_pa=torque_nm / (alpha * long_side_m
                                         * short_side_m ** 2),
        torsion_constant_m4=constant,
        twist_rad_per_m=torque_nm / (shear_modulus_pa * constant),
        section_kind="solid_rectangle")


def thin_open_section(torque_nm: float, segments: "list[tuple[float, float]]",
                      shear_modulus_pa: float) -> TorsionResult:
    """An open section as a sum of thin strips: J = sum(b t^3)/3.

    `segments` are (length, thickness) pairs. A channel, an angle, an I beam or
    a slit tube are all this. The peak shear occurs in the THICKEST strip,
    since tau = 3T t_max / J.
    """
    if not segments:
        raise ValueError("an open section needs at least one strip")
    constant = sum(length * thickness ** 3
                   for length, thickness in segments) / 3.0
    if constant <= 0.0:
        raise ValueError("strip dimensions must be positive")
    thickest = max(thickness for _, thickness in segments)
    return TorsionResult(
        max_shear_stress_pa=torque_nm * thickest / constant,
        torsion_constant_m4=constant,
        twist_rad_per_m=torque_nm / (shear_modulus_pa * constant),
        section_kind="thin_open")


def thin_closed_section(torque_nm: float, enclosed_area_m2: float,
                        wall_thickness_m: float, perimeter_m: float,
                        shear_modulus_pa: float) -> TorsionResult:
    """Bredt's formulas for a single closed cell.

        tau = T / (2 A_m t)        J = 4 A_m^2 t / s

    `enclosed_area_m2` is the area enclosed by the MIDLINE of the wall, not the
    outer area. The shear flow q = T/(2 A_m) is constant around the cell, so
    the thinnest wall carries the highest stress.
    """
    if min(enclosed_area_m2, wall_thickness_m, perimeter_m) <= 0.0:
        raise ValueError("area, thickness and perimeter must be positive")
    constant = 4.0 * enclosed_area_m2 ** 2 * wall_thickness_m / perimeter_m
    return TorsionResult(
        max_shear_stress_pa=torque_nm / (2.0 * enclosed_area_m2
                                         * wall_thickness_m),
        torsion_constant_m4=constant,
        twist_rad_per_m=torque_nm / (shear_modulus_pa * constant),
        section_kind="thin_closed")
