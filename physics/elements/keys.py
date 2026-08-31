"""Parallel keys and splines: transmitting torque between a shaft and a hub.

VALIDITY, before the implementation:

* **Uniform load along the key is assumed and it is optimistic.** A real key
  carries most of its load near the ends of its engagement, because the shaft
  twists relative to the hub along the length. Beyond roughly 1.5 shaft
  diameters a longer key adds very little capacity, and this model would keep
  crediting it linearly. Length is capped for that reason rather than trusted.

* **Half the key height bears on each side.** The key sits half in the shaft
  keyway and half in the hub, so the crushing area is h/2 times the length, not
  h times it. Using the full height doubles the apparent bearing capacity.

* **The keyway's effect on the SHAFT is not included here.** Cutting a keyway
  removes material and adds a stress concentration, and the shaft check is
  where that belongs. A key that is strong enough in a shaft that breaks
  through its own keyway is not a solution.

* **Standard key dimensions come from the published DIN 6885 / ISO 773 series.**
  They are dimensional standards, not vendor parts.

* **Spline load sharing is ASSUMED.** Indexing and spacing errors mean not all
  teeth carry equally, and the fraction that does depends on manufacturing
  accuracy that nothing here knows. The conventional 0.75 is used and stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# DIN 6885 / ISO 773 parallel key sections, as
# (shaft diameter upper bound in mm, width mm, height mm). A published
# dimensional standard.
_KEY_SECTIONS: tuple[tuple[float, float, float], ...] = (
    (8.0, 2.0, 2.0), (10.0, 3.0, 3.0), (12.0, 4.0, 4.0), (17.0, 5.0, 5.0),
    (22.0, 6.0, 6.0), (30.0, 8.0, 7.0), (38.0, 10.0, 8.0), (44.0, 12.0, 8.0),
    (50.0, 14.0, 9.0), (58.0, 16.0, 10.0), (65.0, 18.0, 11.0),
    (75.0, 20.0, 12.0), (85.0, 22.0, 14.0), (95.0, 25.0, 14.0),
)

# Beyond this multiple of the shaft diameter, extra key length stops adding
# capacity because the load concentrates at the ends. [ASSUMED] as the usual
# design guidance rather than derived.
MAX_EFFECTIVE_LENGTH_RATIO = 1.5

# Fraction of spline teeth actually sharing the load. [ASSUMED], conventional.
SPLINE_LOAD_SHARING = 0.75


def standard_key_section(shaft_diameter_m: float) -> tuple[float, float]:
    """(width, height) in metres for this shaft, from the standard series."""
    diameter_mm = shaft_diameter_m * 1000.0
    if diameter_mm <= 6.0:
        raise ValueError(
            f"the standard key series starts above 6 mm; {diameter_mm:.1f} mm "
            f"needs a different retention method")
    for upper, width, height in _KEY_SECTIONS:
        if diameter_mm <= upper:
            return width / 1000.0, height / 1000.0
    raise ValueError(
        f"{diameter_mm:.1f} mm is beyond the tabulated series, which ends at "
        f"{_KEY_SECTIONS[-1][0]:.0f} mm")


def effective_length_m(length_m: float, shaft_diameter_m: float) -> float:
    """Length credited with carrying load, capped at the ratio above."""
    return min(length_m, MAX_EFFECTIVE_LENGTH_RATIO * shaft_diameter_m)


@dataclass(frozen=True)
class KeyResult:
    """Shear and bearing on a parallel key, and which governs."""

    width_m: float
    height_m: float
    length_m: float
    effective_length_m: float
    tangential_force_n: float
    shear_stress_pa: float
    bearing_stress_pa: float
    shear_safety_factor: float
    bearing_safety_factor: float
    governing_mode: str
    length_was_capped: bool

    @property
    def safety_factor(self) -> float:
        return min(self.shear_safety_factor, self.bearing_safety_factor)

    @property
    def passes(self) -> bool:
        return self.safety_factor >= 1.0


def analyze_key(shaft_diameter_m: float, length_m: float, torque_nm: float,
                allowable_shear_pa: float, allowable_bearing_pa: float,
                width_m: float | None = None,
                height_m: float | None = None) -> KeyResult:
    """Check a parallel key in shear and in bearing.

        F = 2 T / d,  tau = F / (b L),  sigma = F / ((h/2) L)
    """
    if shaft_diameter_m <= 0.0 or length_m <= 0.0 or torque_nm < 0.0:
        raise ValueError("diameter and length must be positive, torque non-negative")
    if width_m is None or height_m is None:
        width_m, height_m = standard_key_section(shaft_diameter_m)

    effective = effective_length_m(length_m, shaft_diameter_m)
    force = 2.0 * torque_nm / shaft_diameter_m
    shear = force / (width_m * effective)
    bearing = force / (0.5 * height_m * effective)
    shear_safety = (math.inf if shear <= 0.0 else allowable_shear_pa / shear)
    bearing_safety = (math.inf if bearing <= 0.0
                      else allowable_bearing_pa / bearing)
    return KeyResult(
        width_m=width_m, height_m=height_m, length_m=length_m,
        effective_length_m=effective, tangential_force_n=force,
        shear_stress_pa=shear, bearing_stress_pa=bearing,
        shear_safety_factor=shear_safety, bearing_safety_factor=bearing_safety,
        governing_mode=("shear" if shear_safety < bearing_safety
                        else "bearing"),
        length_was_capped=effective < length_m)


def spline_torque_capacity_nm(pitch_diameter_m: float, tooth_count: int,
                              tooth_height_m: float, length_m: float,
                              allowable_bearing_pa: float,
                              load_sharing: float = SPLINE_LOAD_SHARING
                              ) -> float:
    """T = p N phi h L d / 2, the bearing capacity of the tooth flanks.

    A spline spreads the same torque over many teeth, which is why it carries
    far more than one key of the same diameter. `load_sharing` is the fraction
    of teeth actually in contact and is ASSUMED, not computed.
    """
    if min(pitch_diameter_m, tooth_height_m, length_m) <= 0.0 or tooth_count < 1:
        raise ValueError("spline geometry must be positive")
    if not 0.0 < load_sharing <= 1.0:
        raise ValueError("load sharing is a fraction in (0, 1]")
    return (allowable_bearing_pa * tooth_count * load_sharing * tooth_height_m
            * length_m * pitch_diameter_m / 2.0)
