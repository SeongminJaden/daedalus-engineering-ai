"""Fillet weld capacity on the throat.

VALIDITY, before the implementation:

* **Nominal throat stress, with no concentration.** The stress is taken as
  uniform over the throat area. A real weld has a notch at the root and another
  at the toe, and those are where fatigue cracks start.

* **Static only, and weld fatigue is much worse than parent metal fatigue.**
  A welded joint's endurance strength is a fraction of the plate's, largely
  independent of the steel's own strength, because the geometry dominates.
  Applying a parent-metal fatigue check to a weld overstates its life
  substantially. This module does not do fatigue at all.

* **Full penetration to the throat is assumed.** Lack of fusion at the root is
  the common real defect and it removes throat area that this calculation
  believes is there.

* **Residual stress from welding is ignored**, and it is not small: a weld
  cools into tension near yield.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A 45 degree fillet's throat is its leg divided by root two.
THROAT_FACTOR = 1.0 / math.sqrt(2.0)


def throat_thickness_m(leg_m: float) -> float:
    if leg_m <= 0.0:
        raise ValueError("the weld leg must be positive")
    return THROAT_FACTOR * leg_m


@dataclass(frozen=True)
class WeldResult:
    leg_m: float
    throat_m: float
    length_m: float
    throat_area_m2: float
    stress_pa: float
    safety_factor: float

    @property
    def passes(self) -> bool:
        return self.safety_factor >= 1.0


def analyze_fillet_weld(force_n: float, leg_m: float, length_m: float,
                        allowable_pa: float) -> WeldResult:
    """Nominal stress on the throat: tau = F / (a L)."""
    if length_m <= 0.0:
        raise ValueError("weld length must be positive")
    if allowable_pa <= 0.0:
        raise ValueError("the allowable must be positive")
    throat = throat_thickness_m(leg_m)
    area = throat * length_m
    stress = abs(force_n) / area
    return WeldResult(
        leg_m=leg_m, throat_m=throat, length_m=length_m, throat_area_m2=area,
        stress_pa=stress,
        safety_factor=math.inf if stress <= 0.0 else allowable_pa / stress)
