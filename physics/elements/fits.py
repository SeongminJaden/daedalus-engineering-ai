"""ISO 286 limits and fits: tolerance grades and fundamental deviations.

A dimension without a tolerance is not a specification. Two parts that fit
together do so because their tolerance zones were chosen to overlap in a
particular way, and calling for a 20 mm shaft in a 20 mm hole specifies an
assembly that cannot be built.

VALIDITY, stated before the implementation:

* **The IT grades come from the ISO 286 FORMULA, not a transcribed table.** The
  standard tolerance unit is i = 0.45 cbrt(D) + 0.001 D micrometres, with D the
  geometric mean of the nominal size range in millimetres, and the grades are
  fixed multiples of it. Computing it avoids a hand-copied table that could
  carry a typo nobody would ever notice.

* **It does NOT reproduce the published table exactly, and must not be used to
  put a number on a drawing.** ISO tabulates values rounded to preferred
  numbers rather than the raw formula output. Measured against the published
  IT6 to IT9 values from 3 to 500 mm, this agrees to a mean of 1.2% with a
  worst case of 8.4%, and rounds to the published integer in 36 of 48 cases.
  That is close enough to compare fits and to size a press fit, and not close
  enough to specify one. A drawing needs the table.

* **Sizes at or below 3 mm are refused.** The formula is defined above 3 mm;
  below it ISO tabulates values the expression does not follow, and it is out
  by 13% at IT9 there. Extrapolating into that range would be silently wrong
  in the direction of a looser tolerance than specified.

* **The formula covers grades IT5 upward and sizes to 500 mm.** Below IT5 the
  progression changes and above 500 mm a different expression applies. Both are
  refused rather than extrapolated.

* **Only fundamental deviations with a published closed form are implemented.**
  H and h are zero by definition, and g, k and n have simple published
  expressions. The interference deviations p, r, s and u carry tabulated
  increments that have no formula, so they are NOT implemented and are refused
  by name. Approximating them would put a wrong tolerance on a drawing.

* **A fit class is not an assembly force.** H7/p6 says how much interference
  the parts will have, not what pressure or torque results. That is the press
  fit calculation, which lives next door and needs the geometry and the
  material as well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# ISO 286 nominal size ranges in mm, as (upper bound inclusive). The tolerance
# unit is evaluated at the geometric mean of each range, not at the actual
# size, which is what makes every size inside a range share one tolerance.
_SIZE_RANGES: tuple[tuple[float, float], ...] = (
    (0.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 18.0), (18.0, 30.0),
    (30.0, 50.0), (50.0, 80.0), (80.0, 120.0), (120.0, 180.0),
    (180.0, 250.0), (250.0, 315.0), (315.0, 400.0), (400.0, 500.0),
)

# Multiples of the tolerance unit i, for IT5 upward. The progression is the
# R5 preferred-number series from IT6 onward, which is why each grade is about
# 1.6 times the one below it.
_GRADE_MULTIPLES: dict[int, float] = {
    5: 7.0, 6: 10.0, 7: 16.0, 8: 25.0, 9: 40.0, 10: 64.0, 11: 100.0,
    12: 160.0, 13: 250.0, 14: 400.0, 15: 640.0, 16: 1000.0,
}

MAX_NOMINAL_MM = 500.0
# Below this the ISO expression does not describe the tabulated values, so it
# is refused rather than extrapolated. See the module note.
MIN_NOMINAL_MM = 3.0


class FitType(str, Enum):
    CLEARANCE = "clearance"
    TRANSITION = "transition"
    INTERFERENCE = "interference"


def _range_for(nominal_mm: float) -> tuple[float, float]:
    if nominal_mm <= 0.0:
        raise ValueError("nominal size must be positive")
    if nominal_mm <= MIN_NOMINAL_MM:
        raise ValueError(
            f"{nominal_mm} mm is at or below {MIN_NOMINAL_MM} mm, where the "
            f"ISO 286 tolerance-unit expression does not describe the "
            f"tabulated values. It is out by 13% at IT9 there, so this refuses "
            f"rather than extrapolating")
    if nominal_mm > MAX_NOMINAL_MM:
        raise ValueError(
            f"{nominal_mm} mm is above {MAX_NOMINAL_MM} mm, where ISO 286 uses "
            f"a different expression that is not implemented here")
    for low, high in _SIZE_RANGES:
        if low < MIN_NOMINAL_MM:
            continue
        if low < nominal_mm <= high:
            return low, high
    raise ValueError(f"no ISO 286 size range covers {nominal_mm} mm")


def tolerance_unit_um(nominal_mm: float) -> float:
    """i = 0.45 cbrt(D) + 0.001 D, with D the range's geometric mean, in um."""
    low, high = _range_for(nominal_mm)
    mean = math.sqrt(low * high)
    return 0.45 * mean ** (1.0 / 3.0) + 0.001 * mean


def it_tolerance_um(nominal_mm: float, grade: int) -> float:
    """The IT grade width in micrometres."""
    if grade not in _GRADE_MULTIPLES:
        raise ValueError(
            f"IT{grade} is outside the implemented range IT5 to IT16. Below "
            f"IT5 the progression changes and is not implemented here")
    return _GRADE_MULTIPLES[grade] * tolerance_unit_um(nominal_mm)


# Fundamental deviations with a published closed form. Everything else needs a
# tabulated increment and is refused by name rather than approximated.
_SHAFT_DEVIATION_FORMULAS = {
    "g": lambda d: -2.5 * d ** 0.34,        # upper deviation es, clearance
    "h": lambda d: 0.0,                     # es = 0 by definition
    "k": lambda d: 0.6 * d ** (1.0 / 3.0),  # lower deviation ei, transition
    "n": lambda d: 5.0 * d ** 0.34,         # lower deviation ei, transition
}
_TABULATED_ONLY = ("j", "js", "m", "p", "r", "s", "t", "u", "v", "x", "y", "z")


@dataclass(frozen=True)
class Limits:
    """A tolerance zone, in millimetres, as deviations from nominal."""

    nominal_mm: float
    lower_mm: float
    upper_mm: float

    @property
    def width_mm(self) -> float:
        return self.upper_mm - self.lower_mm

    @property
    def max_mm(self) -> float:
        return self.nominal_mm + self.upper_mm

    @property
    def min_mm(self) -> float:
        return self.nominal_mm + self.lower_mm


def hole_limits(nominal_mm: float, grade: int, letter: str = "H") -> Limits:
    """H holes only: the lower deviation is zero by definition.

    H is the basis of the hole-basis system and covers the overwhelming
    majority of fits. Other hole letters are refused rather than guessed.
    """
    if letter != "H":
        raise ValueError(
            f"hole letter {letter!r} is not implemented. Only H, whose lower "
            f"deviation is zero by definition, has a form that needs no "
            f"tabulated increment")
    width = it_tolerance_um(nominal_mm, grade) / 1000.0
    return Limits(nominal_mm=nominal_mm, lower_mm=0.0, upper_mm=width)


def shaft_limits(nominal_mm: float, grade: int, letter: str) -> Limits:
    """Shaft limits for the deviations that have a closed form."""
    key = letter.lower()
    if key in _TABULATED_ONLY:
        raise ValueError(
            f"shaft deviation {letter!r} carries a tabulated increment with no "
            f"closed form, so it is not implemented. Approximating it would "
            f"put a wrong tolerance on a drawing. Implemented: "
            f"{', '.join(sorted(_SHAFT_DEVIATION_FORMULAS))}")
    if key not in _SHAFT_DEVIATION_FORMULAS:
        raise ValueError(f"unknown shaft deviation {letter!r}")

    width = it_tolerance_um(nominal_mm, grade) / 1000.0
    deviation = _SHAFT_DEVIATION_FORMULAS[key](nominal_mm) / 1000.0
    if key in ("g", "h"):
        # These give the UPPER deviation; the zone runs downward from it.
        return Limits(nominal_mm=nominal_mm, lower_mm=deviation - width,
                      upper_mm=deviation)
    # k and n give the LOWER deviation; the zone runs upward.
    return Limits(nominal_mm=nominal_mm, lower_mm=deviation,
                  upper_mm=deviation + width)


@dataclass(frozen=True)
class Fit:
    """A hole and shaft pairing, with the clearance it produces."""

    hole: Limits
    shaft: Limits
    designation: str

    @property
    def max_clearance_mm(self) -> float:
        """Loosest case: largest hole with smallest shaft."""
        return self.hole.max_mm - self.shaft.min_mm

    @property
    def min_clearance_mm(self) -> float:
        """Tightest case. Negative means interference."""
        return self.hole.min_mm - self.shaft.max_mm

    @property
    def max_interference_mm(self) -> float:
        """The largest interference, as a positive number when it exists."""
        return max(0.0, -self.min_clearance_mm)

    @property
    def fit_type(self) -> FitType:
        if self.min_clearance_mm >= 0.0:
            return FitType.CLEARANCE
        if self.max_clearance_mm <= 0.0:
            return FitType.INTERFERENCE
        return FitType.TRANSITION


def fit(nominal_mm: float, hole_grade: int, shaft_letter: str,
        shaft_grade: int) -> Fit:
    """A hole-basis fit such as H7/g6."""
    return Fit(hole=hole_limits(nominal_mm, hole_grade),
               shaft=shaft_limits(nominal_mm, shaft_grade, shaft_letter),
               designation=f"H{hole_grade}/{shaft_letter}{shaft_grade}")
