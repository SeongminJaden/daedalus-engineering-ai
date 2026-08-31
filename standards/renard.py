"""Preferred numbers: the ISO 3 Renard series, and snapping to them.

A design that calls for a 37.4 mm bar specifies something nobody stocks. The
preferred number series exist so that dimensions cluster on a manageable set of
sizes, and a design snapped onto them is buildable from stock rather than from
a special order.

VALIDITY, before the implementation:

* **SNAPPING CHANGES THE DESIGN, and the safe direction depends on what the
  dimension is.** Rounding a load-bearing thickness DOWN removes material and
  can invalidate the check that sized it. Rounding a clearance UP can close a
  gap that had to stay open. There is no universally safe direction, so the
  direction is a required argument rather than a default, and the caller states
  which way is conservative for their quantity.

* **The published series are ROUNDED geometric progressions, not the
  progressions themselves.** R10 nominally steps by the tenth root of ten,
  which gives 1.2589, and the standard says 1.25. Computing the progression
  and using it directly would put unstandard numbers on a drawing while
  appearing to follow the standard, so the published values are used and the
  rounding is measured against the theory rather than assumed away.

* **A "standard" size is not universally stocked.** The series says which sizes
  are preferred, not which a supplier has, and availability is regional and
  changes. Snapping makes a design more likely to be buildable, not certainly
  so.

* **Compliance with a dimensional standard is not fitness for purpose.** A
  design can sit exactly on the preferred series, use the correct thread
  series, and still be wrong for its duty. This layer checks conformance, and
  conformance is not correctness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Series(str, Enum):
    """The Renard series, coarsest first. Each doubles the number of steps."""

    R5 = "R5"
    R10 = "R10"
    R20 = "R20"
    R40 = "R40"


# The PUBLISHED values over one decade, which are rounded from the geometric
# progression. These are the standard; the progression is only its basis.
_SERIES_VALUES: dict[Series, tuple[float, ...]] = {
    Series.R5: (1.00, 1.60, 2.50, 4.00, 6.30),
    Series.R10: (1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00),
    Series.R20: (1.00, 1.12, 1.25, 1.40, 1.60, 1.80, 2.00, 2.24, 2.50, 2.80,
                 3.15, 3.55, 4.00, 4.50, 5.00, 5.60, 6.30, 7.10, 8.00, 9.00),
    Series.R40: (1.00, 1.06, 1.12, 1.18, 1.25, 1.32, 1.40, 1.50, 1.60, 1.70,
                 1.80, 1.90, 2.00, 2.12, 2.24, 2.36, 2.50, 2.65, 2.80, 3.00,
                 3.15, 3.35, 3.55, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.30,
                 5.60, 6.00, 6.30, 6.70, 7.10, 7.50, 8.00, 8.50, 9.00, 9.50),
}


class SnapDirection(str, Enum):
    """Which way to move. Required, because neither way is universally safe."""

    UP = "up"
    DOWN = "down"
    NEAREST = "nearest"


def series_values(series: Series, decades: int = 3,
                  start_decade: int = -1) -> list[float]:
    """The series expanded over several decades, ascending."""
    if decades < 1:
        raise ValueError("at least one decade is needed")
    values: list[float] = []
    for decade in range(start_decade, start_decade + decades):
        scale = 10.0 ** decade
        values.extend(value * scale for value in _SERIES_VALUES[series])
    return values


def theoretical_step(series: Series) -> float:
    """The geometric ratio the series is built on: 10^(1/n)."""
    return 10.0 ** (1.0 / len(_SERIES_VALUES[series]))


def rounding_deviation(series: Series) -> float:
    """The worst relative gap between the published values and the progression.

    Measured rather than asserted, because the published series is what a
    drawing must use and the progression is only where it came from. If this
    were zero the distinction would not matter, and it is not zero.
    """
    step = theoretical_step(series)
    worst = 0.0
    for index, published in enumerate(_SERIES_VALUES[series]):
        theoretical = step ** index
        worst = max(worst, abs(published - theoretical) / theoretical)
    return worst


def snap(value: float, series: Series, direction: SnapDirection,
         decades: int = 4, start_decade: int = -2) -> float:
    """Move a dimension onto the nearest preferred value in the given direction.

    `direction` is required. Rounding a load-bearing thickness DOWN can
    invalidate the check that sized it, and rounding a clearance UP can close a
    gap that had to stay open, so the caller states which way is conservative
    for the quantity in hand.
    """
    if value <= 0.0:
        raise ValueError("a preferred number is positive")
    values = series_values(series, decades, start_decade)
    if not values[0] <= value <= values[-1]:
        raise ValueError(
            f"{value:g} lies outside the expanded series "
            f"[{values[0]:g}, {values[-1]:g}]; widen the decades rather than "
            f"snapping to an endpoint")

    if direction is SnapDirection.UP:
        return min(v for v in values if v >= value * (1.0 - 1e-12))
    if direction is SnapDirection.DOWN:
        return max(v for v in values if v <= value * (1.0 + 1e-12))
    return min(values, key=lambda v: (abs(v - value), v))


@dataclass(frozen=True)
class SnapReport:
    """What a snap did to a dimension, and by how much."""

    original: float
    snapped: float
    series: Series
    direction: SnapDirection

    @property
    def changed(self) -> bool:
        return abs(self.snapped - self.original) > 1e-12 * max(
            abs(self.original), 1.0)

    @property
    def relative_change(self) -> float:
        return (self.snapped - self.original) / self.original

    @property
    def is_conservative_for_strength(self) -> bool:
        """A strength dimension is safe to grow and unsafe to shrink."""
        return self.snapped >= self.original


def snap_with_report(value: float, series: Series,
                     direction: SnapDirection) -> SnapReport:
    return SnapReport(original=value,
                      snapped=snap(value, series, direction),
                      series=series, direction=direction)
