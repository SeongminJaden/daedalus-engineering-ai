"""Rolling bearing life: the equivalent load and the L10 relation.

**L10 is a statistic, not a lifetime.** It is the number of revolutions that
ninety percent of a population of identical bearings will reach before the
first sign of subsurface fatigue. One bearing in ten is expected to fail
sooner, and a specific bearing has no promised life at all. Reporting it as
"the life" is the most common way this calculation is misread.

The ISO 281 adjustment factors for reliability, lubricant film, contamination
and temperature are NOT applied here. Each of them can move the answer by more
than an order of magnitude in either direction: clean, well-lubricated,
lightly-loaded bearings routinely exceed L10 by a large factor, and
contaminated or marginally-lubricated ones fall far short of it. The number
here is the unadjusted basic rating life and nothing more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .catalog import BearingSpec, BearingType

# Radial and thrust factors for single-row deep groove ball bearings, indexed
# by the axial load as a fraction of the static rating. This is the standard
# tabulation (ISO 281, reproduced in the usual design texts); the values are a
# published standard rather than a per-manufacturer catalogue, which is why
# they can appear here when a part number could not.
#
# When Fa/Fr is at or below e, the thrust load is carried without changing the
# contact geometry and the equivalent load is simply Fr.
_DEEP_GROOVE_TABLE: tuple[tuple[float, float, float], ...] = (
    # (Fa/C0, e, Y for Fa/Fr > e)
    (0.014, 0.19, 2.30),
    (0.021, 0.21, 2.15),
    (0.028, 0.22, 1.99),
    (0.042, 0.24, 1.85),
    (0.056, 0.26, 1.71),
    (0.070, 0.27, 1.63),
    (0.084, 0.28, 1.55),
    (0.110, 0.30, 1.45),
    (0.170, 0.34, 1.31),
    (0.280, 0.38, 1.15),
    (0.420, 0.42, 1.04),
    (0.560, 0.44, 1.00),
)

# X for a deep groove ball bearing once the thrust load passes e.
_DEEP_GROOVE_X = 0.56


def _interpolate(fraction: float) -> tuple[float, float]:
    """(e, Y) at this Fa/C0, linearly between tabulated rows.

    Clamped at both ends rather than extrapolated. Past the last row the
    factors have flattened, and below the first there is no data to extend
    into; inventing a trend outside a standard table would be inventing the
    standard.
    """
    first, last = _DEEP_GROOVE_TABLE[0], _DEEP_GROOVE_TABLE[-1]
    if fraction <= first[0]:
        return first[1], first[2]
    if fraction >= last[0]:
        return last[1], last[2]
    for (low, e_low, y_low), (high, e_high, y_high) in zip(
            _DEEP_GROOVE_TABLE, _DEEP_GROOVE_TABLE[1:]):
        if low <= fraction <= high:
            span = high - low
            weight = 0.0 if span == 0.0 else (fraction - low) / span
            return (e_low + weight * (e_high - e_low),
                    y_low + weight * (y_high - y_low))
    return last[1], last[2]      # pragma: no cover - covered by the clamps


@dataclass(frozen=True)
class EquivalentLoad:
    """P = X Fr + Y Fa, with the factors that produced it."""

    equivalent_load_n: float
    radial_load_n: float
    axial_load_n: float
    x_factor: float
    y_factor: float
    e_ratio: float
    thrust_governs: bool


def equivalent_dynamic_load(bearing: BearingSpec, radial_load_n: float,
                            axial_load_n: float = 0.0) -> EquivalentLoad:
    """The single radial load equivalent to the applied combination.

    For a deep groove ball bearing the factors come from the standard table and
    depend on how large the thrust is relative to the static rating, because
    that is what sets the contact angle the balls run at.

    A cylindrical roller bearing carries no thrust: its rollers run between
    straight raceways, and a bearing without flanges to react an axial load
    cannot be given one. That is raised rather than absorbed into an equivalent
    load, because silently ignoring the thrust would return a life for a
    loading the bearing is not carrying.
    """
    if radial_load_n < 0.0 or axial_load_n < 0.0:
        raise ValueError("bearing loads are magnitudes and cannot be negative")

    if bearing.bearing_type is BearingType.CYLINDRICAL_ROLLER:
        if axial_load_n > 0.0:
            raise ValueError(
                f"{bearing.designation} is a cylindrical roller bearing and "
                f"carries no thrust; an axial load of {axial_load_n:.6g} N "
                f"needs a bearing type that can react it")
        return EquivalentLoad(
            equivalent_load_n=radial_load_n, radial_load_n=radial_load_n,
            axial_load_n=0.0, x_factor=1.0, y_factor=0.0, e_ratio=math.inf,
            thrust_governs=False)

    e_ratio, y_factor = _interpolate(axial_load_n / bearing.static_rating_n)
    if radial_load_n > 0.0 and axial_load_n / radial_load_n > e_ratio:
        load = _DEEP_GROOVE_X * radial_load_n + y_factor * axial_load_n
        return EquivalentLoad(
            equivalent_load_n=load, radial_load_n=radial_load_n,
            axial_load_n=axial_load_n, x_factor=_DEEP_GROOVE_X,
            y_factor=y_factor, e_ratio=e_ratio, thrust_governs=True)

    # Thrust below e, or no radial load to compare against: the radial load
    # alone is the equivalent, except that a pure thrust load still has to be
    # carried, so it is taken as the larger of the two.
    load = max(radial_load_n, axial_load_n * y_factor) if radial_load_n == 0.0 \
        else radial_load_n
    return EquivalentLoad(
        equivalent_load_n=load, radial_load_n=radial_load_n,
        axial_load_n=axial_load_n, x_factor=1.0, y_factor=0.0,
        e_ratio=e_ratio, thrust_governs=False)


def l10_revolutions(dynamic_rating_n: float, equivalent_load_n: float,
                    life_exponent: float) -> float:
    """L10 = (C / P)^p * 1e6 revolutions."""
    if dynamic_rating_n <= 0.0:
        raise ValueError("the dynamic rating must be positive")
    if equivalent_load_n <= 0.0:
        raise ValueError(
            "the equivalent load must be positive; an unloaded bearing has no "
            "fatigue life to compute")
    return (dynamic_rating_n / equivalent_load_n) ** life_exponent * 1.0e6


def l10_hours(revolutions: float, speed_rad_s: float) -> float:
    """Revolutions converted to hours at a running speed.

        L10h = L10 / (60 n),  n in rpm

    The speed comes in as rad/s, the project convention, and is converted here.
    """
    if speed_rad_s <= 0.0:
        raise ValueError("speed must be positive to convert a life to hours")
    rpm = speed_rad_s * 60.0 / (2.0 * math.pi)
    return revolutions / (60.0 * rpm)


@dataclass(frozen=True)
class BearingLifeResult:
    """A bearing life verdict, with the load that produced it."""

    bearing: BearingSpec
    load: EquivalentLoad
    l10_revolutions: float
    l10_hours: float
    static_safety_factor: float
    speed_rad_s: float
    required_hours: float | None
    life_margin: float | None
    speed_within_limit: bool

    @property
    def passes(self) -> bool:
        """Meets the required life and stays inside the speed limit."""
        if not self.speed_within_limit:
            return False
        if self.required_hours is None:
            return True
        return self.l10_hours >= self.required_hours


def rate_bearing(bearing: BearingSpec, radial_load_n: float,
                 speed_rad_s: float, axial_load_n: float = 0.0,
                 required_hours: float | None = None) -> BearingLifeResult:
    """Full check: equivalent load, L10 life, static capacity and speed limit.

    The static safety factor C0/P is reported alongside the life because a
    lightly-spun, heavily-loaded bearing can be indented to failure without
    ever accumulating the revolutions a fatigue life is counted in.
    """
    load = equivalent_dynamic_load(bearing, radial_load_n, axial_load_n)
    revolutions = l10_revolutions(bearing.dynamic_rating_n,
                                  load.equivalent_load_n,
                                  bearing.life_exponent)
    hours = l10_hours(revolutions, speed_rad_s)
    margin = None if required_hours is None else hours / required_hours
    return BearingLifeResult(
        bearing=bearing, load=load, l10_revolutions=revolutions,
        l10_hours=hours,
        static_safety_factor=bearing.static_rating_n / load.equivalent_load_n,
        speed_rad_s=speed_rad_s, required_hours=required_hours,
        life_margin=margin,
        speed_within_limit=speed_rad_s <= bearing.limiting_speed_rad_s)
