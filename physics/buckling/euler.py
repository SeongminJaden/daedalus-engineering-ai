"""Elastic column buckling: the Euler load and when it may be used.

A slender compression member fails by going sideways long before its material
yields. The static stress check has nothing to say about this, because the
stress at buckling can be a small fraction of yield. Thin walls and long links
are exactly the shapes an optimiser produces when minimising mass, so the
failure mode a mass optimiser drives towards is the one a yield check cannot
see.

**The Euler load is an upper bound on a real column and should be treated as
one.** It is derived for a perfectly straight, perfectly centred, homogeneous
elastic column. Real members are crooked, eccentrically loaded and carry
residual stress from forming, and all three lower the collapse load. Design
practice applies a knock-down factor for this; none is applied here, so the
number reported is optimistic and a safety factor near 1 is not a safe design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class EndCondition(str, Enum):
    """How the column's ends are held, which sets the effective length.

    The values are the THEORETICAL effective-length factors. Design codes
    recommend higher ones (AISC suggests 0.65 for fixed-fixed and 0.80 for
    fixed-pinned) because real end fixity is never ideal. The theoretical
    values are used here so the arithmetic is checkable against the closed
    form, and a caller wanting code values passes `k_factor` directly.
    """

    FIXED_FREE = "fixed_free"          # a cantilever column, K = 2.0
    PINNED_PINNED = "pinned_pinned"    # the reference case, K = 1.0
    FIXED_PINNED = "fixed_pinned"      # K = 0.699
    FIXED_FIXED = "fixed_fixed"        # K = 0.5


EFFECTIVE_LENGTH_FACTORS: dict[EndCondition, float] = {
    EndCondition.FIXED_FREE: 2.0,
    EndCondition.PINNED_PINNED: 1.0,
    EndCondition.FIXED_PINNED: 0.699,
    EndCondition.FIXED_FIXED: 0.5,
}


def effective_length_factor(condition: EndCondition) -> float:
    return EFFECTIVE_LENGTH_FACTORS[condition]


def euler_critical_load_n(youngs_modulus_pa: float, second_moment_m4: float,
                          length_m: float, k_factor: float) -> float:
    """P_cr = pi^2 E I / (K L)^2, in newtons.

    `second_moment_m4` must be the SMALLEST second moment of the section: a
    column buckles about its weak axis, and using the strong-axis value
    overstates the critical load by exactly the ratio of the two.
    """
    if min(youngs_modulus_pa, second_moment_m4, length_m, k_factor) <= 0.0:
        raise ValueError("E, I, length and K must all be positive")
    return (math.pi ** 2 * youngs_modulus_pa * second_moment_m4
            / (k_factor * length_m) ** 2)


def radius_of_gyration_m(second_moment_m4: float, area_m2: float) -> float:
    """r = sqrt(I / A)."""
    if second_moment_m4 <= 0.0 or area_m2 <= 0.0:
        raise ValueError("I and A must be positive")
    return math.sqrt(second_moment_m4 / area_m2)


def slenderness_ratio(length_m: float, k_factor: float,
                      radius_of_gyration: float) -> float:
    """The dimensionless column slenderness, K L / r."""
    if radius_of_gyration <= 0.0:
        raise ValueError("radius of gyration must be positive")
    return k_factor * length_m / radius_of_gyration


def critical_slenderness(youngs_modulus_pa: float,
                         yield_strength_pa: float) -> float:
    """C_c = sqrt(2 pi^2 E / Sy), the Euler-to-inelastic transition.

    It is the slenderness at which the Euler stress equals half the yield
    strength, the conventional point below which the elastic derivation stops
    describing the column: yielding begins before the elastic critical load is
    reached, so Euler over-predicts.
    """
    if youngs_modulus_pa <= 0.0 or yield_strength_pa <= 0.0:
        raise ValueError("E and yield strength must be positive")
    return math.sqrt(2.0 * math.pi ** 2 * youngs_modulus_pa / yield_strength_pa)


@dataclass(frozen=True)
class BucklingResult:
    """A buckling verdict and the reasoning behind it."""

    critical_load_n: float
    applied_load_n: float
    safety_factor: float
    slenderness: float
    critical_slenderness: float
    k_factor: float
    euler_valid: bool
    governing_mode: str          # "buckling", "yield" or "not_in_compression"
    notes: str = ""

    @property
    def passes(self) -> bool:
        return self.safety_factor >= 1.0


def analyze_column(youngs_modulus_pa: float, yield_strength_pa: float,
                   area_m2: float, min_second_moment_m4: float,
                   length_m: float, applied_load_n: float,
                   condition: EndCondition = EndCondition.FIXED_FREE,
                   k_factor: float | None = None) -> BucklingResult:
    """Check a compression member against elastic buckling.

    `applied_load_n` is positive in COMPRESSION. A member in tension cannot
    buckle, and that is reported rather than returned as an enormous safety
    factor: a large number would read as a well-designed column instead of an
    inapplicable check.

    Whether Euler applies is decided and reported, not assumed. Below the
    critical slenderness the column yields or fails inelastically before the
    elastic critical load is reached, so the Euler number is an over-estimate
    and `euler_valid` is False. The safety factor is still computed, because
    suppressing it would leave the caller with nothing, but it must not be read
    as a buckling margin in that regime.
    """
    k = k_factor if k_factor is not None else effective_length_factor(condition)
    gyration = radius_of_gyration_m(min_second_moment_m4, area_m2)
    slenderness = slenderness_ratio(length_m, k, gyration)
    transition = critical_slenderness(youngs_modulus_pa, yield_strength_pa)
    critical = euler_critical_load_n(youngs_modulus_pa, min_second_moment_m4,
                                     length_m, k)

    if applied_load_n <= 0.0:
        return BucklingResult(
            critical_load_n=critical, applied_load_n=applied_load_n,
            safety_factor=math.inf, slenderness=slenderness,
            critical_slenderness=transition, k_factor=k,
            euler_valid=slenderness >= transition,
            governing_mode="not_in_compression",
            notes="the member is not in compression, so buckling does not "
                  "apply; this is not a margin")

    safety = critical / applied_load_n
    euler_valid = slenderness >= transition
    if euler_valid:
        mode = "buckling" if safety < (yield_strength_pa * area_m2
                                       / applied_load_n) else "yield"
        note = ("slender column: the elastic Euler derivation applies. The "
                "result is still an upper bound, since initial crookedness, "
                "load eccentricity and residual stress all lower the real "
                "collapse load and no knock-down factor is applied.")
    else:
        mode = "yield"
        note = (f"short column: slenderness {slenderness:.1f} is below the "
                f"transition {transition:.1f}, so the member yields or fails "
                f"inelastically before the elastic critical load is reached. "
                f"The Euler load above OVER-predicts and must not be read as a "
                f"buckling margin.")

    return BucklingResult(
        critical_load_n=critical, applied_load_n=applied_load_n,
        safety_factor=safety, slenderness=slenderness,
        critical_slenderness=transition, k_factor=k, euler_valid=euler_valid,
        governing_mode=mode, notes=note)
