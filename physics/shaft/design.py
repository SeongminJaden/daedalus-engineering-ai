"""Shaft sizing under combined bending, torsion and axial load.

**A diameter chosen from torque alone is wrong.** A shaft carrying a gear or a
pulley is bent by the load it transmits at the same time as it is twisted by
it, and the bending is what usually decides. Worse, a rotating shaft under a
steady transverse load sees that bending as FULLY REVERSED: every revolution
takes a given fibre from maximum tension to maximum compression. So the shaft
is a fatigue problem even when nothing about the duty appears to change, and a
static torsion check will not see it.

That is why the sizing here is the DE-Goodman relation rather than a static
stress limit: it charges the reversed bending against the endurance strength
and the steady torque against the ultimate, in one expression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.materials.db import MaterialSpec

# Stress-concentration factors when the caller does not supply them.
#
# [ASSUMED] and deliberately not 1.0. A real shaft has a shoulder fillet, a
# keyway or a retaining-ring groove wherever it locates something, and every
# one of them raises the local stress. Defaulting to 1.0 would model a
# featureless bar and return a diameter that is too small for anything that
# actually holds a gear. These are mid-range values for a well-radiused
# shoulder and a profiled keyway; a real design measures its own geometry.
DEFAULT_BENDING_KF = 1.7
DEFAULT_TORSION_KFS = 1.5


def bending_stress_pa(moment_nm: float, diameter_m: float) -> float:
    """sigma = 32 M / (pi d^3), the outer fibre of a solid round shaft."""
    if diameter_m <= 0.0:
        raise ValueError("diameter must be positive")
    return 32.0 * moment_nm / (math.pi * diameter_m ** 3)


def torsional_stress_pa(torque_nm: float, diameter_m: float) -> float:
    """tau = 16 T / (pi d^3), the surface of a solid round shaft."""
    if diameter_m <= 0.0:
        raise ValueError("diameter must be positive")
    return 16.0 * torque_nm / (math.pi * diameter_m ** 3)


def axial_stress_pa(force_n: float, diameter_m: float) -> float:
    """sigma = 4 F / (pi d^2)."""
    if diameter_m <= 0.0:
        raise ValueError("diameter must be positive")
    return 4.0 * force_n / (math.pi * diameter_m ** 2)


def von_mises_pa(normal_pa: float, shear_pa: float) -> float:
    """sqrt(sigma^2 + 3 tau^2) for the combined bending-torsion state."""
    return math.sqrt(normal_pa ** 2 + 3.0 * shear_pa ** 2)


def max_shear_pa(normal_pa: float, shear_pa: float) -> float:
    """The maximum shear stress theory: sqrt((sigma/2)^2 + tau^2).

    More conservative than von Mises by up to 15%. Both are offered because
    which one a shop uses is a house convention, and quietly picking one would
    change a safety factor without the caller knowing.
    """
    return math.sqrt((0.5 * normal_pa) ** 2 + shear_pa ** 2)


@dataclass(frozen=True)
class ShaftLoads:
    """The moment and torque at a section, split into mean and alternating.

    For the common case of a rotating shaft under a steady transverse load and
    a steady torque, use `rotating`: the bending alternates and the torque does
    not, and getting that the wrong way round is the single most consequential
    mistake available here.
    """

    moment_alternating_nm: float = 0.0
    moment_mean_nm: float = 0.0
    torque_alternating_nm: float = 0.0
    torque_mean_nm: float = 0.0
    axial_force_n: float = 0.0

    @classmethod
    def rotating(cls, bending_moment_nm: float, torque_nm: float,
                 axial_force_n: float = 0.0) -> "ShaftLoads":
        """A rotating shaft: steady transverse load, steady torque.

        The bending moment is constant in space and therefore fully reversed in
        the material, so it is entirely alternating. The torque is steady and
        therefore entirely mean.
        """
        return cls(moment_alternating_nm=bending_moment_nm,
                   moment_mean_nm=0.0,
                   torque_alternating_nm=0.0,
                   torque_mean_nm=torque_nm,
                   axial_force_n=axial_force_n)

    @classmethod
    def stationary(cls, bending_moment_nm: float, torque_nm: float,
                   axial_force_n: float = 0.0) -> "ShaftLoads":
        """A non-rotating shaft under a steady load: nothing alternates."""
        return cls(moment_alternating_nm=0.0, moment_mean_nm=bending_moment_nm,
                   torque_alternating_nm=0.0, torque_mean_nm=torque_nm,
                   axial_force_n=axial_force_n)


@dataclass(frozen=True)
class ShaftResult:
    """A shaft verdict: the static state, the fatigue factor, and the geometry."""

    diameter_m: float
    bending_stress_pa: float
    torsional_stress_pa: float
    axial_stress_pa: float
    von_mises_pa: float
    static_safety_factor: float
    fatigue_safety_factor: float
    governing_mode: str
    bending_kf: float
    torsion_kfs: float

    @property
    def passes(self) -> bool:
        return min(self.static_safety_factor, self.fatigue_safety_factor) >= 1.0

    @property
    def governing_safety_factor(self) -> float:
        return min(self.static_safety_factor, self.fatigue_safety_factor)


def de_goodman_inverse_factor(loads: ShaftLoads, material: MaterialSpec,
                              diameter_m: float,
                              bending_kf: float = DEFAULT_BENDING_KF,
                              torsion_kfs: float = DEFAULT_TORSION_KFS
                              ) -> float:
    """1/n for the distortion-energy Goodman shaft relation.

        1/n = (16 / (pi d^3)) * [ sqrt(4 (Kf Ma)^2 + 3 (Kfs Ta)^2) / Se
                                + sqrt(4 (Kf Mm)^2 + 3 (Kfs Tm)^2) / Sut ]

    The alternating group is charged against the endurance strength and the
    mean group against the ultimate, which is the Goodman line written for a
    combined stress state through the distortion-energy equivalent.

    For the rotating case with steady torque this collapses to

        1/n = (16 / (pi d^3)) * [ 2 Kf M / Se + sqrt(3) Kfs T / Sut ]

    which is worth knowing because it makes the arithmetic checkable by hand.
    """
    if diameter_m <= 0.0:
        raise ValueError("diameter must be positive")
    alternating = math.sqrt(
        4.0 * (bending_kf * loads.moment_alternating_nm) ** 2
        + 3.0 * (torsion_kfs * loads.torque_alternating_nm) ** 2)
    mean = math.sqrt(
        4.0 * (bending_kf * loads.moment_mean_nm) ** 2
        + 3.0 * (torsion_kfs * loads.torque_mean_nm) ** 2)
    return (16.0 / (math.pi * diameter_m ** 3)) * (
        alternating / material.fatigue_strength_pa
        + mean / material.ultimate_strength_pa)


def de_goodman_diameter_m(loads: ShaftLoads, material: MaterialSpec,
                          safety_factor: float,
                          bending_kf: float = DEFAULT_BENDING_KF,
                          torsion_kfs: float = DEFAULT_TORSION_KFS) -> float:
    """The smallest diameter meeting `safety_factor` by DE-Goodman.

        d = { (16 n / pi) [ ... ] }^(1/3)

    The same relation solved for d, which is how a shaft is actually sized:
    the factor is chosen and the diameter follows.
    """
    if safety_factor <= 0.0:
        raise ValueError("safety factor must be positive")
    alternating = math.sqrt(
        4.0 * (bending_kf * loads.moment_alternating_nm) ** 2
        + 3.0 * (torsion_kfs * loads.torque_alternating_nm) ** 2)
    mean = math.sqrt(
        4.0 * (bending_kf * loads.moment_mean_nm) ** 2
        + 3.0 * (torsion_kfs * loads.torque_mean_nm) ** 2)
    bracket = (alternating / material.fatigue_strength_pa
               + mean / material.ultimate_strength_pa)
    if bracket <= 0.0:
        raise ValueError(
            "no bending or torque was given, so there is nothing to size the "
            "shaft against")
    return ((16.0 * safety_factor / math.pi) * bracket) ** (1.0 / 3.0)


def analyze_shaft(loads: ShaftLoads, material: MaterialSpec, diameter_m: float,
                  bending_kf: float = DEFAULT_BENDING_KF,
                  torsion_kfs: float = DEFAULT_TORSION_KFS) -> ShaftResult:
    """Check a shaft diameter against both static yielding and fatigue.

    Both, because they can disagree and the fatigue one usually wins on a
    rotating shaft. Reporting the static factor alone is how a shaft sized on
    torque passes review and then breaks in service.
    """
    peak_moment = (abs(loads.moment_mean_nm) + abs(loads.moment_alternating_nm))
    peak_torque = (abs(loads.torque_mean_nm) + abs(loads.torque_alternating_nm))
    bending = bending_stress_pa(peak_moment, diameter_m)
    torsion = torsional_stress_pa(peak_torque, diameter_m)
    axial = axial_stress_pa(loads.axial_force_n, diameter_m)
    equivalent = von_mises_pa(bending + axial, torsion)
    static = (math.inf if equivalent <= 0.0
              else material.yield_strength_pa / equivalent)

    inverse = de_goodman_inverse_factor(loads, material, diameter_m,
                                        bending_kf, torsion_kfs)
    fatigue = math.inf if inverse <= 0.0 else 1.0 / inverse

    return ShaftResult(
        diameter_m=diameter_m, bending_stress_pa=bending,
        torsional_stress_pa=torsion, axial_stress_pa=axial,
        von_mises_pa=equivalent, static_safety_factor=static,
        fatigue_safety_factor=fatigue,
        governing_mode="fatigue" if fatigue < static else "static",
        bending_kf=bending_kf, torsion_kfs=torsion_kfs)


def first_critical_speed_rad_s(material: MaterialSpec, diameter_m: float,
                               length_m: float) -> float:
    """First bending natural frequency of a uniform simply supported shaft.

        omega_1 = (pi / L)^2 * sqrt(E I / (rho A))

    **A deliberate simplification and it flatters the answer.** It is a bare
    uniform shaft: no gear, pulley or rotor mass, which are exactly the heavy
    things a shaft carries and which lower the critical speed substantially. It
    also assumes simple supports, while real bearings add stiffness. Use it to
    find out whether the operating speed is anywhere near resonance, not to
    conclude that it is safely below it.
    """
    if diameter_m <= 0.0 or length_m <= 0.0:
        raise ValueError("diameter and length must be positive")
    area = math.pi * diameter_m ** 2 / 4.0
    second_moment = math.pi * diameter_m ** 4 / 64.0
    return ((math.pi / length_m) ** 2
            * math.sqrt(material.youngs_modulus_pa * second_moment
                        / (material.density_kg_m3 * area)))
