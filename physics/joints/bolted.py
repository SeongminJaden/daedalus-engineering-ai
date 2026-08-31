"""Bolted joints: preload, load sharing, separation and bolt fatigue.

**A bolted joint is a preload problem, not a bolt-strength problem.** The thing
that decides whether the joint works is how hard the bolt was pulled up before
any external load arrived. A properly preloaded bolt sees only a small fraction
of an external load, because most of it is taken by relieving the compression
in the clamped members instead. An under-preloaded one sees nearly all of it,
and fails in fatigue at a load the same bolt would carry indefinitely if it had
been tightened correctly.

That is why the load factor C is the centre of this module, and why the fatigue
check is written in terms of preload rather than in terms of the external load
alone.

**Torque control is the weakest link in the whole calculation.** The nut factor
K relating torque to preload scatters by roughly plus or minus 30 percent
depending on lubrication, surface finish and how many times the bolt has been
used, so the achieved preload is uncertain by that much even when the torque
wrench is perfect. Angle control and direct tension measurement are better, and
neither is modelled here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# --- ISO 898-1 property classes ---------------------------------------------
#
# A published standard, not a manufacturer catalogue: class 8.8 means the same
# thing from any supplier. That is why these values can appear here when a part
# number could not. Strengths in Pa.


class PropertyClass(str, Enum):
    C8_8 = "8.8"
    C10_9 = "10.9"
    C12_9 = "12.9"


@dataclass(frozen=True)
class BoltGrade:
    proof_strength_pa: float
    yield_strength_pa: float
    ultimate_strength_pa: float
    source: str = "ISO 898-1 property class"


BOLT_GRADES: dict[PropertyClass, BoltGrade] = {
    PropertyClass.C8_8: BoltGrade(580e6, 640e6, 800e6),
    PropertyClass.C10_9: BoltGrade(830e6, 940e6, 1040e6),
    PropertyClass.C12_9: BoltGrade(970e6, 1100e6, 1220e6),
}

# Tensile stress area of ISO metric coarse threads, m^2. Standard values, and
# NOT the plain shank area: the thread cuts material away, and using the
# nominal diameter would overstate a bolt's capacity by roughly 20 percent.
THREAD_STRESS_AREA_M2: dict[str, float] = {
    "M3": 5.03e-6,
    "M4": 8.78e-6,
    "M5": 14.2e-6,
    "M6": 20.1e-6,
    "M8": 36.6e-6,
    "M10": 58.0e-6,
    "M12": 84.3e-6,
}

NOMINAL_DIAMETER_M: dict[str, float] = {
    "M3": 0.003, "M4": 0.004, "M5": 0.005, "M6": 0.006,
    "M8": 0.008, "M10": 0.010, "M12": 0.012,
}

# Fraction of proof load used as the preload target. 0.75 is the usual figure
# for a reusable connection; 0.90 is used for permanent ones. Preloading to
# less than this is the common cause of fatigue failure, not the safe choice it
# looks like.
PRELOAD_FRACTION = 0.75

# Nut factor in T = K F d. 0.2 is the conventional dry-steel value and is
# [ASSUMED]. It scatters by about plus or minus 30 percent with lubrication and
# surface condition, so a torque figure derived from it carries that same
# uncertainty into the preload.
NUT_FACTOR_DRY = 0.2

# Wileman, Choudhury and Green's empirical fit for member stiffness,
# km = A E d exp(B d / l). Published constants, per member material.
_WILEMAN = {"steel": (0.78715, 0.62873), "aluminium": (0.79670, 0.63816)}


def thread_stress_area_m2(size: str) -> float:
    try:
        return THREAD_STRESS_AREA_M2[size]
    except KeyError:
        raise KeyError(f"unknown thread size {size!r}. Known: "
                       f"{', '.join(THREAD_STRESS_AREA_M2)}") from None


def proof_load_n(size: str, grade: PropertyClass) -> float:
    """A_t times the proof strength: the load the bolt takes without set."""
    return thread_stress_area_m2(size) * BOLT_GRADES[grade].proof_strength_pa


def target_preload_n(size: str, grade: PropertyClass,
                     fraction: float = PRELOAD_FRACTION) -> float:
    return fraction * proof_load_n(size, grade)


def tightening_torque_nm(preload_n: float, size: str,
                         nut_factor: float = NUT_FACTOR_DRY) -> float:
    """T = K F d.

    The relation is an approximation with a wide spread, and it is the step at
    which most of the uncertainty in a bolted joint enters.
    """
    return nut_factor * preload_n * NOMINAL_DIAMETER_M[size]


def bolt_stiffness_n_m(size: str, grip_length_m: float,
                       youngs_modulus_pa: float = 207e9) -> float:
    """k_b = A_t E / l, the bolt as a bar in tension over its grip."""
    if grip_length_m <= 0.0:
        raise ValueError("grip length must be positive")
    return thread_stress_area_m2(size) * youngs_modulus_pa / grip_length_m


def member_stiffness_n_m(size: str, grip_length_m: float,
                         youngs_modulus_pa: float = 207e9,
                         material: str = "steel") -> float:
    """k_m by the Wileman fit: A E d exp(B d / l).

    The clamped members are much stiffer than the bolt, typically by a factor
    of three to five, and that ratio is the whole reason a preloaded joint
    protects its bolt.
    """
    if grip_length_m <= 0.0:
        raise ValueError("grip length must be positive")
    try:
        a, b = _WILEMAN[material]
    except KeyError:
        raise KeyError(f"no Wileman constants for {material!r}. Known: "
                       f"{', '.join(_WILEMAN)}") from None
    diameter = NOMINAL_DIAMETER_M[size]
    return (a * youngs_modulus_pa * diameter
            * math.exp(b * diameter / grip_length_m))


def load_factor(bolt_stiffness: float, member_stiffness: float) -> float:
    """C = k_b / (k_b + k_m), the share of an external load the BOLT sees.

    The rest of the load does not go into the bolt at all: it relieves the
    compression already in the members. A typical joint has C around 0.2, so
    the bolt feels a fifth of what is applied to the joint, and that is what
    makes preloaded joints survive fatigue loading.
    """
    total = bolt_stiffness + member_stiffness
    if total <= 0.0:
        raise ValueError("stiffnesses must be positive")
    return bolt_stiffness / total


@dataclass(frozen=True)
class JointResult:
    """Everything a bolted joint verdict rests on."""

    size: str
    grade: PropertyClass
    preload_n: float
    tightening_torque_nm: float
    load_factor: float
    external_load_n: float
    bolt_load_n: float
    clamp_load_n: float
    separation_load_n: float
    separation_margin: float
    separated: bool
    yield_safety_factor: float
    fatigue_safety_factor: float | None
    governing_mode: str

    @property
    def passes(self) -> bool:
        return self.governing_safety_factor >= 1.0

    @property
    def governing_safety_factor(self) -> float:
        factors = [self.separation_margin, self.yield_safety_factor]
        if self.fatigue_safety_factor is not None:
            factors.append(self.fatigue_safety_factor)
        return min(factors)

    def summary(self) -> str:
        return (f"{self.governing_mode} governs at "
                f"{self.governing_safety_factor:.3f}")


def analyze_joint(size: str, grade: PropertyClass, grip_length_m: float,
                  external_load_n: float,
                  external_load_min_n: float = 0.0,
                  preload_fraction: float = PRELOAD_FRACTION,
                  member_material: str = "steel",
                  member_modulus_pa: float = 207e9,
                  endurance_strength_pa: float = 129e6) -> JointResult:
    """Check one preloaded bolt against separation, yield and fatigue.

    `external_load_n` is the maximum tensile load on the joint and
    `external_load_min_n` the minimum over the cycle, so a joint loaded and
    unloaded is (P, 0) and a steady one is (P, P).

    `endurance_strength_pa` defaults to a rolled-thread value. Rolled threads
    are substantially better in fatigue than cut ones because the rolling
    leaves compressive residual stress in the root, and the root is where
    bolts break.
    """
    area = thread_stress_area_m2(size)
    strengths = BOLT_GRADES[grade]
    preload = target_preload_n(size, grade, preload_fraction)
    torque = tightening_torque_nm(preload, size)

    bolt_k = bolt_stiffness_n_m(size, grip_length_m)
    member_k = member_stiffness_n_m(size, grip_length_m, member_modulus_pa,
                                    member_material)
    factor = load_factor(bolt_k, member_k)

    separation_load = preload / (1.0 - factor) if factor < 1.0 else math.inf
    clamp_load = preload - (1.0 - factor) * external_load_n

    # F_b = F_i + C P holds only WHILE THE JOINT IS STILL CLAMPED. Once the
    # members separate there is nothing left to share the load with and the
    # bolt carries all of it. Applying the clamped formula past separation
    # understates the bolt load badly, and it understates it exactly in the
    # case that matters: an under-preloaded joint, which is the usual way
    # bolted joints fail in fatigue.
    separated = external_load_n > separation_load
    bolt_load = (external_load_n if separated
                 else preload + factor * external_load_n)
    separation_margin = (separation_load / external_load_n
                         if external_load_n > 0.0 else math.inf)

    yield_safety = strengths.proof_strength_pa / (bolt_load / area)

    fatigue: float | None = None
    if external_load_n != external_load_min_n:
        # While clamped, only the bolt's SHARE of the external load alternates,
        # about a mean the preload sets. That share is C, typically a fifth, so
        # a preloaded joint turns a large external range into a small stress
        # range at the bolt. This is the entire reason preloaded joints survive
        # cyclic loading.
        #
        # Once separated the bolt takes the whole range, and the alternating
        # stress jumps by a factor of 1/C. Computing the clamped case anyway
        # would report a comfortable fatigue life for a joint that is about to
        # break.
        maximum = (external_load_n if separated
                   else preload + factor * external_load_n)
        minimum_separated = external_load_min_n > separation_load
        minimum = (external_load_min_n if minimum_separated
                   else preload + factor * external_load_min_n)
        alternating = max(0.0, (maximum - minimum) / (2.0 * area))
        mean = (maximum + minimum) / (2.0 * area)
        initial = preload / area

        # The Goodman line with the LOAD LINE STARTING AT THE PRELOAD, which is
        # the standard treatment for a bolted joint and not the same as the
        # plain Goodman used for a shaft:
        #
        #     n = Se (Sut - sigma_i) / (Sut sigma_a + Se (sigma_m - sigma_i))
        #
        # The preload is a static stress that exists before the cycle begins,
        # so the cyclic component rides on it rather than being averaged with
        # it. Using plain Goodman here reports that REDUCING preload improves
        # fatigue life, because it lowers the mean stress term. That is
        # backwards as engineering advice: the reason to preload a joint is to
        # keep it clamped, and an under-preloaded joint separates and hands the
        # bolt the entire external range.
        numerator = endurance_strength_pa * (strengths.ultimate_strength_pa
                                             - initial)
        denominator = (strengths.ultimate_strength_pa * alternating
                       + endurance_strength_pa * (mean - initial))
        if numerator <= 0.0:
            fatigue = 0.0        # preloaded past the ultimate strength
        elif denominator <= 0.0:
            fatigue = math.inf
        else:
            fatigue = numerator / denominator

    candidates = [("separation", separation_margin), ("yield", yield_safety)]
    if fatigue is not None:
        candidates.append(("fatigue", fatigue))
    mode, _ = min(candidates, key=lambda pair: pair[1])

    return JointResult(
        size=size, grade=grade, preload_n=preload, tightening_torque_nm=torque,
        load_factor=factor, external_load_n=external_load_n,
        bolt_load_n=bolt_load, clamp_load_n=clamp_load,
        separation_load_n=separation_load, separated=separated,
        separation_margin=separation_margin, yield_safety_factor=yield_safety,
        fatigue_safety_factor=fatigue, governing_mode=mode)
