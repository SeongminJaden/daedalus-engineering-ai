"""Minimum section height for a rectangular cantilever, and what governs it.

Every check in this project answers "is this design safe". This answers the
question a designer actually asks first, which is "how small can it be", and
just as importantly which failure mode is the reason it cannot be smaller.

Each mode is inverted in closed form rather than searched, so the answer is
exact for the mode and the governing one can be identified by comparison
instead of by iteration. The composition is new; every formula it composes is
already verified elsewhere in this project.

VALIDITY, stated before use:

* Euler-Bernoulli bending. Shear deflection is neglected, which is
  reasonable while the beam is slender and understates deflection when it is
  not. The span to depth ratio is returned so the caller can see whether the
  assumption held, and a ratio below about 10 means the answer is optimistic.
* Small deflection. The moment arm is taken as the undeformed length.
* A solid rectangle, bending about the strong axis, with a tip point load.
RELATED, and deliberately not merged with it: `integration.minimum_dimension`
sizes by BISECTION on any callable returning a safety factor, which is the
right tool when the relationship has no closed form and the wrong one here.
This module inverts each mode exactly and identifies which one governs, which
bisection on a single combined factor cannot do. Its result type was renamed
to SectionSizing so the two are not confused; both existing at once is
deliberate, and either being mistaken for the other is not.

* Lateral torsional buckling is NOT checked. A tall thin section can fail
  that way at a load below its yield capacity, so a result with a large
  height to width ratio is not safe on the strength of this function alone.
  The ratio is returned and a flag is raised past a stated threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from core.materials.db import MaterialSpec
from physics.fatigue import MeanStressCriterion, StressCycle

#: Below this span to depth ratio the beam is no longer slender and neglecting
#: shear deflection understates the real deflection. Ten is the conventional
#: dividing line; it is a judgement, not a measurement.
SLENDER_SPAN_TO_DEPTH = 10.0

#: Above this height to width ratio a rectangular beam becomes vulnerable to
#: lateral torsional buckling, which this function does not check. Reported so
#: the omission is visible rather than silent.
LATERAL_BUCKLING_ASPECT = 4.0


class FailureMode(str, Enum):
    YIELD = "yield"
    DEFLECTION = "deflection"
    FATIGUE = "fatigue"
    MANUFACTURING = "manufacturing"


@dataclass(frozen=True)
class ModeRequirement:
    """The height one mode on its own would demand."""

    mode: FailureMode
    height_m: float
    basis: str


@dataclass(frozen=True)
class SectionSizing:
    """The smallest height that satisfies every mode, and why."""

    height_m: float
    governing: FailureMode
    requirements: tuple
    width_m: float
    length_m: float
    span_to_depth: float
    aspect_ratio: float

    @property
    def is_slender(self) -> bool:
        return self.span_to_depth >= SLENDER_SPAN_TO_DEPTH

    @property
    def may_buckle_laterally(self) -> bool:
        return self.aspect_ratio > LATERAL_BUCKLING_ASPECT

    @property
    def warnings(self) -> tuple:
        out = []
        if not self.is_slender:
            out.append(
                f"span to depth is {self.span_to_depth:.1f}, below "
                f"{SLENDER_SPAN_TO_DEPTH:.0f}. Shear deflection is neglected "
                f"here, so the real deflection is LARGER than computed and "
                f"this height is optimistic.")
        if self.may_buckle_laterally:
            out.append(
                f"height to width is {self.aspect_ratio:.1f}, above "
                f"{LATERAL_BUCKLING_ASPECT:.0f}. Lateral torsional buckling "
                f"is not checked by this function and can govern before "
                f"yield.")
        return tuple(out)

    def required_for(self, mode: FailureMode) -> float:
        for requirement in self.requirements:
            if requirement.mode is mode:
                return requirement.height_m
        raise KeyError(f"{mode} was not evaluated")


def height_for_yield_m(load_n: float, length_m: float, width_m: float,
                       allowable_pa: float) -> float:
    """sigma = 6 M / (b h^2), so h = sqrt(6 M / (b sigma))."""
    moment = load_n * length_m
    return math.sqrt(6.0 * moment / (width_m * allowable_pa))


def height_for_deflection_m(load_n: float, length_m: float, width_m: float,
                            youngs_modulus_pa: float, limit_m: float) -> float:
    """delta = P L^3 / (3 E I) with I = b h^3 / 12, so

           h = (4 P L^3 / (E b delta))^(1/3)
    """
    return (4.0 * load_n * length_m ** 3
            / (youngs_modulus_pa * width_m * limit_m)) ** (1.0 / 3.0)


def size_rectangular_cantilever(
        load_n: float, length_m: float, width_m: float,
        material: MaterialSpec, safety_factor: float = 1.5,
        deflection_limit_m: float | None = None,
        fully_reversed: bool = False,
        fatigue_criterion: MeanStressCriterion = MeanStressCriterion.GOODMAN,
        minimum_height_m: float | None = None) -> SectionSizing:
    """The smallest section height that satisfies every mode requested.

    `fully_reversed` states that the tip load reverses, which makes fatigue a
    live mode rather than an omitted one. It is not inferred: a load that
    reverses and a load that does not look identical as a number, and guessing
    would either invent a fatigue limit or silently drop one.

    `minimum_height_m` is a manufacturing floor, such as the thinnest wall a
    process can hold. It is a mode like any other and can govern.
    """
    if load_n <= 0.0:
        raise ValueError(f"the load must be positive, got {load_n}")
    if length_m <= 0.0 or width_m <= 0.0:
        raise ValueError("length and width must be positive")
    if safety_factor < 1.0:
        raise ValueError(
            f"a safety factor below one is a request to fail, got "
            f"{safety_factor}")

    requirements = [ModeRequirement(
        mode=FailureMode.YIELD,
        height_m=height_for_yield_m(
            load_n, length_m, width_m,
            material.yield_strength_pa / safety_factor),
        basis=f"bending stress reaches yield / {safety_factor:g}")]

    if deflection_limit_m is not None:
        if deflection_limit_m <= 0.0:
            raise ValueError("the deflection limit must be positive")
        requirements.append(ModeRequirement(
            mode=FailureMode.DEFLECTION,
            height_m=height_for_deflection_m(
                load_n, length_m, width_m, material.youngs_modulus_pa,
                deflection_limit_m),
            basis=f"tip deflection reaches {deflection_limit_m * 1e3:g} mm"))

    if fully_reversed:
        # A fully reversed cycle has zero mean, so the Goodman and Soderberg
        # lines both reduce to the endurance limit itself. Dividing by the
        # safety factor is the same operation as for yield, applied to the
        # endurance strength instead.
        allowable = material.fatigue_strength_pa / safety_factor
        requirements.append(ModeRequirement(
            mode=FailureMode.FATIGUE,
            height_m=height_for_yield_m(load_n, length_m, width_m, allowable),
            basis=f"alternating bending stress reaches the endurance "
                  f"strength / {safety_factor:g}"))

    if minimum_height_m is not None:
        if minimum_height_m <= 0.0:
            raise ValueError("the manufacturing floor must be positive")
        requirements.append(ModeRequirement(
            mode=FailureMode.MANUFACTURING,
            height_m=minimum_height_m,
            basis="the thinnest section the process can hold"))

    governing = max(requirements, key=lambda r: r.height_m)
    height = governing.height_m
    return SectionSizing(
        height_m=height, governing=governing.mode,
        requirements=tuple(requirements), width_m=width_m, length_m=length_m,
        span_to_depth=length_m / height, aspect_ratio=height / width_m)


def tip_stress_pa(load_n: float, length_m: float, width_m: float,
                  height_m: float) -> float:
    """The forward check: 6 M / (b h^2)."""
    return 6.0 * load_n * length_m / (width_m * height_m ** 2)


def tip_deflection_m(load_n: float, length_m: float, width_m: float,
                     height_m: float, youngs_modulus_pa: float) -> float:
    """The forward check: P L^3 / (3 E I)."""
    inertia = width_m * height_m ** 3 / 12.0
    return load_n * length_m ** 3 / (3.0 * youngs_modulus_pa * inertia)


def stress_cycle_at(load_n: float, length_m: float, width_m: float,
                    height_m: float) -> StressCycle:
    """The fully reversed cycle a reversing tip load produces."""
    peak = tip_stress_pa(load_n, length_m, width_m, height_m)
    return StressCycle.fully_reversed(peak)
