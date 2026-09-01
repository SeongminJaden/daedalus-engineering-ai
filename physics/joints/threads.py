"""physics.joints.threads - thread stripping and the engagement length for it.

A bolted joint has two ways to come apart in tension: the bolt breaks, or the
threads strip. The first is the designed one, because a broken bolt is
obvious and a stripped thread often is not. This computes the engagement
length at which the two capacities are equal, which is the length below which
stripping governs.

The geometry here is not assumed. ISO 68-1 fixes the basic profile of an ISO
metric thread exactly, so the pitch diameter, the minor diameter and the
shear cylinders all follow from the nominal diameter and the pitch. The
0.57735 appearing below is cot(60 degrees), from the 60 degree flank angle,
not a fitted constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .bolted import (NOMINAL_DIAMETER_M, THREAD_STRESS_AREA_M2, BOLT_GRADES,
                     PropertyClass)

#: Pitch of ISO metric COARSE threads, metres. Standard values from ISO 261,
#: in the same category as the tensile stress areas already tabulated: they
#: define the thread rather than describing a measurement.
ISO_COARSE_PITCH_M: dict[str, float] = {
    "M3": 0.0005, "M4": 0.0007, "M5": 0.0008, "M6": 0.0010,
    "M8": 0.00125, "M10": 0.0015, "M12": 0.00175,
}

#: cot(60 degrees). The flank half angle of an ISO metric thread is 30
#: degrees from the axis, and this factor converts the radial overlap between
#: the two members into the axial extent of engaged material.
FLANK_COTANGENT = 1.0 / math.sqrt(3.0)

#: [ASSUMED] shear strength as a fraction of ultimate tensile strength.
#: 0.6 is the usual figure for steel. The von Mises value is 0.577 and the
#: Tresca value is 0.5; using a higher fraction predicts a SHORTER required
#: engagement, which is the unsafe direction, so this is the one number here
#: that is not geometry and it is deliberately not the most optimistic.
SHEAR_STRENGTH_FRACTION = 0.6

#: A standard ISO hex nut is about 0.8 times the nominal diameter tall. It is
#: not a safety margin; it is the height at which the standard expects the
#: bolt to break before the threads strip.
STANDARD_NUT_HEIGHT_RATIO = 0.8


def pitch_m(size: str) -> float:
    try:
        return ISO_COARSE_PITCH_M[size]
    except KeyError:
        raise KeyError(
            f"unknown thread size {size!r}. Known: "
            f"{', '.join(ISO_COARSE_PITCH_M)}") from None


def pitch_diameter_m(size: str) -> float:
    """d2 = d - 0.6495 P, from the ISO 68-1 basic profile."""
    return NOMINAL_DIAMETER_M[size] - 0.6495 * pitch_m(size)


def minor_diameter_m(size: str) -> float:
    """d1 = d - 1.0825 P, the basic minor diameter."""
    return NOMINAL_DIAMETER_M[size] - 1.0825 * pitch_m(size)


def bolt_shear_area_per_length_m(size: str) -> float:
    """Area of BOLT thread sheared per unit engagement length.

    The bolt's threads shear on the cylinder at the nut's minor diameter, and
    the engaged fraction of each pitch is one half plus the flank overlap:

        A/L = pi d1 [ 1/2 + cot(60) (d2 - d1) / P ]
    """
    d2 = pitch_diameter_m(size)
    d1 = minor_diameter_m(size)
    return math.pi * d1 * (0.5 + FLANK_COTANGENT * (d2 - d1) / pitch_m(size))


def nut_shear_area_per_length_m(size: str) -> float:
    """Area of NUT thread sheared per unit engagement length.

    The nut's threads shear on the larger cylinder at the bolt's major
    diameter, which is why a nut of the same material as the bolt is not the
    member that governs:

        A/L = pi d [ 1/2 + cot(60) (d - d2) / P ]
    """
    d = NOMINAL_DIAMETER_M[size]
    d2 = pitch_diameter_m(size)
    return math.pi * d * (0.5 + FLANK_COTANGENT * (d - d2) / pitch_m(size))


@dataclass(frozen=True)
class StrippingResult:
    """Where a joint fails in tension, and the length that decides it."""

    size: str
    bolt_tensile_capacity_n: float
    required_engagement_m: float
    governing: str                    # "bolt_thread" or "nut_thread"
    bolt_thread_length_m: float
    nut_thread_length_m: float

    @property
    def required_engagement_diameters(self) -> float:
        return self.required_engagement_m / NOMINAL_DIAMETER_M[self.size]

    def strips_before_breaking(self, engagement_m: float) -> bool:
        return engagement_m < self.required_engagement_m


def required_engagement_length(
        size: str, grade: PropertyClass,
        nut_ultimate_strength_pa: float | None = None) -> StrippingResult:
    """The engagement at which stripping and breaking are equally likely.

    NO MARGIN is included. This is the length at which the two capacities are
    equal, so at exactly this length the joint is indifferent between the two
    failures. Design practice adds margin on top, and the usual rules of thumb
    already contain it.

    `nut_ultimate_strength_pa` defaults to the bolt's own ultimate, which is
    the case of a nut of the same grade. Passing a softer material is the
    interesting case, since a tapped hole in aluminium is where stripping
    usually governs.
    """
    if size not in ISO_COARSE_PITCH_M:
        raise KeyError(f"unknown thread size {size!r}")

    bolt = BOLT_GRADES[grade]
    capacity = THREAD_STRESS_AREA_M2[size] * bolt.ultimate_strength_pa

    bolt_shear = SHEAR_STRENGTH_FRACTION * bolt.ultimate_strength_pa
    nut_ultimate = (bolt.ultimate_strength_pa if nut_ultimate_strength_pa
                    is None else nut_ultimate_strength_pa)
    if nut_ultimate <= 0.0:
        raise ValueError("the nut material needs a positive ultimate strength")
    nut_shear = SHEAR_STRENGTH_FRACTION * nut_ultimate

    bolt_length = capacity / (bolt_shear_area_per_length_m(size) * bolt_shear)
    nut_length = capacity / (nut_shear_area_per_length_m(size) * nut_shear)

    if nut_length >= bolt_length:
        governing, required = "nut_thread", nut_length
    else:
        governing, required = "bolt_thread", bolt_length

    return StrippingResult(
        size=size, bolt_tensile_capacity_n=capacity,
        required_engagement_m=required, governing=governing,
        bolt_thread_length_m=bolt_length, nut_thread_length_m=nut_length)
