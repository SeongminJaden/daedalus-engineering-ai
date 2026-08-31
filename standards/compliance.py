"""Checking a design's dimensions against published standards.

An "ISO compliant design" here means two separate things and they should not be
confused. The first is that every value drawn from a standard IS in that
standard: a thread size that exists, a bolt grade that is defined, a tolerance
grade in range. The second is that free dimensions sit on preferred numbers so
the thing can be built from stock. This module reports both, separately.

VALIDITY, before the implementation:

* **Conformance is not correctness, and this is the important limit.** A design
  can use only standard threads, only defined bolt grades, only preferred
  dimensions, and be entirely unfit for its duty. Nothing here evaluates
  whether the design works. It evaluates whether its dimensions are ones a
  supplier recognises.

* **The catalogue of what is standard is only as complete as what has been
  implemented.** A dimension reported as NON-STANDARD may simply be one this
  project has not tabulated. The report distinguishes "not in the standard"
  from "not checkable here", because those call for different responses.

* **Snapping is a proposal, not an edit.** The report says what a dimension
  would become; applying it is the caller's decision, because moving a
  dimension can invalidate the analysis that produced it and only the caller
  knows which direction is conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from physics.elements.fits import MAX_NOMINAL_MM, MIN_NOMINAL_MM
from physics.elements.keys import _KEY_SECTIONS
from physics.joints.bolted import BOLT_GRADES, THREAD_STRESS_AREA_M2

from .renard import Series, SnapDirection, snap_with_report


class Conformance(str, Enum):
    STANDARD = "standard"
    NON_STANDARD = "non_standard"
    # The value may be perfectly standard and this project has no table for it.
    NOT_CHECKABLE = "not_checkable"


@dataclass(frozen=True)
class DimensionCheck:
    """One dimension, its conformance, and what it would snap to."""

    name: str
    value: float
    conformance: Conformance
    standard: str
    detail: str = ""
    suggested: float | None = None
    relative_change: float | None = None


def check_thread(name: str, designation: str) -> DimensionCheck:
    """Whether a thread designation is in the ISO metric coarse series here."""
    known = designation in THREAD_STRESS_AREA_M2
    return DimensionCheck(
        name=name, value=float("nan"),
        conformance=Conformance.STANDARD if known else Conformance.NOT_CHECKABLE,
        standard="ISO 261 / ISO 262 metric coarse",
        detail=("in the tabulated series"
                if known else
                f"{designation} is not in this project's table, which holds "
                f"{', '.join(sorted(THREAD_STRESS_AREA_M2))}. It may still be "
                f"a standard size"))


def check_bolt_grade(name: str, grade: str) -> DimensionCheck:
    """Whether a property class is defined in ISO 898-1 as implemented here."""
    known = any(grade == g.value for g in BOLT_GRADES)
    return DimensionCheck(
        name=name, value=float("nan"),
        conformance=Conformance.STANDARD if known else Conformance.NOT_CHECKABLE,
        standard="ISO 898-1 property class",
        detail=("defined" if known else
                f"class {grade} is not implemented here; "
                f"{', '.join(g.value for g in BOLT_GRADES)} are"))


def check_fit_size(name: str, nominal_mm: float) -> DimensionCheck:
    """Whether a nominal size is inside the ISO 286 expression's valid range."""
    if nominal_mm <= MIN_NOMINAL_MM:
        return DimensionCheck(
            name=name, value=nominal_mm, conformance=Conformance.NOT_CHECKABLE,
            standard="ISO 286 limits and fits",
            detail=f"{nominal_mm:g} mm is at or below {MIN_NOMINAL_MM:g} mm, "
                   f"where the tolerance expression does not describe the "
                   f"tabulated values")
    if nominal_mm > MAX_NOMINAL_MM:
        return DimensionCheck(
            name=name, value=nominal_mm, conformance=Conformance.NOT_CHECKABLE,
            standard="ISO 286 limits and fits",
            detail=f"{nominal_mm:g} mm is above {MAX_NOMINAL_MM:g} mm, where a "
                   f"different expression applies")
    return DimensionCheck(
        name=name, value=nominal_mm, conformance=Conformance.STANDARD,
        standard="ISO 286 limits and fits",
        detail="inside the range where tolerances can be computed")


def check_key_shaft(name: str, shaft_diameter_mm: float) -> DimensionCheck:
    """Whether a shaft diameter has a tabulated standard key section."""
    lowest, highest = _KEY_SECTIONS[0][0], _KEY_SECTIONS[-1][0]
    if 6.0 < shaft_diameter_mm <= highest:
        return DimensionCheck(
            name=name, value=shaft_diameter_mm,
            conformance=Conformance.STANDARD,
            standard="DIN 6885 / ISO 773 parallel keys",
            detail="a standard key section exists for this shaft")
    return DimensionCheck(
        name=name, value=shaft_diameter_mm,
        conformance=Conformance.NOT_CHECKABLE,
        standard="DIN 6885 / ISO 773 parallel keys",
        detail=f"outside the tabulated range of {lowest:g} to {highest:g} mm")


def check_preferred(name: str, value: float, series: Series = Series.R20,
                    direction: SnapDirection = SnapDirection.UP,
                    tolerance: float = 1e-9) -> DimensionCheck:
    """Whether a free dimension already sits on a preferred number.

    `direction` defaults to UP because most free dimensions in this project are
    load-bearing, where growing is the safe way to be wrong. A caller snapping a
    clearance must say DOWN.
    """
    report = snap_with_report(value, series, SnapDirection.NEAREST)
    if abs(report.relative_change) <= tolerance:
        return DimensionCheck(
            name=name, value=value, conformance=Conformance.STANDARD,
            standard=f"ISO 3 preferred numbers, {series.value}",
            detail="already a preferred value")
    proposed = snap_with_report(value, series, direction)
    return DimensionCheck(
        name=name, value=value, conformance=Conformance.NON_STANDARD,
        standard=f"ISO 3 preferred numbers, {series.value}",
        detail=f"not on the {series.value} series; nearest is "
               f"{report.snapped:g}",
        suggested=proposed.snapped,
        relative_change=proposed.relative_change)


@dataclass
class ComplianceReport:
    """Every dimension checked, split by outcome."""

    checks: list[DimensionCheck] = field(default_factory=list)

    def add(self, check: DimensionCheck) -> None:
        self.checks.append(check)

    def by(self, conformance: Conformance) -> list[DimensionCheck]:
        return [c for c in self.checks if c.conformance is conformance]

    @property
    def non_standard(self) -> list[DimensionCheck]:
        return self.by(Conformance.NON_STANDARD)

    @property
    def not_checkable(self) -> list[DimensionCheck]:
        return self.by(Conformance.NOT_CHECKABLE)

    @property
    def fully_conformant(self) -> bool:
        """True only when nothing is non-standard AND nothing is unverifiable.

        An unverifiable dimension is not a pass. It may be perfectly standard,
        and this project cannot say so, which is a different statement from
        having checked it.
        """
        return not self.non_standard and not self.not_checkable

    def summary(self) -> str:
        return (f"{len(self.by(Conformance.STANDARD))} standard, "
                f"{len(self.non_standard)} non-standard, "
                f"{len(self.not_checkable)} not checkable")
