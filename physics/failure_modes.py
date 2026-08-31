"""Running every failure check a duty makes possible, and reporting the worst.

Until now a design passed if its peak stress was under yield and its deflection
under the limit. That is a statement about one load applied once. A part that is
loaded and unloaded a million times, or that carries compression in a slender
member, has failure modes those two checks cannot see, and reporting only them
calls such a design safe.

This runs the checks the duty makes applicable and reports the governing one,
which is the smallest safety factor rather than the one that happens to be
familiar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.design_genome.section import HollowRectangleSection
from core.materials.db import MaterialSpec
from physics.buckling.euler import (BucklingResult, EndCondition,
                                    analyze_column)
from physics.fatigue.sn import (FatigueResult, MeanStressCriterion,
                                StressCycle, fatigue_safety_factor)


@dataclass(frozen=True)
class DutyCycle:
    """How a design is actually used, as opposed to the one load it was sized for.

    `bending_load_max_n` and `bending_load_min_n` are the extremes of the
    transverse load over the operating cycle, signed, so a fully reversed duty
    is (+P, -P) and a lift-and-lower duty is (+P, 0).

    `compressive_load_n` is a separate axial compression, positive in
    compression. Zero means the member carries no axial compression and the
    buckling check does not apply.
    """

    bending_load_max_n: float
    bending_load_min_n: float = 0.0
    compressive_load_n: float = 0.0
    end_condition: EndCondition = EndCondition.FIXED_FREE
    cycles: float | None = None

    @property
    def is_cyclic(self) -> bool:
        """Whether the load actually varies. A constant load has no cycle."""
        return self.bending_load_max_n != self.bending_load_min_n

    @property
    def in_compression(self) -> bool:
        return self.compressive_load_n > 0.0


@dataclass(frozen=True)
class FailureModeReport:
    """Every applicable check, and which one governs."""

    static_safety_factor: float
    max_bending_stress_pa: float
    axial_safety_factor: float | None
    axial_stress_pa: float | None
    fatigue: FatigueResult | None
    buckling: BucklingResult | None
    governing_mode: str
    governing_safety_factor: float

    @property
    def passes(self) -> bool:
        return self.governing_safety_factor >= 1.0

    def summary(self) -> str:
        return (f"{self.governing_mode} governs at safety factor "
                f"{self.governing_safety_factor:.3f}")


def bending_stress_pa(load_n: float, length_m: float,
                      section_modulus_m3: float) -> float:
    """Root bending stress of a tip-loaded cantilever: sigma = P L / S."""
    if section_modulus_m3 <= 0.0:
        raise ValueError("section modulus must be positive")
    return load_n * length_m / section_modulus_m3


def check_design(section: HollowRectangleSection, material: MaterialSpec,
                 length_m: float, duty: DutyCycle,
                 criterion: MeanStressCriterion = MeanStressCriterion.GOODMAN,
                 ) -> FailureModeReport:
    """Run static, fatigue and buckling as the duty makes them applicable.

    A check that the duty does not make possible is left as None rather than
    returned with an enormous safety factor. An inapplicable check reporting a
    large number reads as a well-designed part, when in fact nothing was
    examined.
    """
    if not section.is_valid():
        raise ValueError(f"invalid section: {section.validity_reason()}")
    properties = section.section_properties()

    peak_load = max(abs(duty.bending_load_max_n), abs(duty.bending_load_min_n))
    peak_stress = bending_stress_pa(peak_load, length_m, properties.s_x_m3)
    static = (math.inf if peak_stress <= 0.0
              else material.yield_strength_pa / peak_stress)

    # Axial compression has to be checked against yield in its own right. Left
    # out, a member with no transverse load reported an infinite static safety
    # factor while carrying compression, and the only number left to govern was
    # a buckling factor that the short-column case had already invalidated.
    axial_stress: float | None = None
    axial: float | None = None
    if duty.in_compression:
        axial_stress = duty.compressive_load_n / properties.area_m2
        axial = material.yield_strength_pa / axial_stress

    fatigue: FatigueResult | None = None
    if duty.is_cyclic:
        cycle = StressCycle(
            max_pa=bending_stress_pa(duty.bending_load_max_n, length_m,
                                     properties.s_x_m3),
            min_pa=bending_stress_pa(duty.bending_load_min_n, length_m,
                                     properties.s_x_m3))
        fatigue = fatigue_safety_factor(cycle, material, criterion)

    buckling: BucklingResult | None = None
    if duty.in_compression:
        # A column buckles about its weak axis, so the smaller second moment is
        # the one that decides.
        buckling = analyze_column(
            youngs_modulus_pa=material.youngs_modulus_pa,
            yield_strength_pa=material.yield_strength_pa,
            area_m2=properties.area_m2,
            min_second_moment_m4=min(properties.i_x_m4, properties.i_y_m4),
            length_m=length_m, applied_load_n=duty.compressive_load_n,
            condition=duty.end_condition)

    candidates = [("static", static)]
    if axial is not None:
        candidates.append(("axial_yield", axial))
    if fatigue is not None:
        candidates.append(("fatigue", fatigue.safety_factor))
    # Only a VALID Euler result may govern. Below the critical slenderness the
    # elastic derivation over-predicts, so letting that number compete would
    # let an over-estimate present itself as the binding margin, which is the
    # exact failure the validity flag exists to prevent. The number is still
    # reported on the result, with its warning.
    if (buckling is not None
            and buckling.governing_mode != "not_in_compression"
            and buckling.euler_valid):
        candidates.append(("buckling", buckling.safety_factor))
    mode, safety = min(candidates, key=lambda pair: pair[1])

    return FailureModeReport(
        static_safety_factor=static, max_bending_stress_pa=peak_stress,
        axial_safety_factor=axial, axial_stress_pa=axial_stress,
        fatigue=fatigue, buckling=buckling, governing_mode=mode,
        governing_safety_factor=safety)
