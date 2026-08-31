"""High-cycle fatigue: stress-life with a mean-stress correction.

A part that survives its static load can still fail under repetition. The
static check compares a peak stress to yield; this one compares the ALTERNATING
stress to the material's fatigue strength, corrected for the mean stress the
cycle sits on. They are different questions and they can disagree, which is the
reason this exists: a design that passes the yield check can be governed by
fatigue.

**What this is not.** It is stress-life (S-N) and therefore high-cycle only.
Low-cycle fatigue, where plastic strain dominates and a strain-life model is
needed, is not covered and this module will not tell you it is out of its
depth: the caller has to know that a few thousand cycles at near-yield stress
is not what S-N describes. There is no notch factor, no surface finish factor,
no size factor and no temperature correction; a real endurance limit is the
handbook value multiplied by all of those, each below 1. The numbers here are
therefore optimistic against a real part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from core.materials.db import MaterialSpec


class MeanStressCriterion(str, Enum):
    """How the mean stress is charged against the alternating stress."""

    GOODMAN = "goodman"        # against ultimate strength
    SODERBERG = "soderberg"    # against yield strength, more conservative


# Materials whose S-N curve genuinely flattens, so that a stress below the knee
# implies unlimited life. Carbon and low-alloy steels do. Aluminium, magnesium
# and the polymers here do NOT: their curve keeps falling, so a "fatigue
# strength" is a stress at a stated life and nothing more.
#
# [ASSUMED] the membership of this set. It reflects the usual materials-science
# position rather than a measurement in this project, and two entries are
# deliberately left out of it despite being metals often described as having a
# limit: austenitic stainless (ss_316) and titanium (ti_6al_4v) are both
# contested in the literature, so they are treated as having none. An id that
# is not listed is treated as having no endurance limit, which is the
# conservative direction: an unknown material cannot silently claim infinite
# life.
ENDURANCE_LIMIT_MATERIALS = frozenset({"steel_s45c", "steel_scm440"})

# The life the tabulated fatigue strength refers to for materials without a
# true endurance limit. Stated in data/materials.yaml as an R.R. Moore rotating
# beam value at about this many cycles.
REFERENCE_LIFE_CYCLES = 5.0e8


def has_endurance_limit(material: MaterialSpec) -> bool:
    return material.id in ENDURANCE_LIMIT_MATERIALS


@dataclass(frozen=True)
class StressCycle:
    """One fully-described stress cycle. Tension positive, SI units."""

    max_pa: float
    min_pa: float

    def __post_init__(self) -> None:
        if self.min_pa > self.max_pa:
            raise ValueError(
                f"cycle minimum {self.min_pa:.6g} Pa exceeds its maximum "
                f"{self.max_pa:.6g} Pa")

    @property
    def alternating_pa(self) -> float:
        """sigma_a = (sigma_max - sigma_min) / 2. Never negative."""
        return 0.5 * (self.max_pa - self.min_pa)

    @property
    def mean_pa(self) -> float:
        """sigma_m = (sigma_max + sigma_min) / 2. Sign carries meaning."""
        return 0.5 * (self.max_pa + self.min_pa)

    @property
    def range_pa(self) -> float:
        return self.max_pa - self.min_pa

    @property
    def ratio(self) -> float:
        """R = sigma_min / sigma_max. Fully reversed is -1, released is 0."""
        if self.max_pa == 0.0:
            return math.nan
        return self.min_pa / self.max_pa

    @classmethod
    def fully_reversed(cls, amplitude_pa: float) -> "StressCycle":
        return cls(max_pa=amplitude_pa, min_pa=-amplitude_pa)

    @classmethod
    def released(cls, peak_pa: float) -> "StressCycle":
        """Zero to peak and back, R = 0. A joint that loads and unloads."""
        return cls(max_pa=peak_pa, min_pa=0.0)


@dataclass(frozen=True)
class FatigueResult:
    """A fatigue verdict, with everything needed to argue with it."""

    safety_factor: float
    alternating_pa: float
    mean_pa: float
    endurance_pa: float
    criterion: MeanStressCriterion
    infinite_life: bool
    reference_life_cycles: float | None
    mean_stress_charged: bool
    notes: str = ""

    @property
    def passes(self) -> bool:
        return self.safety_factor >= 1.0


def fatigue_safety_factor(
        cycle: StressCycle, material: MaterialSpec,
        criterion: MeanStressCriterion = MeanStressCriterion.GOODMAN,
) -> FatigueResult:
    """Safety factor against high-cycle fatigue, by the chosen criterion.

    Goodman:    sigma_a / Se + sigma_m / Su = 1 / n
    Soderberg:  sigma_a / Se + sigma_m / Sy = 1 / n

    so n is the factor by which the whole cycle can be scaled before the design
    point reaches the failure line.

    **A compressive mean stress is not credited.** Both lines are derived for
    tensile mean stress, and extending them to negative sigma_m would predict a
    safety factor above the fully-reversed one, which is the unsafe direction to
    be wrong in. When sigma_m <= 0 the mean term is dropped and the result is
    the fully-reversed check, Se / sigma_a. `mean_stress_charged` records which
    happened.
    """
    alternating = cycle.alternating_pa
    mean = cycle.mean_pa
    endurance = material.fatigue_strength_pa

    if alternating <= 0.0 and mean <= 0.0:
        return FatigueResult(
            safety_factor=math.inf, alternating_pa=alternating, mean_pa=mean,
            endurance_pa=endurance, criterion=criterion,
            infinite_life=has_endurance_limit(material),
            reference_life_cycles=_reference_life(material),
            mean_stress_charged=False,
            notes="no tensile stress in the cycle; fatigue does not apply")

    limit = (material.ultimate_strength_pa
             if criterion is MeanStressCriterion.GOODMAN
             else material.yield_strength_pa)

    if mean > 0.0:
        damage = alternating / endurance + mean / limit
        charged = True
    else:
        # Compressive mean: the fully-reversed check, no credit taken.
        damage = alternating / endurance
        charged = False

    safety = math.inf if damage <= 0.0 else 1.0 / damage
    return FatigueResult(
        safety_factor=safety, alternating_pa=alternating, mean_pa=mean,
        endurance_pa=endurance, criterion=criterion,
        infinite_life=has_endurance_limit(material),
        reference_life_cycles=_reference_life(material),
        mean_stress_charged=charged,
        notes=_life_note(material))


def _reference_life(material: MaterialSpec) -> float | None:
    return None if has_endurance_limit(material) else REFERENCE_LIFE_CYCLES


def _life_note(material: MaterialSpec) -> str:
    """Say what surviving this check actually buys."""
    if has_endurance_limit(material):
        return (f"{material.id} has a true endurance limit, so a stress below "
                f"it implies unlimited life under this model")
    return (f"{material.id} has no true endurance limit: its fatigue strength "
            f"is a stress at about {REFERENCE_LIFE_CYCLES:.0e} cycles, not a "
            f"guarantee of infinite life. Longer service needs a lower stress.")


def governing_failure_mode(static_safety_factor: float,
                           fatigue: FatigueResult) -> str:
    """Which check is closer to failing: 'static' or 'fatigue'.

    The whole point of running both. A design tuned against yield alone can sit
    comfortably on the static check and be governed by fatigue, and reporting
    only the static number would call that design safe.
    """
    return "fatigue" if fatigue.safety_factor < static_safety_factor else "static"
