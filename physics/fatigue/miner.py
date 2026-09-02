"""physics.fatigue.miner - cumulative damage under a spectrum of load blocks.

The stress-life check in `sn.py` answers whether one repeated cycle is safe
forever. Real duty is a mixture of cycles, and this answers what fraction of
the part's life a stated mixture consumes.

The validity of each step is stated where it is used, because Miner's rule is
the most confidently misapplied formula in fatigue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.materials.db import MaterialSpec

from .sn import (MeanStressCriterion, StressCycle, has_endurance_limit)

#: The high cycle S-N line is anchored at 1000 cycles, where the fatigue
#: strength is taken as this fraction of the ultimate. Below that the part is
#: in LOW cycle fatigue, where life is governed by plastic strain rather than
#: by stress, and a stress-life curve does not apply at all.
LOW_CYCLE_LIMIT = 1.0e3

#: [ASSUMED] the 0.9 factor at 1000 cycles. It is the usual bending value in
#: the machine design literature. For axial loading the customary figure is
#: 0.75, which gives a shorter life, so using the bending value on an axially
#: loaded part is unconservative. Nothing in this project measured it.
STRENGTH_FRACTION_AT_1000 = 0.9

#: Where the S-N line meets the endurance limit for materials that have one.
KNEE_CYCLES = 1.0e6

#: Miner predicts failure when the damage sum reaches one. Observed sums at
#: failure scatter roughly between 0.3 and 3 depending on the ORDER the blocks
#: are applied in, which the rule cannot see. A sum below one is therefore not
#: a guarantee, and this is recorded on every result rather than buried here.
DAMAGE_AT_FAILURE = 1.0


class LowCycleRegime(ValueError):
    """The stress implies a life below the stress-life curve's domain.

    Raised rather than extrapolated. Below about a thousand cycles the part is
    yielding every cycle and life is governed by plastic strain; a stress-life
    answer there is not conservative or unconservative, it is inapplicable.
    """


@dataclass(frozen=True)
class LoadBlock:
    """A number of repetitions of one stress cycle."""

    cycle: StressCycle
    cycles: float

    def __post_init__(self) -> None:
        if self.cycles < 0.0:
            raise ValueError(f"cycles must not be negative, got {self.cycles}")


@dataclass(frozen=True)
class BlockDamage:
    """What one block consumed."""

    alternating_pa: float
    mean_pa: float
    equivalent_alternating_pa: float
    cycles: float
    cycles_to_failure: float | None      # None means below the endurance limit
    damage: float


@dataclass(frozen=True)
class MinerResult:
    """The damage a spectrum consumes, with what it rests on."""

    damage: float
    blocks: tuple
    criterion: MeanStressCriterion
    material_has_endurance_limit: bool
    notes: str

    @property
    def survives(self) -> bool:
        return self.damage < DAMAGE_AT_FAILURE

    @property
    def remaining_fraction(self) -> float:
        return max(0.0, DAMAGE_AT_FAILURE - self.damage)


def equivalent_alternating_stress(
        cycle: StressCycle, material: MaterialSpec,
        criterion: MeanStressCriterion = MeanStressCriterion.GOODMAN) -> float:
    """The fully reversed stress that damages as fast as this cycle does.

    A mean stress is converted rather than ignored, by the same line the
    safety factor uses:

        sigma_ar = sigma_a / (1 - sigma_m / S)

    with S the ultimate for Goodman and the yield for Soderberg.

    A COMPRESSIVE mean stress is not credited, exactly as in the safety factor
    check: the lines are derived for tensile mean stress, and extending them to
    negative sigma_m would predict a longer life than the fully reversed case,
    which is the unsafe direction to be wrong in.
    """
    alternating = cycle.alternating_pa
    mean = cycle.mean_pa
    if mean <= 0.0:
        return alternating

    limit = (material.ultimate_strength_pa
             if criterion is MeanStressCriterion.GOODMAN
             else material.yield_strength_pa)
    if mean >= limit:
        return math.inf
    return alternating / (1.0 - mean / limit)


def cycles_to_failure(alternating_pa: float, material: MaterialSpec) -> float | None:
    """Life at a fully reversed stress, from the log-log S-N line.

    The line runs between 0.9 Su at 1000 cycles and the tabulated fatigue
    strength at the knee, so

        S = a N^b,   b = log10(S1/Se) / log10(1000 / N_knee)

    Returns None when the stress is at or below the endurance limit AND the
    material has one, meaning unlimited life.

    Raises LowCycleRegime above the 1000 cycle anchor, rather than
    extrapolating a stress-life curve into the plastic regime where it does
    not apply.
    """
    if alternating_pa <= 0.0:
        return None

    endurance = material.require_fatigue_strength_pa()
    if alternating_pa <= endurance:
        if has_endurance_limit(material):
            return None
        # No true endurance limit: the curve keeps falling, so a stress below
        # the tabulated value still spends life. Extrapolating the same line is
        # the honest continuation of the data, not a claim of infinite life.

    strength_at_1000 = STRENGTH_FRACTION_AT_1000 * material.ultimate_strength_pa
    if alternating_pa >= strength_at_1000:
        raise LowCycleRegime(
            f"an alternating stress of {alternating_pa:.3e} Pa is at or above "
            f"the {LOW_CYCLE_LIMIT:.0f} cycle anchor of "
            f"{strength_at_1000:.3e} Pa, which is 0.9 times the ultimate "
            f"strength. Below about a thousand cycles the part yields every "
            f"cycle and life is set by plastic strain, so a stress-life curve "
            f"does not apply. Use a strain-life method, which this project "
            f"does not implement.")

    exponent = (math.log10(strength_at_1000 / endurance)
                / math.log10(LOW_CYCLE_LIMIT / KNEE_CYCLES))
    coefficient = strength_at_1000 / (LOW_CYCLE_LIMIT ** exponent)
    return (alternating_pa / coefficient) ** (1.0 / exponent)


def cumulative_damage(
        blocks, material: MaterialSpec,
        criterion: MeanStressCriterion = MeanStressCriterion.GOODMAN
) -> MinerResult:
    """Miner's sum over a spectrum of blocks.

    D = sum(n_i / N_i), with failure predicted at D = 1.

    What this cannot see, stated on every result: Miner is INDEPENDENT OF
    ORDER. Running the high blocks first and the low blocks second gives the
    same answer as the reverse, and real parts do not behave that way. Observed
    damage sums at failure scatter roughly between 0.3 and 3 for that reason,
    so a sum below one is evidence rather than a guarantee.
    """
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("a damage sum over no blocks is not a result")

    details = []
    total = 0.0
    for block in blocks:
        equivalent = equivalent_alternating_stress(block.cycle, material,
                                                   criterion)
        life = (None if math.isinf(equivalent)
                else cycles_to_failure(equivalent, material))
        if math.isinf(equivalent):
            damage = math.inf
        elif life is None:
            damage = 0.0
        else:
            damage = block.cycles / life
        total += damage
        details.append(BlockDamage(
            alternating_pa=block.cycle.alternating_pa,
            mean_pa=block.cycle.mean_pa,
            equivalent_alternating_pa=equivalent,
            cycles=block.cycles, cycles_to_failure=life, damage=damage))

    limited = has_endurance_limit(material)
    notes = ("Miner's sum ignores the order the blocks are applied in, and "
             "order matters: observed sums at failure scatter roughly between "
             "0.3 and 3. A sum below one is evidence, not a guarantee.")
    if not limited:
        notes += (f" {material.id} has no endurance limit here, so stresses "
                  f"below the tabulated fatigue strength still consume life "
                  f"rather than being free.")
    return MinerResult(damage=total, blocks=tuple(details),
                       criterion=criterion,
                       material_has_endurance_limit=limited, notes=notes)
