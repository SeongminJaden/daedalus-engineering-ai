"""Cumulative damage under a spectrum, checked against hand calculation.

Miner's rule is easy to compute and easy to misapply, so the tests here are as
much about what it refuses and what it admits it cannot see as about the sum
itself.
"""

from __future__ import annotations

import math

import pytest

from core.materials.db import load_materials
from physics.fatigue import (DAMAGE_AT_FAILURE, KNEE_CYCLES, LOW_CYCLE_LIMIT,
                             STRENGTH_FRACTION_AT_1000, LoadBlock,
                             LowCycleRegime, MeanStressCriterion, StressCycle,
                             cumulative_damage, cycles_to_failure,
                             equivalent_alternating_stress,
                             has_endurance_limit)


@pytest.fixture(scope="module")
def steel():
    return load_materials().get("steel_s45c")


@pytest.fixture(scope="module")
def aluminium():
    db = load_materials()
    for identifier in db.ids():
        material = db.get(identifier)
        if not has_endurance_limit(material):
            return material
    pytest.skip("no material without an endurance limit in the database")


# ------------------------------------------------------------- the S-N line

def test_the_line_passes_through_both_anchors(steel):
    """It is built from two points, so it must return them.

    Trivial only until the exponent is derived with a sign error, which this
    would catch and a plausibility check on a middle value would not.
    """
    top = STRENGTH_FRACTION_AT_1000 * steel.ultimate_strength_pa
    assert cycles_to_failure(top * (1 - 1e-12), steel) == pytest.approx(
        LOW_CYCLE_LIMIT, rel=1e-6)

    just_above = steel.fatigue_strength_pa * (1 + 1e-9)
    assert cycles_to_failure(just_above, steel) == pytest.approx(
        KNEE_CYCLES, rel=1e-6)


def test_the_line_matches_a_hand_calculation(steel):
    """S = a N^b worked out independently of the implementation."""
    s1 = STRENGTH_FRACTION_AT_1000 * steel.ultimate_strength_pa
    se = steel.fatigue_strength_pa
    b = math.log10(s1 / se) / math.log10(LOW_CYCLE_LIMIT / KNEE_CYCLES)
    a = s1 / LOW_CYCLE_LIMIT ** b

    for stress in (300e6, 400e6, 500e6):
        expected = (stress / a) ** (1.0 / b)
        assert cycles_to_failure(stress, steel) == pytest.approx(expected,
                                                                 rel=1e-12)


def test_life_falls_as_stress_rises(steel):
    lives = [cycles_to_failure(s, steel) for s in (260e6, 300e6, 400e6, 500e6)]
    assert lives == sorted(lives, reverse=True)


def test_low_cycle_fatigue_is_refused_not_extrapolated(steel):
    """Below about a thousand cycles the part yields every cycle and life is
    set by plastic strain. A stress-life answer there is not conservative or
    unconservative, it is inapplicable."""
    with pytest.raises(LowCycleRegime, match="strain-life"):
        cycles_to_failure(0.95 * steel.ultimate_strength_pa, steel)


def test_a_steel_below_its_endurance_limit_lives_forever(steel):
    assert has_endurance_limit(steel)
    assert cycles_to_failure(steel.fatigue_strength_pa, steel) is None


def test_a_material_without_an_endurance_limit_keeps_spending_life(aluminium):
    """Its curve does not flatten, so a stress below the tabulated fatigue
    strength is not free. Returning infinite life there would be the unsafe
    direction to be wrong in."""
    assert not has_endurance_limit(aluminium)
    life = cycles_to_failure(0.8 * aluminium.fatigue_strength_pa, aluminium)
    assert life is not None and math.isfinite(life)


# ------------------------------------------------------ mean stress handling

def test_a_tensile_mean_stress_shortens_life(steel):
    reversed_cycle = StressCycle.fully_reversed(200e6)
    with_mean = StressCycle(max_pa=300e6, min_pa=-100e6)
    assert with_mean.alternating_pa == pytest.approx(200e6)
    assert with_mean.mean_pa == pytest.approx(100e6)

    plain = equivalent_alternating_stress(reversed_cycle, steel)
    charged = equivalent_alternating_stress(with_mean, steel)
    assert charged > plain
    expected = 200e6 / (1.0 - 100e6 / steel.ultimate_strength_pa)
    assert charged == pytest.approx(expected, rel=1e-12)


def test_soderberg_is_harsher_than_goodman(steel):
    cycle = StressCycle(max_pa=300e6, min_pa=-100e6)
    goodman = equivalent_alternating_stress(cycle, steel,
                                            MeanStressCriterion.GOODMAN)
    soderberg = equivalent_alternating_stress(cycle, steel,
                                              MeanStressCriterion.SODERBERG)
    assert soderberg > goodman


def test_a_compressive_mean_stress_earns_no_credit(steel):
    """The lines are derived for tensile mean stress. Extending them to
    negative mean would predict a LONGER life than fully reversed, which is
    the unsafe direction."""
    compressive = StressCycle(max_pa=100e6, min_pa=-300e6)
    assert compressive.mean_pa < 0
    assert equivalent_alternating_stress(compressive, steel) == pytest.approx(
        compressive.alternating_pa)


# ------------------------------------------------------------ the damage sum

def test_two_half_life_blocks_sum_to_failure(steel):
    """The definition of the rule, arranged so the answer is exactly one."""
    stress = 350e6
    life = cycles_to_failure(stress, steel)
    blocks = [LoadBlock(StressCycle.fully_reversed(stress), life / 2),
              LoadBlock(StressCycle.fully_reversed(stress), life / 2)]
    result = cumulative_damage(blocks, steel)
    assert result.damage == pytest.approx(DAMAGE_AT_FAILURE, rel=1e-12)
    assert not result.survives


def test_damage_adds_across_different_stresses(steel):
    high = 400e6
    low = 300e6
    n_high, n_low = 1000.0, 20000.0
    result = cumulative_damage(
        [LoadBlock(StressCycle.fully_reversed(high), n_high),
         LoadBlock(StressCycle.fully_reversed(low), n_low)], steel)
    expected = (n_high / cycles_to_failure(high, steel)
                + n_low / cycles_to_failure(low, steel))
    assert result.damage == pytest.approx(expected, rel=1e-12)
    assert len(result.blocks) == 2


def test_blocks_below_the_endurance_limit_consume_nothing_for_steel(steel):
    result = cumulative_damage(
        [LoadBlock(StressCycle.fully_reversed(steel.fatigue_strength_pa * 0.9),
                   1e9)], steel)
    assert result.damage == 0.0
    assert result.survives
    assert result.blocks[0].cycles_to_failure is None


def test_the_order_independence_is_stated_on_the_result(steel):
    """Miner cannot see load ORDER, and real parts can.

    Reversing the blocks must give an identical answer, which is precisely the
    limitation, so the result says so rather than leaving the caller to assume
    the sum is a guarantee.
    """
    blocks = [LoadBlock(StressCycle.fully_reversed(400e6), 500.0),
              LoadBlock(StressCycle.fully_reversed(300e6), 5000.0)]
    forward = cumulative_damage(blocks, steel)
    backward = cumulative_damage(list(reversed(blocks)), steel)
    assert forward.damage == pytest.approx(backward.damage, rel=1e-15)
    assert "order" in forward.notes.lower()
    assert "guarantee" in forward.notes.lower()


def test_a_sum_over_no_blocks_is_refused(steel):
    with pytest.raises(ValueError, match="no blocks"):
        cumulative_damage([], steel)


def test_negative_repetitions_are_refused(steel):
    with pytest.raises(ValueError, match="must not be negative"):
        LoadBlock(StressCycle.fully_reversed(300e6), -1.0)


def test_a_material_without_a_limit_says_so_in_the_notes(aluminium):
    result = cumulative_damage(
        [LoadBlock(StressCycle.fully_reversed(
            aluminium.fatigue_strength_pa * 0.5), 1e5)], aluminium)
    assert not result.material_has_endurance_limit
    assert "no endurance limit" in result.notes
    assert result.damage > 0.0


def test_the_capability_refuses_a_single_repeated_cycle():
    """A spectrum method should not be selected for a problem that is one
    repeated cycle: the stress-life check answers that more directly, and
    Miner adds only its own order-blindness on top."""
    from core.registry.context import ProblemContext
    from nodes.roster import build_roster

    method = next(c.method for c in build_roster().all()
                  if c.name == "fatigue_cumulative_damage")
    single = ProblemContext(has_cyclic_load=True)
    reasons = " ".join(str(r) for r in method.applicability(single).failed)
    assert "has_duty_cycle" in reasons

    spectrum = ProblemContext(has_cyclic_load=True, has_duty_cycle=True)
    assert not list(method.applicability(spectrum).failed)


def test_the_registry_note_records_the_bending_assumption():
    """0.9 of the ultimate at a thousand cycles is the BENDING figure. The
    axial value is 0.75, so using this on an axially loaded part is
    unconservative, and that has to be findable from the registry."""
    from nodes.roster import build_roster

    method = next(c.method for c in build_roster().all()
                  if c.name == "fatigue_cumulative_damage")
    assert "BENDING" in method.notes
    assert "0.75" in method.notes
    assert "unconservative" in method.notes
