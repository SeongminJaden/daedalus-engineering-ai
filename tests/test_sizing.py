"""Minimum sizing, checked by forward substitution.

Every check elsewhere asks whether a design is safe. This asks how small it
can be, so the verification is the reverse: put the answer back through the
forward formulas and require the governing mode to sit EXACTLY at its limit
while every other mode still has margin. A sizing routine that returned
something merely safe would pass a safety check and fail this.
"""

from __future__ import annotations

import math

import pytest

from core.materials.db import load_materials
from physics.sizing import FailureMode, size_rectangular_cantilever
from physics.sizing.cantilever import (LATERAL_BUCKLING_ASPECT,
                                       SLENDER_SPAN_TO_DEPTH,
                                       height_for_deflection_m,
                                       height_for_yield_m, tip_deflection_m,
                                       tip_stress_pa)

LOAD = 500.0
LENGTH = 0.3
WIDTH = 0.02


@pytest.fixture(scope="module")
def steel():
    return load_materials().get("steel_s45c")


# --------------------------------------------------- each inverse on its own

def test_the_yield_inverse_matches_its_closed_form(steel):
    """h = sqrt(6 M / (b sigma)), worked independently."""
    allowable = steel.yield_strength_pa / 1.5
    expected = math.sqrt(6.0 * LOAD * LENGTH / (WIDTH * allowable))
    assert height_for_yield_m(LOAD, LENGTH, WIDTH, allowable) == pytest.approx(
        expected, rel=1e-12)


def test_the_deflection_inverse_matches_its_closed_form(steel):
    """h = (4 P L^3 / (E b delta))^(1/3), worked independently."""
    limit = 1e-3
    expected = (4.0 * LOAD * LENGTH ** 3
                / (steel.youngs_modulus_pa * WIDTH * limit)) ** (1.0 / 3.0)
    assert height_for_deflection_m(
        LOAD, LENGTH, WIDTH, steel.youngs_modulus_pa, limit) == pytest.approx(
        expected, rel=1e-12)


def test_each_inverse_is_the_exact_undo_of_its_forward_formula(steel):
    """The point of inverting in closed form rather than searching."""
    allowable = steel.yield_strength_pa / 1.5
    height = height_for_yield_m(LOAD, LENGTH, WIDTH, allowable)
    assert tip_stress_pa(LOAD, LENGTH, WIDTH, height) == pytest.approx(
        allowable, rel=1e-12)

    limit = 1e-3
    height = height_for_deflection_m(LOAD, LENGTH, WIDTH,
                                     steel.youngs_modulus_pa, limit)
    assert tip_deflection_m(LOAD, LENGTH, WIDTH, height,
                            steel.youngs_modulus_pa) == pytest.approx(
        limit, rel=1e-12)


# ------------------------------------------------- the composition, forwards

def test_the_governing_mode_is_exactly_tight_and_others_have_margin(steel):
    """The central check.

    At the returned height the binding constraint must be met exactly, to
    round off, and every other mode must be strictly satisfied. Any larger
    height would be safe but not minimal; any smaller would violate something.
    """
    result = size_rectangular_cantilever(
        LOAD, LENGTH, WIDTH, steel, safety_factor=1.5,
        deflection_limit_m=1e-3, fully_reversed=True, minimum_height_m=0.002)

    assert result.governing is FailureMode.DEFLECTION
    assert tip_deflection_m(LOAD, LENGTH, WIDTH, result.height_m,
                            steel.youngs_modulus_pa) == pytest.approx(
        1e-3, rel=1e-12)

    stress = tip_stress_pa(LOAD, LENGTH, WIDTH, result.height_m)
    assert stress < steel.yield_strength_pa / 1.5
    assert stress < steel.fatigue_strength_pa / 1.5
    assert result.height_m > 0.002


def test_the_answer_is_the_largest_requirement_not_the_sum(steel):
    """Modes are alternatives, not contributions. Taking anything other than
    the maximum would either oversize or violate a mode."""
    result = size_rectangular_cantilever(
        LOAD, LENGTH, WIDTH, steel, deflection_limit_m=1e-3,
        fully_reversed=True, minimum_height_m=0.002)
    assert result.height_m == pytest.approx(
        max(r.height_m for r in result.requirements), rel=1e-15)


def test_tightening_the_deflection_limit_changes_what_governs(steel):
    """A sizing routine that always reported the same governing mode would
    pass a single case and be useless."""
    loose = size_rectangular_cantilever(
        LOAD, LENGTH, WIDTH, steel, deflection_limit_m=0.05)
    tight = size_rectangular_cantilever(
        LOAD, LENGTH, WIDTH, steel, deflection_limit_m=1e-4)
    assert loose.governing is FailureMode.YIELD
    assert tight.governing is FailureMode.DEFLECTION
    assert tight.height_m > loose.height_m


def test_a_manufacturing_floor_can_govern(steel):
    """It is a mode like any other. A tiny load sized purely on stress would
    return something no process can make."""
    result = size_rectangular_cantilever(
        1.0, 0.05, 0.02, steel, minimum_height_m=0.003)
    assert result.governing is FailureMode.MANUFACTURING
    assert result.height_m == pytest.approx(0.003)


def test_fatigue_is_not_included_unless_the_load_is_stated_to_reverse(steel):
    """A reversing load and a steady one are the same number.

    Inferring would either invent a fatigue requirement that is not there or,
    worse, silently drop one that is.
    """
    steady = size_rectangular_cantilever(LOAD, LENGTH, WIDTH, steel)
    reversing = size_rectangular_cantilever(LOAD, LENGTH, WIDTH, steel,
                                            fully_reversed=True)
    assert FailureMode.FATIGUE not in {r.mode for r in steady.requirements}
    assert FailureMode.FATIGUE in {r.mode for r in reversing.requirements}
    assert reversing.height_m > steady.height_m


def test_fatigue_governs_over_yield_when_the_load_reverses(steel):
    """The endurance strength is below the yield strength, so a reversing load
    needs more section than a steady one of the same magnitude. A result that
    said otherwise would be the dangerous direction."""
    result = size_rectangular_cantilever(LOAD, LENGTH, WIDTH, steel,
                                         fully_reversed=True)
    assert steel.fatigue_strength_pa < steel.yield_strength_pa
    assert result.governing is FailureMode.FATIGUE
    assert (result.required_for(FailureMode.FATIGUE)
            > result.required_for(FailureMode.YIELD))


def test_a_heavier_load_never_needs_less_section(steel):
    heights = [size_rectangular_cantilever(p, LENGTH, WIDTH, steel,
                                           deflection_limit_m=1e-3).height_m
               for p in (100.0, 300.0, 900.0, 2700.0)]
    assert heights == sorted(heights)


# ------------------------------------------------- what it admits it skipped

def test_a_stubby_beam_says_the_answer_is_optimistic(steel):
    """Euler-Bernoulli neglects shear deflection, which makes a short deep
    beam deflect MORE than computed. The warning has to be visible."""
    result = size_rectangular_cantilever(
        20000.0, 0.05, 0.02, steel, deflection_limit_m=1e-5)
    assert result.span_to_depth < SLENDER_SPAN_TO_DEPTH
    assert not result.is_slender
    assert any("optimistic" in w for w in result.warnings)


def test_a_tall_thin_section_flags_the_buckling_it_does_not_check(steel):
    """Lateral torsional buckling can govern before yield, and this function
    does not check it. Silence there would be the unsafe kind."""
    result = size_rectangular_cantilever(4000.0, 0.5, 0.004, steel)
    assert result.aspect_ratio > LATERAL_BUCKLING_ASPECT
    assert result.may_buckle_laterally
    assert any("buckling" in w for w in result.warnings)


def test_a_slender_well_proportioned_beam_warns_about_nothing(steel):
    result = size_rectangular_cantilever(
        200.0, 0.4, 0.03, steel, deflection_limit_m=2e-3)
    assert result.is_slender
    assert not result.may_buckle_laterally
    assert result.warnings == ()


# ---------------------------------------------------------------- refusals

def test_a_safety_factor_below_one_is_refused(steel):
    with pytest.raises(ValueError, match="request to fail"):
        size_rectangular_cantilever(LOAD, LENGTH, WIDTH, steel,
                                    safety_factor=0.9)


@pytest.mark.parametrize("bad", [0.0, -100.0])
def test_a_non_positive_load_is_refused(steel, bad):
    with pytest.raises(ValueError, match="load must be positive"):
        size_rectangular_cantilever(bad, LENGTH, WIDTH, steel)


def test_a_non_positive_geometry_is_refused(steel):
    with pytest.raises(ValueError, match="must be positive"):
        size_rectangular_cantilever(LOAD, LENGTH, 0.0, steel)


def test_a_non_positive_deflection_limit_is_refused(steel):
    with pytest.raises(ValueError, match="deflection limit must be positive"):
        size_rectangular_cantilever(LOAD, LENGTH, WIDTH, steel,
                                    deflection_limit_m=0.0)


def test_asking_for_a_mode_that_was_not_evaluated_raises(steel):
    result = size_rectangular_cantilever(LOAD, LENGTH, WIDTH, steel)
    with pytest.raises(KeyError, match="was not evaluated"):
        result.required_for(FailureMode.DEFLECTION)


def test_the_capability_refuses_without_a_load_case():
    """A minimum size with no load is not a question.

    Stated rather than inferred from the presence of a geometry, because a
    context can carry a shape with no load attached to it.
    """
    from core.registry.context import ProblemContext
    from nodes.roster import build_roster

    method = next(c.method for c in build_roster().all()
                  if c.name == "minimum_sizing")
    reasons = " ".join(str(r) for r
                       in method.applicability(ProblemContext()).failed)
    assert "has_load_case" in reasons
    assert not list(method.applicability(
        ProblemContext(has_load_case=True)).failed)


def test_the_registry_note_records_what_is_not_checked():
    """Lateral torsional buckling and shear deflection are both omitted, and
    an omission that is not findable from the registry is a trap."""
    from nodes.roster import build_roster

    method = next(c.method for c in build_roster().all()
                  if c.name == "minimum_sizing")
    assert "lateral torsional buckling" in method.notes
    assert "shear deflection" in method.notes
    assert "optimistic" in method.notes
