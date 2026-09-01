"""Thread stripping, checked against ISO geometry and the standard nut height.

The useful anchor here is not a tolerance. A standard ISO hex nut is about
0.8 d tall BECAUSE the standard intends the bolt to break before the threads
strip. So a calculation that says a same grade steel nut needs less than 0.8 d
is agreeing with a decision made by the standard, independently of this code.
"""

from __future__ import annotations

import math

import pytest

from physics.joints.bolted import (BOLT_GRADES, NOMINAL_DIAMETER_M,
                                   THREAD_STRESS_AREA_M2, PropertyClass)
from physics.joints.threads import (FLANK_COTANGENT, ISO_COARSE_PITCH_M,
                                    SHEAR_STRENGTH_FRACTION,
                                    STANDARD_NUT_HEIGHT_RATIO,
                                    bolt_shear_area_per_length_m,
                                    minor_diameter_m,
                                    nut_shear_area_per_length_m, pitch_m,
                                    pitch_diameter_m,
                                    required_engagement_length)

SIZES = ("M6", "M8", "M10", "M12")


# ------------------------------------------------------------- the geometry

@pytest.mark.parametrize("size", SIZES)
def test_the_basic_profile_diameters_follow_iso(size):
    """d2 = d - 0.6495 P and d1 = d - 1.0825 P, from ISO 68-1.

    These are definitions of the thread, so they are checked as arithmetic
    rather than approximated.
    """
    d = NOMINAL_DIAMETER_M[size]
    p = pitch_m(size)
    assert pitch_diameter_m(size) == pytest.approx(d - 0.6495 * p, rel=1e-12)
    assert minor_diameter_m(size) == pytest.approx(d - 1.0825 * p, rel=1e-12)
    assert minor_diameter_m(size) < pitch_diameter_m(size) < d


def test_the_flank_factor_is_the_cotangent_of_sixty_degrees():
    """It comes from the 60 degree thread form, not from a fit."""
    assert FLANK_COTANGENT == pytest.approx(1.0 / math.tan(math.radians(60.0)),
                                            rel=1e-12)


@pytest.mark.parametrize("size", SIZES)
def test_the_nut_shears_on_a_larger_cylinder_than_the_bolt(size):
    """The nut's threads shear at the bolt's major diameter and the bolt's at
    the nut's minor diameter, so for equal materials the bolt governs. This is
    why a nut is not simply made of the same stuff and left short."""
    assert nut_shear_area_per_length_m(size) > bolt_shear_area_per_length_m(size)


# ------------------------------------------------- the engagement it implies

@pytest.mark.parametrize("size", SIZES)
def test_a_same_grade_steel_nut_is_shorter_than_the_standard_nut(size):
    """The independent anchor.

    A standard hex nut is about 0.8 d tall, and the standard chose that so the
    bolt breaks first. A required engagement below it agrees with that
    decision without using it as an input.
    """
    result = required_engagement_length(size, PropertyClass.C8_8)
    assert result.governing == "bolt_thread"
    assert result.required_engagement_diameters < STANDARD_NUT_HEIGHT_RATIO


@pytest.mark.parametrize("size", SIZES)
def test_a_tapped_hole_in_aluminium_needs_more_and_the_nut_governs(size):
    """Where stripping actually bites in practice.

    With a soft member the nut side becomes the weaker one despite its larger
    shear cylinder, and the required engagement rises past the standard nut
    height, which is the reason a steel nut height is not a general rule.
    """
    steel = required_engagement_length(size, PropertyClass.C8_8)
    soft = required_engagement_length(size, PropertyClass.C8_8,
                                      nut_ultimate_strength_pa=310e6)
    assert soft.governing == "nut_thread"
    assert soft.required_engagement_m > steel.required_engagement_m
    assert soft.required_engagement_diameters > STANDARD_NUT_HEIGHT_RATIO


def test_the_required_length_matches_a_hand_calculation():
    """Worked independently for M10 class 8.8."""
    size, grade = "M10", PropertyClass.C8_8
    capacity = THREAD_STRESS_AREA_M2[size] * BOLT_GRADES[grade].ultimate_strength_pa
    area_per_length = bolt_shear_area_per_length_m(size)
    shear = SHEAR_STRENGTH_FRACTION * BOLT_GRADES[grade].ultimate_strength_pa
    expected = capacity / (area_per_length * shear)

    result = required_engagement_length(size, grade)
    assert result.bolt_thread_length_m == pytest.approx(expected, rel=1e-12)
    assert result.bolt_tensile_capacity_n == pytest.approx(capacity, rel=1e-12)


def test_the_requirement_scales_with_diameter():
    """Threads are geometrically similar, so the answer in DIAMETERS should be
    nearly constant. A result that drifted strongly with size would mean the
    areas were not scaling as areas."""
    ratios = [required_engagement_length(s, PropertyClass.C8_8)
              .required_engagement_diameters for s in SIZES]
    assert max(ratios) - min(ratios) < 0.05


def test_a_stronger_bolt_in_the_same_nut_material_needs_more_thread():
    """A 12.9 bolt can pull harder, so the same tapped hole must hold more."""
    weak = required_engagement_length("M10", PropertyClass.C8_8,
                                      nut_ultimate_strength_pa=310e6)
    strong = required_engagement_length("M10", PropertyClass.C12_9,
                                        nut_ultimate_strength_pa=310e6)
    assert strong.required_engagement_m > weak.required_engagement_m


@pytest.mark.parametrize("size", SIZES)
def test_it_reports_whether_a_given_engagement_strips(size):
    result = required_engagement_length(size, PropertyClass.C8_8)
    assert result.strips_before_breaking(result.required_engagement_m * 0.5)
    assert not result.strips_before_breaking(result.required_engagement_m * 2.0)


# ---------------------------------------------------------------- refusals

def test_an_unknown_size_is_refused():
    with pytest.raises(KeyError, match="unknown thread size"):
        required_engagement_length("M42", PropertyClass.C8_8)


def test_a_nut_material_with_no_strength_is_refused():
    with pytest.raises(ValueError, match="positive ultimate strength"):
        required_engagement_length("M10", PropertyClass.C8_8,
                                   nut_ultimate_strength_pa=0.0)


def test_the_shear_fraction_is_not_the_most_optimistic_choice():
    """A higher fraction predicts a SHORTER engagement, which is the unsafe
    direction. 0.6 sits above Tresca's 0.5 and above von Mises' 0.577, so this
    asserts it has not drifted upward."""
    assert 0.5 <= SHEAR_STRENGTH_FRACTION <= 0.62


def test_every_tabulated_size_has_a_pitch():
    """A size with a stress area but no pitch would raise deep inside the
    geometry rather than at the boundary."""
    assert set(ISO_COARSE_PITCH_M) == set(NOMINAL_DIAMETER_M)
    assert set(ISO_COARSE_PITCH_M) == set(THREAD_STRESS_AREA_M2)
