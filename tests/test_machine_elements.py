"""Keys, splines, press fits, welds and ISO 286 limits and fits.

The test that carries the ISO work is
`test_the_tolerance_formula_agrees_with_the_published_table_only_approximately`,
which measures the disagreement rather than asserting agreement. The formula is
close enough to compare fits and not close enough to put on a drawing, and that
distinction is the useful output.
"""

import math

import pytest

from core.registry import ProblemContext, build_default_registry
from physics.elements import (FRICTION_PRESSED_STEEL,
                              MAX_EFFECTIVE_LENGTH_RATIO, MIN_NOMINAL_MM,
                              FitType, analyze_fillet_weld, analyze_key,
                              analyze_press_fit, contact_pressure_pa,
                              effective_length_m, fit, hole_limits,
                              hub_hoop_stress_pa, it_tolerance_um,
                              shaft_limits, spline_torque_capacity_nm,
                              standard_key_section, throat_thickness_m,
                              tolerance_unit_um)

STEEL_E, STEEL_NU, STEEL_YIELD = 207e9, 0.29, 655e6

# The published ISO 286 IT values in micrometres, by size range upper bound.
PUBLISHED_IT = {
    6: {6: 8, 7: 12, 8: 18, 9: 30}, 10: {6: 9, 7: 15, 8: 22, 9: 36},
    18: {6: 11, 7: 18, 8: 27, 9: 43}, 30: {6: 13, 7: 21, 8: 33, 9: 52},
    50: {6: 16, 7: 25, 8: 39, 9: 62}, 80: {6: 19, 7: 30, 8: 46, 9: 74},
    120: {6: 22, 7: 35, 8: 54, 9: 87}, 180: {6: 25, 7: 40, 8: 63, 9: 100},
    250: {6: 29, 7: 46, 8: 72, 9: 115}, 315: {6: 32, 7: 52, 8: 81, 9: 130},
    400: {6: 36, 7: 57, 8: 89, 9: 140}, 500: {6: 40, 7: 63, 8: 97, 9: 155},
}


# --- ISO 286 -----------------------------------------------------------------

def test_the_tolerance_unit_is_the_published_expression():
    """i = 0.45 cbrt(D) + 0.001 D at the range's geometric mean."""
    mean = math.sqrt(30.0 * 50.0)
    assert tolerance_unit_um(50.0) == pytest.approx(
        0.45 * mean ** (1 / 3) + 0.001 * mean, rel=1e-12)
    # Every size inside a range shares the range's tolerance.
    assert tolerance_unit_um(31.0) == pytest.approx(tolerance_unit_um(50.0))


def test_the_tolerance_formula_agrees_with_the_published_table_only_approximately():
    """Measured, not asserted, because the standard rounds and the formula does not.

    Being precise about HOW close matters more than claiming closeness: this is
    good enough to compare two fits and not good enough to specify one.
    """
    errors = []
    for size, grades in PUBLISHED_IT.items():
        for grade, published in grades.items():
            errors.append(abs(it_tolerance_um(size, grade) - published)
                          / published)
    mean = sum(errors) / len(errors)
    assert mean < 0.02, f"mean deviation {mean:.3%} is worse than measured"
    assert max(errors) < 0.09, f"worst deviation {max(errors):.3%}"
    # And it is NOT exact, which is the point of measuring.
    exact = sum(1 for size, grades in PUBLISHED_IT.items()
                for grade, published in grades.items()
                if round(it_tolerance_um(size, grade)) == published)
    assert exact < len(errors), "the formula is not expected to be exact"


def test_grades_grow_in_the_expected_progression():
    """Each grade is about 1.6 times the one below, the R5 series."""
    for grade in range(6, 16):
        ratio = it_tolerance_um(50.0, grade + 1) / it_tolerance_um(50.0, grade)
        assert 1.5 < ratio < 1.7


def test_small_sizes_are_refused_rather_than_extrapolated():
    """The expression does not describe the tabulated values below 3 mm."""
    with pytest.raises(ValueError, match="at or below"):
        it_tolerance_um(MIN_NOMINAL_MM, 7)
    with pytest.raises(ValueError, match="at or below"):
        it_tolerance_um(2.0, 7)


def test_large_sizes_and_fine_grades_are_refused():
    with pytest.raises(ValueError, match="above"):
        it_tolerance_um(600.0, 7)
    with pytest.raises(ValueError, match="IT4|outside"):
        it_tolerance_um(50.0, 4)


def test_the_hole_basis_zone_starts_at_nominal():
    """H is defined by a zero lower deviation, which is what makes it the basis."""
    hole = hole_limits(50.0, 7)
    assert hole.lower_mm == 0.0
    assert hole.min_mm == 50.0
    assert hole.width_mm == pytest.approx(it_tolerance_um(50.0, 7) / 1000.0)


def test_the_fit_types_come_out_as_expected():
    """g clears, n interferes at the tight end, h touches at zero."""
    assert fit(50.0, 7, "g", 6).fit_type is FitType.CLEARANCE
    assert fit(50.0, 7, "h", 6).min_clearance_mm == pytest.approx(0.0)
    assert fit(50.0, 7, "n", 6).fit_type is FitType.TRANSITION
    loose, tight = fit(50.0, 7, "g", 6), fit(50.0, 7, "n", 6)
    assert loose.min_clearance_mm > tight.min_clearance_mm


def test_tabulated_deviations_are_refused_by_name():
    """p, r, s and u have no closed form, and guessing would be a wrong drawing."""
    for letter in ("p", "s", "u", "m"):
        with pytest.raises(ValueError, match="tabulated increment"):
            shaft_limits(50.0, 6, letter)


def test_non_h_holes_are_refused():
    with pytest.raises(ValueError, match="not implemented"):
        hole_limits(50.0, 7, "G")


# --- keys and splines --------------------------------------------------------

def test_standard_key_sections_follow_the_published_series():
    assert standard_key_section(0.030) == (0.008, 0.007)
    assert standard_key_section(0.020) == (0.006, 0.006)
    assert standard_key_section(0.045) == (0.014, 0.009)
    with pytest.raises(ValueError, match="starts above"):
        standard_key_section(0.005)


def test_key_stresses_match_the_hand_calculation():
    """F = 2T/d, tau = F/(bL), sigma = F/((h/2)L)."""
    diameter, length, torque = 0.030, 0.040, 150.0
    width, height = standard_key_section(diameter)
    result = analyze_key(diameter, length, torque, 90e6, 180e6)
    force = 2.0 * torque / diameter
    assert result.tangential_force_n == pytest.approx(force)
    assert result.shear_stress_pa == pytest.approx(force / (width * length))
    assert result.bearing_stress_pa == pytest.approx(
        force / (0.5 * height * length))


def test_only_half_the_key_height_bears():
    """Using the full height would double the apparent bearing capacity."""
    diameter, length, torque = 0.030, 0.040, 150.0
    _, height = standard_key_section(diameter)
    result = analyze_key(diameter, length, torque, 90e6, 180e6)
    force = 2.0 * torque / diameter
    assert result.bearing_stress_pa == pytest.approx(force / (0.5 * height * length))
    assert result.bearing_stress_pa > force / (height * length)


def test_extra_key_length_stops_being_credited():
    """The load concentrates at the ends, so a very long key is not stronger.

    A five times longer key must not give five times the capacity.
    """
    diameter, torque = 0.030, 150.0
    short = analyze_key(diameter, 0.040, torque, 90e6, 180e6)
    very_long = analyze_key(diameter, 0.200, torque, 90e6, 180e6)
    assert very_long.length_was_capped
    assert not short.length_was_capped
    assert very_long.effective_length_m == pytest.approx(
        MAX_EFFECTIVE_LENGTH_RATIO * diameter)
    assert very_long.safety_factor < 2.0 * short.safety_factor
    assert effective_length_m(0.2, 0.03) == pytest.approx(0.045)


def test_a_spline_carries_far_more_than_one_key():
    """Which is the reason to use one."""
    capacity = spline_torque_capacity_nm(0.030, 12, 0.002, 0.040, 180e6)
    key = analyze_key(0.030, 0.040, 1.0, 90e6, 180e6)
    key_capacity = 1.0 * key.bearing_safety_factor
    assert capacity > key_capacity


def test_spline_load_sharing_must_be_a_fraction():
    with pytest.raises(ValueError, match="fraction"):
        spline_torque_capacity_nm(0.030, 12, 0.002, 0.040, 180e6,
                                  load_sharing=1.5)


# --- press fit ---------------------------------------------------------------

def test_the_general_lame_form_reduces_to_the_same_material_closed_form():
    """An independent cross-check of the general expression.

    For a solid shaft and identical materials the whole bracket collapses to
    p = delta E (D^2 - d^2) / (2 d D^2). Two derivations agreeing is the check.
    """
    diameter, outer, interference = 0.030, 0.060, 30e-6
    general = contact_pressure_pa(interference, diameter, outer, STEEL_E,
                                  STEEL_NU, STEEL_E, STEEL_NU)
    closed = (interference * STEEL_E * (outer ** 2 - diameter ** 2)
              / (2.0 * diameter * outer ** 2))
    assert general == pytest.approx(closed, rel=1e-12)


def test_capacity_and_hoop_stress_match_the_hand_calculation():
    diameter, outer, length, interference = 0.030, 0.060, 0.040, 30e-6
    result = analyze_press_fit(interference, diameter, outer, length, STEEL_E,
                               STEEL_NU, STEEL_E, STEEL_NU,
                               hub_yield_pa=STEEL_YIELD)
    pressure = result.contact_pressure_pa
    assert result.torque_capacity_nm == pytest.approx(
        FRICTION_PRESSED_STEEL * pressure * math.pi * diameter ** 2 * length / 2.0)
    assert result.axial_capacity_n == pytest.approx(
        FRICTION_PRESSED_STEEL * pressure * math.pi * diameter * length)
    assert result.hub_hoop_stress_pa == pytest.approx(
        hub_hoop_stress_pa(pressure, diameter, outer))


def test_the_hub_hoop_stress_always_exceeds_the_contact_pressure():
    """Which is why a thin hub splits before the pressure looks alarming."""
    for outer in (0.035, 0.060, 0.120):
        result = analyze_press_fit(30e-6, 0.030, outer, 0.040, STEEL_E,
                                   STEEL_NU, STEEL_E, STEEL_NU)
        assert result.hub_hoop_stress_pa > result.contact_pressure_pa
    # And a thinner hub is worse off, not better.
    thin = analyze_press_fit(30e-6, 0.030, 0.035, 0.040, STEEL_E, STEEL_NU,
                             STEEL_E, STEEL_NU, hub_yield_pa=STEEL_YIELD)
    thick = analyze_press_fit(30e-6, 0.030, 0.120, 0.040, STEEL_E, STEEL_NU,
                              STEEL_E, STEEL_NU, hub_yield_pa=STEEL_YIELD)
    assert thin.hub_yield_safety_factor < thick.hub_yield_safety_factor


def test_a_thin_hub_can_yield_before_it_grips():
    """And the pressure it reports then is no longer trustworthy.

    The Lame relation is elastic. Once the bore yields the pressure stops
    following the interference, so the torque capacity computed from it is
    fiction. Reporting the hoop stress is what lets a caller notice.

    The numbers matter: a 30 mm shaft in a 32 mm hub at 80 micrometres does NOT
    yield (hoop 519 MPa against 655), which was this test's first guess. It
    takes 120 micrometres, which is already an extreme interference at this
    diameter.
    """
    safe = analyze_press_fit(80e-6, 0.030, 0.032, 0.040, STEEL_E, STEEL_NU,
                             STEEL_E, STEEL_NU, hub_yield_pa=STEEL_YIELD)
    assert not safe.hub_yields

    excessive = analyze_press_fit(120e-6, 0.030, 0.032, 0.040, STEEL_E,
                                  STEEL_NU, STEEL_E, STEEL_NU,
                                  hub_yield_pa=STEEL_YIELD)
    assert excessive.hub_yields
    assert excessive.hub_hoop_stress_pa > STEEL_YIELD


def test_torque_capacity_is_proportional_to_friction():
    """Which is why an uncertain coefficient makes an uncertain capacity."""
    low = analyze_press_fit(30e-6, 0.030, 0.060, 0.040, STEEL_E, STEEL_NU,
                            STEEL_E, STEEL_NU, friction=0.08)
    high = analyze_press_fit(30e-6, 0.030, 0.060, 0.040, STEEL_E, STEEL_NU,
                             STEEL_E, STEEL_NU, friction=0.16)
    assert high.torque_capacity_nm == pytest.approx(
        2.0 * low.torque_capacity_nm)


def test_a_hollow_shaft_grips_less_than_a_solid_one():
    """It deflects inward under the same interference, relieving the pressure."""
    solid = contact_pressure_pa(30e-6, 0.030, 0.060, STEEL_E, STEEL_NU,
                                STEEL_E, STEEL_NU)
    hollow = contact_pressure_pa(30e-6, 0.030, 0.060, STEEL_E, STEEL_NU,
                                 STEEL_E, STEEL_NU, shaft_bore_m=0.020)
    assert hollow < solid


def test_impossible_press_fit_geometry_is_refused():
    with pytest.raises(ValueError, match="must exceed"):
        contact_pressure_pa(30e-6, 0.060, 0.030, STEEL_E, STEEL_NU, STEEL_E,
                            STEEL_NU)
    with pytest.raises(ValueError, match="bore"):
        contact_pressure_pa(30e-6, 0.030, 0.060, STEEL_E, STEEL_NU, STEEL_E,
                            STEEL_NU, shaft_bore_m=0.030)


# --- welds -------------------------------------------------------------------

def test_the_throat_is_the_leg_over_root_two():
    assert throat_thickness_m(0.006) == pytest.approx(0.006 / math.sqrt(2.0))


def test_weld_stress_matches_the_hand_calculation():
    result = analyze_fillet_weld(50000.0, 0.006, 0.080, 120e6)
    throat = 0.006 / math.sqrt(2.0)
    assert result.stress_pa == pytest.approx(50000.0 / (throat * 0.080))
    assert result.safety_factor == pytest.approx(120e6 / result.stress_pa)
    assert not result.passes


def test_a_larger_leg_helps_linearly():
    small = analyze_fillet_weld(50000.0, 0.006, 0.080, 120e6)
    large = analyze_fillet_weld(50000.0, 0.012, 0.080, 120e6)
    assert large.safety_factor == pytest.approx(2.0 * small.safety_factor)


# --- registry ----------------------------------------------------------------

def test_the_new_methods_are_gated():
    registry = build_default_registry()
    none = ProblemContext(geometry="assembly", representations=("assembly",),
                          has_shaft_hub_connection=False,
                          has_welded_joint=False, requires_tolerances=False)
    candidates = registry.query(none)
    for name in ("key_joint", "press_fit", "fillet_weld", "iso_fit"):
        assert name not in candidates.names()
        assert candidates.reason(name)

    present = ProblemContext(geometry="assembly",
                             representations=("assembly",),
                             has_shaft_hub_connection=True,
                             has_welded_joint=True, requires_tolerances=True)
    names = registry.query(present).names()
    for name in ("key_joint", "press_fit", "fillet_weld", "iso_fit"):
        assert name in names


def test_unimplemented_element_methods_are_not_registered():
    """Nothing may be registered before it exists.

    thread_stripping was on this list and has been removed from it, because it
    is now implemented in physics.joints.threads and verified against the
    reason a standard nut is 0.8 diameters tall. Removing a name from here is
    the ONLY way a capability should become registered: the guard failed first
    and was updated deliberately, rather than the list being kept vague enough
    never to fail.
    """
    registry = build_default_registry()
    for absent in ("weld_fatigue", "gdt_tolerance_stackup", "fretting"):
        assert absent not in registry


def test_thread_stripping_is_registered_and_has_an_implementation():
    """The other half of the guard above.

    Deleting a name from the absent list must not be enough on its own; the
    method has to resolve to code that actually exists.
    """
    import importlib

    registry = build_default_registry()
    assert "thread_stripping" in registry

    method = registry.get("thread_stripping")
    module_path, _, function = method.implementation.rpartition(".")
    module = importlib.import_module(module_path)
    assert callable(getattr(module, function))
