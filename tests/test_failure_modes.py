"""Fatigue and buckling, and the designs they catch that a yield check does not.

The tests that justify the phase are the two governance ones: a design that
sits comfortably on the static check and is condemned by fatigue, and one that
is condemned by buckling. If those did not exist, adding the checks would have
changed no verdict.
"""

import math

import pytest

from core.design_genome.section import HollowRectangleSection
from core.materials import get_material
from core.registry import Category, ProblemContext, build_default_registry
from physics.buckling import (BucklingResult, EndCondition, analyze_column,
                              critical_slenderness, effective_length_factor,
                              euler_critical_load_n, radius_of_gyration_m,
                              slenderness_ratio)
from physics.failure_modes import DutyCycle, bending_stress_pa, check_design
from physics.fatigue import (REFERENCE_LIFE_CYCLES, MeanStressCriterion,
                             StressCycle, fatigue_safety_factor,
                             governing_failure_mode, has_endurance_limit)

ALUMINIUM = "al_7075_t6"
STEEL = "steel_s45c"


@pytest.fixture(scope="module")
def aluminium():
    return get_material(ALUMINIUM)


def tube(width=0.04, height=0.04, thickness=0.002) -> HollowRectangleSection:
    return HollowRectangleSection(outer_width_m=width, outer_height_m=height,
                                  wall_thickness_m=thickness)


# --- fatigue -----------------------------------------------------------------

def test_cycle_algebra():
    cycle = StressCycle(max_pa=120e6, min_pa=20e6)
    assert cycle.alternating_pa == pytest.approx(50e6)
    assert cycle.mean_pa == pytest.approx(70e6)
    assert cycle.range_pa == pytest.approx(100e6)
    assert cycle.ratio == pytest.approx(20.0 / 120.0)

    reversed_cycle = StressCycle.fully_reversed(80e6)
    assert reversed_cycle.mean_pa == pytest.approx(0.0)
    assert reversed_cycle.ratio == pytest.approx(-1.0)

    released = StressCycle.released(90e6)
    assert released.alternating_pa == pytest.approx(45e6)
    assert released.mean_pa == pytest.approx(45e6)
    assert released.ratio == pytest.approx(0.0)


def test_a_cycle_cannot_have_its_minimum_above_its_maximum():
    with pytest.raises(ValueError, match="exceeds"):
        StressCycle(max_pa=10e6, min_pa=20e6)


def test_goodman_matches_the_hand_calculation(aluminium):
    """n = 1 / (sigma_a/Se + sigma_m/Su), computed by hand from the datasheet.

    Se = 159 MPa, Su = 572 MPa, sigma_a = 50 MPa, sigma_m = 70 MPa:
        50/159 + 70/572 = 0.314465 + 0.122378 = 0.436843
        n = 1 / 0.436843 = 2.28914
    """
    result = fatigue_safety_factor(StressCycle(max_pa=120e6, min_pa=20e6),
                                   aluminium, MeanStressCriterion.GOODMAN)
    expected = 1.0 / (50e6 / 159e6 + 70e6 / 572e6)
    assert result.safety_factor == pytest.approx(expected, rel=1e-12)
    assert result.safety_factor == pytest.approx(2.28914, rel=1e-5)
    assert result.mean_stress_charged


def test_soderberg_is_more_conservative_than_goodman(aluminium):
    """It charges the mean against yield, which is below ultimate."""
    cycle = StressCycle(max_pa=120e6, min_pa=20e6)
    goodman = fatigue_safety_factor(cycle, aluminium,
                                    MeanStressCriterion.GOODMAN)
    soderberg = fatigue_safety_factor(cycle, aluminium,
                                      MeanStressCriterion.SODERBERG)
    assert soderberg.safety_factor < goodman.safety_factor
    expected = 1.0 / (50e6 / 159e6 + 70e6 / 503e6)
    assert soderberg.safety_factor == pytest.approx(expected, rel=1e-12)


def test_a_fully_reversed_cycle_is_the_endurance_ratio(aluminium):
    """With no mean stress the criterion reduces to Se / sigma_a."""
    result = fatigue_safety_factor(StressCycle.fully_reversed(80e6), aluminium)
    assert result.safety_factor == pytest.approx(159.0 / 80.0, rel=1e-12)


def test_a_compressive_mean_stress_is_not_credited(aluminium):
    """Extending the line into compression would predict the unsafe direction.

    A compressive mean stress genuinely helps, but neither Goodman nor
    Soderberg is derived for it, and evaluating them there returns a safety
    factor ABOVE the fully-reversed one. Being wrong in that direction ships
    overstressed parts, so the mean term is dropped instead.
    """
    amplitude = 80e6
    reversed_result = fatigue_safety_factor(
        StressCycle.fully_reversed(amplitude), aluminium)
    compressive = fatigue_safety_factor(
        StressCycle(max_pa=amplitude - 40e6, min_pa=-amplitude - 40e6),
        aluminium)
    assert compressive.mean_pa < 0
    assert not compressive.mean_stress_charged
    assert compressive.safety_factor == pytest.approx(
        reversed_result.safety_factor, rel=1e-12)


def test_aluminium_is_not_credited_with_infinite_life(aluminium):
    """Its S-N curve keeps falling, so passing does not mean unlimited life."""
    result = fatigue_safety_factor(StressCycle.fully_reversed(40e6), aluminium)
    assert result.safety_factor > 1.0
    assert not result.infinite_life
    assert result.reference_life_cycles == REFERENCE_LIFE_CYCLES
    assert "not a guarantee of infinite life" in result.notes


def test_carbon_steel_is_credited_with_an_endurance_limit():
    steel = get_material(STEEL)
    assert has_endurance_limit(steel)
    result = fatigue_safety_factor(StressCycle.fully_reversed(40e6), steel)
    assert result.infinite_life
    assert result.reference_life_cycles is None


def test_a_material_not_on_the_list_is_assumed_to_have_no_endurance_limit():
    """The conservative default: an unknown material cannot claim infinite life."""
    for material_id in ("ti_6al_4v", "ss_316", "mg_az31b", "pla"):
        assert not has_endurance_limit(get_material(material_id))


def test_a_constant_compressive_stress_is_not_a_fatigue_case(aluminium):
    """No variation and no tension: there is no cycle to check."""
    result = fatigue_safety_factor(StressCycle(max_pa=-50e6, min_pa=-50e6),
                                   aluminium)
    assert result.alternating_pa == 0.0
    assert result.safety_factor == math.inf
    assert "does not apply" in result.notes


def test_a_compression_to_zero_cycle_is_still_checked(aluminium):
    """It alternates, so it is a fatigue case even with no tensile peak.

    Cycling between 0 and -50 MPa has an alternating stress of 25 MPa. Treating
    "never in tension" as "no fatigue" would skip it, and compression-
    compression cycling still drives cracks. The mean is compressive, so it is
    not credited and the check is Se / sigma_a.
    """
    result = fatigue_safety_factor(StressCycle(max_pa=0.0, min_pa=-50e6),
                                   aluminium)
    assert result.alternating_pa == pytest.approx(25e6)
    assert result.mean_pa == pytest.approx(-25e6)
    assert not result.mean_stress_charged
    assert result.safety_factor == pytest.approx(159.0 / 25.0, rel=1e-12)


def test_governing_mode_picks_the_smaller_factor(aluminium):
    fatigue = fatigue_safety_factor(StressCycle.fully_reversed(180e6),
                                    aluminium)
    assert governing_failure_mode(2.8, fatigue) == "fatigue"
    assert governing_failure_mode(0.5, fatigue) == "static"


# --- buckling ----------------------------------------------------------------

def test_euler_load_matches_the_hand_calculation():
    """P_cr = pi^2 E I / (K L)^2 with E=200 GPa, I=1e-8 m^4, L=2 m, K=1.

        pi^2 * 200e9 * 1e-8 / 4 = 4934.8022 N
    """
    load = euler_critical_load_n(200e9, 1e-8, 2.0, 1.0)
    assert load == pytest.approx(math.pi ** 2 * 200e9 * 1e-8 / 4.0, rel=1e-15)
    assert load == pytest.approx(4934.8022, rel=1e-6)


def test_the_critical_load_scales_as_one_over_k_squared():
    reference = euler_critical_load_n(200e9, 1e-8, 2.0, 1.0)
    for condition, factor in ((EndCondition.FIXED_FREE, 2.0),
                              (EndCondition.PINNED_PINNED, 1.0),
                              (EndCondition.FIXED_PINNED, 0.699),
                              (EndCondition.FIXED_FIXED, 0.5)):
        assert effective_length_factor(condition) == factor
        load = euler_critical_load_n(200e9, 1e-8, 2.0, factor)
        assert load == pytest.approx(reference / factor ** 2, rel=1e-12)
    # A cantilever column carries a quarter of the pinned-pinned load.
    assert (euler_critical_load_n(200e9, 1e-8, 2.0, 2.0)
            == pytest.approx(reference / 4.0, rel=1e-12))


def test_radius_of_gyration_and_slenderness():
    assert radius_of_gyration_m(4e-8, 1e-4) == pytest.approx(0.02)
    assert slenderness_ratio(2.0, 2.0, 0.02) == pytest.approx(200.0)


def test_critical_slenderness_matches_its_definition():
    """C_c = sqrt(2 pi^2 E / Sy), where the Euler stress is half of yield."""
    e, yield_strength = 71.7e9, 503e6
    value = critical_slenderness(e, yield_strength)
    assert value == pytest.approx(
        math.sqrt(2.0 * math.pi ** 2 * e / yield_strength), rel=1e-15)
    # And at that slenderness the Euler stress is indeed Sy/2.
    euler_stress = math.pi ** 2 * e / value ** 2
    assert euler_stress == pytest.approx(yield_strength / 2.0, rel=1e-12)


def test_a_short_column_reports_that_euler_does_not_apply(aluminium):
    properties = tube().section_properties()
    result = analyze_column(
        aluminium.youngs_modulus_pa, aluminium.yield_strength_pa,
        properties.area_m2, min(properties.i_x_m4, properties.i_y_m4),
        length_m=0.2, applied_load_n=7000.0,
        condition=EndCondition.FIXED_FREE)
    assert result.slenderness < result.critical_slenderness
    assert not result.euler_valid
    assert "OVER-predicts" in result.notes


def test_a_member_in_tension_is_not_given_a_buckling_margin(aluminium):
    """An inapplicable check must not read as a well-designed column."""
    properties = tube().section_properties()
    result = analyze_column(
        aluminium.youngs_modulus_pa, aluminium.yield_strength_pa,
        properties.area_m2, properties.i_x_m4, length_m=1.5,
        applied_load_n=-5000.0)
    assert result.governing_mode == "not_in_compression"
    assert "not a margin" in result.notes


def test_the_weak_axis_decides():
    """Using the strong-axis second moment overstates the critical load."""
    section = tube(width=0.02, height=0.06, thickness=0.002)
    properties = section.section_properties()
    assert properties.i_y_m4 < properties.i_x_m4
    weak = euler_critical_load_n(71.7e9, properties.i_y_m4, 1.0, 1.0)
    strong = euler_critical_load_n(71.7e9, properties.i_x_m4, 1.0, 1.0)
    assert weak < strong


# --- the point of the phase --------------------------------------------------

def test_a_design_can_pass_the_static_check_and_fail_on_fatigue(aluminium):
    """The verdict a yield check alone would have got wrong.

    Fully reversed bending at 180 MPa in 7075-T6: yield is 503 MPa so the
    static factor is comfortable, while the endurance strength is 159 MPa and
    the part is condemned.
    """
    section = tube()
    length = 0.5
    load = section.section_properties().s_x_m3 * 180e6 / length
    duty = DutyCycle(bending_load_max_n=load, bending_load_min_n=-load)

    report = check_design(section, aluminium, length, duty)
    assert report.static_safety_factor > 2.5
    assert report.fatigue is not None
    assert report.fatigue.safety_factor < 1.0
    assert report.governing_mode == "fatigue"
    assert not report.passes


def test_a_design_can_pass_the_static_check_and_fail_on_buckling(aluminium):
    """A slender member in compression, nowhere near yielding."""
    section = tube()
    duty = DutyCycle(bending_load_max_n=0.0, compressive_load_n=7000.0,
                     end_condition=EndCondition.FIXED_FREE)
    report = check_design(section, aluminium, 1.5, duty)

    assert report.axial_safety_factor > 20.0     # yield is not close
    assert report.buckling is not None and report.buckling.euler_valid
    assert report.buckling.safety_factor < 1.0
    assert report.governing_mode == "buckling"
    assert not report.passes


def test_slenderness_decides_which_mode_governs(aluminium):
    """The same section and load, governed differently by length alone."""
    section = tube()
    duty = DutyCycle(bending_load_max_n=0.0, compressive_load_n=7000.0,
                     end_condition=EndCondition.FIXED_FREE)
    long_column = check_design(section, aluminium, 1.5, duty)
    short_column = check_design(section, aluminium, 0.2, duty)
    assert long_column.governing_mode == "buckling"
    assert short_column.governing_mode == "axial_yield"


def test_an_invalid_euler_result_is_never_the_governing_margin(aluminium):
    """It over-predicts, so letting it compete would hide the real limit."""
    section = tube()
    duty = DutyCycle(bending_load_max_n=0.0, compressive_load_n=7000.0,
                     end_condition=EndCondition.FIXED_FREE)
    report = check_design(section, aluminium, 0.2, duty)
    assert not report.buckling.euler_valid
    assert report.buckling.safety_factor > report.governing_safety_factor
    assert report.governing_mode != "buckling"


def test_checks_the_duty_does_not_make_possible_are_absent_not_huge(aluminium):
    """None, rather than a large number that reads as a good design."""
    section = tube()
    static_only = DutyCycle(bending_load_max_n=500.0, bending_load_min_n=500.0)
    report = check_design(section, aluminium, 0.5, static_only)
    assert not static_only.is_cyclic
    assert report.fatigue is None
    assert report.buckling is None
    assert report.axial_safety_factor is None
    assert report.governing_mode == "static"


def test_bending_stress_is_the_cantilever_formula():
    assert bending_stress_pa(200.0, 0.5, 4e-6) == pytest.approx(25e6)


# --- registry ----------------------------------------------------------------

def test_the_new_methods_are_gated_by_the_duty():
    """A static single load has no fatigue; a tension member cannot buckle."""
    registry = build_default_registry()
    static = ProblemContext(
        geometry="prismatic_beam", representations=("prismatic_beam",),
        slenderness=30.0, needs_stress_field=False,
        has_cyclic_load=False, has_compressive_load=False)
    candidates = registry.query(static, Category.ANALYSIS)
    assert "fatigue_sn" not in candidates.names()
    assert "buckling_euler" not in candidates.names()
    assert "repeated loading" in candidates.reason("fatigue_sn")[0]
    assert "compression" in candidates.reason("buckling_euler")[0]

    cyclic = ProblemContext(
        geometry="prismatic_beam", representations=("prismatic_beam",),
        slenderness=30.0, needs_stress_field=False,
        has_cyclic_load=True, has_compressive_load=True)
    names = registry.query(cyclic, Category.ANALYSIS).names()
    assert "fatigue_sn" in names and "buckling_euler" in names


def test_unimplemented_fem_buckling_is_not_registered():
    """A registry entry is a claim the method exists. It does not."""
    registry = build_default_registry()
    assert "buckling_fem" not in registry
    assert "buckling_linear_fem" not in registry
