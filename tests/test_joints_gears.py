"""Bolted joints and gear teeth.

Two tests carry the phase. `test_an_under_preloaded_joint_separates_and_the_bolt_takes_everything`
pins the regime change that makes bolted joints fail: while clamped a bolt sees
only its stiffness share of an external load, and once separated it sees all of
it. `test_module_decides_which_gear_mode_governs` pins that bending and pitting
are genuinely different checks that bind on different designs.
"""

import math

import pytest

from core.registry import Category, ProblemContext, build_default_registry
from physics.gears import (GearMesh, analyze_mesh, elastic_coefficient,
                           geometry_factor_i, hertz_contact_stress_pa,
                           lewis_bending_stress_pa, lewis_form_factor,
                           pitch_diameter_m, tangential_load_n)
from physics.joints import (BOLT_GRADES, NUT_FACTOR_DRY, PRELOAD_FRACTION,
                            PropertyClass, analyze_joint, bolt_stiffness_n_m,
                            load_factor, member_stiffness_n_m, proof_load_n,
                            target_preload_n, thread_stress_area_m2,
                            tightening_torque_nm)

STEEL_BENDING_PA, STEEL_CONTACT_PA = 200e6, 700e6


# --- bolted joint fundamentals -----------------------------------------------

def test_preload_and_torque_match_the_hand_calculation():
    """A_t 36.6 mm^2 at class 8.8 proof 580 MPa, preloaded to 75 percent."""
    area = thread_stress_area_m2("M8")
    assert area == pytest.approx(36.6e-6)
    assert proof_load_n("M8", PropertyClass.C8_8) == pytest.approx(
        36.6e-6 * 580e6, rel=1e-12)
    preload = target_preload_n("M8", PropertyClass.C8_8)
    assert preload == pytest.approx(0.75 * 36.6e-6 * 580e6, rel=1e-12)
    assert preload == pytest.approx(15921.0, rel=1e-4)
    # T = K F d
    assert tightening_torque_nm(preload, "M8") == pytest.approx(
        NUT_FACTOR_DRY * preload * 0.008, rel=1e-12)


def test_the_thread_stress_area_is_not_the_shank_area():
    """The thread cuts material away; using the nominal diameter overstates it."""
    for size, diameter in (("M6", 0.006), ("M8", 0.008), ("M10", 0.010)):
        shank = math.pi * diameter ** 2 / 4.0
        assert thread_stress_area_m2(size) < shank
        assert thread_stress_area_m2(size) > 0.7 * shank


def test_property_classes_are_ordered_and_internally_consistent():
    previous = None
    for grade in (PropertyClass.C8_8, PropertyClass.C10_9, PropertyClass.C12_9):
        strengths = BOLT_GRADES[grade]
        assert (strengths.proof_strength_pa < strengths.yield_strength_pa
                < strengths.ultimate_strength_pa)
        if previous is not None:
            assert strengths.proof_strength_pa > previous
        previous = strengths.proof_strength_pa


def test_stiffnesses_and_the_load_factor_match_the_hand_calculation():
    """k_b = A_t E / l, k_m by the Wileman fit, C = k_b / (k_b + k_m)."""
    bolt = bolt_stiffness_n_m("M8", 0.020)
    assert bolt == pytest.approx(36.6e-6 * 207e9 / 0.020, rel=1e-12)
    member = member_stiffness_n_m("M8", 0.020)
    assert member == pytest.approx(
        0.78715 * 207e9 * 0.008 * math.exp(0.62873 * 0.008 / 0.020), rel=1e-12)
    assert load_factor(bolt, member) == pytest.approx(bolt / (bolt + member))


def test_the_members_are_much_stiffer_than_the_bolt():
    """Which is the entire reason a preloaded joint protects its bolt.

    C lands around 0.2, so the bolt feels about a fifth of an external load and
    the rest merely relieves compression in the members.
    """
    bolt = bolt_stiffness_n_m("M8", 0.020)
    member = member_stiffness_n_m("M8", 0.020)
    assert member > 3.0 * bolt
    assert 0.1 < load_factor(bolt, member) < 0.3


# --- the regime change -------------------------------------------------------

def test_a_clamped_joint_gives_the_bolt_only_its_share():
    result = analyze_joint("M8", PropertyClass.C8_8, grip_length_m=0.020,
                           external_load_n=8000.0)
    assert not result.separated
    expected = result.preload_n + result.load_factor * 8000.0
    assert result.bolt_load_n == pytest.approx(expected, rel=1e-12)
    # The bolt load rises far less than the applied load.
    assert result.bolt_load_n - result.preload_n < 0.3 * 8000.0


def test_an_under_preloaded_joint_separates_and_the_bolt_takes_everything():
    """The regime change, and the usual way bolted joints fail.

    F_b = F_i + C P is only valid WHILE CLAMPED. Past separation there is
    nothing left to share with and the bolt carries the whole external load.
    Applying the clamped formula anyway understates the bolt load exactly in
    the case that matters.
    """
    tight = analyze_joint("M8", PropertyClass.C8_8, grip_length_m=0.020,
                          external_load_n=8000.0, external_load_min_n=0.0,
                          preload_fraction=0.75)
    loose = analyze_joint("M8", PropertyClass.C8_8, grip_length_m=0.020,
                          external_load_n=8000.0, external_load_min_n=0.0,
                          preload_fraction=0.15)

    assert not tight.separated and tight.separation_margin > 1.0
    assert loose.separated and loose.separation_margin < 1.0
    # The loose joint hands the bolt the entire external load.
    assert loose.bolt_load_n == pytest.approx(8000.0)
    # Which is what the clamped formula would NOT have said.
    clamped_formula = loose.preload_n + loose.load_factor * 8000.0
    assert loose.bolt_load_n > clamped_formula

    assert tight.passes
    assert not loose.passes
    assert loose.governing_mode == "separation"


def test_separation_multiplies_the_alternating_stress():
    """Below separation the bolt sees C of the range; above it, all of it.

    That jump is roughly 1/C, a factor of five here, and it is what destroys
    the fatigue life of an under-preloaded joint.
    """
    area = thread_stress_area_m2("M8")

    def alternating(fraction):
        result = analyze_joint("M8", PropertyClass.C8_8, grip_length_m=0.020,
                               external_load_n=8000.0, external_load_min_n=0.0,
                               preload_fraction=fraction)
        low = (result.preload_n if not result.separated
               else result.preload_n)
        return (result.bolt_load_n - low) / (2.0 * area), result

    clamped, tight = alternating(0.75)
    separated, loose = alternating(0.15)
    assert not tight.separated and loose.separated
    assert separated > 3.0 * clamped


def test_the_separation_load_is_the_preload_over_one_minus_c():
    result = analyze_joint("M8", PropertyClass.C8_8, grip_length_m=0.020,
                           external_load_n=1000.0)
    assert result.separation_load_n == pytest.approx(
        result.preload_n / (1.0 - result.load_factor), rel=1e-12)


def test_a_steady_load_has_no_bolt_fatigue_to_check():
    result = analyze_joint("M8", PropertyClass.C8_8, grip_length_m=0.020,
                           external_load_n=5000.0, external_load_min_n=5000.0)
    assert result.fatigue_safety_factor is None
    assert result.governing_mode in ("separation", "yield")


def test_a_higher_grade_raises_preload_and_therefore_capacity():
    joints = {grade: analyze_joint("M8", grade, grip_length_m=0.020,
                                   external_load_n=12000.0)
              for grade in PropertyClass}
    assert (joints[PropertyClass.C8_8].preload_n
            < joints[PropertyClass.C10_9].preload_n
            < joints[PropertyClass.C12_9].preload_n)
    # A stronger bolt can be pulled up harder, so it separates later.
    assert (joints[PropertyClass.C8_8].separation_load_n
            < joints[PropertyClass.C12_9].separation_load_n)


def test_an_unknown_thread_size_is_refused():
    with pytest.raises(KeyError, match="unknown thread size"):
        thread_stress_area_m2("M7")


# --- gear teeth --------------------------------------------------------------

def test_gear_geometry_and_load_match_the_hand_calculation():
    """d = m N, W_t = 2 T / d."""
    assert pitch_diameter_m(0.002, 20) == pytest.approx(0.040)
    assert tangential_load_n(25.0, 0.040) == pytest.approx(1250.0)


def test_lewis_bending_matches_the_hand_calculation():
    """sigma = W_t / (b m Y): 1250 N, 20 mm face, 2 mm module, Y = 0.322."""
    stress = lewis_bending_stress_pa(1250.0, 0.020, 0.002, 0.322)
    assert stress == pytest.approx(1250.0 / (0.020 * 0.002 * 0.322), rel=1e-12)
    assert stress == pytest.approx(97.05e6, rel=1e-3)


def test_the_form_factor_penalises_few_teeth():
    """A small pinion has an undercut, weaker root."""
    assert lewis_form_factor(12) < lewis_form_factor(20) < lewis_form_factor(50)
    assert lewis_form_factor(20) == pytest.approx(0.322)
    # Clamped rather than extrapolated at both ends.
    assert lewis_form_factor(5) == lewis_form_factor(12)
    assert lewis_form_factor(10000) == lewis_form_factor(400)


def test_the_elastic_coefficient_matches_the_steel_pair_value():
    """Z_E for steel on steel is about 190 sqrt(MPa)."""
    coefficient = elastic_coefficient(207e9, 0.29, 207e9, 0.29)
    assert coefficient == pytest.approx(
        math.sqrt(1.0 / (math.pi * 2.0 * (1 - 0.29 ** 2) / 207e9)), rel=1e-12)
    assert 185e3 < coefficient < 195e3
    # A softer partner spreads the contact and lowers it substantially.
    assert elastic_coefficient(3.0e9, 0.35, 207e9, 0.29) < 0.25 * coefficient


def test_the_geometry_factor_rises_with_ratio():
    assert geometry_factor_i(3.0) == pytest.approx(
        math.cos(math.radians(20)) * math.sin(math.radians(20)) / 2.0 * 0.75,
        rel=1e-12)
    assert geometry_factor_i(1.0) < geometry_factor_i(3.0) < geometry_factor_i(9.0)


def test_hertz_contact_matches_the_hand_calculation():
    coefficient = elastic_coefficient(207e9, 0.29, 207e9, 0.29)
    factor = geometry_factor_i(3.0)
    stress = hertz_contact_stress_pa(1250.0, 0.020, 0.040, coefficient, factor)
    assert stress == pytest.approx(
        coefficient * math.sqrt(1250.0 / (0.020 * 0.040 * factor)), rel=1e-12)


def test_contact_stress_grows_as_the_square_root_of_load():
    """Bending is linear in load and contact is not, which is why the governing
    mode can flip as a design is scaled."""
    coefficient = elastic_coefficient(207e9, 0.29, 207e9, 0.29)
    factor = geometry_factor_i(3.0)
    single = hertz_contact_stress_pa(1000.0, 0.020, 0.040, coefficient, factor)
    quadruple = hertz_contact_stress_pa(4000.0, 0.020, 0.040, coefficient,
                                        factor)
    assert quadruple == pytest.approx(2.0 * single, rel=1e-12)
    bending_single = lewis_bending_stress_pa(1000.0, 0.020, 0.002, 0.322)
    bending_quadruple = lewis_bending_stress_pa(4000.0, 0.020, 0.002, 0.322)
    assert bending_quadruple == pytest.approx(4.0 * bending_single, rel=1e-12)


def test_module_decides_which_gear_mode_governs():
    """Bending and pitting are different checks that bind on different gears.

    Bending goes as 1/(b m Y) and contact as the square root of 1/(b m N), so a
    fine-pitch gear is bending-critical and a coarse-pitch one is
    pitting-critical. Checking only one mode is how a gear set passes review
    and then fails in the way nobody checked.
    """
    def governing(module_mm):
        mesh = GearMesh(module_m=module_mm / 1000.0, pinion_teeth=20,
                        gear_teeth=60, face_width_m=0.010, torque_nm=25.0)
        return analyze_mesh(mesh, STEEL_BENDING_PA, STEEL_CONTACT_PA)

    fine = governing(1.0)
    coarse = governing(4.0)
    assert fine.governing_mode == "bending"
    assert coarse.governing_mode == "pitting"
    # And the ratio moves monotonically between them.
    ratios = [governing(m).bending_stress_pa / governing(m).contact_stress_pa
              for m in (0.6, 1.0, 2.0, 4.0)]
    assert ratios == sorted(ratios, reverse=True)


def test_the_correction_factors_default_to_the_optimistic_value():
    """1.0 is not neutral: every AGMA factor this stands in for is at least 1.

    An uncorrected result therefore runs high against a real gear, which is why
    the corrections are visible arguments rather than absent.
    """
    mesh = GearMesh(module_m=0.002, pinion_teeth=20, gear_teeth=60,
                    face_width_m=0.020, torque_nm=25.0)
    optimistic = analyze_mesh(mesh, STEEL_BENDING_PA, STEEL_CONTACT_PA)
    realistic = analyze_mesh(mesh, STEEL_BENDING_PA, STEEL_CONTACT_PA,
                             bending_correction=1.8, contact_correction=1.3)
    assert optimistic.passes
    assert not realistic.passes
    assert (realistic.bending_stress_pa
            == pytest.approx(1.8 * optimistic.bending_stress_pa))


def test_the_two_allowable_stresses_are_separate_inputs():
    """A single number for both would make the comparison meaningless."""
    mesh = GearMesh(module_m=0.002, pinion_teeth=20, gear_teeth=60,
                    face_width_m=0.020, torque_nm=25.0)
    result = analyze_mesh(mesh, STEEL_BENDING_PA, STEEL_CONTACT_PA)
    assert result.bending_allowable_pa != result.contact_allowable_pa
    assert result.bending_safety_factor == pytest.approx(
        STEEL_BENDING_PA / result.bending_stress_pa)
    assert result.contact_safety_factor == pytest.approx(
        STEEL_CONTACT_PA / result.contact_stress_pa)


def test_a_mesh_needs_real_teeth_and_a_real_face():
    with pytest.raises(ValueError):
        GearMesh(module_m=0.002, pinion_teeth=0, gear_teeth=60,
                 face_width_m=0.020, torque_nm=25.0)
    with pytest.raises(ValueError):
        GearMesh(module_m=0.002, pinion_teeth=20, gear_teeth=60,
                 face_width_m=0.0, torque_nm=25.0)


# --- registry ----------------------------------------------------------------

def test_the_new_methods_are_gated():
    registry = build_default_registry()
    none = ProblemContext(geometry="assembly", representations=("assembly",),
                          has_bolted_joint=False, has_gear_mesh=False)
    candidates = registry.query(none)
    assert "bolted_joint" not in candidates.names()
    assert "gear_tooth" not in candidates.names()
    assert "bolted" in candidates.reason("bolted_joint")[0]
    assert "gear mesh" in candidates.reason("gear_tooth")[0]

    present = ProblemContext(geometry="assembly", representations=("assembly",),
                             has_bolted_joint=True, has_gear_mesh=True)
    names = registry.query(present, Category.ANALYSIS).names()
    assert "bolted_joint" in names and "gear_tooth" in names


def test_unimplemented_methods_are_not_registered():
    registry = build_default_registry()
    for absent in ("gear_agma_full", "bolt_group", "planetary_internal",
                   "harmonic_drive"):
        assert absent not in registry
