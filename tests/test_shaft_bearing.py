"""Shaft sizing and bearing life, and the load path that connects them.

The tests that justify the phase are the discriminating ones: a shaft diameter
that the static check accepts and fatigue condemns, and a bearing that passes
on one side of a gearbox and fails on the other. Without those, adding the
checks would have changed no verdict.
"""

import math

import pytest

from core.materials import get_material
from core.registry import Category, ProblemContext, build_default_registry
from drivetrain.bearings import (BearingType, all_bearings,
                                 equivalent_dynamic_load, get_bearing,
                                 l10_hours, l10_revolutions, rate_bearing)
from drivetrain.bearings.catalog import BearingSpec
from drivetrain.loadpath import (ShaftLayout, bearing_reactions_n,
                                 bending_moment_nm, trace)
from drivetrain.motors.catalog import PartStatus
from drivetrain.selection.select import Requirement, select_drivetrain
from physics.shaft import (DEFAULT_BENDING_KF, DEFAULT_TORSION_KFS, ShaftLoads,
                           analyze_shaft, axial_stress_pa, bending_stress_pa,
                           de_goodman_diameter_m, de_goodman_inverse_factor,
                           first_critical_speed_rad_s, max_shear_pa,
                           torsional_stress_pa, von_mises_pa)


@pytest.fixture(scope="module")
def steel():
    return get_material("steel_scm440")


# --- shaft stresses ----------------------------------------------------------

def test_shaft_stress_formulas():
    """sigma = 32M/(pi d^3), tau = 16T/(pi d^3), sigma_axial = 4F/(pi d^2)."""
    d = 0.020
    assert bending_stress_pa(50.0, d) == pytest.approx(
        32.0 * 50.0 / (math.pi * d ** 3), rel=1e-15)
    assert torsional_stress_pa(80.0, d) == pytest.approx(
        16.0 * 80.0 / (math.pi * d ** 3), rel=1e-15)
    assert axial_stress_pa(1000.0, d) == pytest.approx(
        4.0 * 1000.0 / (math.pi * d ** 2), rel=1e-15)
    # Torsion is exactly half of bending for equal magnitudes, since the polar
    # section modulus is twice the bending one.
    assert (torsional_stress_pa(50.0, d)
            == pytest.approx(0.5 * bending_stress_pa(50.0, d), rel=1e-15))


def test_failure_theories():
    assert von_mises_pa(100e6, 0.0) == pytest.approx(100e6)
    assert von_mises_pa(0.0, 100e6) == pytest.approx(math.sqrt(3.0) * 100e6)
    # Maximum shear is the more conservative of the two in pure tension.
    assert max_shear_pa(100e6, 0.0) == pytest.approx(50e6)
    assert max_shear_pa(0.0, 100e6) == pytest.approx(100e6)


def test_a_rotating_shaft_puts_bending_in_the_alternating_term(steel):
    """The most consequential single fact about shaft fatigue.

    A steady transverse load on a rotating shaft is fully reversed in the
    material: each fibre goes from maximum tension to maximum compression once
    per revolution. The torque, being steady, is entirely mean. Swapping the
    two would charge the reversed stress against the ultimate strength instead
    of the endurance strength and roughly double the apparent safety factor.
    """
    loads = ShaftLoads.rotating(bending_moment_nm=50.0, torque_nm=80.0)
    assert loads.moment_alternating_nm == 50.0
    assert loads.moment_mean_nm == 0.0
    assert loads.torque_mean_nm == 80.0
    assert loads.torque_alternating_nm == 0.0

    stationary = ShaftLoads.stationary(bending_moment_nm=50.0, torque_nm=80.0)
    assert stationary.moment_mean_nm == 50.0
    assert stationary.moment_alternating_nm == 0.0

    # And it matters: the rotating case is the more severe of the two.
    rotating_factor = de_goodman_inverse_factor(loads, steel, 0.020)
    stationary_factor = de_goodman_inverse_factor(stationary, steel, 0.020)
    assert rotating_factor > stationary_factor


# --- DE-Goodman --------------------------------------------------------------

def test_de_goodman_matches_the_collapsed_rotating_form(steel):
    """For a rotating shaft with steady torque the general expression becomes

        1/n = (16 / (pi d^3)) * [ 2 Kf M / Se + sqrt(3) Kfs T / Sut ]

    which is checkable by hand, unlike the general form.
    """
    moment, torque, d = 50.0, 80.0, 0.020
    loads = ShaftLoads.rotating(bending_moment_nm=moment, torque_nm=torque)
    computed = de_goodman_inverse_factor(loads, steel, d, DEFAULT_BENDING_KF,
                                         DEFAULT_TORSION_KFS)
    expected = (16.0 / (math.pi * d ** 3)) * (
        2.0 * DEFAULT_BENDING_KF * moment / steel.fatigue_strength_pa
        + math.sqrt(3.0) * DEFAULT_TORSION_KFS * torque
        / steel.ultimate_strength_pa)
    assert computed == pytest.approx(expected, rel=1e-15)


def test_the_diameter_equation_inverts_the_factor_equation(steel):
    """Sizing and checking have to be the same relation, or one of them is wrong."""
    loads = ShaftLoads.rotating(bending_moment_nm=50.0, torque_nm=80.0)
    for diameter in (0.012, 0.020, 0.035):
        factor = 1.0 / de_goodman_inverse_factor(loads, steel, diameter)
        assert de_goodman_diameter_m(loads, steel, factor) == pytest.approx(
            diameter, rel=1e-12)


def test_pure_reversed_bending_reduces_to_the_endurance_check(steel):
    """An independent cross-check against the stress-life module.

    With no torque, DE-Goodman must collapse to n = Se / (Kf sigma_a), which is
    the fully-reversed fatigue check from Phase 16 with a concentration factor.
    Two derivations arriving at the same number is the point.
    """
    moment, d, kf = 50.0, 0.020, 1.7
    loads = ShaftLoads.rotating(bending_moment_nm=moment, torque_nm=0.0)
    factor = 1.0 / de_goodman_inverse_factor(loads, steel, d, kf, 1.0)
    alternating = bending_stress_pa(moment, d)
    assert factor == pytest.approx(
        steel.fatigue_strength_pa / (kf * alternating), rel=1e-12)


def test_pure_steady_torsion_reduces_to_a_static_ultimate_check(steel):
    """With nothing alternating, the mean term alone remains.

    n = Sut / (Kfs * sqrt(3) * tau), which is the von Mises equivalent of pure
    shear charged against the ultimate strength.
    """
    torque, d, kfs = 80.0, 0.020, 1.5
    loads = ShaftLoads.rotating(bending_moment_nm=0.0, torque_nm=torque)
    factor = 1.0 / de_goodman_inverse_factor(loads, steel, d, 1.0, kfs)
    shear = torsional_stress_pa(torque, d)
    assert factor == pytest.approx(
        steel.ultimate_strength_pa / (kfs * math.sqrt(3.0) * shear), rel=1e-12)


def test_sizing_with_nothing_to_size_against_is_refused(steel):
    with pytest.raises(ValueError, match="nothing to size"):
        de_goodman_diameter_m(ShaftLoads(), steel, 2.0)


def test_the_default_concentration_factors_are_not_one():
    """A shaft that holds anything has a shoulder or a keyway at that point.

    Defaulting to 1.0 would model a featureless bar and return a diameter too
    small for the thing it has to carry.
    """
    assert DEFAULT_BENDING_KF > 1.0
    assert DEFAULT_TORSION_KFS > 1.0


# --- the discriminating shaft case -------------------------------------------

def test_a_shaft_can_pass_the_static_check_and_fail_on_fatigue(steel):
    """A 10 mm shaft under 20 N m bending and 25.6 N m torque.

    Static safety factor 2.15 against yield, which any review would accept, and
    a fatigue factor of 0.78. Sizing this shaft on its static stress would have
    shipped it.
    """
    loads = ShaftLoads.rotating(bending_moment_nm=20.0, torque_nm=25.6)
    marginal = analyze_shaft(loads, steel, 0.010)
    assert marginal.static_safety_factor > 2.0
    assert marginal.fatigue_safety_factor < 1.0
    assert marginal.governing_mode == "fatigue"
    assert not marginal.passes

    # One millimetre more and fatigue is satisfied.
    assert analyze_shaft(loads, steel, 0.011).passes


def test_fatigue_governs_a_rotating_shaft_at_every_size(steel):
    """Not an artefact of one diameter: the two factors keep a fixed ratio.

    Both scale as d^3, so no diameter makes the static check the binding one.
    A design process that only ever looked at static stress would be wrong at
    every size, not just near the limit.
    """
    loads = ShaftLoads.rotating(bending_moment_nm=20.0, torque_nm=25.6)
    for diameter in (0.010, 0.016, 0.025, 0.040):
        result = analyze_shaft(loads, steel, diameter)
        assert result.governing_mode == "fatigue"


def test_critical_speed_is_the_uniform_beam_formula(steel):
    """omega_1 = (pi/L)^2 sqrt(E I / (rho A)) for a simply supported shaft."""
    d, length = 0.016, 0.13
    area = math.pi * d ** 2 / 4.0
    second_moment = math.pi * d ** 4 / 64.0
    expected = ((math.pi / length) ** 2
                * math.sqrt(steel.youngs_modulus_pa * second_moment
                            / (steel.density_kg_m3 * area)))
    assert first_critical_speed_rad_s(steel, d, length) == pytest.approx(
        expected, rel=1e-15)
    # It falls as the shaft gets longer, which is the behaviour that matters.
    assert (first_critical_speed_rad_s(steel, d, 0.26)
            < first_critical_speed_rad_s(steel, d, 0.13))


# --- bearings ----------------------------------------------------------------

def test_l10_matches_the_hand_calculation():
    """L10 = (C/P)^p * 1e6. C = 20300 N, P = 2000 N, p = 3."""
    revolutions = l10_revolutions(20300.0, 2000.0, 3.0)
    assert revolutions == pytest.approx((20300.0 / 2000.0) ** 3 * 1e6, rel=1e-15)
    assert revolutions == pytest.approx(1.045678375e9, rel=1e-9)


def test_life_in_hours_converts_through_rpm():
    """L10h = L10 / (60 n), with the speed given in rad/s."""
    revolutions = 1.045678375e9
    speed = 1500.0 * 2.0 * math.pi / 60.0
    assert l10_hours(revolutions, speed) == pytest.approx(
        revolutions / (60.0 * 1500.0), rel=1e-12)


def test_the_roller_exponent_gains_life_faster_than_the_ball():
    """10/3 against 3: halving the load buys a roller more than a ball."""
    ball = get_bearing("bearing_6206")
    roller = get_bearing("bearing_nu206")
    assert ball.life_exponent == 3.0
    assert roller.life_exponent == pytest.approx(10.0 / 3.0)
    ball_gain = l10_revolutions(1.0, 0.5, ball.life_exponent) / l10_revolutions(
        1.0, 1.0, ball.life_exponent)
    roller_gain = l10_revolutions(1.0, 0.5, roller.life_exponent) / l10_revolutions(
        1.0, 1.0, roller.life_exponent)
    assert roller_gain > ball_gain


def test_an_unloaded_bearing_has_no_life_to_report():
    with pytest.raises(ValueError, match="positive"):
        l10_revolutions(20300.0, 0.0, 3.0)


def test_a_small_thrust_does_not_change_the_equivalent_load():
    """Below the e ratio the thrust is carried without altering the geometry."""
    bearing = get_bearing("bearing_6206")
    load = equivalent_dynamic_load(bearing, radial_load_n=1000.0,
                                   axial_load_n=100.0)
    assert not load.thrust_governs
    assert load.equivalent_load_n == pytest.approx(1000.0)
    assert load.x_factor == 1.0 and load.y_factor == 0.0


def test_a_large_thrust_is_charged_through_x_and_y():
    """P = 0.56 Fr + Y Fa once Fa/Fr passes e."""
    bearing = get_bearing("bearing_6206")
    load = equivalent_dynamic_load(bearing, radial_load_n=1000.0,
                                   axial_load_n=2000.0)
    assert load.thrust_governs
    assert load.x_factor == pytest.approx(0.56)
    assert load.y_factor > 1.0
    assert load.equivalent_load_n == pytest.approx(
        0.56 * 1000.0 + load.y_factor * 2000.0)
    assert load.equivalent_load_n > 1000.0


def test_a_cylindrical_roller_bearing_refuses_a_thrust_load():
    """It has no flanges to react one, so a life for it would be fiction."""
    roller = get_bearing("bearing_nu206")
    assert equivalent_dynamic_load(roller, 1000.0, 0.0).equivalent_load_n == 1000.0
    with pytest.raises(ValueError, match="carries no thrust"):
        equivalent_dynamic_load(roller, 1000.0, 500.0)


def test_negative_bearing_loads_are_refused():
    with pytest.raises(ValueError, match="magnitudes"):
        equivalent_dynamic_load(get_bearing("bearing_6206"), -100.0, 0.0)


def test_the_bearing_catalogue_is_tagged_illustrative():
    """Standard dimensions, representative ratings, no vendor part numbers."""
    for bearing in all_bearings():
        assert bearing.status is PartStatus.ILLUSTRATIVE
        assert "verify vs manufacturer catalog" in bearing.source
        assert bearing.outer_diameter_m > bearing.bore_m
        assert 0.1 <= bearing.dynamic_rating_n / bearing.static_rating_n <= 20.0


def test_implausibly_paired_ratings_are_rejected():
    """Catches a units error or two swapped fields."""
    with pytest.raises(ValueError, match="plausible band"):
        BearingSpec(
            id="bad", designation="bad",
            bearing_type=BearingType.DEEP_GROOVE_BALL,
            bore_m=0.03, outer_diameter_m=0.062, width_m=0.016,
            dynamic_rating_n=20300.0, static_rating_n=11.2,   # kN against N
            limiting_speed_rad_s=1466.0, mass_kg=0.2, source="test")


def test_a_bearing_beyond_its_speed_limit_fails_whatever_its_life():
    bearing = get_bearing("bearing_6206")
    result = rate_bearing(bearing, 500.0,
                          speed_rad_s=bearing.limiting_speed_rad_s * 2.0,
                          required_hours=1.0)
    assert result.l10_hours > 1.0
    assert not result.speed_within_limit
    assert not result.passes


# --- the load path -----------------------------------------------------------

def test_overhung_reactions_satisfy_equilibrium():
    """R_B = P(L+a)/L and R_A = P a / L, and they differ by the applied load.

    The near bearing carries MORE than the whole load; the far one is lifted.
    """
    layout = ShaftLayout(bearing_span_m=0.08, overhang_m=0.05,
                         radial_load_n=400.0)
    near, far = bearing_reactions_n(layout)
    assert near == pytest.approx(400.0 * 0.13 / 0.08)
    assert far == pytest.approx(400.0 * 0.05 / 0.08)
    assert near - far == pytest.approx(400.0, rel=1e-12)
    assert near > 400.0
    assert bending_moment_nm(layout) == pytest.approx(20.0)


def test_a_longer_overhang_raises_both_the_moment_and_the_reaction():
    short = ShaftLayout(bearing_span_m=0.08, overhang_m=0.02,
                        radial_load_n=400.0)
    long = ShaftLayout(bearing_span_m=0.08, overhang_m=0.08,
                       radial_load_n=400.0)
    assert bending_moment_nm(long) > bending_moment_nm(short)
    assert bearing_reactions_n(long)[0] > bearing_reactions_n(short)[0]


def test_the_load_path_runs_from_the_selected_drivetrain_to_the_parts():
    """Phase 12 torque reaches a shaft diameter and a bearing life."""
    requirement = Requirement(joint="elbow", continuous_torque_nm=25.0,
                              peak_torque_nm=60.0, max_speed_rad_s=3.0,
                              load_inertia_kg_m2=0.4)
    best, _ = select_drivetrain(requirement)
    assert best is not None

    layout = ShaftLayout(bearing_span_m=0.08, overhang_m=0.05,
                         radial_load_n=400.0)
    path = trace(best, layout)

    # The torque is the gearbox output, not the motor torque.
    assert path.output_torque_nm == pytest.approx(
        best.motor.continuous_torque_nm * best.gearbox.ratio
        * best.gearbox.efficiency)
    assert path.output_torque_nm > best.motor.continuous_torque_nm
    assert path.speed_rad_s == requirement.max_speed_rad_s

    loads = path.shaft_loads()
    assert loads.moment_alternating_nm == pytest.approx(20.0)
    assert loads.torque_mean_nm == pytest.approx(path.output_torque_nm)

    steel = get_material("steel_scm440")
    diameter = de_goodman_diameter_m(loads, steel, safety_factor=2.0)
    assert 0.005 < diameter < 0.050
    assert analyze_shaft(loads, steel, diameter).fatigue_safety_factor == \
        pytest.approx(2.0, rel=1e-9)


def test_choosing_peak_torque_changes_the_load_path():
    """Which rating is used has to be a decision, not a default.

    Running a life calculation on the peak torque understates it badly, so the
    two must not be silently interchangeable.
    """
    requirement = Requirement(joint="elbow", continuous_torque_nm=25.0,
                              peak_torque_nm=60.0, max_speed_rad_s=3.0)
    best, _ = select_drivetrain(requirement)
    layout = ShaftLayout(bearing_span_m=0.08, overhang_m=0.05,
                         radial_load_n=400.0)
    assert (trace(best, layout, use_peak_torque=True).output_torque_nm
            > trace(best, layout).output_torque_nm)


def test_the_same_bearing_passes_on_one_side_of_the_gearbox_and_fails_on_the_other():
    """The discriminating bearing case, and a real design lesson.

    A gearbox trades torque for speed. The motor-side bearing sees the same
    kind of load at a hundred times the speed, so it accumulates its
    revolutions a hundred times faster and its life in hours is a hundredth.
    Identical part, identical load, opposite verdict.
    """
    bearing = get_bearing("bearing_608")
    joint_speed, ratio, load, required = 3.0, 100.0, 650.0, 20000.0

    joint_side = rate_bearing(bearing, load, joint_speed,
                              required_hours=required)
    motor_side = rate_bearing(bearing, load, joint_speed * ratio,
                              required_hours=required)

    assert joint_side.passes
    assert not motor_side.passes
    assert joint_side.l10_revolutions == pytest.approx(
        motor_side.l10_revolutions, rel=1e-12)      # same load, same revolutions
    assert joint_side.l10_hours == pytest.approx(
        motor_side.l10_hours * ratio, rel=1e-9)     # only the rate differs


# --- registry ----------------------------------------------------------------

def test_the_drivetrain_methods_are_gated_by_the_duty():
    registry = build_default_registry()
    none = ProblemContext(geometry="prismatic_beam",
                          representations=("prismatic_beam",),
                          transmits_torque=False, has_rotating_support=False)
    candidates = registry.query(none)
    assert "shaft_combined" not in candidates.names()
    assert "bearing_l10" not in candidates.names()
    assert "torque" in candidates.reason("shaft_combined")[0]
    assert "rotating support" in candidates.reason("bearing_l10")[0]

    drive = ProblemContext(geometry="assembly", representations=("assembly",),
                           transmits_torque=True, has_rotating_support=True)
    names = registry.query(drive).names()
    assert "shaft_combined" in names and "bearing_l10" in names


def test_unimplemented_drivetrain_methods_are_not_registered():
    registry = build_default_registry()
    for absent in ("shaft_fea", "bearing_contact_fatigue", "gear_agma",
                   "bolted_joint"):
        assert absent not in registry
