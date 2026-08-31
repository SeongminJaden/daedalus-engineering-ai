"""Conduction, convection, radiation, resistance networks and transients.

Two tests carry the phase. `test_the_biot_number_rejects_a_lumped_model_that_does_not_apply`
pins the hard validity condition on transient analysis, which is the one that
looks reasonable when violated. `test_radiation_is_a_large_share_at_moderate_temperature`
pins the omission that most often makes a thermal model wrong in the
comfortable direction.
"""

import math

import pytest

from core.registry import ProblemContext, build_default_registry
from physics.thermal import (ABSOLUTE_ZERO_C, BIOT_LUMPED_LIMIT,
                             CONVECTION_RANGES, STEFAN_BOLTZMANN, Resistance,
                             ThermalPath, biot_number, celsius_to_kelvin,
                             convection_resistance_k_w,
                             cylinder_resistance_k_w, fourier_heat_w,
                             lumped_response, natural_convection_vertical_plate_w_m2k,
                             newton_cooling_w, parallel_resistance_k_w,
                             plane_wall_resistance_k_w,
                             radiation_coefficient_w_m2k, radiation_heat_w,
                             series_resistance_k_w, sphere_resistance_k_w,
                             surface_loss)


# --- conduction --------------------------------------------------------------

def test_conduction_resistances_match_their_definitions():
    assert plane_wall_resistance_k_w(0.010, 0.01, 200.0) == pytest.approx(
        0.010 / (200.0 * 0.01))
    assert cylinder_resistance_k_w(0.010, 0.020, 0.1, 50.0) == pytest.approx(
        math.log(2.0) / (2.0 * math.pi * 50.0 * 0.1))
    assert sphere_resistance_k_w(0.010, 0.020, 50.0) == pytest.approx(
        (1 / 0.010 - 1 / 0.020) / (4.0 * math.pi * 50.0))


def test_fourier_and_the_plane_wall_resistance_are_consistent():
    """Two routes to the same heat flow, which is the check on both."""
    conductivity, area, thickness, difference = 200.0, 0.01, 0.010, 50.0
    direct = fourier_heat_w(conductivity, area, difference, thickness)
    through_resistance = difference / plane_wall_resistance_k_w(
        thickness, area, conductivity)
    assert direct == pytest.approx(through_resistance)


def test_cylinder_resistance_is_logarithmic_not_linear():
    """Which is where the critical insulation radius comes from.

    Doubling the wall thickness of a pipe does not double its resistance.
    """
    thin = cylinder_resistance_k_w(0.010, 0.015, 1.0, 0.05)
    thick = cylinder_resistance_k_w(0.010, 0.020, 1.0, 0.05)
    assert thick < 2.0 * thin


# --- convection and radiation ------------------------------------------------

def test_newton_cooling_and_its_resistance_agree():
    coefficient, area, surface, ambient = 15.0, 0.02, 80.0, 25.0
    heat = newton_cooling_w(coefficient, area, surface, ambient)
    assert heat == pytest.approx(
        (surface - ambient) / convection_resistance_k_w(coefficient, area))


def test_radiation_must_be_evaluated_in_kelvin():
    """Using celsius gives a wildly wrong answer, not a slightly wrong one."""
    correct = radiation_heat_w(0.8, 0.01, 100.0, 25.0)
    surface_k, ambient_k = celsius_to_kelvin(100.0), celsius_to_kelvin(25.0)
    assert correct == pytest.approx(
        0.8 * STEFAN_BOLTZMANN * 0.01 * (surface_k ** 4 - ambient_k ** 4))
    naive = 0.8 * STEFAN_BOLTZMANN * 0.01 * (100.0 ** 4 - 25.0 ** 4)
    assert naive < 0.01 * correct


def test_the_linearised_coefficient_reproduces_the_quartic_law_at_its_point():
    """Exact where it is evaluated, and only there."""
    coefficient = radiation_coefficient_w_m2k(0.8, 100.0, 25.0)
    assert coefficient * 0.01 * (100.0 - 25.0) == pytest.approx(
        radiation_heat_w(0.8, 0.01, 100.0, 25.0), rel=1e-12)


def test_radiation_is_a_large_share_at_moderate_temperature():
    """The omission that makes a thermal model wrong in the comfortable direction.

    A surface at 100 C in a 25 C room with modest natural convection loses a
    comparable amount by radiation. Leaving it out OVER-predicts the
    temperature rise, so a model without it looks safe and is not describing
    the part.
    """
    loss = surface_loss(0.01, 100.0, 25.0, 10.0, 0.8)
    assert loss.radiation_share > 0.3
    assert loss.total_w > loss.convection_w
    # A polished surface radiates far less: emissivity is a finish, not a
    # metal, and polished against anodised aluminium is a factor of sixteen.
    # The radiated POWER scales with emissivity exactly; the SHARE does not,
    # because the convective denominator is unchanged. Asserting the share
    # scaled by sixteen was wrong arithmetic and it failed at 0.1015 against
    # 0.1000.
    polished = surface_loss(0.01, 100.0, 25.0, 10.0, 0.05)
    assert polished.radiation_w == pytest.approx(loss.radiation_w * 0.05 / 0.8)
    assert polished.radiation_share < 0.05


def test_absolute_zero_is_refused():
    with pytest.raises(ValueError, match="absolute zero"):
        celsius_to_kelvin(ABSOLUTE_ZERO_C - 1.0)


def test_the_convection_ranges_are_wide_and_say_so():
    """They are the spread, not values to use."""
    for name, (low, high) in CONVECTION_RANGES.items():
        assert high > 2.0 * low, f"{name} looks too narrow to be honest"


def test_the_natural_convection_correlation_grows_with_temperature_difference():
    small = natural_convection_vertical_plate_w_m2k(30.0, 25.0, 0.2)
    large = natural_convection_vertical_plate_w_m2k(100.0, 25.0, 0.2)
    assert large > small
    assert natural_convection_vertical_plate_w_m2k(25.0, 25.0, 0.2) == 0.0
    # And it lands inside the stated natural-air range.
    low, high = CONVECTION_RANGES["natural_air"]
    assert low <= large <= high


# --- networks ----------------------------------------------------------------

def test_series_and_parallel_combine_as_expected():
    resistances = [Resistance("a", 2.0), Resistance("b", 3.0)]
    assert series_resistance_k_w(resistances) == pytest.approx(5.0)
    equal = [Resistance("a", 2.0), Resistance("b", 2.0)]
    assert parallel_resistance_k_w(equal) == pytest.approx(1.0)
    # A parallel combination is always smaller than its smallest branch.
    assert parallel_resistance_k_w(resistances) < 2.0


def test_a_network_names_the_resistance_worth_attacking():
    """Which is most of a network's value.

    This is the Phase 18 claim made concrete: the mounting dominates, and here
    it is 47% of the total rather than an assertion.
    """
    path = (ThermalPath().add("winding to iron", 0.35)
            .add("iron to case", 0.55).add("case to bracket", 0.90)
            .add("bracket to air", 1.60))
    assert path.total_k_w == pytest.approx(3.40)
    assert path.temperature_rise_k(40.0) == pytest.approx(136.0)
    assert path.dominant().name == "bracket to air"
    assert path.shares()["bracket to air"] > 0.45


def test_a_negative_resistance_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        Resistance("bad", -1.0)


# --- transients --------------------------------------------------------------

def test_the_biot_number_matches_its_definition():
    assert biot_number(25.0, 0.005, 167.0) == pytest.approx(
        25.0 * 0.005 / 167.0)


def test_the_biot_number_rejects_a_lumped_model_that_does_not_apply():
    """The hard condition, and the one that looks reasonable when violated.

    The same geometry and the same cooling, in aluminium and in a polymer. The
    aluminium body is effectively isothermal and the polymer one is not, and
    nothing about the calculation would announce that without the check.
    """
    aluminium = lumped_response(0.5, 900.0, 1.0, 25.0, 0.005, 167.0)
    polymer = lumped_response(0.5, 1500.0, 1.0, 25.0, 0.005, 0.25)

    assert aluminium.biot_number < BIOT_LUMPED_LIMIT
    assert aluminium.lumped_valid
    assert polymer.biot_number > BIOT_LUMPED_LIMIT
    assert not polymer.lumped_valid
    # The number is still returned when the model does not apply, because it is
    # informative, and the caller has been told.
    assert polymer.time_constant_s > 0.0


def test_the_time_constant_is_resistance_times_capacitance():
    response = lumped_response(0.5, 900.0, 1.0, 25.0, 0.005, 167.0)
    assert response.capacitance_j_k == pytest.approx(450.0)
    assert response.time_constant_s == pytest.approx(1.0 * 450.0)


def test_one_time_constant_covers_the_familiar_fraction():
    """63.2% of the way to steady state, which is 1 - 1/e."""
    response = lumped_response(0.5, 900.0, 1.0, 25.0, 0.005, 167.0)
    steady = 25.0 + 40.0 * response.resistance_k_w
    at_tau = response.temperature_c(25.0, 25.0, 40.0, response.time_constant_s)
    assert (at_tau - 25.0) / (steady - 25.0) == pytest.approx(
        1.0 - math.exp(-1.0))
    # And it converges to the steady state given long enough.
    assert response.temperature_c(25.0, 25.0, 40.0,
                                  20.0 * response.time_constant_s) == \
        pytest.approx(steady, rel=1e-6)


def test_a_hot_body_cools_toward_ambient():
    response = lumped_response(0.5, 900.0, 1.0, 25.0, 0.005, 167.0)
    cooling = response.temperature_c(150.0, 25.0, 0.0,
                                     response.time_constant_s)
    assert 25.0 < cooling < 150.0


# --- registry ----------------------------------------------------------------

def test_the_thermal_methods_are_gated():
    registry = build_default_registry()
    none = ProblemContext(geometry="assembly", representations=("assembly",),
                          has_heat_path=False, has_thermal_transient=False)
    candidates = registry.query(none)
    for name in ("thermal_network", "lumped_transient"):
        assert name not in candidates.names()
        assert candidates.reason(name)

    present = ProblemContext(geometry="assembly",
                             representations=("assembly",),
                             has_heat_path=True, has_thermal_transient=True)
    names = registry.query(present).names()
    assert "thermal_network" in names and "lumped_transient" in names


def test_unimplemented_thermal_methods_are_still_not_registered():
    registry = build_default_registry()
    for absent in ("thermal_cfd", "contact_thermal_resistance",
                   "spreading_resistance", "thermal_fem"):
        assert absent not in registry
