"""Internal flow, drag, fluid power actuators and cooling flow.

The test that carries the phase is
`test_haaland_agrees_with_an_iterated_colebrook_solution`: an explicit
correlation checked against the implicit relation it approximates, so the
approximation is measured rather than trusted. The second is
`test_the_transitional_band_is_flagged_rather_than_answered`, since there is no
valid friction law between laminar and turbulent and returning one silently is
the failure available here.
"""

import math

import pytest

from core.registry import ProblemContext, build_default_registry
from physics.fluids import (LAMINAR_LIMIT, SPHERE_CD_RANGE, TURBULENT_ONSET,
                            FlowRegime, colebrook_friction_factor,
                            cooling_flow, cylinder_flow_m3_s, cylinder_force,
                            darcy_weisbach_pa, drag_force_n, drag_power_w,
                            flow_regime, haaland_friction_factor,
                            laminar_friction_factor, minor_loss_pa,
                            reynolds_number, solve_pipe_flow,
                            sphere_cd_is_in_range)

WATER_DENSITY, WATER_VISCOSITY, WATER_CP = 997.0, 8.9e-4, 4186.0


# --- internal flow -----------------------------------------------------------

def test_reynolds_number_matches_its_definition():
    assert reynolds_number(2.0, 0.010, 997.0, 8.9e-4) == pytest.approx(
        997.0 * 2.0 * 0.010 / 8.9e-4)


def test_the_laminar_friction_factor_is_exact_not_a_correlation():
    """It falls out of the analytical velocity profile."""
    for reynolds in (100.0, 1000.0, 2000.0):
        assert laminar_friction_factor(reynolds) == pytest.approx(
            64.0 / reynolds, rel=1e-15)


def test_haaland_agrees_with_an_iterated_colebrook_solution():
    """The explicit correlation measured against the implicit relation.

    Haaland exists because Colebrook has to be iterated. Checking one against
    the other turns "approximately" into a number, and 1.3% is what it is over
    four decades of Reynolds number and four of roughness.
    """
    worst = 0.0
    for reynolds in (1e4, 1e5, 1e6, 1e7):
        for roughness in (0.0, 1e-4, 1e-3, 1e-2):
            explicit = haaland_friction_factor(reynolds, roughness)
            implicit = colebrook_friction_factor(reynolds, roughness)
            worst = max(worst, abs(explicit - implicit) / implicit)
    assert worst < 0.02, f"worst deviation {worst:.3%}"
    assert worst > 0.001, "the two are not expected to agree exactly"


def test_fully_rough_flow_stops_depending_on_reynolds_number():
    """Where roughness alone sets the friction factor."""
    rough = 0.01
    assert haaland_friction_factor(1e7, rough) == pytest.approx(
        haaland_friction_factor(1e8, rough), rel=0.02)
    # A smooth pipe keeps falling with Reynolds number.
    assert haaland_friction_factor(1e8, 0.0) < haaland_friction_factor(1e7, 0.0)


def test_the_transitional_band_is_flagged_rather_than_answered():
    """There is no valid friction law between laminar and turbulent.

    Something must be returned, so the turbulent correlation is used and the
    result is marked as interpolated. Returning it silently would present an
    interpolation as physics.
    """
    velocity = 3000.0 * WATER_VISCOSITY / (WATER_DENSITY * 0.008)
    flow = velocity * math.pi * 0.008 ** 2 / 4.0

    transitional = solve_pipe_flow(flow, 0.008, 2.0, WATER_DENSITY,
                                   WATER_VISCOSITY)
    assert transitional.regime is FlowRegime.TRANSITIONAL
    assert transitional.friction_is_interpolated
    assert not transitional.regime_is_certain

    laminar = solve_pipe_flow(flow / 3.0, 0.008, 2.0, WATER_DENSITY,
                              WATER_VISCOSITY)
    assert laminar.regime is FlowRegime.LAMINAR
    assert laminar.regime_is_certain
    assert laminar.friction_factor == pytest.approx(
        64.0 / laminar.reynolds)


def test_the_regime_boundaries_are_where_they_are_stated():
    assert flow_regime(LAMINAR_LIMIT - 1.0) is FlowRegime.LAMINAR
    assert flow_regime(LAMINAR_LIMIT + 1.0) is FlowRegime.TRANSITIONAL
    assert flow_regime(TURBULENT_ONSET + 1.0) is FlowRegime.TURBULENT


def test_darcy_weisbach_matches_its_definition():
    assert darcy_weisbach_pa(0.02, 2.0, 0.008, 997.0, 3.0) == pytest.approx(
        0.02 * 2.0 / 0.008 * 997.0 * 9.0 / 2.0)


def test_pressure_drop_rises_faster_than_flow():
    """Quadratic in velocity, softened slightly by the friction factor falling.

    Which is why oversizing a cooling line is so effective, and why a slightly
    undersized one is much worse than it looks.
    """
    full = solve_pipe_flow(2e-4, 0.008, 2.0, WATER_DENSITY, WATER_VISCOSITY,
                           roughness_m=1.5e-6)
    half = solve_pipe_flow(1e-4, 0.008, 2.0, WATER_DENSITY, WATER_VISCOSITY,
                           roughness_m=1.5e-6)
    ratio = full.pressure_drop_pa / half.pressure_drop_pa
    assert 3.0 < ratio < 4.0


def test_minor_losses_are_minor_only_by_convention():
    """A short run with several fittings is dominated by them."""
    short = solve_pipe_flow(2e-4, 0.008, 0.3, WATER_DENSITY, WATER_VISCOSITY)
    with_fittings = solve_pipe_flow(2e-4, 0.008, 0.3, WATER_DENSITY,
                                    WATER_VISCOSITY,
                                    minor_loss_coefficients=6.0)
    assert with_fittings.pressure_drop_pa > 2.0 * short.pressure_drop_pa
    assert minor_loss_pa(1.0, 997.0, 2.0) == pytest.approx(997.0 * 4.0 / 2.0)


# --- actuators ---------------------------------------------------------------

def test_cylinder_forces_match_the_hand_calculation():
    result = cylinder_force(6e5, 0.040, 0.016)
    assert result.extend_n == pytest.approx(
        6e5 * math.pi * 0.040 ** 2 / 4.0)
    assert result.retract_n == pytest.approx(
        6e5 * math.pi * (0.040 ** 2 - 0.016 ** 2) / 4.0)


def test_retracting_is_always_weaker_than_extending():
    """The rod occupies part of the annulus, so the same pressure acts on less.

    A cylinder sized on its extend force can be badly undersized on the return.
    """
    for rod in (0.010, 0.016, 0.025):
        result = cylinder_force(6e5, 0.040, rod)
        assert result.retract_n < result.extend_n
        assert result.retract_ratio < 1.0
    # A fatter rod costs more of the return stroke.
    assert (cylinder_force(6e5, 0.040, 0.025).retract_ratio
            < cylinder_force(6e5, 0.040, 0.010).retract_ratio)


def test_the_efficiency_default_is_optimistic_and_visible():
    """1.0 is the theoretical value, and a real cylinder does not reach it."""
    theoretical = cylinder_force(6e5, 0.040, 0.016)
    realistic = cylinder_force(6e5, 0.040, 0.016, efficiency=0.85)
    assert theoretical.efficiency == 1.0
    assert realistic.extend_n == pytest.approx(0.85 * theoretical.extend_n)
    with pytest.raises(ValueError, match="efficiency"):
        cylinder_force(6e5, 0.040, 0.016, efficiency=1.2)


def test_a_rod_cannot_be_larger_than_its_bore():
    with pytest.raises(ValueError, match="rod"):
        cylinder_force(6e5, 0.040, 0.040)


def test_flow_demand_scales_with_speed():
    result = cylinder_force(6e5, 0.040, 0.016)
    assert cylinder_flow_m3_s(result.extend_area_m2, 0.2) == pytest.approx(
        result.extend_area_m2 * 0.2)


# --- drag --------------------------------------------------------------------

def test_drag_is_quadratic_and_its_power_cubic():
    """Doubling speed quadruples the force and octuples the power."""
    single = drag_force_n(1.2, 10.0, 0.47, 0.01)
    double = drag_force_n(1.2, 20.0, 0.47, 0.01)
    assert double / single == pytest.approx(4.0)
    assert (drag_power_w(1.2, 20.0, 0.47, 0.01)
            / drag_power_w(1.2, 10.0, 0.47, 0.01)) == pytest.approx(8.0)


def test_the_quoted_sphere_coefficient_states_its_range():
    """It drops by about threefold above the range when the boundary layer
    turns turbulent, so quoting it across that transition is badly wrong."""
    low, high = SPHERE_CD_RANGE
    assert sphere_cd_is_in_range(1e4)
    assert not sphere_cd_is_in_range(low / 10.0)
    assert not sphere_cd_is_in_range(high * 10.0)


# --- cooling -----------------------------------------------------------------

def test_cooling_flow_matches_the_energy_balance():
    requirement = cooling_flow(500.0, 10.0, WATER_CP, WATER_DENSITY)
    assert requirement.mass_flow_kg_s == pytest.approx(
        500.0 / (WATER_CP * 10.0))
    assert requirement.volume_flow_m3_s == pytest.approx(
        requirement.mass_flow_kg_s / WATER_DENSITY)


def test_a_smaller_allowed_rise_needs_proportionally_more_flow():
    generous = cooling_flow(500.0, 20.0, WATER_CP, WATER_DENSITY)
    tight = cooling_flow(500.0, 5.0, WATER_CP, WATER_DENSITY)
    assert tight.mass_flow_kg_s == pytest.approx(4.0 * generous.mass_flow_kg_s)


def test_a_coolant_that_does_not_warm_up_is_refused():
    """It would need infinite flow, which is a modelling error not a design."""
    with pytest.raises(ValueError, match="temperature rise"):
        cooling_flow(500.0, 0.0, WATER_CP, WATER_DENSITY)


# --- registry ----------------------------------------------------------------

def test_the_fluid_methods_are_gated():
    registry = build_default_registry()
    none = ProblemContext(geometry="assembly", representations=("assembly",),
                          has_internal_flow=False, has_external_flow=False,
                          has_fluid_actuator=False)
    candidates = registry.query(none)
    for name in ("pipe_flow", "external_drag", "fluid_actuator"):
        assert name not in candidates.names()
        assert candidates.reason(name)

    present = ProblemContext(geometry="assembly",
                             representations=("assembly",),
                             has_internal_flow=True, has_external_flow=True,
                             has_fluid_actuator=True)
    names = registry.query(present).names()
    for name in ("pipe_flow", "external_drag", "fluid_actuator"):
        assert name in names


def test_unimplemented_fluid_methods_are_not_registered():
    registry = build_default_registry()
    for absent in ("fluid_cfd", "compressible_flow", "pump_curve_matching",
                   "cavitation"):
        assert absent not in registry
