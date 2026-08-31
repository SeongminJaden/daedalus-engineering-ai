"""Motor heating and thermal stress.

Two tests carry the phase. `test_a_duty_the_torque_proxy_accepts_can_overheat`
shows the continuous-torque check Phase 12 relied on accepting a duty that
cooks the winding, which is what "subject to thermal validation" was standing
in for. `test_cooling_adds_to_a_tensile_mechanical_stress` shows the sign
convention earning its keep: heating relieves a tensile part and cooling
endangers it, and a model returning magnitudes would have this backwards.
"""

import math

import pytest

from core.materials import get_material
from core.materials.db import MaterialSpec
from core.registry import Category, ProblemContext, build_default_registry
from drivetrain.motors.catalog import (WINDING_LIMIT_C, InsulationClass,
                                       get_motor, motors)
from physics.thermal import (DutySegment, check_motor_thermal,
                             check_thermal_stress,
                             constrained_thermal_stress_pa, differential_strain,
                             free_expansion_strain, losses_w, mean_speed_rad_s,
                             rms_torque_nm, stress_per_kelvin_pa,
                             temperature_rise_k)


@pytest.fixture(scope="module")
def aluminium():
    return get_material("al_7075_t6")


# --- duty cycle arithmetic ---------------------------------------------------

def test_rms_torque_is_not_the_mean():
    """Copper loss goes as torque squared, so the RMS is what heats the motor.

    Half a cycle at 1 N m and half at rest has a mean of 0.5 and an RMS of
    0.7071. Using the mean would understate the heating by a factor of two, and
    the error grows with how peaky the duty is.
    """
    duty = [DutySegment(1.0, 100.0, 0.5), DutySegment(0.0, 0.0, 0.5)]
    assert rms_torque_nm(duty) == pytest.approx(math.sqrt(0.5))
    mean = sum(s.torque_nm * s.fraction for s in duty)
    assert mean == pytest.approx(0.5)
    assert rms_torque_nm(duty) > mean
    # And the loss ratio is the square of the torque ratio.
    assert rms_torque_nm(duty) ** 2 / mean ** 2 == pytest.approx(2.0)


def test_fractions_are_normalised_not_required_to_sum_to_one():
    """Durations in seconds work as well as fractions."""
    seconds = [DutySegment(1.0, 100.0, 3.0), DutySegment(0.0, 0.0, 1.0)]
    fractions = [DutySegment(1.0, 100.0, 0.75), DutySegment(0.0, 0.0, 0.25)]
    assert rms_torque_nm(seconds) == pytest.approx(rms_torque_nm(fractions))
    assert mean_speed_rad_s(seconds) == pytest.approx(75.0)


def test_a_duty_with_no_duration_is_refused():
    with pytest.raises(ValueError, match="no duration"):
        rms_torque_nm([DutySegment(1.0, 100.0, 0.0)])


# --- motor thermal -----------------------------------------------------------

def test_losses_and_rise_match_the_hand_calculation():
    """P_cu = k T^2, P_fe = k omega, T_rise = P R_th."""
    motor = get_motor("bldc_100w")
    torque, speed = 0.32, 314.0
    loss = losses_w(motor, torque, speed)
    assert loss.copper_w == pytest.approx(
        motor.copper_loss_coefficient_w_nm2 * torque ** 2, rel=1e-15)
    assert loss.iron_w == pytest.approx(
        motor.iron_loss_coefficient_w_s_rad * speed, rel=1e-15)
    rise = temperature_rise_k(loss.total_w, motor.thermal_resistance_k_w)
    assert rise == pytest.approx(loss.total_w * motor.thermal_resistance_k_w,
                                 rel=1e-15)


def test_the_archetypes_are_thermally_self_consistent():
    """A motor's continuous rating IS its thermal rating.

    Running each archetype at its continuous torque and rated speed must land
    the winding near, and below, its insulation limit. Coefficients chosen
    independently of the torque rating produced a 6 K rise, which would have
    made every thermal check pass and the calculation decorative.
    """
    for motor in motors():
        duty = [DutySegment(motor.continuous_torque_nm,
                            motor.rated_speed_rad_s, 1.0)]
        result = check_motor_thermal(motor, duty, ambient_c=40.0)
        assert result.passes, f"{motor.id} exceeds its limit at its own rating"
        assert 70.0 < result.temperature_rise_k < 120.0, (
            f"{motor.id} rises {result.temperature_rise_k:.1f} K at its "
            f"continuous rating, which is not a thermally defined rating")
        # And the implied efficiency is plausible for a small BLDC machine.
        output = motor.continuous_torque_nm * motor.rated_speed_rad_s
        efficiency = output / (output + result.losses.total_w)
        assert 0.6 < efficiency < 0.85


def test_a_duty_the_torque_proxy_accepts_can_overheat():
    """The proxy Phase 12 left as "subject to thermal validation".

    Half a cycle at 0.60 N m and half at rest has a MEAN of 0.30, inside the
    0.32 rating, so a continuous-torque check accepts it. The RMS is 0.424, and
    the winding reaches 178 C against a 155 C limit. The proxy cannot see the
    shape of a duty, only its average.
    """
    motor = get_motor("bldc_100w")
    peaky = [DutySegment(0.60, 300.0, 0.5), DutySegment(0.0, 0.0, 0.5)]

    mean_torque = sum(s.torque_nm * s.fraction for s in peaky)
    assert mean_torque <= motor.continuous_torque_nm      # the proxy accepts it
    assert rms_torque_nm(peaky) > motor.continuous_torque_nm

    result = check_motor_thermal(motor, peaky, ambient_c=40.0)
    assert not result.passes
    assert result.winding_c > result.limit_c
    assert result.margin_k < 0.0


def test_ambient_temperature_eats_the_margin_one_for_one():
    """Every degree of ambient is a degree of winding margin gone.

    A drive sized on a bench at 25 C and installed next to its own driver at
    70 C has lost 45 K of margin without anything about the duty changing.
    """
    motor = get_motor("bldc_100w")
    duty = [DutySegment(0.30, 300.0, 1.0)]
    cool = check_motor_thermal(motor, duty, ambient_c=25.0)
    hot = check_motor_thermal(motor, duty, ambient_c=70.0)

    assert cool.passes and not hot.passes
    assert hot.temperature_rise_k == pytest.approx(cool.temperature_rise_k)
    assert cool.margin_k - hot.margin_k == pytest.approx(45.0)


def test_a_motor_without_thermal_data_is_refused_not_guessed():
    """Reporting a temperature for it would be inventing one."""
    motor = get_motor("bldc_100w").model_copy(
        update={"copper_loss_coefficient_w_nm2": None,
                "thermal_resistance_k_w": None})
    assert not motor.has_thermal_data
    with pytest.raises(ValueError, match="no thermal data"):
        check_motor_thermal(motor, [DutySegment(0.3, 300.0, 1.0)])


def test_insulation_classes_are_ordered():
    limits = [WINDING_LIMIT_C[c] for c in
              (InsulationClass.A, InsulationClass.E, InsulationClass.B,
               InsulationClass.F, InsulationClass.H)]
    assert limits == sorted(limits)
    assert WINDING_LIMIT_C[InsulationClass.F] == 155.0


# --- thermal expansion in the material database ------------------------------

def test_every_material_carries_an_expansion_coefficient():
    for material_id in ("al_7075_t6", "steel_scm440", "ti_6al_4v", "ss_316",
                        "mg_az31b", "alumina_al2o3", "abs", "pla", "pc",
                        "pa12", "petg", "cfrp_ud"):
        assert get_material(material_id).thermal_expansion_1_k is not None


def test_carbon_fibre_contracts_along_its_length():
    """A negative coefficient, which a positive-only field would have rejected.

    CFRP is about -0.5e-6 along the fibres and +25e-6 across them: a factor of
    fifty apart and of opposite sign. A single value for this material would be
    wrong in one direction whichever value was chosen.
    """
    cfrp = get_material("cfrp_ud")
    assert cfrp.cte1_1_k < 0.0
    assert cfrp.cte2_1_k > 0.0
    assert abs(cfrp.cte2_1_k / cfrp.cte1_1_k) > 40.0
    # The isotropic field holds the axis-1 value, as it does for the moduli.
    assert cfrp.thermal_expansion_1_k == pytest.approx(cfrp.cte1_1_k)


def test_an_orthotropic_material_must_declare_directional_coefficients():
    cfrp = get_material("cfrp_ud")
    with pytest.raises(ValueError, match="SIGN"):
        MaterialSpec(**{**cfrp.model_dump(), "cte2_1_k": None})


def test_the_expected_ordering_of_expansion_coefficients():
    """Magnesium expands most, ceramics and titanium least, polymers far more."""
    alpha = {m: get_material(m).thermal_expansion_1_k
             for m in ("mg_az31b", "al_7075_t6", "ss_316", "steel_scm440",
                       "ti_6al_4v", "alumina_al2o3", "abs")}
    assert alpha["mg_az31b"] > alpha["al_7075_t6"] > alpha["ss_316"]
    assert alpha["ss_316"] > alpha["steel_scm440"] > alpha["ti_6al_4v"]
    assert alpha["ti_6al_4v"] > alpha["alumina_al2o3"]
    assert alpha["abs"] > alpha["mg_az31b"]


# --- thermal stress ----------------------------------------------------------

def test_free_expansion_produces_strain_and_no_stress(aluminium):
    """Restraint, not temperature, is what makes a thermal stress."""
    assert free_expansion_strain(aluminium.thermal_expansion_1_k, 60.0) == \
        pytest.approx(23.6e-6 * 60.0)
    assert constrained_thermal_stress_pa(
        aluminium.youngs_modulus_pa, aluminium.thermal_expansion_1_k, 60.0,
        constraint=0.0) == pytest.approx(0.0)


def test_full_restraint_matches_the_hand_calculation(aluminium):
    """sigma = -E alpha dT: 71.7 GPa, 23.6e-6, 60 K gives -101.527 MPa."""
    stress = constrained_thermal_stress_pa(
        aluminium.youngs_modulus_pa, aluminium.thermal_expansion_1_k, 60.0)
    assert stress == pytest.approx(-71.7e9 * 23.6e-6 * 60.0, rel=1e-15)
    assert stress == pytest.approx(-101.527e6, rel=1e-4)
    assert stress < 0.0                       # heating restrained means compression


def test_partial_restraint_scales_linearly(aluminium):
    full = constrained_thermal_stress_pa(
        aluminium.youngs_modulus_pa, aluminium.thermal_expansion_1_k, 60.0, 1.0)
    half = constrained_thermal_stress_pa(
        aluminium.youngs_modulus_pa, aluminium.thermal_expansion_1_k, 60.0, 0.5)
    assert half == pytest.approx(0.5 * full)


def test_a_constraint_factor_outside_zero_to_one_is_refused(aluminium):
    with pytest.raises(ValueError, match="0 \\(free\\) to 1"):
        constrained_thermal_stress_pa(aluminium.youngs_modulus_pa, 23.6e-6,
                                      60.0, constraint=1.5)


def test_steel_develops_more_stress_per_kelvin_than_aluminium():
    """Counterintuitive, and the reason the modulus cannot be left out.

    Steel expands half as much as aluminium and is three times as stiff, so
    restraining it costs MORE stress per kelvin, not less. Ranking materials by
    expansion coefficient alone gets this backwards.
    """
    steel = get_material("steel_scm440")
    aluminium = get_material("al_7075_t6")
    assert steel.thermal_expansion_1_k < aluminium.thermal_expansion_1_k
    steel_rate = stress_per_kelvin_pa(steel.youngs_modulus_pa,
                                      steel.thermal_expansion_1_k)
    aluminium_rate = stress_per_kelvin_pa(aluminium.youngs_modulus_pa,
                                          aluminium.thermal_expansion_1_k)
    assert steel_rate > aluminium_rate


def test_heating_relieves_a_tensile_mechanical_stress(aluminium):
    """Superposition with signs, not magnitudes.

    Adding magnitudes would report a problem where the physics removes one.
    """
    hot = check_thermal_stress(aluminium, delta_t_k=100.0,
                               mechanical_stress_pa=350e6)
    assert hot.thermal_stress_pa < 0.0
    assert abs(hot.combined_stress_pa) < 350e6
    assert hot.safety_factor > aluminium.yield_strength_pa / 350e6


def test_cooling_adds_to_a_tensile_mechanical_stress(aluminium):
    """The dangerous direction, and it is the counterintuitive one.

    A restrained part that is COOLED goes into tension, which adds to a tensile
    mechanical stress. A 7075 bracket comfortable at 350 MPa at room
    temperature is over yield after a 100 K cool-down, with nothing about the
    mechanical load having changed.
    """
    warm = check_thermal_stress(aluminium, delta_t_k=0.0,
                                mechanical_stress_pa=350e6)
    cold = check_thermal_stress(aluminium, delta_t_k=-100.0,
                                mechanical_stress_pa=350e6)
    assert warm.passes
    assert cold.thermal_stress_pa > 0.0
    assert cold.combined_stress_pa > warm.combined_stress_pa
    assert not cold.passes
    assert cold.governing_contribution == "mechanical"    # 169 against 350


def test_thermal_can_be_the_governing_contribution(aluminium):
    result = check_thermal_stress(aluminium, delta_t_k=150.0,
                                  mechanical_stress_pa=80e6)
    assert abs(result.thermal_stress_pa) > abs(result.mechanical_stress_pa)
    assert result.governing_contribution == "thermal"


def test_a_material_without_a_coefficient_is_refused(aluminium):
    stripped = aluminium.model_copy(update={"thermal_expansion_1_k": None})
    with pytest.raises(ValueError, match="no thermal expansion"):
        check_thermal_stress(stripped, delta_t_k=60.0)


def test_an_orthotropic_direction_can_be_supplied(aluminium):
    """Using the axis-1 value across the fibres would be wrong by fifty times."""
    cfrp = get_material("cfrp_ud")
    along = check_thermal_stress(cfrp, delta_t_k=100.0,
                                 alpha_1_k=cfrp.cte1_1_k)
    across = check_thermal_stress(cfrp, delta_t_k=100.0,
                                  alpha_1_k=cfrp.cte2_1_k)
    assert along.thermal_stress_pa > 0.0     # negative alpha, heating: tension
    assert across.thermal_stress_pa < 0.0
    assert abs(across.thermal_stress_pa) > abs(along.thermal_stress_pa)


def test_dissimilar_materials_carry_their_difference():
    """Neither part needs external restraint: each restrains the other."""
    aluminium = get_material("al_7075_t6")
    steel = get_material("steel_scm440")
    mismatch = differential_strain(aluminium.thermal_expansion_1_k,
                                   steel.thermal_expansion_1_k, 100.0)
    assert mismatch == pytest.approx((23.6e-6 - 11.7e-6) * 100.0)
    assert mismatch > 0.5 * free_expansion_strain(
        aluminium.thermal_expansion_1_k, 100.0)
    # Identical materials have no mismatch at all.
    assert differential_strain(aluminium.thermal_expansion_1_k,
                               aluminium.thermal_expansion_1_k, 100.0) == 0.0


# --- registry ----------------------------------------------------------------

def test_the_thermal_methods_are_gated():
    registry = build_default_registry()
    none = ProblemContext(geometry="assembly", representations=("assembly",),
                          has_duty_cycle=False, has_temperature_change=False)
    candidates = registry.query(none)
    assert "motor_thermal" not in candidates.names()
    assert "thermal_stress" not in candidates.names()
    assert "duty cycle" in candidates.reason("motor_thermal")[0]
    assert "temperature change" in candidates.reason("thermal_stress")[0]

    thermal = ProblemContext(geometry="assembly", representations=("assembly",),
                             has_duty_cycle=True, has_temperature_change=True)
    names = registry.query(thermal, Category.ANALYSIS).names()
    assert "motor_thermal" in names and "thermal_stress" in names


def test_unimplemented_thermal_methods_are_not_registered():
    registry = build_default_registry()
    for absent in ("thermal_transient", "thermal_network", "thermal_cfd",
                   "contact_thermal_resistance"):
        assert absent not in registry
