"""Phase 12 verification: motor and gearbox selection.

The catalogue is illustrative archetypes, so there is nothing to verify about
the parts themselves. What must be right is the **selection logic**: the gear
relations, the margins, and the refusal to pick something that does not meet the
requirement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from drivetrain.gearboxes import GearboxFamily, GearboxSpec, gearboxes, get_gearbox  # noqa: E402
from drivetrain.motors import MotorSpec, PartStatus, get_motor, motors  # noqa: E402
from drivetrain.selection import (  # noqa: E402
    Requirement, compare_alternatives, evaluate_candidate, infeasibility_report,
    output_torque_nm, required_motor_speed_rad_s, select_drivetrain,
)


def requirement(continuous=5.0, peak=12.0, speed=1.0, inertia=0.01, **kw):
    return Requirement(joint="test", continuous_torque_nm=continuous,
                       peak_torque_nm=peak, max_speed_rad_s=speed,
                       load_inertia_kg_m2=inertia, **kw)


# =========================================================================== #
# the catalogue is honest about what it is
# =========================================================================== #
def test_every_part_is_tagged_illustrative():
    """No invented vendor part numbers. A fabricated catalogue would be read
    later as if it had been sourced."""
    for part in list(motors()) + list(gearboxes()):
        assert part.status is PartStatus.ILLUSTRATIVE
        assert "replace with vendor datasheet" in part.source


def test_motor_power_matches_its_size_class():
    """Continuous torque times rated speed must reproduce the nominal power, or
    the catalogue is internally inconsistent."""
    for motor, expected_w in ((get_motor("bldc_50w"), 50),
                              (get_motor("bldc_100w"), 100),
                              (get_motor("bldc_200w"), 200),
                              (get_motor("bldc_400w"), 400)):
        assert motor.continuous_power_w == pytest.approx(expected_w, rel=0.02)


def test_peak_exceeds_continuous_for_every_motor():
    for motor in motors():
        assert motor.peak_torque_nm > motor.continuous_torque_nm
        assert 2.5 < motor.peak_torque_ratio < 3.5


def test_planetary_is_more_efficient_and_harmonic_has_less_backlash():
    """The two families have to trade differently or comparing them is pointless."""
    planetary = [g for g in gearboxes() if g.family is GearboxFamily.PLANETARY]
    harmonic = [g for g in gearboxes() if g.family is GearboxFamily.HARMONIC]
    assert max(g.efficiency for g in planetary) > max(g.efficiency for g in harmonic)
    assert (min(g.backlash_arcmin for g in harmonic)
            < min(g.backlash_arcmin for g in planetary))


def test_planetary_efficiency_falls_with_ratio():
    """More stages, more losses."""
    planetary = sorted([g for g in gearboxes()
                        if g.family is GearboxFamily.PLANETARY],
                       key=lambda g: g.ratio)
    efficiencies = [g.efficiency for g in planetary]
    assert efficiencies == sorted(efficiencies, reverse=True)


def test_inconsistent_part_is_rejected():
    with pytest.raises(Exception, match="peak torque"):
        MotorSpec(id="bad", name="bad", continuous_torque_nm=1.0,
                  peak_torque_nm=0.5, rated_speed_rad_s=100.0,
                  max_speed_rad_s=200.0, rotor_inertia_kg_m2=1e-5, mass_kg=0.5)
    with pytest.raises(Exception, match="peak torque"):
        GearboxSpec(id="bad", family=GearboxFamily.PLANETARY, ratio=10.0,
                    efficiency=0.9, rated_output_torque_nm=10.0,
                    peak_output_torque_nm=5.0, backlash_arcmin=10.0,
                    input_inertia_kg_m2=1e-6, mass_kg=0.3)


# =========================================================================== #
# 1 & 2. the gear relations, hand-checked and swept
# =========================================================================== #
def test_output_torque_hand_check():
    """0.32 N m through ratio 50 at 85% efficiency = 13.6 N m."""
    gearbox = get_gearbox("planetary_50")
    assert output_torque_nm(0.32, gearbox) == pytest.approx(0.32 * 50 * 0.85)
    assert output_torque_nm(0.32, gearbox) == pytest.approx(13.6)


def test_motor_speed_hand_check():
    """A joint turning at 1 rad/s through ratio 50 needs 50 rad/s at the motor."""
    assert required_motor_speed_rad_s(1.0, get_gearbox("planetary_50")) == 50.0


def test_reflected_inertia_hand_check():
    """Load inertia at the motor shaft is J_load / ratio^2.

    0.02 kg m^2 through ratio 50 is 8e-6 kg m^2, which against a 3e-5 rotor is
    an inertia ratio of 0.267.
    """
    candidate = evaluate_candidate(get_motor("bldc_100w"),
                                   get_gearbox("planetary_50"),
                                   requirement(inertia=0.02))
    assert candidate.reflected_load_inertia_kg_m2 == pytest.approx(0.02 / 2500)
    assert candidate.reflected_load_inertia_kg_m2 == pytest.approx(8e-6)
    assert candidate.inertia_ratio == pytest.approx(8e-6 / 3.0e-5, rel=1e-12)


def test_output_referred_inertia_hand_check():
    """J_out = (J_rotor + J_gearbox) * ratio^2 + J_load."""
    motor, gearbox = get_motor("bldc_100w"), get_gearbox("planetary_50")
    candidate = evaluate_candidate(motor, gearbox, requirement(inertia=0.02))
    expected = ((motor.rotor_inertia_kg_m2 + gearbox.input_inertia_kg_m2)
                * 50 ** 2 + 0.02)
    assert candidate.output_inertia_kg_m2 == pytest.approx(expected, rel=1e-12)


def test_ratio_sweep_scaling_laws():
    """Across the planetary family at fixed efficiency the relations must hold:
    output torque grows with ratio, speed falls as 1/ratio, and reflected
    inertia falls as 1/ratio^2."""
    motor = get_motor("bldc_100w")
    ratios, torques, speeds, reflected = [], [], [], []
    for ratio in (10.0, 25.0, 50.0, 100.0):
        gearbox = GearboxSpec(
            id=f"test_{ratio:.0f}", family=GearboxFamily.PLANETARY, ratio=ratio,
            efficiency=0.85, rated_output_torque_nm=1e6,
            peak_output_torque_nm=2e6, backlash_arcmin=10.0,
            input_inertia_kg_m2=1e-6, mass_kg=0.5)
        candidate = evaluate_candidate(motor, gearbox, requirement(inertia=0.02))
        ratios.append(ratio)
        torques.append(output_torque_nm(motor.continuous_torque_nm, gearbox))
        speeds.append(required_motor_speed_rad_s(1.0, gearbox))
        reflected.append(candidate.reflected_load_inertia_kg_m2)

    ratios = np.array(ratios)
    assert np.allclose(np.array(torques) / ratios, torques[0] / ratios[0])
    assert np.allclose(np.array(speeds) / ratios, speeds[0] / ratios[0])
    assert np.allclose(np.array(reflected) * ratios ** 2,
                       reflected[0] * ratios[0] ** 2)


def test_margin_is_available_over_required():
    candidate = evaluate_candidate(get_motor("bldc_100w"),
                                   get_gearbox("planetary_50"),
                                   requirement(continuous=6.8))
    check = [c for c in candidate.checks if c.name == "continuous torque"][0]
    assert check.available == pytest.approx(13.6)
    assert check.margin == pytest.approx(13.6 / 6.8)
    assert check.margin == pytest.approx(2.0)


# =========================================================================== #
# 3. selection picks something that works, or reports that nothing does
# =========================================================================== #
def test_selection_returns_a_feasible_candidate():
    best, feasible = select_drivetrain(requirement(continuous=5.0, peak=12.0))
    assert best is not None
    assert best.feasible
    assert all(c.passes for c in best.checks)
    assert best in feasible


def test_selection_picks_the_lightest_feasible_option():
    """Mass at a joint is carried by every joint inboard of it, so it is the
    cost that compounds on a serial arm."""
    best, feasible = select_drivetrain(requirement(continuous=5.0, peak=12.0))
    assert best.total_mass_kg == min(c.total_mass_kg for c in feasible)


def test_impossible_requirement_is_reported_as_infeasible():
    """Nothing in the catalogue can deliver this, and the selector must say so
    rather than returning the least bad option."""
    best, feasible = select_drivetrain(
        requirement(continuous=5000.0, peak=12000.0))
    assert best is None
    assert feasible == []


def test_infeasibility_report_names_the_failing_check():
    req = requirement(continuous=5000.0, peak=12000.0)
    candidates = [evaluate_candidate(m, g, req)
                  for m in motors() for g in gearboxes()]
    report = infeasibility_report(req, candidates)
    assert "no feasible drivetrain" in report
    assert "continuous torque" in report
    assert "short by up to" in report


def test_speed_requirement_can_make_a_high_ratio_infeasible():
    """A high ratio buys torque and costs speed. At a fast joint the ratio that
    solves the torque problem cannot keep up."""
    fast = requirement(continuous=1.0, peak=2.0, speed=20.0)
    candidate = evaluate_candidate(get_motor("bldc_100w"),
                                   get_gearbox("harmonic_160"), fast)
    speed_check = [c for c in candidate.checks if c.name == "motor speed"][0]
    assert not speed_check.passes
    assert speed_check.required == pytest.approx(20.0 * 160)
    assert not candidate.feasible


def test_backlash_requirement_excludes_planetary():
    """A one-arcminute requirement rules out the planetary family entirely."""
    precise = requirement(continuous=5.0, peak=12.0, max_backlash_arcmin=2.0)
    best, feasible = select_drivetrain(precise)
    assert best is not None
    assert best.gearbox.family is GearboxFamily.HARMONIC
    assert all(c.gearbox.family is GearboxFamily.HARMONIC for c in feasible)


# =========================================================================== #
# 4. continuous and peak must BOTH be checked
# =========================================================================== #
def test_a_drive_can_pass_peak_and_fail_continuous():
    """The case that justifies carrying both numbers: a brief acceleration is
    within reach while the holding load would cook the motor."""
    motor, gearbox = get_motor("bldc_50w"), get_gearbox("planetary_50")
    available_continuous = output_torque_nm(motor.continuous_torque_nm, gearbox)
    available_peak = output_torque_nm(motor.peak_torque_nm, gearbox)
    # Ask for more than continuous but less than peak.
    req = requirement(continuous=available_continuous * 1.2,
                      peak=available_peak * 0.8)
    candidate = evaluate_candidate(motor, gearbox, req)
    continuous_check = [c for c in candidate.checks
                        if c.name == "continuous torque"][0]
    peak_check = [c for c in candidate.checks if c.name == "peak torque"][0]
    assert peak_check.passes
    assert not continuous_check.passes
    assert not candidate.feasible, "checking peak alone would have accepted this"


def test_a_drive_can_pass_continuous_and_fail_peak():
    """The mirror case: fine at steady state, stalls on acceleration."""
    motor, gearbox = get_motor("bldc_50w"), get_gearbox("planetary_50")
    available_continuous = output_torque_nm(motor.continuous_torque_nm, gearbox)
    available_peak = output_torque_nm(motor.peak_torque_nm, gearbox)
    req = requirement(continuous=available_continuous * 0.5,
                      peak=available_peak * 1.3)
    candidate = evaluate_candidate(motor, gearbox, req)
    assert [c for c in candidate.checks
            if c.name == "continuous torque"][0].passes
    assert not [c for c in candidate.checks if c.name == "peak torque"][0].passes
    assert not candidate.feasible


def test_gearbox_rating_is_independent_of_the_motor():
    """A big motor cannot rescue an undersized gearbox."""
    req = requirement(continuous=30.0, peak=45.0)
    candidate = evaluate_candidate(get_motor("bldc_400w"),
                                   get_gearbox("planetary_50"), req)
    motor_check = [c for c in candidate.checks if c.name == "continuous torque"][0]
    gearbox_check = [c for c in candidate.checks
                     if c.name == "gearbox rated torque"][0]
    assert motor_check.passes           # 1.27 * 50 * 0.85 = 53.98
    assert not gearbox_check.passes     # gearbox rated at 25
    assert not candidate.feasible


def test_limiting_check_is_the_one_with_least_headroom():
    candidate = evaluate_candidate(get_motor("bldc_100w"),
                                   get_gearbox("planetary_50"),
                                   requirement(continuous=11.08, peak=27.04))
    limiting = candidate.limiting_check
    assert limiting.margin == min(c.margin for c in candidate.checks)
    assert limiting.name == "continuous torque"


# =========================================================================== #
# 5. alternatives carry reasons
# =========================================================================== #
def test_alternatives_are_ranked_and_explained():
    best, feasible = select_drivetrain(requirement(continuous=5.0, peak=12.0))
    comparison = compare_alternatives(feasible, count=3)
    assert len(comparison) >= 2, "need at least two options to compare"
    winner, winner_reason = comparison[0]
    assert winner is best
    assert "selected" in winner_reason
    assert "lightest feasible" in winner_reason
    for candidate, reason in comparison[1:]:
        assert "kg" in reason
        assert "limited by" in reason


def test_comparison_of_no_candidates_is_empty():
    assert compare_alternatives([]) == []


# =========================================================================== #
# the end to end demonstration
# =========================================================================== #
def test_two_link_arm_joints_both_get_a_drivetrain():
    from core.materials import get_material
    from physics.dynamics import evaluate_duty_cycle, mass_matrix
    from projects.robotic_arm.arm import build_arm

    arm = build_arm()
    density = get_material(arm.material_id).density_kg_m3
    duty = evaluate_duty_cycle(arm, density)
    inertia = mass_matrix(arm, [np.pi / 4, 0.2], density)
    peak, continuous = duty.peak_torque_nm(), duty.continuous_torque_nm()

    for i, joint in enumerate(arm.actuated_joints()):
        req = Requirement(joint=joint.name,
                          continuous_torque_nm=float(continuous[i]),
                          peak_torque_nm=float(peak[i]), max_speed_rad_s=1.0,
                          load_inertia_kg_m2=float(inertia[i, i]))
        best, feasible = select_drivetrain(req)
        assert best is not None, f"{joint.name}: no feasible drivetrain"
        assert len(feasible) >= 2, "expected alternatives to compare"
        assert best.inertia_ratio < 10.0
