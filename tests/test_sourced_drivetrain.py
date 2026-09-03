"""Motors and gear units whose numbers came from the manufacturer's page.

The catalogue in drivetrain/motors and drivetrain/gearboxes is archetypes and
says so. These entries are real parts, and the tests here are about the
discipline that makes them worth more than archetypes: every stored number
names the document and the printed figure, a value that was not published is
absent rather than estimated, and a conversion that needs a missing value
refuses with the field name.
"""

from __future__ import annotations

import math

import pytest

from drivetrain.gearboxes.catalog import GearboxFamily
from drivetrain.motors.catalog import PartStatus
from drivetrain.sourced import (GCM2_TO_KG_M2, MNM_TO_NM, RPM_TO_RAD_S,
                                SOURCED_GEARBOXES, SOURCED_MOTORS,
                                MissingDatasheetValue, sourced_gearbox,
                                sourced_motor, unsourced_report)


# --- provenance --------------------------------------------------------------

def test_every_stored_number_names_a_document_and_a_printed_figure():
    for part in list(SOURCED_MOTORS) + list(SOURCED_GEARBOXES):
        part.check_provenance()
        assert part.documents, part.id
        for document in part.documents:
            assert document.url.startswith("https://")
            assert document.read_on.startswith("2026-")
        for value in part.value_sources:
            assert value.printed_as.strip(), (part.id, value.field)


def test_a_value_without_a_source_is_refused():
    """The validator that makes the rule enforceable rather than aspirational."""
    motor = sourced_motor("cubemars_ak80_9_v3")
    copy = type(motor)(**{**motor.__dict__,
                          "value_sources": list(motor.value_sources)[:-1]})
    with pytest.raises(ValueError, match="no source"):
        copy.check_provenance()


def test_a_source_naming_a_document_that_is_not_listed_is_refused():
    from drivetrain.sourced import ValueSource

    motor = sourced_motor("cubemars_ak80_9_v3")
    copy = type(motor)(**{**motor.__dict__,
                          "value_sources": list(motor.value_sources)
                          + [ValueSource("mass_kg", "490 g", "some other page")]})
    with pytest.raises(ValueError, match="not in the document list"):
        copy.check_provenance()


# --- the conversions the printed data supports, and the ones it does not -----

def test_the_unit_conversions_are_the_printed_figures():
    """Every stored SI value must be the printed one converted, and the
    printed form is kept so the conversion can be checked by eye."""
    maxon = sourced_motor("maxon_ec_i_40_100w_48v")
    assert maxon.nominal_torque_nm == pytest.approx(0.224)
    assert maxon.rotor_inertia_kg_m2 == pytest.approx(44.0 * 1e-7)
    assert maxon.rotor_inertia_kg_m2 == pytest.approx(4.4e-6)
    assert maxon.nominal_speed_rad_s == pytest.approx(4390.0 * 2 * math.pi / 60)
    assert maxon.mass_kg == 0.390
    assert GCM2_TO_KG_M2 == 1e-7 and MNM_TO_NM == 1e-3
    assert RPM_TO_RAD_S == pytest.approx(0.10471975511965978)

    ak80 = sourced_motor("cubemars_ak80_9_v3")
    assert ak80.nominal_speed_rad_s == pytest.approx(390.0 * RPM_TO_RAD_S)
    assert ak80.rotor_inertia_kg_m2 == pytest.approx(1118.3238e-7)


def test_a_motor_with_a_published_peak_becomes_a_selectable_spec():
    spec = sourced_motor("cubemars_ak80_9_v3").as_motor_spec()
    assert spec.status is PartStatus.VENDOR_DATASHEET
    assert spec.continuous_torque_nm == 9.0
    assert spec.peak_torque_nm == 22.0
    assert "cubemars.com" in spec.source
    assert "2026-09-03" in spec.source


def test_a_motor_without_a_published_peak_is_refused_by_name():
    """The maxon page prints a nominal torque and a stall torque and no peak.
    A stall torque is not a peak rating, and choosing between them is the
    vendor's statement to make."""
    with pytest.raises(MissingDatasheetValue, match="peak_torque_nm"):
        sourced_motor("maxon_ec_i_40_100w_48v").as_motor_spec()


def test_a_gear_unit_without_an_efficiency_is_refused_by_name():
    """The inertia came later from the catalogue rating table. The efficiency
    did not, and cannot: the catalogue gives it as curves against ambient
    temperature for each ratio and input speed at rated torque, with about
    three percent scatter and a compensation coefficient below rated torque.
    Collapsing that into one number would be inventing an operating point."""
    gearbox = sourced_gearbox("harmonic_csf_17_50_2uh")
    assert gearbox.input_inertia_kg_m2 == pytest.approx(7.9e-6)
    assert gearbox.efficiency is None
    assert "EFFICIENCY IS NOT A NUMBER" in gearbox.notes
    with pytest.raises(MissingDatasheetValue, match="efficiency"):
        gearbox.as_gearbox_spec()


def test_a_borrowed_value_needs_its_own_source():
    gearbox = sourced_gearbox("harmonic_csf_17_50_2uh")
    with pytest.raises(ValueError, match="its own source"):
        gearbox.as_gearbox_spec(input_inertia_kg_m2=3.3e-6, efficiency=0.8)
    spec = gearbox.as_gearbox_spec(input_inertia_kg_m2=3.3e-6, efficiency=0.8,
                                   inertia_source="component catalogue table, to be replaced")
    assert spec.status is PartStatus.VENDOR_DATASHEET
    assert spec.ratio == 50.0
    assert spec.rated_output_torque_nm == 16.0
    assert spec.peak_output_torque_nm == 186.0 or spec.peak_output_torque_nm == 70.0
    assert "to be replaced" in spec.source


# --- what the pages did not print --------------------------------------------

def test_the_gaps_are_reported_rather_than_filled():
    report = unsourced_report()
    assert "peak_torque_nm" in report["motors"]["maxon_ec_i_40_100w_48v"]
    assert "efficiency" in report["gearboxes"]["harmonic_csf_17_50_2uh"]
    assert "input_inertia_kg_m2" in report["gearboxes"]["nabtesco_rv_42n"]
    # The Nabtesco page lists eight ratios and one rated torque, so no single
    # ratio is stored and the caller has to say which one it is using.
    assert "ratio" in report["gearboxes"]["nabtesco_rv_42n"]
    assert sourced_gearbox("nabtesco_rv_42n").family is GearboxFamily.CYCLOIDAL


def test_the_harmonic_stiffness_is_the_k3_value_and_says_so():
    gearbox = sourced_gearbox("harmonic_csf_17_50_2uh")
    assert gearbox.torsional_stiffness_nm_rad == 1.3e4
    assert "K3" in gearbox.notes or any("K3" in v.printed_as
                                        for v in gearbox.value_sources)
    assert "K1" in gearbox.notes


def test_the_cycloidal_stiffness_is_converted_from_newton_metres_per_arcminute():
    """113 Nm/arc.min is 3.885e5 Nm/rad, and getting that conversion wrong by
    the 3438 factor would make the unit look eight orders of magnitude stiffer
    than a harmonic drive rather than one."""
    gearbox = sourced_gearbox("nabtesco_rv_42n")
    per_arcmin = 113.0
    assert gearbox.torsional_stiffness_nm_rad == pytest.approx(
        per_arcmin * 180.0 * 60.0 / math.pi)
    assert gearbox.torsional_stiffness_nm_rad == pytest.approx(3.885e5, rel=1e-3)


def test_the_integrated_actuator_says_its_ratings_are_at_the_output():
    ak80 = sourced_motor("cubemars_ak80_9_v3")
    assert ak80.gear_ratio == 9.0
    assert "OUTPUT" in ak80.notes


# --- the wider pool, its grades and its conditions ---------------------------

def test_the_pool_covers_several_manufacturers_and_both_paths():
    from drivetrain.sourced import PartGrade

    makers = {motor.manufacturer for motor in SOURCED_MOTORS}
    assert len(SOURCED_MOTORS) >= 14
    assert {"ROBOTIS", "CubeMars", "Kollmorgen", "maxon", "mjbots",
            "DAMIAO"} <= makers
    assert len(SOURCED_GEARBOXES) >= 7
    assert {g.family.value for g in SOURCED_GEARBOXES if g.family} >= {
        "planetary", "harmonic", "cycloidal"}
    assert {m.grade for m in SOURCED_MOTORS} >= {PartGrade.INDUSTRIAL,
                                                PartGrade.ROBOTICS_MODULE}


def test_a_distributor_listing_is_labelled_as_one():
    """A retailer's page is a document and it is not a data sheet."""
    from drivetrain.sourced import DocumentKind, sourced_motor

    damiao = sourced_motor("damiao_dm_j8009_2ec")
    assert damiao.documents[0].kind is DocumentKind.DISTRIBUTOR_PAGE
    assert "DISTRIBUTOR" in damiao.notes
    robotis = sourced_motor("robotis_ph54_200_s500_r")
    assert robotis.documents[0].kind is DocumentKind.MANUFACTURER_MANUAL


def test_every_peak_carries_its_condition_or_says_it_has_none():
    for motor in SOURCED_MOTORS:
        assert motor.peak_torque_condition.strip()
        if motor.peak_torque_nm is None:
            lowered = motor.peak_torque_condition.lower()
            assert ("no peak" in lowered or "stall" in lowered
                    or "not stated" in lowered), motor.id
    from drivetrain.sourced import sourced_motor
    assert "less than 1 second" in sourced_motor(
        "mjbots_qdd100_beta3").peak_torque_condition
    assert "25 C winding" in sourced_motor(
        "kollmorgen_tbm_6051_a").peak_torque_condition


def test_a_stall_torque_is_not_stored_as_a_rating():
    """The XM540 page prints a stall torque and no continuous one, so the
    entry has no nominal torque at all and cannot be selected on."""
    from drivetrain.sourced import sourced_motor

    servo = sourced_motor("robotis_xm540_w270")
    assert servo.nominal_torque_nm is None
    assert servo.stall_torque_nm == 10.6
    assert servo.bus_voltage_v == 12.0
    assert "not a continuous rating" in servo.peak_torque_condition


def test_the_bus_voltage_is_stored_with_the_figures_it_belongs_to():
    from drivetrain.sourced import sourced_motor

    assert sourced_motor("robotis_ph54_200_s500_r").bus_voltage_v == 24.0
    assert sourced_motor("cubemars_ak80_64_kv80").bus_voltage_v == 48.0
    assert sourced_motor("mjbots_qdd100_beta3").bus_voltage_v == 36.0
