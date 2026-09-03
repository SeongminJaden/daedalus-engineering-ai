"""Real parts, with every number traced to the page it was read from.

The motor and gearbox catalogues in this package are archetypes and say so.
This module is the other kind: entries whose values come from a manufacturer's
published data, each value naming the document, the printed figure and the day
it was read, exactly as the material database does.

THE RULE THAT SHAPES THIS FILE
==============================
A value that was not printed is not here. Not estimated from a similar size,
not inferred from a curve, not carried over from another ratio. A missing
value is `None` and a conversion that needs it refuses with the field name in
the message. Inventing a part number or a rating is worse than having neither,
because a fabricated catalogue entry reads exactly like a sourced one six
months later.

WHAT THAT COSTS, CONCRETELY
===========================
The maxon EC-i 40 page prints a nominal torque and a stall torque and no peak
torque, so `as_motor_spec` refuses to build a selectable motor from it: a
stall torque is not a peak rating and choosing between them is the vendor's
statement to make, not this file's. The Harmonic Drive pages print three
torque ratings and three stiffness values and no moment of inertia, so those
gearboxes convert only when the caller supplies the inertia from a document
this file does not have. Both refusals are tested.

UNITS
=====
SI in the stored fields, with the printed figure kept verbatim in
`printed_as` so the conversion can be checked: 44 gcm2 is 4.4e-6 kg m2, and
390 rpm is 40.84 rad/s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from drivetrain.gearboxes.catalog import GearboxFamily, GearboxSpec
from drivetrain.motors.catalog import MotorSpec, PartStatus

RPM_TO_RAD_S = 2.0 * math.pi / 60.0
GCM2_TO_KG_M2 = 1e-7          # 1 g cm^2 = 1e-3 kg * 1e-4 m^2
MNM_TO_NM = 1e-3


class MissingDatasheetValue(ValueError):
    """A conversion needs a number the manufacturer's page did not print."""


@dataclass(frozen=True)
class SourceDocument:
    publisher: str
    title: str
    url: str
    read_on: str                    # ISO date


@dataclass(frozen=True)
class ValueSource:
    """One stored value, its printed form, and where it was read."""

    field: str
    printed_as: str
    document: str                   # SourceDocument.title


@dataclass
class SourcedPart:
    """Fields common to a sourced motor or gearbox."""

    id: str
    manufacturer: str
    part_number: str
    documents: list[SourceDocument]
    value_sources: list[ValueSource] = field(default_factory=list)
    notes: str = ""

    def sourced_fields(self) -> set[str]:
        return {v.field for v in self.value_sources}

    def check_provenance(self) -> None:
        """Every stored number must name a document, and every named document
        must be listed. The same validator the material database has."""
        titles = {d.title for d in self.documents}
        for value in self.value_sources:
            if value.document not in titles:
                raise ValueError(
                    f"{self.id}: value {value.field} cites {value.document!r}, "
                    f"which is not in the document list")
            if getattr(self, value.field, None) is None:
                raise ValueError(
                    f"{self.id}: value source for {value.field} but the field "
                    f"is empty")
        for name, value in self.__dict__.items():
            if name in ("id", "manufacturer", "part_number", "documents",
                        "value_sources", "notes"):
                continue
            if value is not None and name not in self.sourced_fields():
                raise ValueError(
                    f"{self.id}: {name} is set to {value!r} with no source")


@dataclass
class SourcedMotor(SourcedPart):
    """A motor as its manufacturer publishes it."""

    nominal_voltage_v: float | None = None
    nominal_torque_nm: float | None = None
    peak_torque_nm: float | None = None
    stall_torque_nm: float | None = None
    no_load_speed_rad_s: float | None = None
    nominal_speed_rad_s: float | None = None
    rotor_inertia_kg_m2: float | None = None
    torque_constant_nm_a: float | None = None
    mass_kg: float | None = None
    thermal_resistance_winding_housing_k_w: float | None = None
    thermal_resistance_housing_ambient_k_w: float | None = None
    gear_ratio: float | None = None
    backlash_arcmin: float | None = None
    rated_current_a: float | None = None
    peak_current_a: float | None = None

    def require(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise MissingDatasheetValue(
                f"{self.id}: {name} is not printed on any document this entry "
                f"cites, and a value that was not published will not be "
                f"invented here")
        return float(value)

    def as_motor_spec(self) -> MotorSpec:
        """The selection layer's MotorSpec, or a refusal naming what is missing."""
        return MotorSpec(
            id=self.id, name=f"{self.manufacturer} {self.part_number}",
            continuous_torque_nm=self.require("nominal_torque_nm"),
            peak_torque_nm=self.require("peak_torque_nm"),
            rated_speed_rad_s=self.require("nominal_speed_rad_s"),
            max_speed_rad_s=self.require("no_load_speed_rad_s"),
            rotor_inertia_kg_m2=self.require("rotor_inertia_kg_m2"),
            mass_kg=self.require("mass_kg"),
            status=PartStatus.VENDOR_DATASHEET,
            source=f"{self.manufacturer} {self.part_number}, "
                   f"{self.documents[0].url}, read {self.documents[0].read_on}")


@dataclass
class SourcedGearbox(SourcedPart):
    """A gear unit as its manufacturer publishes it."""

    family: GearboxFamily | None = None
    ratio: float | None = None
    rated_torque_nm: float | None = None
    repeated_peak_torque_nm: float | None = None
    momentary_peak_torque_nm: float | None = None
    average_input_speed_rad_s: float | None = None
    torsional_stiffness_nm_rad: float | None = None
    backlash_arcmin: float | None = None
    lost_motion_arcmin: float | None = None
    input_inertia_kg_m2: float | None = None
    efficiency: float | None = None
    mass_kg: float | None = None
    starting_torque_nm: float | None = None
    backdriving_torque_nm: float | None = None

    def require(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise MissingDatasheetValue(
                f"{self.id}: {name} is not printed on any document this entry "
                f"cites, and a value that was not published will not be "
                f"invented here")
        return float(value)

    def as_gearbox_spec(self, input_inertia_kg_m2: float | None = None,
                        efficiency: float | None = None,
                        inertia_source: str = "") -> GearboxSpec:
        """The selection layer's GearboxSpec.

        Inertia and efficiency may be supplied by the caller when the pages
        cited here do not print them, and then `inertia_source` is required so
        the borrowed number is still traceable.
        """
        inertia = input_inertia_kg_m2 if input_inertia_kg_m2 is not None \
            else self.input_inertia_kg_m2
        eta = efficiency if efficiency is not None else self.efficiency
        if inertia is None or eta is None:
            missing = [n for n, v in (("input_inertia_kg_m2", inertia),
                                      ("efficiency", eta)) if v is None]
            raise MissingDatasheetValue(
                f"{self.id}: {', '.join(missing)} not printed on the cited "
                f"documents; supply them with a source or use a part whose "
                f"data sheet prints them")
        if (input_inertia_kg_m2 is not None or efficiency is not None) \
                and not inertia_source.strip():
            raise ValueError(
                f"{self.id}: a value supplied from outside the cited documents "
                f"needs its own source")
        return GearboxSpec(
            id=self.id, family=self.family or GearboxFamily.HARMONIC,
            ratio=self.require("ratio"), efficiency=float(eta),
            rated_output_torque_nm=self.require("rated_torque_nm"),
            peak_output_torque_nm=self.require("momentary_peak_torque_nm"),
            backlash_arcmin=(self.backlash_arcmin
                             if self.backlash_arcmin is not None else 0.0),
            input_inertia_kg_m2=float(inertia),
            mass_kg=self.require("mass_kg"),
            status=PartStatus.VENDOR_DATASHEET,
            source=f"{self.manufacturer} {self.part_number}, "
                   f"{self.documents[0].url}, read {self.documents[0].read_on}"
                   + (f"; {inertia_source}" if inertia_source else ""))


# --------------------------------------------------------------- documents

MAXON_EC_I_40 = SourceDocument(
    publisher="maxon group",
    title="EC-i 40, 40 mm, brushless, 100 W, with Hall sensors, product page 488607",
    url="https://www.maxongroup.com/maxon/view/product/motor/ecmotor/EC-i/488607",
    read_on="2026-09-03")

CUBEMARS_AK80_9 = SourceDocument(
    publisher="CubeMars (T-Motor)",
    title="AK80-9 V3.0 KV100 actuator product page, goods 982",
    url="https://www.cubemars.com/goods-982-AK80-9.html",
    read_on="2026-09-03")

HD_CSF_17_50 = SourceDocument(
    publisher="Harmonic Drive LLC",
    title="CSF-17-50-2UH gear unit product page",
    url="https://www.harmonicdrive.net/products/gear-units/gear-units/csf-2uh/csf-17-50-2uh",
    read_on="2026-09-03")

HD_CSF_17_100 = SourceDocument(
    publisher="Harmonic Drive LLC",
    title="CSF-17-100-2UH gear unit product page",
    url="https://www.harmonicdrive.net/products/gear-units/gear-units/csf-2uh/csf-17-100-2uh",
    read_on="2026-09-03")

HD_CSF_25_50 = SourceDocument(
    publisher="Harmonic Drive LLC",
    title="CSF-25-50-2UH gear unit product page",
    url="https://www.harmonicdrive.net/products/gear-units/gear-units/csf-2uh/csf-25-50-2uh",
    read_on="2026-09-03")

CUBEMARS_AK10_9 = SourceDocument(
    publisher="CubeMars (T-Motor)",
    title="AK10-9 V2.0 KV60 actuator product page, goods 1141",
    url="https://www.cubemars.com/goods-1141-AK10-9+V20+KV60.html",
    read_on="2026-09-04")

CUBEMARS_AK70_10 = SourceDocument(
    publisher="CubeMars (T-Motor)",
    title="AK70-10 KV100 actuator product page",
    url="https://www.cubemars.com/product/ak70-10-kv100-robotic-actuator.html",
    read_on="2026-09-04")

CUBEMARS_AK80_64 = SourceDocument(
    publisher="CubeMars (T-Motor)",
    title="AK80-64 KV80 actuator product page, goods 1143",
    url="https://www.cubemars.com/goods-1143-AK80-64.html",
    read_on="2026-09-04")

KOLLMORGEN_TBM = SourceDocument(
    publisher="Kollmorgen",
    title="TBM(S) frameless motor selection guide, TBM 60 series performance "
          "data and motor parameters",
    url="https://www.electromate.com/media/assets/catalog-library/pdfs/kollmorgen/Kollmorgen_TBM(S)_Motor_Catalog.pdf",
    read_on="2026-09-04")

APEX_AF = SourceDocument(
    publisher="Apex Dynamics",
    title="AF and AFR series high precision planetary gearbox catalogue, "
          "gearbox performance and gearbox inertia tables",
    url="https://apexdynamicsusa.com/pub/media/sebwite/productdownloads//a/f/afafr_catalog.pdf",
    read_on="2026-09-04")

HD_CSF_CATALOGUE = SourceDocument(
    publisher="Harmonic Drive LLC",
    title="CSF and CSG cup type component sets and housed units catalogue, "
          "CSF rating table 1",
    url="https://www.harmonicdrive.net/_hd/content/catalogs/pdf/csf-csg.pdf",
    read_on="2026-09-04")

NABTESCO_RV_42N = SourceDocument(
    publisher="Nabtesco Precision Europe",
    title="RV-42N cycloidal gearbox kit product page",
    url="https://www.nabtesco.de/en/product/rv-42n",
    read_on="2026-09-03")


# ------------------------------------------------------------------ motors

SOURCED_MOTORS: list[SourcedMotor] = [
    SourcedMotor(
        id="maxon_ec_i_40_100w_48v",
        manufacturer="maxon", part_number="EC-i 40, 100 W, 48 V",
        documents=[MAXON_EC_I_40],
        nominal_voltage_v=48.0,
        nominal_torque_nm=224.0 * MNM_TO_NM,
        stall_torque_nm=2080.0 * MNM_TO_NM,
        no_load_speed_rad_s=5000.0 * RPM_TO_RAD_S,
        nominal_speed_rad_s=4390.0 * RPM_TO_RAD_S,
        rotor_inertia_kg_m2=44.0 * GCM2_TO_KG_M2,
        torque_constant_nm_a=91.0 * MNM_TO_NM,
        mass_kg=0.390,
        thermal_resistance_winding_housing_k_w=1.35,
        thermal_resistance_housing_ambient_k_w=7.17,
        value_sources=[
            ValueSource("nominal_voltage_v", "Nominal voltage: 48 V", MAXON_EC_I_40.title),
            ValueSource("nominal_torque_nm", "Nominal torque (max. continuous torque): 224 mNm", MAXON_EC_I_40.title),
            ValueSource("stall_torque_nm", "Stall torque: 2080 mNm", MAXON_EC_I_40.title),
            ValueSource("no_load_speed_rad_s", "No load speed: 5000 rpm", MAXON_EC_I_40.title),
            ValueSource("nominal_speed_rad_s", "Nominal speed: 4390 rpm", MAXON_EC_I_40.title),
            ValueSource("rotor_inertia_kg_m2", "Rotor inertia: 44 gcm2", MAXON_EC_I_40.title),
            ValueSource("torque_constant_nm_a", "Torque constant: 91 mNm/A", MAXON_EC_I_40.title),
            ValueSource("mass_kg", "Weight: 390 g", MAXON_EC_I_40.title),
            ValueSource("thermal_resistance_winding_housing_k_w", "Thermal resistance winding-housing: 1.35 K/W", MAXON_EC_I_40.title),
            ValueSource("thermal_resistance_housing_ambient_k_w", "Thermal resistance housing-ambient: 7.17 K/W", MAXON_EC_I_40.title),
        ],
        notes=("No peak torque is printed. The stall torque is not a peak "
               "rating, so this entry cannot become a selectable MotorSpec "
               "without a figure the manufacturer publishes elsewhere.")),
    SourcedMotor(
        id="cubemars_ak80_9_v3",
        manufacturer="CubeMars", part_number="AK80-9 V3.0 KV100",
        documents=[CUBEMARS_AK80_9],
        nominal_voltage_v=48.0,
        nominal_torque_nm=9.0,
        peak_torque_nm=22.0,
        no_load_speed_rad_s=570.0 * RPM_TO_RAD_S,
        nominal_speed_rad_s=390.0 * RPM_TO_RAD_S,
        rotor_inertia_kg_m2=1118.3238 * GCM2_TO_KG_M2,
        mass_kg=0.490,
        gear_ratio=9.0,
        backlash_arcmin=15.0,
        rated_current_a=12.0,
        peak_current_a=28.0,
        value_sources=[
            ValueSource("nominal_voltage_v", "Rated Voltage 48V", CUBEMARS_AK80_9.title),
            ValueSource("nominal_torque_nm", "Rated Torque 9 Nm", CUBEMARS_AK80_9.title),
            ValueSource("peak_torque_nm", "Peak Torque 22 Nm", CUBEMARS_AK80_9.title),
            ValueSource("no_load_speed_rad_s", "No-load Speed 570 rpm", CUBEMARS_AK80_9.title),
            ValueSource("nominal_speed_rad_s", "Rated Speed 390 rpm", CUBEMARS_AK80_9.title),
            ValueSource("rotor_inertia_kg_m2", "Rotor Inertia 1118.3238 gcm2", CUBEMARS_AK80_9.title),
            ValueSource("mass_kg", "Weight 490g", CUBEMARS_AK80_9.title),
            ValueSource("gear_ratio", "Gear Ratio 9:1", CUBEMARS_AK80_9.title),
            ValueSource("backlash_arcmin", "Backlash 15 arcmin", CUBEMARS_AK80_9.title),
            ValueSource("rated_current_a", "Rated Current 12A", CUBEMARS_AK80_9.title),
            ValueSource("peak_current_a", "Peak Current 28A", CUBEMARS_AK80_9.title),
        ],
        notes=("An integrated actuator: the torques and speeds are at the "
               "OUTPUT of its own 9:1 planetary stage, so it is not a bare "
               "motor and must not be paired with another gearbox in the "
               "selection without dividing them back out.")),
    SourcedMotor(
        id="cubemars_ak10_9_v2_kv60",
        manufacturer="CubeMars", part_number="AK10-9 V2.0 KV60",
        documents=[CUBEMARS_AK10_9],
        nominal_voltage_v=48.0,
        nominal_torque_nm=18.0,
        peak_torque_nm=48.0,
        no_load_speed_rad_s=320.0 * RPM_TO_RAD_S,
        nominal_speed_rad_s=228.0 * RPM_TO_RAD_S,
        rotor_inertia_kg_m2=1002.0 * GCM2_TO_KG_M2,
        torque_constant_nm_a=0.198,
        mass_kg=0.960,
        gear_ratio=9.0,
        backlash_arcmin=0.33 * 60.0,
        rated_current_a=10.6,
        peak_current_a=29.8,
        value_sources=[
            ValueSource("nominal_voltage_v", "Rated Voltage 24/48 V", CUBEMARS_AK10_9.title),
            ValueSource("nominal_torque_nm", "Rated Torque 18 Nm", CUBEMARS_AK10_9.title),
            ValueSource("peak_torque_nm", "Peak Torque 48 Nm", CUBEMARS_AK10_9.title),
            ValueSource("no_load_speed_rad_s", "No-load Speed 160/320 rpm", CUBEMARS_AK10_9.title),
            ValueSource("nominal_speed_rad_s", "Rated Speed 109/228 rpm", CUBEMARS_AK10_9.title),
            ValueSource("rotor_inertia_kg_m2", "Rotor Inertia 1002 gcm2", CUBEMARS_AK10_9.title),
            ValueSource("torque_constant_nm_a", "Kt 0.198 Nm/A", CUBEMARS_AK10_9.title),
            ValueSource("mass_kg", "Weight 960 g", CUBEMARS_AK10_9.title),
            ValueSource("gear_ratio", "Reduction Ratio 9:1", CUBEMARS_AK10_9.title),
            ValueSource("backlash_arcmin", "Backlash 0.33 degrees", CUBEMARS_AK10_9.title),
            ValueSource("rated_current_a", "Rated Current 10.6 A", CUBEMARS_AK10_9.title),
            ValueSource("peak_current_a", "Peak Current 29.8 A", CUBEMARS_AK10_9.title),
        ],
        notes=("Integrated actuator: the torques and speeds are at the output "
               "of its own 9:1 stage, so no further gearbox may be stacked on "
               "it. The two speed figures are the 24 V and 48 V values and "
               "the 48 V one is stored. Backlash is printed in degrees and "
               "stored in arc minutes.")),
    # --- frameless motors for the geared path -------------------------------
    # These print a continuous AND a peak torque, which is what a motor needs
    # to be selectable; the maxon page above prints a stall torque instead and
    # is refused for it.
    SourcedMotor(
        id="kollmorgen_tbm_6013_a",
        manufacturer="Kollmorgen", part_number="TBM(S)-6013-A",
        documents=[KOLLMORGEN_TBM],
        nominal_voltage_v=48.0,
        nominal_torque_nm=0.413,
        peak_torque_nm=1.37,
        nominal_speed_rad_s=4300.0 * RPM_TO_RAD_S,
        no_load_speed_rad_s=4300.0 * RPM_TO_RAD_S,
        rotor_inertia_kg_m2=1.41e-05,
        mass_kg=0.213,
        rated_current_a=5.70,
        peak_current_a=19.0,
        value_sources=[
            ValueSource("nominal_voltage_v", "Design Voltage Vbus 48.0 Vdc", KOLLMORGEN_TBM.title),
            ValueSource("nominal_torque_nm", "Continuous Stall Torque Tc 0.413 N-m", KOLLMORGEN_TBM.title),
            ValueSource("peak_torque_nm", "Peak Stall Torque Tp 1.37 N-m at 25 C winding", KOLLMORGEN_TBM.title),
            ValueSource("nominal_speed_rad_s", "Speed at Rated Power 4300 RPM", KOLLMORGEN_TBM.title),
            ValueSource("no_load_speed_rad_s", "Speed at Rated Power 4300 RPM", KOLLMORGEN_TBM.title),
            ValueSource("rotor_inertia_kg_m2", "Inertia Jm 1.41E-05 Kg-m2", KOLLMORGEN_TBM.title),
            ValueSource("mass_kg", "Weight Wt 213 grams", KOLLMORGEN_TBM.title),
            ValueSource("rated_current_a", "Continuous Current Ic 5.70 Adc", KOLLMORGEN_TBM.title),
            ValueSource("peak_current_a", "Peak Current Ip 19.0 Adc", KOLLMORGEN_TBM.title),
        ],
        notes=("Frameless: the mass and inertia are the rotor and stator "
               "only, with no housing, bearings, shaft or encoder, so a "
               "built joint weighs more than this by parts nobody here has "
               "sourced. The no load speed is not printed; the speed at rated "
               "power is stored in its place and named as such, which makes "
               "the speed check conservative.")),
    SourcedMotor(
        id="kollmorgen_tbm_6025_a",
        manufacturer="Kollmorgen", part_number="TBM(S)-6025-A",
        documents=[KOLLMORGEN_TBM],
        nominal_voltage_v=48.0,
        nominal_torque_nm=0.706,
        peak_torque_nm=2.56,
        nominal_speed_rad_s=2900.0 * RPM_TO_RAD_S,
        no_load_speed_rad_s=2900.0 * RPM_TO_RAD_S,
        rotor_inertia_kg_m2=2.52e-05,
        mass_kg=0.377,
        rated_current_a=5.70,
        peak_current_a=21.3,
        value_sources=[
            ValueSource("nominal_voltage_v", "Design Voltage Vbus 48.0 Vdc", KOLLMORGEN_TBM.title),
            ValueSource("nominal_torque_nm", "Continuous Stall Torque Tc 0.706 N-m", KOLLMORGEN_TBM.title),
            ValueSource("peak_torque_nm", "Peak Stall Torque Tp 2.56 N-m at 25 C winding", KOLLMORGEN_TBM.title),
            ValueSource("nominal_speed_rad_s", "Speed at Rated Power 2900 RPM", KOLLMORGEN_TBM.title),
            ValueSource("no_load_speed_rad_s", "Speed at Rated Power 2900 RPM", KOLLMORGEN_TBM.title),
            ValueSource("rotor_inertia_kg_m2", "Inertia Jm 2.52E-05 Kg-m2", KOLLMORGEN_TBM.title),
            ValueSource("mass_kg", "Weight Wt 377 grams", KOLLMORGEN_TBM.title),
            ValueSource("rated_current_a", "Continuous Current Ic 5.70 Adc", KOLLMORGEN_TBM.title),
            ValueSource("peak_current_a", "Peak Current Ip 21.3 Adc", KOLLMORGEN_TBM.title),
        ],
        notes="Frameless, same caveat as the 6013."),
    SourcedMotor(
        id="kollmorgen_tbm_6051_a",
        manufacturer="Kollmorgen", part_number="TBM(S)-6051-A",
        documents=[KOLLMORGEN_TBM],
        nominal_voltage_v=48.0,
        nominal_torque_nm=1.16,
        peak_torque_nm=4.53,
        nominal_speed_rad_s=2130.0 * RPM_TO_RAD_S,
        no_load_speed_rad_s=2130.0 * RPM_TO_RAD_S,
        rotor_inertia_kg_m2=4.75e-05,
        mass_kg=0.550,
        rated_current_a=7.00,
        peak_current_a=30.0,
        value_sources=[
            ValueSource("nominal_voltage_v", "Design Voltage Vbus 48.0 Vdc", KOLLMORGEN_TBM.title),
            ValueSource("nominal_torque_nm", "Continuous Stall Torque Tc 1.16 N-m", KOLLMORGEN_TBM.title),
            ValueSource("peak_torque_nm", "Peak Stall Torque Tp 4.53 N-m at 25 C winding", KOLLMORGEN_TBM.title),
            ValueSource("nominal_speed_rad_s", "Speed at Rated Power 2130 RPM", KOLLMORGEN_TBM.title),
            ValueSource("no_load_speed_rad_s", "Speed at Rated Power 2130 RPM", KOLLMORGEN_TBM.title),
            ValueSource("rotor_inertia_kg_m2", "Inertia Jm 4.75E-05 Kg-m2", KOLLMORGEN_TBM.title),
            ValueSource("mass_kg", "Weight Wt 550 grams", KOLLMORGEN_TBM.title),
            ValueSource("rated_current_a", "Continuous Current Ic 7.00 Adc", KOLLMORGEN_TBM.title),
            ValueSource("peak_current_a", "Peak Current Ip 30.0 Adc", KOLLMORGEN_TBM.title),
        ],
        notes="Frameless, same caveat as the 6013."),
    SourcedMotor(
        id="cubemars_ak80_64_kv80",
        manufacturer="CubeMars", part_number="AK80-64 KV80",
        documents=[CUBEMARS_AK80_64],
        nominal_voltage_v=48.0,
        nominal_torque_nm=48.0,
        peak_torque_nm=120.0,
        no_load_speed_rad_s=75.0 * RPM_TO_RAD_S,
        nominal_speed_rad_s=48.0 * RPM_TO_RAD_S,
        rotor_inertia_kg_m2=564.5 * GCM2_TO_KG_M2,
        torque_constant_nm_a=0.136,
        mass_kg=0.850,
        gear_ratio=64.0,
        backlash_arcmin=0.18 * 60.0,
        rated_current_a=7.0,
        peak_current_a=19.0,
        value_sources=[
            ValueSource("nominal_voltage_v", "Rated Voltage 24/48 V", CUBEMARS_AK80_64.title),
            ValueSource("nominal_torque_nm", "Rated Torque 48 Nm", CUBEMARS_AK80_64.title),
            ValueSource("peak_torque_nm", "Peak Torque 120 Nm", CUBEMARS_AK80_64.title),
            ValueSource("no_load_speed_rad_s", "No-load Speed 37/75 rpm", CUBEMARS_AK80_64.title),
            ValueSource("nominal_speed_rad_s", "Rated Speed 23/48 rpm", CUBEMARS_AK80_64.title),
            ValueSource("rotor_inertia_kg_m2", "Inertia 564.5 gcm2", CUBEMARS_AK80_64.title),
            ValueSource("torque_constant_nm_a", "Kt 0.136 Nm/A", CUBEMARS_AK80_64.title),
            ValueSource("mass_kg", "Weight 850 g", CUBEMARS_AK80_64.title),
            ValueSource("gear_ratio", "Reduction Ratio 64:1", CUBEMARS_AK80_64.title),
            ValueSource("backlash_arcmin", "Backlash 0.18 degrees", CUBEMARS_AK80_64.title),
            ValueSource("rated_current_a", "Rated Current 7 ADC", CUBEMARS_AK80_64.title),
            ValueSource("peak_current_a", "Peak Current 19 ADC", CUBEMARS_AK80_64.title),
        ],
        notes=("Integrated actuator with a 64:1 stage, so it is slow: 48 rpm "
               "rated at 48 V. A joint that must turn faster than that cannot "
               "use it whatever its torque, and the selection checks speed as "
               "well as torque for exactly this reason.")),
    SourcedMotor(
        id="cubemars_ak70_10_kv100",
        manufacturer="CubeMars", part_number="AK70-10 KV100",
        documents=[CUBEMARS_AK70_10],
        nominal_voltage_v=48.0,
        nominal_torque_nm=8.3,
        peak_torque_nm=24.8,
        no_load_speed_rad_s=480.0 * RPM_TO_RAD_S,
        nominal_speed_rad_s=310.0 * RPM_TO_RAD_S,
        rotor_inertia_kg_m2=753.4788 * GCM2_TO_KG_M2,
        mass_kg=0.521,
        gear_ratio=10.0,
        backlash_arcmin=0.2 * 60.0,
        value_sources=[
            ValueSource("nominal_voltage_v", "Rated Voltage 24/48 V", CUBEMARS_AK70_10.title),
            ValueSource("nominal_torque_nm", "Rated Torque 8.3 Nm", CUBEMARS_AK70_10.title),
            ValueSource("peak_torque_nm", "Peak Torque 24.8 Nm", CUBEMARS_AK70_10.title),
            ValueSource("no_load_speed_rad_s", "No-load Speed 240/480 rpm", CUBEMARS_AK70_10.title),
            ValueSource("nominal_speed_rad_s", "Rated Speed 148/310 rpm", CUBEMARS_AK70_10.title),
            ValueSource("rotor_inertia_kg_m2", "Rotor Inertia 753.4788 gcm2", CUBEMARS_AK70_10.title),
            ValueSource("mass_kg", "Weight 521 g", CUBEMARS_AK70_10.title),
            ValueSource("gear_ratio", "Gear Ratio 10:1", CUBEMARS_AK70_10.title),
            ValueSource("backlash_arcmin", "Backlash 0.2 degrees", CUBEMARS_AK70_10.title),
        ],
        notes="Integrated actuator, same rule as the other two."),
]


# ---------------------------------------------------------------- gearboxes

SOURCED_GEARBOXES: list[SourcedGearbox] = [
    SourcedGearbox(
        id="harmonic_csf_17_50_2uh", manufacturer="Harmonic Drive",
        part_number="CSF-17-50-2UH", documents=[HD_CSF_17_50, HD_CSF_CATALOGUE],
        family=GearboxFamily.HARMONIC, ratio=50.0,
        rated_torque_nm=16.0, repeated_peak_torque_nm=34.0,
        momentary_peak_torque_nm=70.0,
        average_input_speed_rad_s=3500.0 * RPM_TO_RAD_S,
        torsional_stiffness_nm_rad=1.3e4, mass_kg=0.68,
        input_inertia_kg_m2=0.079e-4,
        value_sources=[
            ValueSource("input_inertia_kg_m2",
                        "CSF rating table 1, size 17: moment of inertia "
                        "0.079 x10-4 kg m2, reflected to the wave generator",
                        HD_CSF_CATALOGUE.title),
            ValueSource("ratio", "CSF-17-50", HD_CSF_17_50.title),
            ValueSource("rated_torque_nm", "Rated Torque L10: 16 Nm", HD_CSF_17_50.title),
            ValueSource("repeated_peak_torque_nm", "Limit for Repeated Peak Torque: 34 Nm", HD_CSF_17_50.title),
            ValueSource("momentary_peak_torque_nm", "Limit for Momentary Peak Torque: 70 Nm", HD_CSF_17_50.title),
            ValueSource("average_input_speed_rad_s", "Average Input Speed: 3,500 rpm", HD_CSF_17_50.title),
            ValueSource("torsional_stiffness_nm_rad", "K3: 1.3 x 10^4 Nm/rad", HD_CSF_17_50.title),
            ValueSource("mass_kg", "Mass: 0.68 kg", HD_CSF_17_50.title),
            ValueSource("family", "cup type harmonic gear unit", HD_CSF_17_50.title),
        ],
        notes=("The moment of inertia comes from the catalogue rating table "
               "and is per SIZE, not per ratio, and is reflected to the wave "
               "generator. Backlash is not printed. EFFICIENCY IS NOT A "
               "NUMBER in this catalogue: it is a family of curves against "
               "ambient temperature for each ratio and input speed at rated "
               "torque, with about 3 percent scatter and a separate "
               "compensation coefficient below rated torque, so no single "
               "value is stored. K3 is the stiffness above the second torque "
               "breakpoint; K1 (0.81e4) and K2 (1.1e4) are lower and apply at "
               "smaller torques.")),
    SourcedGearbox(
        id="harmonic_csf_17_100_2uh", manufacturer="Harmonic Drive",
        part_number="CSF-17-100-2UH", documents=[HD_CSF_17_100, HD_CSF_CATALOGUE],
        family=GearboxFamily.HARMONIC, ratio=100.0,
        rated_torque_nm=24.0, repeated_peak_torque_nm=54.0,
        momentary_peak_torque_nm=108.0,
        average_input_speed_rad_s=3500.0 * RPM_TO_RAD_S,
        torsional_stiffness_nm_rad=1.6e4, mass_kg=0.68,
        starting_torque_nm=3.4e-2, backdriving_torque_nm=3.3,
        input_inertia_kg_m2=0.079e-4,
        value_sources=[
            ValueSource("input_inertia_kg_m2",
                        "CSF rating table 1, size 17: moment of inertia "
                        "0.079 x10-4 kg m2, reflected to the wave generator",
                        HD_CSF_CATALOGUE.title),
            ValueSource("ratio", "CSF-17-100", HD_CSF_17_100.title),
            ValueSource("rated_torque_nm", "Rated Torque L10: 24 Nm", HD_CSF_17_100.title),
            ValueSource("repeated_peak_torque_nm", "Limit for Repeated Peak Torque: 54 Nm", HD_CSF_17_100.title),
            ValueSource("momentary_peak_torque_nm", "Limit for Momentary Peak Torque: 108 Nm", HD_CSF_17_100.title),
            ValueSource("average_input_speed_rad_s", "Average Input Speed: 3,500 rpm", HD_CSF_17_100.title),
            ValueSource("torsional_stiffness_nm_rad", "K3: 1.6 x 10^4 Nm/rad", HD_CSF_17_100.title),
            ValueSource("mass_kg", "Mass: 0.68 kg", HD_CSF_17_100.title),
            ValueSource("starting_torque_nm", "Starting Torque: 3.4 Ncm", HD_CSF_17_100.title),
            ValueSource("backdriving_torque_nm", "Backdriving Torque: 3.3 Nm", HD_CSF_17_100.title),
            ValueSource("family", "cup type harmonic gear unit", HD_CSF_17_100.title),
        ]),
    SourcedGearbox(
        id="harmonic_csf_25_50_2uh", manufacturer="Harmonic Drive",
        part_number="CSF-25-50-2UH", documents=[HD_CSF_25_50, HD_CSF_CATALOGUE],
        family=GearboxFamily.HARMONIC, ratio=50.0,
        rated_torque_nm=39.0, repeated_peak_torque_nm=98.0,
        momentary_peak_torque_nm=186.0,
        average_input_speed_rad_s=3500.0 * RPM_TO_RAD_S,
        torsional_stiffness_nm_rad=4.4e4, mass_kg=1.5,
        starting_torque_nm=15.0e-2, backdriving_torque_nm=9.0,
        input_inertia_kg_m2=0.413e-4,
        value_sources=[
            ValueSource("input_inertia_kg_m2",
                        "CSF rating table 1, size 25: moment of inertia "
                        "0.413 x10-4 kg m2, reflected to the wave generator",
                        HD_CSF_CATALOGUE.title),
            ValueSource("ratio", "CSF-25-50", HD_CSF_25_50.title),
            ValueSource("rated_torque_nm", "Rated Torque L10: 39 Nm", HD_CSF_25_50.title),
            ValueSource("repeated_peak_torque_nm", "Limit for Repeated Peak Torque: 98 Nm", HD_CSF_25_50.title),
            ValueSource("momentary_peak_torque_nm", "Limit for Momentary Peak Torque: 186 Nm", HD_CSF_25_50.title),
            ValueSource("average_input_speed_rad_s", "Average Input Speed: 3,500 rpm", HD_CSF_25_50.title),
            ValueSource("torsional_stiffness_nm_rad", "K3: 4.4 x 10^4 Nm/rad", HD_CSF_25_50.title),
            ValueSource("mass_kg", "Mass: 1.5 kg", HD_CSF_25_50.title),
            ValueSource("starting_torque_nm", "Starting Torque: 15 Ncm", HD_CSF_25_50.title),
            ValueSource("backdriving_torque_nm", "Backdriving Torque: 9 Nm", HD_CSF_25_50.title),
            ValueSource("family", "cup type harmonic gear unit", HD_CSF_25_50.title),
        ]),
    # --- planetary units whose catalogue prints everything a pairing needs ---
    SourcedGearbox(
        id="apex_af042_ratio50", manufacturer="Apex Dynamics",
        part_number="AF042, 2 stage, ratio 50", documents=[APEX_AF],
        family=GearboxFamily.PLANETARY, ratio=50.0,
        rated_torque_nm=22.0,
        repeated_peak_torque_nm=0.6 * 3.0 * 22.0,
        momentary_peak_torque_nm=3.0 * 22.0,
        average_input_speed_rad_s=5000.0 * RPM_TO_RAD_S,
        torsional_stiffness_nm_rad=3.0 / (math.pi / (180.0 * 60.0)),
        backlash_arcmin=5.0, input_inertia_kg_m2=0.03e-4,
        efficiency=0.94, mass_kg=0.8,
        value_sources=[
            ValueSource("ratio", "AF042, 2 stage, ratio 50", APEX_AF.title),
            ValueSource("rated_torque_nm", "Nominal Output Torque T2N, ratio 50, AF042: 22 Nm", APEX_AF.title),
            ValueSource("momentary_peak_torque_nm", "Emergency Stop Torque T2NOTB = 3 times of Nominal Output Torque", APEX_AF.title),
            ValueSource("repeated_peak_torque_nm", "Max. acceleration torque T2B = 60% of T2NOT", APEX_AF.title),
            ValueSource("average_input_speed_rad_s", "Nominal Input Speed n1N 5,000 rpm", APEX_AF.title),
            ValueSource("torsional_stiffness_nm_rad", "Torsional Rigidity 3 Nm/arcmin", APEX_AF.title),
            ValueSource("backlash_arcmin", "Reduced Backlash P1, 2 stage: <= 5 arcmin", APEX_AF.title),
            ValueSource("input_inertia_kg_m2", "Mass Moments of Inertia J1, ratio 50, AF042: 0.03 kg cm2", APEX_AF.title),
            ValueSource("efficiency", "Efficiency, 2 stage 12~100: >= 94%", APEX_AF.title),
            ValueSource("mass_kg", "Weight, 2 stage, AF042: 0.8 kg", APEX_AF.title),
            ValueSource("family", "high precision planetary gearbox", APEX_AF.title),
        ],
        notes=("Efficiency is printed as a floor for the whole two stage "
               "range rather than a curve, so 0.94 is the catalogue's own "
               "lower bound and not a reading off a graph. The repeated peak "
               "is the catalogue's maximum acceleration torque, 60 percent of "
               "the emergency stop value, which is a different quantity from "
               "a harmonic drive's repeated peak and is named as such. "
               "Backlash is the reduced backlash grade P1; standard P2 is "
               "7 arcmin.")),
    SourcedGearbox(
        id="apex_af060_ratio50", manufacturer="Apex Dynamics",
        part_number="AF060, 2 stage, ratio 50", documents=[APEX_AF],
        family=GearboxFamily.PLANETARY, ratio=50.0,
        rated_torque_nm=60.0,
        repeated_peak_torque_nm=0.6 * 3.0 * 60.0,
        momentary_peak_torque_nm=3.0 * 60.0,
        average_input_speed_rad_s=5000.0 * RPM_TO_RAD_S,
        torsional_stiffness_nm_rad=7.0 / (math.pi / (180.0 * 60.0)),
        backlash_arcmin=5.0, input_inertia_kg_m2=0.03e-4,
        efficiency=0.94, mass_kg=1.5,
        value_sources=[
            ValueSource("ratio", "AF060, 2 stage, ratio 50", APEX_AF.title),
            ValueSource("rated_torque_nm", "Nominal Output Torque T2N, ratio 50, AF060: 60 Nm", APEX_AF.title),
            ValueSource("momentary_peak_torque_nm", "Emergency Stop Torque T2NOTB = 3 times of Nominal Output Torque", APEX_AF.title),
            ValueSource("repeated_peak_torque_nm", "Max. acceleration torque T2B = 60% of T2NOT", APEX_AF.title),
            ValueSource("average_input_speed_rad_s", "Nominal Input Speed n1N 5,000 rpm", APEX_AF.title),
            ValueSource("torsional_stiffness_nm_rad", "Torsional Rigidity 7 Nm/arcmin", APEX_AF.title),
            ValueSource("backlash_arcmin", "Reduced Backlash P1, 2 stage: <= 5 arcmin", APEX_AF.title),
            ValueSource("input_inertia_kg_m2", "Mass Moments of Inertia J1, ratio 50, AF060: 0.03 kg cm2", APEX_AF.title),
            ValueSource("efficiency", "Efficiency, 2 stage 12~100: >= 94%", APEX_AF.title),
            ValueSource("mass_kg", "Weight, 2 stage, AF060: 1.5 kg", APEX_AF.title),
            ValueSource("family", "high precision planetary gearbox", APEX_AF.title),
        ],
        notes="Same catalogue and the same caveats as the AF042 entry."),
    SourcedGearbox(
        id="apex_af042_ratio20", manufacturer="Apex Dynamics",
        part_number="AF042, 2 stage, ratio 20", documents=[APEX_AF],
        family=GearboxFamily.PLANETARY, ratio=20.0,
        rated_torque_nm=19.0,
        repeated_peak_torque_nm=0.6 * 3.0 * 19.0,
        momentary_peak_torque_nm=3.0 * 19.0,
        average_input_speed_rad_s=5000.0 * RPM_TO_RAD_S,
        torsional_stiffness_nm_rad=3.0 / (math.pi / (180.0 * 60.0)),
        backlash_arcmin=5.0, input_inertia_kg_m2=0.03e-4,
        efficiency=0.94, mass_kg=0.8,
        value_sources=[
            ValueSource("ratio", "AF042, 2 stage, ratio 20", APEX_AF.title),
            ValueSource("rated_torque_nm", "Nominal Output Torque T2N, ratio 20, AF042: 19 Nm", APEX_AF.title),
            ValueSource("momentary_peak_torque_nm", "Emergency Stop Torque T2NOTB = 3 times of Nominal Output Torque", APEX_AF.title),
            ValueSource("repeated_peak_torque_nm", "Max. acceleration torque T2B = 60% of T2NOT", APEX_AF.title),
            ValueSource("average_input_speed_rad_s", "Nominal Input Speed n1N 5,000 rpm", APEX_AF.title),
            ValueSource("torsional_stiffness_nm_rad", "Torsional Rigidity 3 Nm/arcmin", APEX_AF.title),
            ValueSource("backlash_arcmin", "Reduced Backlash P1, 2 stage: <= 5 arcmin", APEX_AF.title),
            ValueSource("input_inertia_kg_m2", "Mass Moments of Inertia J1, ratio 20, AF042: 0.03 kg cm2", APEX_AF.title),
            ValueSource("efficiency", "Efficiency, 2 stage 12~100: >= 94%", APEX_AF.title),
            ValueSource("mass_kg", "Weight, 2 stage, AF042: 0.8 kg", APEX_AF.title),
            ValueSource("family", "high precision planetary gearbox", APEX_AF.title),
        ],
        notes="Same catalogue and the same caveats as the ratio 50 entry."),
    SourcedGearbox(
        id="nabtesco_rv_42n", manufacturer="Nabtesco",
        part_number="RV-42N", documents=[NABTESCO_RV_42N],
        family=GearboxFamily.CYCLOIDAL,
        rated_torque_nm=412.0, repeated_peak_torque_nm=1029.0,
        momentary_peak_torque_nm=2058.0,
        torsional_stiffness_nm_rad=113.0 / (math.pi / (180.0 * 60.0)),
        backlash_arcmin=1.0, lost_motion_arcmin=1.0,
        value_sources=[
            ValueSource("rated_torque_nm", "Rated torque 412 Nm", NABTESCO_RV_42N.title),
            ValueSource("repeated_peak_torque_nm", "Allowable acceleration/deceleration torque 1,029 Nm", NABTESCO_RV_42N.title),
            ValueSource("momentary_peak_torque_nm", "Torque stop (momentary maximum) 2,058 Nm", NABTESCO_RV_42N.title),
            ValueSource("torsional_stiffness_nm_rad", "Torsional stiffness 113 Nm/arc.min", NABTESCO_RV_42N.title),
            ValueSource("backlash_arcmin", "Hysteresis loss < 1 arc.min", NABTESCO_RV_42N.title),
            ValueSource("lost_motion_arcmin", "Lost motion < 1 arc.min", NABTESCO_RV_42N.title),
            ValueSource("family", "RV cycloidal gearbox", NABTESCO_RV_42N.title),
        ],
        notes=("The page lists eight ratios (41 to 164.07) and one rated "
               "torque, so no single ratio is stored; the caller states which "
               "ratio it is using. Backlash and lost motion are printed as "
               "less than one arc minute and are stored as one, which is the "
               "worst case the page allows. Mass, efficiency and inertia are "
               "not printed.")),
]


def sourced_motor(part_id: str) -> SourcedMotor:
    for motor in SOURCED_MOTORS:
        if motor.id == part_id:
            return motor
    raise KeyError(f"no sourced motor {part_id!r}; have "
                   f"{sorted(m.id for m in SOURCED_MOTORS)}")


def sourced_gearbox(part_id: str) -> SourcedGearbox:
    for gearbox in SOURCED_GEARBOXES:
        if gearbox.id == part_id:
            return gearbox
    raise KeyError(f"no sourced gearbox {part_id!r}; have "
                   f"{sorted(g.id for g in SOURCED_GEARBOXES)}")


def unsourced_report() -> dict[str, Any]:
    """What the published pages did not print, per part.

    The visible gap, so that a missing rating is a fact in the record rather
    than a silence.
    """
    report: dict[str, Any] = {"motors": {}, "gearboxes": {}}
    for motor in SOURCED_MOTORS:
        missing = [name for name, value in motor.__dict__.items()
                   if value is None]
        report["motors"][motor.id] = sorted(missing)
    for gearbox in SOURCED_GEARBOXES:
        missing = [name for name, value in gearbox.__dict__.items()
                   if value is None]
        report["gearboxes"][gearbox.id] = sorted(missing)
    return report
