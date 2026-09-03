"""drivetrain.gearboxes.catalog: gearbox archetypes.

Same rule as the motor catalogue: **these are not real parts**. Generic
planetary and harmonic archetypes tagged `illustrative`, standing in for size
and family classes. No vendor part numbers, because a fabricated catalogue
would be read later as if it were sourced.

The two families exist because they trade differently, and the selection has to
be able to show that trade:

  planetary  cheaper, higher efficiency, more backlash (arc-minutes in the tens)
  harmonic   near-zero backlash, high ratio in one stage, lower efficiency,
             heavier, more expensive
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from drivetrain.motors.catalog import PartStatus


class GearboxFamily(str, Enum):
    PLANETARY = "planetary"
    HARMONIC = "harmonic"
    #: Cycloidal units, added with the first sourced entry (Nabtesco RV). No
    #: archetype exists for this family: the only members are real parts.
    CYCLOIDAL = "cycloidal"


class GearboxSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    family: GearboxFamily
    ratio: float = Field(gt=1.0)
    efficiency: float = Field(gt=0.0, le=1.0)
    rated_output_torque_nm: float = Field(gt=0.0)
    peak_output_torque_nm: float = Field(gt=0.0)
    backlash_arcmin: float = Field(ge=0.0)
    input_inertia_kg_m2: float = Field(gt=0.0)
    mass_kg: float = Field(gt=0.0)
    status: PartStatus = PartStatus.ILLUSTRATIVE
    source: str = "representative archetype, replace with vendor datasheet"

    @model_validator(mode="after")
    def _consistent(self) -> "GearboxSpec":
        if self.peak_output_torque_nm <= self.rated_output_torque_nm:
            raise ValueError(f"{self.id}: peak torque must exceed rated torque")
        return self


GEARBOXES: list[GearboxSpec] = [
    # Planetary: efficiency falls as stages are added for higher ratio.
    GearboxSpec(id="planetary_10", family=GearboxFamily.PLANETARY, ratio=10.0,
                efficiency=0.90, rated_output_torque_nm=8.0,
                peak_output_torque_nm=16.0, backlash_arcmin=12.0,
                input_inertia_kg_m2=2.0e-6, mass_kg=0.35),
    GearboxSpec(id="planetary_25", family=GearboxFamily.PLANETARY, ratio=25.0,
                efficiency=0.88, rated_output_torque_nm=15.0,
                peak_output_torque_nm=30.0, backlash_arcmin=12.0,
                input_inertia_kg_m2=2.5e-6, mass_kg=0.45),
    GearboxSpec(id="planetary_50", family=GearboxFamily.PLANETARY, ratio=50.0,
                efficiency=0.85, rated_output_torque_nm=25.0,
                peak_output_torque_nm=50.0, backlash_arcmin=15.0,
                input_inertia_kg_m2=3.0e-6, mass_kg=0.60),
    GearboxSpec(id="planetary_100", family=GearboxFamily.PLANETARY, ratio=100.0,
                efficiency=0.80, rated_output_torque_nm=35.0,
                peak_output_torque_nm=70.0, backlash_arcmin=18.0,
                input_inertia_kg_m2=3.5e-6, mass_kg=0.80),
    # Harmonic: near-zero backlash, high single-stage ratio, lower efficiency.
    GearboxSpec(id="harmonic_50", family=GearboxFamily.HARMONIC, ratio=50.0,
                efficiency=0.80, rated_output_torque_nm=40.0,
                peak_output_torque_nm=90.0, backlash_arcmin=1.0,
                input_inertia_kg_m2=4.0e-6, mass_kg=0.70),
    GearboxSpec(id="harmonic_100", family=GearboxFamily.HARMONIC, ratio=100.0,
                efficiency=0.78, rated_output_torque_nm=60.0,
                peak_output_torque_nm=140.0, backlash_arcmin=1.0,
                input_inertia_kg_m2=4.5e-6, mass_kg=0.90),
    GearboxSpec(id="harmonic_160", family=GearboxFamily.HARMONIC, ratio=160.0,
                efficiency=0.75, rated_output_torque_nm=80.0,
                peak_output_torque_nm=190.0, backlash_arcmin=1.0,
                input_inertia_kg_m2=5.0e-6, mass_kg=1.10),
]


def gearboxes() -> list[GearboxSpec]:
    return list(GEARBOXES)


def get_gearbox(gearbox_id: str) -> GearboxSpec:
    for gearbox in GEARBOXES:
        if gearbox.id == gearbox_id:
            return gearbox
    raise KeyError(f"unknown gearbox {gearbox_id!r}; available: "
                   f"{[g.id for g in GEARBOXES]}")
