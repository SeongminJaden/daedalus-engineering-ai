"""drivetrain.motors.catalog: BLDC motor archetypes.

THESE ARE NOT REAL PARTS. Every entry is a generic archetype standing in for a
size class, tagged `illustrative`, and none of them corresponds to a vendor part
number. Inventing part numbers would put a fabricated catalogue into a record
that later gets read as if it were sourced, which is the same failure mode the
material database avoids by tagging everything `reference_typical`.

The deliverable of this phase is the **selection logic**, not the catalogue.
Replace these entries with real datasheet values before anything is ordered.

SI throughout: N m, rad/s, kg m^2, kg.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PartStatus(str, Enum):
    ILLUSTRATIVE = "illustrative"      # archetype, not a real part
    VENDOR_DATASHEET = "vendor_datasheet"


class MotorSpec(BaseModel):
    """A BLDC motor archetype."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    continuous_torque_nm: float = Field(gt=0.0)
    peak_torque_nm: float = Field(gt=0.0)
    rated_speed_rad_s: float = Field(gt=0.0)
    max_speed_rad_s: float = Field(gt=0.0)
    rotor_inertia_kg_m2: float = Field(gt=0.0)
    mass_kg: float = Field(gt=0.0)
    status: PartStatus = PartStatus.ILLUSTRATIVE
    source: str = "representative archetype, replace with vendor datasheet"

    @model_validator(mode="after")
    def _consistent(self) -> "MotorSpec":
        if self.peak_torque_nm <= self.continuous_torque_nm:
            raise ValueError(
                f"{self.id}: peak torque must exceed continuous torque")
        if self.max_speed_rad_s < self.rated_speed_rad_s:
            raise ValueError(
                f"{self.id}: max speed must be at least the rated speed")
        return self

    @property
    def continuous_power_w(self) -> float:
        """Mechanical power at the rated point, P = tau * omega."""
        return self.continuous_torque_nm * self.rated_speed_rad_s

    @property
    def peak_torque_ratio(self) -> float:
        return self.peak_torque_nm / self.continuous_torque_nm


# Anchored so that continuous torque times rated speed reproduces the nominal
# power in the name, and peak is 3x continuous across the range. Both
# relationships are checked by tests, so a future edit cannot leave the
# catalogue internally inconsistent.
MOTORS: list[MotorSpec] = [
    MotorSpec(id="bldc_50w", name="BLDC 50 W class",
              continuous_torque_nm=0.16, peak_torque_nm=0.48,
              rated_speed_rad_s=314.0, max_speed_rad_s=419.0,
              rotor_inertia_kg_m2=1.5e-5, mass_kg=0.30),
    MotorSpec(id="bldc_100w", name="BLDC 100 W class",
              continuous_torque_nm=0.32, peak_torque_nm=0.95,
              rated_speed_rad_s=314.0, max_speed_rad_s=419.0,
              rotor_inertia_kg_m2=3.0e-5, mass_kg=0.50),
    MotorSpec(id="bldc_200w", name="BLDC 200 W class",
              continuous_torque_nm=0.64, peak_torque_nm=1.90,
              rated_speed_rad_s=314.0, max_speed_rad_s=419.0,
              rotor_inertia_kg_m2=6.0e-5, mass_kg=0.80),
    MotorSpec(id="bldc_400w", name="BLDC 400 W class",
              continuous_torque_nm=1.27, peak_torque_nm=3.80,
              rated_speed_rad_s=314.0, max_speed_rad_s=419.0,
              rotor_inertia_kg_m2=1.2e-4, mass_kg=1.20),
]


def motors() -> list[MotorSpec]:
    return list(MOTORS)


def get_motor(motor_id: str) -> MotorSpec:
    for motor in MOTORS:
        if motor.id == motor_id:
            return motor
    raise KeyError(f"unknown motor {motor_id!r}; available: "
                   f"{[m.id for m in MOTORS]}")
