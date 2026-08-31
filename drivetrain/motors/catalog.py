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


class InsulationClass(str, Enum):
    """Winding insulation class and its maximum temperature, degrees C.

    The limit is on the WINDING, not on the air around it or the case you can
    touch. Exceeding it does not fail the motor immediately; it shortens
    insulation life, roughly halving it for every 10 K over. So a thermal check
    that passes is a statement about lifetime rather than about survival.
    """

    A = "A"
    E = "E"
    B = "B"
    F = "F"
    H = "H"


WINDING_LIMIT_C: dict[InsulationClass, float] = {
    InsulationClass.A: 105.0,
    InsulationClass.E: 120.0,
    InsulationClass.B: 130.0,
    InsulationClass.F: 155.0,
    InsulationClass.H: 180.0,
}


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

    # --- thermal, all illustrative like the rest of this table ---
    #
    # `copper_loss_coefficient_w_nm2` is k in P_cu = k T^2. Torque is
    # proportional to current in a BLDC machine and copper loss goes as I^2, so
    # loss goes as the square of torque. That squaring is why an RMS torque and
    # not a mean torque has to drive the calculation.
    #
    # `iron_loss_coefficient_w_s_rad` is k in P_fe = k omega, a LINEAR
    # simplification. Real core loss splits into hysteresis (roughly linear in
    # frequency) and eddy current (roughly quadratic), so this understates loss
    # at high speed. It is here so that speed appears in the answer at all.
    #
    # `thermal_resistance_k_w` is winding to ambient as a single lumped number.
    # A real motor has a winding-to-case and a case-to-ambient resistance in
    # series and the mounting dominates the second one: bolting the motor to a
    # large aluminium bracket can halve it, and standing it in free air can
    # double it. One number cannot express that.
    copper_loss_coefficient_w_nm2: float | None = Field(default=None, gt=0.0)
    iron_loss_coefficient_w_s_rad: float | None = Field(default=None, ge=0.0)
    thermal_resistance_k_w: float | None = Field(default=None, gt=0.0)
    insulation_class: InsulationClass = InsulationClass.F

    @property
    def winding_limit_c(self) -> float:
        return WINDING_LIMIT_C[self.insulation_class]

    @property
    def has_thermal_data(self) -> bool:
        """Whether a thermal check can run at all on this archetype."""
        return (self.copper_loss_coefficient_w_nm2 is not None
                and self.thermal_resistance_k_w is not None)
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
              rotor_inertia_kg_m2=1.5e-5, mass_kg=0.30,
              copper_loss_coefficient_w_nm2=625.0,
              iron_loss_coefficient_w_s_rad=0.0191,
              thermal_resistance_k_w=4.3),
    MotorSpec(id="bldc_100w", name="BLDC 100 W class",
              continuous_torque_nm=0.32, peak_torque_nm=0.95,
              rated_speed_rad_s=314.0, max_speed_rad_s=419.0,
              rotor_inertia_kg_m2=3.0e-5, mass_kg=0.50,
              copper_loss_coefficient_w_nm2=293.0,
              iron_loss_coefficient_w_s_rad=0.0318,
              thermal_resistance_k_w=2.4),
    MotorSpec(id="bldc_200w", name="BLDC 200 W class",
              continuous_torque_nm=0.64, peak_torque_nm=1.90,
              rated_speed_rad_s=314.0, max_speed_rad_s=419.0,
              rotor_inertia_kg_m2=6.0e-5, mass_kg=0.80,
              copper_loss_coefficient_w_nm2=127.0,
              iron_loss_coefficient_w_s_rad=0.0573,
              thermal_resistance_k_w=1.36),
    MotorSpec(id="bldc_400w", name="BLDC 400 W class",
              continuous_torque_nm=1.27, peak_torque_nm=3.80,
              rated_speed_rad_s=314.0, max_speed_rad_s=419.0,
              rotor_inertia_kg_m2=1.2e-4, mass_kg=1.20,
              copper_loss_coefficient_w_nm2=55.8,
              iron_loss_coefficient_w_s_rad=0.0955,
              thermal_resistance_k_w=0.79),
]


def motors() -> list[MotorSpec]:
    return list(MOTORS)


def get_motor(motor_id: str) -> MotorSpec:
    for motor in MOTORS:
        if motor.id == motor_id:
            return motor
    raise KeyError(f"unknown motor {motor_id!r}; available: "
                   f"{[m.id for m in MOTORS]}")
