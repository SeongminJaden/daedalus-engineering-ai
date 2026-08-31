"""Steady-state thermal check for a motor under a duty cycle.

Phase 12 sized drivetrains against a continuous torque rating and left the
thermal question as a proxy, marked "subject to thermal validation". This
replaces the proxy. A continuous rating is itself a thermal statement, made at
one ambient temperature and one mounting, and a duty that differs from those
assumptions is not covered by it.

**Steady state only.** This is the temperature a motor reaches once it has
stopped heating up, which for a motor of this size takes tens of minutes. A
duty that exceeds the limit briefly and then recovers will pass here and may
well be fine in reality, because the thermal mass has not had time to follow;
equally, a duty that passes on average can cook the winding during a long
overload this model cannot see. Transient analysis is not implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from drivetrain.motors.catalog import MotorSpec


@dataclass(frozen=True)
class DutySegment:
    """One operating point and the share of the cycle spent at it."""

    torque_nm: float
    speed_rad_s: float
    fraction: float

    def __post_init__(self) -> None:
        if self.fraction < 0.0:
            raise ValueError("a duty fraction cannot be negative")
        if self.speed_rad_s < 0.0:
            raise ValueError("speed is a magnitude here and cannot be negative")


def _normalised(segments: list[DutySegment]) -> list[DutySegment]:
    total = sum(segment.fraction for segment in segments)
    if total <= 0.0:
        raise ValueError("the duty cycle has no duration")
    return [DutySegment(s.torque_nm, s.speed_rad_s, s.fraction / total)
            for s in segments]


def rms_torque_nm(segments: list[DutySegment]) -> float:
    """sqrt(sum(f_i T_i^2)), the torque that produces the same copper loss.

    **RMS and not mean, because copper loss goes as the square of torque.** A
    duty that spends half its time at 1 N m and half at rest has a mean torque
    of 0.5 N m and an RMS of 0.707, and using the mean would understate the
    heating by a factor of two. The difference grows with how peaky the duty
    is, so the error is largest exactly where it matters.
    """
    normalised = _normalised(segments)
    return math.sqrt(sum(s.fraction * s.torque_nm ** 2 for s in normalised))


def mean_speed_rad_s(segments: list[DutySegment]) -> float:
    """Duty-weighted mean speed, which is what the linear iron loss sees."""
    return sum(s.fraction * s.speed_rad_s for s in _normalised(segments))


@dataclass(frozen=True)
class ThermalLosses:
    copper_w: float
    iron_w: float

    @property
    def total_w(self) -> float:
        return self.copper_w + self.iron_w


def losses_w(motor: MotorSpec, rms_torque: float,
             mean_speed: float) -> ThermalLosses:
    """Copper loss from the RMS torque, iron loss from the mean speed.

        P_cu = k_cu T_rms^2      (torque is proportional to current, loss to I^2)
        P_fe = k_fe omega        (a linear stand-in for core loss)
    """
    if not motor.has_thermal_data:
        raise ValueError(
            f"{motor.id} carries no thermal data, so no thermal check can be "
            f"run on it. Reporting a temperature for it would be inventing one")
    iron_coefficient = motor.iron_loss_coefficient_w_s_rad or 0.0
    return ThermalLosses(
        copper_w=motor.copper_loss_coefficient_w_nm2 * rms_torque ** 2,
        iron_w=iron_coefficient * mean_speed)


def temperature_rise_k(total_loss_w: float, thermal_resistance_k_w: float
                       ) -> float:
    """T_rise = P_loss R_th."""
    if thermal_resistance_k_w <= 0.0:
        raise ValueError("thermal resistance must be positive")
    return total_loss_w * thermal_resistance_k_w


@dataclass(frozen=True)
class MotorThermalResult:
    """A winding temperature and the verdict on it."""

    motor: MotorSpec
    rms_torque_nm: float
    mean_speed_rad_s: float
    losses: ThermalLosses
    temperature_rise_k: float
    ambient_c: float
    winding_c: float
    limit_c: float
    margin_k: float

    @property
    def passes(self) -> bool:
        return self.winding_c <= self.limit_c

    def summary(self) -> str:
        verdict = "within" if self.passes else "OVER"
        return (f"winding {self.winding_c:.1f} C, {verdict} the "
                f"{self.motor.insulation_class.value} class limit of "
                f"{self.limit_c:.0f} C")


def check_motor_thermal(motor: MotorSpec, segments: list[DutySegment],
                        ambient_c: float = 40.0) -> MotorThermalResult:
    """Steady-state winding temperature for a duty cycle, against the limit.

    `ambient_c` defaults to 40 C rather than room temperature. A motor inside a
    robot arm sits next to its own driver electronics and other motors, and
    sizing against 20 C ambient is how a drive that passed on the bench
    overheats in the machine.
    """
    rms = rms_torque_nm(segments)
    speed = mean_speed_rad_s(segments)
    loss = losses_w(motor, rms, speed)
    rise = temperature_rise_k(loss.total_w, motor.thermal_resistance_k_w)
    winding = ambient_c + rise
    limit = motor.winding_limit_c
    return MotorThermalResult(
        motor=motor, rms_torque_nm=rms, mean_speed_rad_s=speed, losses=loss,
        temperature_rise_k=rise, ambient_c=ambient_c, winding_c=winding,
        limit_c=limit, margin_k=limit - winding)
