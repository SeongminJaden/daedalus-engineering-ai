"""physics.thermal: steady-state motor heating and structural thermal stress."""

from .motor import (
    DutySegment,
    MotorThermalResult,
    ThermalLosses,
    check_motor_thermal,
    losses_w,
    mean_speed_rad_s,
    rms_torque_nm,
    temperature_rise_k,
)
from .stress import (
    ThermalStressResult,
    check_thermal_stress,
    constrained_thermal_stress_pa,
    differential_strain,
    free_expansion_strain,
    stress_per_kelvin_pa,
)

__all__ = [
    "DutySegment", "MotorThermalResult", "ThermalLosses",
    "ThermalStressResult", "check_motor_thermal", "check_thermal_stress",
    "constrained_thermal_stress_pa", "differential_strain",
    "free_expansion_strain", "losses_w", "mean_speed_rad_s", "rms_torque_nm",
    "stress_per_kelvin_pa", "temperature_rise_k",
]
