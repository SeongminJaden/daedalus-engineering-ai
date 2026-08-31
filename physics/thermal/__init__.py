"""physics.thermal: heat transfer, resistance networks, motor heating and thermal stress."""

from .heat_transfer import (
    ABSOLUTE_ZERO_C,
    CONVECTION_RANGES,
    STEFAN_BOLTZMANN,
    SurfaceLoss,
    celsius_to_kelvin,
    convection_resistance_k_w,
    cylinder_resistance_k_w,
    fourier_heat_w,
    natural_convection_vertical_plate_w_m2k,
    newton_cooling_w,
    plane_wall_resistance_k_w,
    radiation_coefficient_w_m2k,
    radiation_heat_w,
    sphere_resistance_k_w,
    surface_loss,
)
from .network import (
    BIOT_LUMPED_LIMIT,
    Resistance,
    ThermalPath,
    TransientResponse,
    biot_number,
    lumped_response,
    parallel_resistance_k_w,
    series_resistance_k_w,
)
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
    "ABSOLUTE_ZERO_C", "BIOT_LUMPED_LIMIT", "CONVECTION_RANGES", "Resistance",
    "STEFAN_BOLTZMANN", "SurfaceLoss", "ThermalPath", "TransientResponse",
    "biot_number", "celsius_to_kelvin", "convection_resistance_k_w",
    "cylinder_resistance_k_w", "fourier_heat_w", "lumped_response",
    "natural_convection_vertical_plate_w_m2k", "newton_cooling_w",
    "parallel_resistance_k_w", "plane_wall_resistance_k_w",
    "radiation_coefficient_w_m2k", "radiation_heat_w",
    "series_resistance_k_w", "sphere_resistance_k_w", "surface_loss",
    "DutySegment", "MotorThermalResult", "ThermalLosses",
    "ThermalStressResult", "check_motor_thermal", "check_thermal_stress",
    "constrained_thermal_stress_pa", "differential_strain",
    "free_expansion_strain", "losses_w", "mean_speed_rad_s", "rms_torque_nm",
    "stress_per_kelvin_pa", "temperature_rise_k",
]
