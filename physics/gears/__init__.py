"""physics.gears: gear tooth bending and contact capacity."""

from .tooth import (
    PRESSURE_ANGLE_DEG,
    GearMesh,
    GearResult,
    analyze_mesh,
    elastic_coefficient,
    geometry_factor_i,
    hertz_contact_stress_pa,
    lewis_bending_stress_pa,
    lewis_form_factor,
    pitch_diameter_m,
    tangential_load_n,
)

__all__ = [
    "GearMesh", "GearResult", "PRESSURE_ANGLE_DEG", "analyze_mesh",
    "elastic_coefficient", "geometry_factor_i", "hertz_contact_stress_pa",
    "lewis_bending_stress_pa", "lewis_form_factor", "pitch_diameter_m",
    "tangential_load_n",
]
