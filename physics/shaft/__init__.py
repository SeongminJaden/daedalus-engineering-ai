"""physics.shaft: shaft sizing under combined bending, torsion and axial load."""

from .design import (
    DEFAULT_BENDING_KF,
    DEFAULT_TORSION_KFS,
    ShaftLoads,
    ShaftResult,
    analyze_shaft,
    axial_stress_pa,
    bending_stress_pa,
    de_goodman_diameter_m,
    de_goodman_inverse_factor,
    first_critical_speed_rad_s,
    max_shear_pa,
    torsional_stress_pa,
    von_mises_pa,
)

__all__ = [
    "DEFAULT_BENDING_KF", "DEFAULT_TORSION_KFS", "ShaftLoads", "ShaftResult",
    "analyze_shaft", "axial_stress_pa", "bending_stress_pa",
    "de_goodman_diameter_m", "de_goodman_inverse_factor",
    "first_critical_speed_rad_s", "max_shear_pa", "torsional_stress_pa",
    "von_mises_pa",
]
