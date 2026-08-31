"""physics.elements: machine elements and ISO limits and fits."""

from .fits import (MAX_NOMINAL_MM, MIN_NOMINAL_MM, Fit, FitType, Limits,
                   fit, hole_limits, it_tolerance_um, shaft_limits,
                   tolerance_unit_um)
from .keys import (MAX_EFFECTIVE_LENGTH_RATIO, SPLINE_LOAD_SHARING, KeyResult,
                   analyze_key, effective_length_m, spline_torque_capacity_nm,
                   standard_key_section)
from .press_fit import (FRICTION_PRESSED_STEEL, PressFitResult,
                        analyze_press_fit, contact_pressure_pa,
                        hub_hoop_stress_pa)
from .welds import THROAT_FACTOR, WeldResult, analyze_fillet_weld, throat_thickness_m

__all__ = [
    "FRICTION_PRESSED_STEEL", "Fit", "FitType", "KeyResult", "Limits",
    "MAX_EFFECTIVE_LENGTH_RATIO", "MAX_NOMINAL_MM", "MIN_NOMINAL_MM",
    "PressFitResult", "SPLINE_LOAD_SHARING", "THROAT_FACTOR", "WeldResult",
    "analyze_fillet_weld", "analyze_key", "analyze_press_fit",
    "contact_pressure_pa", "effective_length_m", "fit", "hole_limits",
    "hub_hoop_stress_pa", "it_tolerance_um", "shaft_limits",
    "spline_torque_capacity_nm", "standard_key_section", "throat_thickness_m",
    "tolerance_unit_um",
]
