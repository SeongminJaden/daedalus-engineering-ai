"""drivetrain.selection: duty-cycle driven motor and gearbox selection."""

from .select import (
    INERTIA_RATIO_GUIDANCE,
    Candidate,
    Check,
    Requirement,
    compare_alternatives,
    evaluate_candidate,
    infeasibility_report,
    output_torque_nm,
    required_motor_speed_rad_s,
    select_drivetrain,
)

__all__ = [
    "INERTIA_RATIO_GUIDANCE", "Candidate", "Check", "Requirement",
    "compare_alternatives", "evaluate_candidate", "infeasibility_report",
    "output_torque_nm", "required_motor_speed_rad_s", "select_drivetrain",
]
