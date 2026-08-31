"""physics.dynamics: rigid-body dynamics, inertia and duty-cycle load cases.

Statics answers "hold this pose". Dynamics answers "make this motion", which is
what an actuator is actually selected against. Friction, backlash and joint
compliance are all zero here: the terms exist, the data does not.
"""

from .equations import (
    CHRISTOFFEL_STEP,
    coriolis_matrix,
    friction_torques,
    gravity_torques,
    inverse_dynamics,
    joint_power_w,
    kinetic_energy_j,
    mass_matrix,
    mass_matrix_derivative,
)
from .inertia import (
    box_inertia_tensor,
    hollow_rect_inertia,
    is_valid_inertia,
    link_inertia,
    parallel_axis,
)
from .load_cases import (
    CaseResult,
    DutyCycle,
    LoadCase,
    evaluate_case,
    evaluate_duty_cycle,
    standard_load_cases,
)

__all__ = [
    "CHRISTOFFEL_STEP", "CaseResult", "DutyCycle", "LoadCase",
    "box_inertia_tensor", "coriolis_matrix", "evaluate_case",
    "evaluate_duty_cycle", "friction_torques", "gravity_torques",
    "hollow_rect_inertia", "inverse_dynamics", "is_valid_inertia",
    "joint_power_w", "kinetic_energy_j", "link_inertia", "mass_matrix",
    "mass_matrix_derivative", "parallel_axis", "standard_load_cases",
]
