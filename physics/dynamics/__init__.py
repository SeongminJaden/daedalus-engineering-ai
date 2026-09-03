"""physics.dynamics: rigid-body dynamics, inertia, trajectories and duty cycles.

Statics answers "hold this pose". Dynamics answers "make this motion", which is
what an actuator is actually selected against: a trajectory, the torque it
demands at every instant, and the peak and RMS that a motor and its thermal
rating are chosen on.

Friction is zero unless measured parameters are supplied, and
`physics.dynamics.friction` refuses to invent them. Backlash and joint
compliance are not modelled at all. A torque computed here is therefore a
lower bound on what a real drivetrain must deliver.
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
from .friction import (
    FrictionDataMissing,
    JointFriction,
    frictionless,
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

from .trajectory import (
    JointTrajectory,
    Profile,
    TorqueProfile,
    UnreachableMove,
    plan_move,
    s_curve,
    torque_profile,
    trapezoidal,
    trapezoidal_duration,
)

__all__ = [
    "CHRISTOFFEL_STEP", "CaseResult", "DutyCycle", "FrictionDataMissing",
    "JointFriction", "JointTrajectory", "LoadCase", "Profile", "TorqueProfile",
    "UnreachableMove", "frictionless", "plan_move", "s_curve",
    "torque_profile", "trapezoidal", "trapezoidal_duration",
    "box_inertia_tensor", "coriolis_matrix", "evaluate_case",
    "evaluate_duty_cycle", "friction_torques", "gravity_torques",
    "hollow_rect_inertia", "inverse_dynamics", "is_valid_inertia",
    "joint_power_w", "kinetic_energy_j", "link_inertia", "mass_matrix",
    "mass_matrix_derivative", "parallel_axis", "standard_load_cases",
]
