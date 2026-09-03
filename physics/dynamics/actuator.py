"""Referring a joint's demand to a motor shaft, and where the ratio is best.

A joint needs a torque and a speed. A motor delivers a different torque at a
different speed, and the gearbox in between decides which. Two consequences
are worth computing rather than guessing: what the motor must supply for a
given joint demand, and the ratio at which a pure acceleration demand is
cheapest, which is not obvious because the rotor's own inertia grows with the
square of the ratio when referred to the joint.

WHAT IS EXACT
=============
The referral itself. With ratio n, efficiency e, joint torque tau_j and joint
acceleration a_j:

    tau_motor = tau_j / (n e) + (J_rotor + J_gearbox) * n * a_j

The first term is the load referred to the motor, the second is the drivetrain
accelerating itself. Both are algebra, and the tests check them against hand
arithmetic.

WHAT IS A CHOICE
================
The optimum. For a demand that is pure inertia (no gravity, no external load),
the motor torque is minimised at the inertia matched ratio, n = sqrt(J_load /
J_rotor). That is a classical result and `inertia_matched_ratio` returns it;
`best_ratio` finds the minimum numerically over a list of available ratios
INCLUDING the gravity and load terms, because a real joint is not pure
inertia and the matched ratio is then no longer the answer. The difference
between the two is measured in the tests rather than asserted away.

EFFICIENCY
==========
A gearbox efficiency below one raises the motor torque for a load and does not
reduce the inertia term. There is no default: the caller passes the number
from the gearbox data sheet, and passing nothing means one, which is
optimistic and stated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriveDemand:
    """What one joint asks of its drivetrain at an instant."""

    joint_torque_nm: float
    joint_speed_rad_s: float
    joint_acceleration_rad_s2: float
    load_inertia_kg_m2: float


def reflected_inertia_kg_m2(load_inertia_kg_m2: float, ratio: float) -> float:
    """Load inertia seen at the motor shaft."""
    if ratio <= 0.0:
        raise ValueError("the gear ratio must be positive")
    return float(load_inertia_kg_m2) / float(ratio) ** 2


def motor_torque_nm(demand: DriveDemand, ratio: float,
                    rotor_inertia_kg_m2: float,
                    gearbox_inertia_kg_m2: float = 0.0,
                    efficiency: float = 1.0) -> float:
    """Torque the motor must supply for this joint demand.

    The joint torque is divided by the ratio and the efficiency; the rotor and
    gearbox inertia are accelerated at the motor's own acceleration, which is
    the joint's times the ratio.
    """
    if ratio <= 0.0:
        raise ValueError("the gear ratio must be positive")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency must be in (0, 1]")
    load = demand.joint_torque_nm / (ratio * efficiency)
    inertia = (rotor_inertia_kg_m2 + gearbox_inertia_kg_m2) * ratio \
        * demand.joint_acceleration_rad_s2
    return float(load + inertia)


def motor_speed_rad_s(demand: DriveDemand, ratio: float) -> float:
    return float(demand.joint_speed_rad_s) * float(ratio)


def inertia_matched_ratio(load_inertia_kg_m2: float,
                          rotor_inertia_kg_m2: float) -> float:
    """The classical optimum for a pure inertia demand: sqrt(J_load/J_rotor)."""
    if rotor_inertia_kg_m2 <= 0.0:
        raise ValueError("the rotor inertia must be positive")
    return float(np.sqrt(load_inertia_kg_m2 / rotor_inertia_kg_m2))


def best_ratio(demand: DriveDemand, ratios, rotor_inertia_kg_m2: float,
               gearbox_inertia_kg_m2: float = 0.0, efficiency: float = 1.0
               ) -> tuple[float, float]:
    """The available ratio that minimises the motor torque, and that torque.

    Searches the ratios the caller actually has rather than a continuum, which
    is what a catalogue offers, and includes every term of the demand rather
    than the inertia alone.
    """
    ratios = [float(r) for r in ratios]
    if not ratios:
        raise ValueError("no ratios to choose from")
    torques = [abs(motor_torque_nm(demand, r, rotor_inertia_kg_m2,
                                   gearbox_inertia_kg_m2, efficiency))
               for r in ratios]
    index = int(np.argmin(torques))
    return ratios[index], torques[index]


def ratio_sweep(demand: DriveDemand, ratios, rotor_inertia_kg_m2: float,
                gearbox_inertia_kg_m2: float = 0.0, efficiency: float = 1.0
                ) -> list[dict]:
    """Motor torque, speed and inertia ratio at each available ratio."""
    rows = []
    for ratio in ratios:
        rows.append({
            "ratio": float(ratio),
            "motor_torque_nm": motor_torque_nm(demand, ratio, rotor_inertia_kg_m2,
                                               gearbox_inertia_kg_m2, efficiency),
            "motor_speed_rad_s": motor_speed_rad_s(demand, ratio),
            "reflected_load_inertia_kg_m2":
                reflected_inertia_kg_m2(demand.load_inertia_kg_m2, ratio),
            "inertia_ratio": reflected_inertia_kg_m2(demand.load_inertia_kg_m2,
                                                     ratio) / rotor_inertia_kg_m2,
        })
    return rows


def demand_from_profile(profile, joint_index: int, load_inertia_kg_m2: float
                        ) -> DriveDemand:
    """The worst instant of a torque profile, as a drive demand.

    The peak torque and the acceleration at that instant, which is what sizes
    the motor's peak; the RMS is a separate number and is read from the
    profile directly.
    """
    torque = profile.torque_nm[:, joint_index]
    index = int(np.argmax(np.abs(torque)))
    return DriveDemand(
        joint_torque_nm=float(torque[index]),
        joint_speed_rad_s=float(profile.trajectory.qd[index, joint_index]),
        joint_acceleration_rad_s2=float(profile.trajectory.qdd[index, joint_index]),
        load_inertia_kg_m2=float(load_inertia_kg_m2))
