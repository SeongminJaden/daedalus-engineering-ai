"""Motion profiles, and the torque a motion actually demands.

A duty cycle built from a handful of poses answers "what torque at this
instant". An actuator is selected against a motion: how hard the peak is, how
hot the continuous part is, and how much of both is the load's own inertia
rather than gravity. That needs a trajectory, and a trajectory needs a
profile.

WHAT IS HERE
============
trapezoidal
    Constant acceleration, constant velocity, constant deceleration. Exact
    closed form, including the triangular case where the move is too short to
    reach the velocity limit. Acceleration is discontinuous at the corners,
    which is what makes it hard on a drivetrain and easy to compute.

s_curve
    The same move with the acceleration ramped linearly (constant jerk), so
    acceleration is continuous. Built as the double integral of a trapezoidal
    acceleration, which keeps the closed forms exact and makes the jerk
    limit explicit rather than implied by a smoothing filter.

Both are scalar profiles on one joint. A multi-joint move is synchronised by
running every joint over the same duration, which is what `synchronise` does:
the slowest joint sets the time and the others are re-planned to fill it.

WHAT THE TORQUE PROFILE IS FOR
==============================
`torque_profile` evaluates the inverse dynamics along a trajectory and returns
the torque at every sample. `peak_torque_nm` and `rms_torque_nm` then separate
the two numbers a motor is chosen against, and they are usually far apart: a
move that peaks at three times its RMS needs a motor with the peak and a
thermal rating near the RMS, not a motor rated for the peak continuously.

WHAT IS NOT MODELLED
====================
Friction, backlash and joint compliance. The terms exist in the equations and
are zero because no source in this repository gives their parameters for any
real joint. `physics.dynamics.friction` refuses to produce a friction torque
without measured coefficients rather than defaulting to a plausible one, and
this module's torques therefore describe a frictionless drivetrain, which is
optimistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.assembly.model import Assembly

from .equations import inverse_dynamics


class UnreachableMove(ValueError):
    """A move that the stated limits cannot perform."""


@dataclass(frozen=True)
class Profile:
    """One joint's motion in time: position, velocity and acceleration."""

    time_s: np.ndarray
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    acceleration_rad_s2: np.ndarray
    duration_s: float
    kind: str

    def sample_count(self) -> int:
        return int(self.time_s.shape[0])


def trapezoidal_duration(distance: float, velocity_limit: float,
                         acceleration_limit: float) -> tuple[float, float, float]:
    """(total time, ramp time, cruise time) for a trapezoidal move.

    Returns the triangular solution when the move is too short to reach the
    velocity limit, which is the case a fixed formula gets wrong.
    """
    distance = abs(float(distance))
    if velocity_limit <= 0.0 or acceleration_limit <= 0.0:
        raise UnreachableMove("velocity and acceleration limits must be positive")
    if distance == 0.0:
        return 0.0, 0.0, 0.0
    ramp = velocity_limit / acceleration_limit
    ramp_distance = velocity_limit * ramp          # both ramps together
    if ramp_distance >= distance:
        ramp = float(np.sqrt(distance / acceleration_limit))
        return 2.0 * ramp, ramp, 0.0
    cruise = (distance - ramp_distance) / velocity_limit
    return 2.0 * ramp + cruise, ramp, cruise


def trapezoidal(distance: float, velocity_limit: float, acceleration_limit: float,
                samples: int = 200, duration_s: float | None = None) -> Profile:
    """A trapezoidal velocity profile, or a triangular one when it must be.

    `duration_s` stretches the move to a longer time by scaling the limits,
    which is how several joints are synchronised without changing the shape.
    """
    total, ramp, cruise = trapezoidal_duration(distance, velocity_limit,
                                               acceleration_limit)
    if duration_s is not None and total > 0.0:
        if duration_s < total - 1e-12:
            raise UnreachableMove(
                f"the move needs {total:.6g} s at these limits and was given "
                f"{duration_s:.6g} s")
        scale = total / duration_s
        return trapezoidal(distance, velocity_limit * scale,
                           acceleration_limit * scale ** 2, samples)
    sign = np.sign(distance) if distance != 0.0 else 1.0
    time = np.linspace(0.0, max(total, 0.0), samples)
    peak_velocity = acceleration_limit * ramp
    position = np.empty_like(time)
    velocity = np.empty_like(time)
    acceleration = np.empty_like(time)
    for i, t in enumerate(time):
        if t <= ramp:
            acceleration[i] = acceleration_limit
            velocity[i] = acceleration_limit * t
            position[i] = 0.5 * acceleration_limit * t ** 2
        elif t <= ramp + cruise:
            acceleration[i] = 0.0
            velocity[i] = peak_velocity
            position[i] = (0.5 * peak_velocity * ramp
                           + peak_velocity * (t - ramp))
        else:
            remaining = max(total - t, 0.0)
            acceleration[i] = -acceleration_limit
            velocity[i] = acceleration_limit * remaining
            position[i] = (abs(distance) - 0.5 * acceleration_limit * remaining ** 2)
    return Profile(time_s=time, position_rad=sign * position,
                   velocity_rad_s=sign * velocity,
                   acceleration_rad_s2=sign * acceleration,
                   duration_s=float(total), kind="trapezoidal")


def s_curve(distance: float, velocity_limit: float, acceleration_limit: float,
            jerk_limit: float, samples: int = 400,
            duration_s: float | None = None) -> Profile:
    """A move with continuous acceleration, by integrating a trapezoidal
    acceleration profile.

    The acceleration itself follows a trapezoid bounded by the jerk limit, and
    position and velocity are its integrals, so the profile is exact at the
    sample points rather than a smoothed trapezoid.
    """
    distance = float(distance)
    if jerk_limit <= 0.0:
        raise UnreachableMove("the jerk limit must be positive")
    magnitude = abs(distance)
    if magnitude == 0.0:
        zeros = np.zeros(samples)
        return Profile(zeros, zeros, zeros, zeros, 0.0, "s_curve")

    jerk_time = acceleration_limit / jerk_limit
    # Distance covered while ramping acceleration up and down, at the limit.
    if velocity_limit < acceleration_limit * jerk_time:
        # Acceleration never reaches its limit: jerk and velocity decide.
        jerk_time = float(np.sqrt(velocity_limit / jerk_limit))
        acceleration_limit = jerk_limit * jerk_time
    accel_time = max(velocity_limit / acceleration_limit - jerk_time, 0.0)
    # The velocity curve over the acceleration segment is symmetric about its
    # midpoint, so the distance it covers is exactly v times half the segment,
    # and the segment is 2 jerk times plus the plateau. Getting this wrong (by
    # leaving out one jerk time) made the profile overshoot and need a
    # correction factor, which then held the move below its stated limits.
    ramp_distance = velocity_limit * (2.0 * jerk_time + accel_time)
    if ramp_distance >= magnitude:
        # Too short to cruise: shrink the peak velocity until it fits, keeping
        # the jerk limit and the shape.
        scale = magnitude / ramp_distance
        velocity_limit = velocity_limit * scale
        jerk_time = min(jerk_time, float(np.sqrt(velocity_limit / jerk_limit)))
        acceleration_limit = jerk_limit * jerk_time
        accel_time = max(velocity_limit / acceleration_limit - jerk_time, 0.0)
        ramp_distance = velocity_limit * (2.0 * jerk_time + accel_time)
    cruise = max((magnitude - ramp_distance) / velocity_limit, 0.0)
    total = 2.0 * (2.0 * jerk_time + accel_time) + cruise

    time = np.linspace(0.0, total, samples)
    acceleration = np.zeros_like(time)
    ramp_up_end = jerk_time
    accel_end = ramp_up_end + accel_time
    ramp_down_end = accel_end + jerk_time
    cruise_end = ramp_down_end + cruise
    for i, t in enumerate(time):
        if t < ramp_up_end:
            acceleration[i] = jerk_limit * t
        elif t < accel_end:
            acceleration[i] = acceleration_limit
        elif t < ramp_down_end:
            acceleration[i] = acceleration_limit - jerk_limit * (t - accel_end)
        elif t < cruise_end:
            acceleration[i] = 0.0
        else:
            mirrored = total - t
            if mirrored < ramp_up_end:
                acceleration[i] = -jerk_limit * mirrored
            elif mirrored < accel_end:
                acceleration[i] = -acceleration_limit
            elif mirrored < ramp_down_end:
                acceleration[i] = -(acceleration_limit
                                    - jerk_limit * (mirrored - accel_end))
    velocity = np.concatenate([[0.0], np.cumsum(
        0.5 * (acceleration[1:] + acceleration[:-1]) * np.diff(time))])
    position = np.concatenate([[0.0], np.cumsum(
        0.5 * (velocity[1:] + velocity[:-1]) * np.diff(time))])
    # The trapezoidal integration of a piecewise linear acceleration is exact
    # up to the sampling of the corners, so the end point lands on the distance
    # to a few parts in ten thousand. Correcting it any further would scale the
    # profile off its own limits, which is what an earlier version did.
    if position[-1] > 0.0:
        drift = abs(position[-1] - magnitude) / magnitude
        if drift > 5e-3:
            raise UnreachableMove(
                f"the s curve integration drifted {drift:.2%} from the "
                f"requested distance; increase the sample count")
        position = position * (magnitude / position[-1])
    sign = np.sign(distance)
    if duration_s is not None and total > 0.0 and duration_s > total:
        stretch = total / duration_s
        return s_curve(distance, velocity_limit * stretch,
                       acceleration_limit * stretch ** 2,
                       jerk_limit * stretch ** 3, samples)
    return Profile(time_s=time, position_rad=sign * position,
                   velocity_rad_s=sign * velocity,
                   acceleration_rad_s2=sign * acceleration,
                   duration_s=float(total), kind="s_curve")


def synchronise(profiles: Sequence[Profile]) -> float:
    """The duration every joint must take: the slowest one's."""
    return max((p.duration_s for p in profiles), default=0.0)


@dataclass
class JointTrajectory:
    """A synchronised move of the whole assembly."""

    time_s: np.ndarray
    q: np.ndarray                 # (samples, dof)
    qd: np.ndarray
    qdd: np.ndarray
    kind: str

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1]) if self.time_s.size else 0.0


def plan_move(start_q, end_q, velocity_limits, acceleration_limits,
              jerk_limits=None, samples: int = 300, kind: str = "trapezoidal"
              ) -> JointTrajectory:
    """A synchronised point to point move for every joint.

    Each joint is planned alone, the slowest sets the duration, and the others
    are re-planned to that duration so they start and stop together. A joint
    with no distance to travel holds its position.
    """
    start = np.asarray(start_q, dtype=float).reshape(-1)
    end = np.asarray(end_q, dtype=float).reshape(-1)
    velocity_limits = np.asarray(velocity_limits, dtype=float).reshape(-1)
    acceleration_limits = np.asarray(acceleration_limits, dtype=float).reshape(-1)
    if kind == "s_curve":
        if jerk_limits is None:
            raise UnreachableMove("an s curve needs a jerk limit per joint")
        jerk_limits = np.asarray(jerk_limits, dtype=float).reshape(-1)

    def plan(index: int, duration: float | None) -> Profile:
        distance = end[index] - start[index]
        if kind == "trapezoidal":
            return trapezoidal(distance, velocity_limits[index],
                               acceleration_limits[index], samples, duration)
        return s_curve(distance, velocity_limits[index],
                       acceleration_limits[index], jerk_limits[index],
                       samples, duration)

    first = [plan(i, None) for i in range(start.size)]
    duration = synchronise(first)
    profiles = [plan(i, duration) if first[i].duration_s < duration else first[i]
                for i in range(start.size)]
    time = np.linspace(0.0, duration, samples)
    q = np.zeros((samples, start.size))
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    for i, profile in enumerate(profiles):
        if profile.duration_s <= 0.0:
            q[:, i] = start[i]
            continue
        q[:, i] = start[i] + np.interp(time, profile.time_s, profile.position_rad)
        qd[:, i] = np.interp(time, profile.time_s, profile.velocity_rad_s)
        qdd[:, i] = np.interp(time, profile.time_s, profile.acceleration_rad_s2)
    return JointTrajectory(time_s=time, q=q, qd=qd, qdd=qdd, kind=kind)


@dataclass
class TorqueProfile:
    """Torque along a trajectory, and the two numbers a motor is chosen on."""

    time_s: np.ndarray
    torque_nm: np.ndarray          # (samples, dof)
    gravity_nm: np.ndarray
    trajectory: JointTrajectory

    @property
    def peak_torque_nm(self) -> np.ndarray:
        return np.max(np.abs(self.torque_nm), axis=0)

    @property
    def rms_torque_nm(self) -> np.ndarray:
        """Root mean square over time, which is what a thermal rating is
        compared against. Integrated over the trajectory rather than averaged
        over samples, so a denser sampling does not change it."""
        if self.time_s.size < 2:
            return np.abs(self.torque_nm).reshape(-1)
        duration = float(self.time_s[-1] - self.time_s[0])
        squared = np.trapezoid(self.torque_nm ** 2, self.time_s, axis=0)
        return np.sqrt(squared / max(duration, 1e-12))

    @property
    def peak_to_rms(self) -> np.ndarray:
        rms = self.rms_torque_nm
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(self.peak_torque_nm / rms, nan=0.0, posinf=0.0)

    @property
    def dynamic_share(self) -> np.ndarray:
        """Share of the peak torque that gravity does not explain."""
        peak = self.peak_torque_nm
        gravity_at_peak = np.abs(self.gravity_nm[np.argmax(np.abs(self.torque_nm),
                                                           axis=0),
                                                 np.arange(peak.size)])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(1.0 - gravity_at_peak / peak, nan=0.0)


def torque_profile(assembly: Assembly, trajectory: JointTrajectory,
                   density_kg_m3: float, tip_force_n=None) -> TorqueProfile:
    """Inverse dynamics at every sample of a trajectory."""
    from .equations import gravity_torques

    samples = trajectory.q.shape[0]
    torque = np.zeros_like(trajectory.q)
    gravity = np.zeros_like(trajectory.q)
    for i in range(samples):
        torque[i] = inverse_dynamics(assembly, trajectory.q[i], trajectory.qd[i],
                                     trajectory.qdd[i], density_kg_m3,
                                     tip_force_n=tip_force_n)
        gravity[i] = gravity_torques(assembly, trajectory.q[i], density_kg_m3)
    return TorqueProfile(time_s=trajectory.time_s, torque_nm=torque,
                         gravity_nm=gravity, trajectory=trajectory)
