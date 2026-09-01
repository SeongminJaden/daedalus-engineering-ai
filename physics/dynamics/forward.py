"""Forward dynamics: what the mechanism DOES, given the torques applied to it.

Everything in equations.py runs the problem backwards, from a commanded motion
to the torque it needs, which is what sizes a motor. This runs it forwards,
from torque to motion, which is what a simulation needs. It is general
multibody, not robot specific: any open chain of rigid links with revolute or
prismatic joints goes through it, a machine linkage as readily as an arm.

    M(q) qdd = tau - C(q, qd) qd - G(q)

VALIDITY DOMAIN
===============
Stated before implementing, per the standing discipline.

The model
    Rigid links, ideal joints, and NO friction. `friction_torques` returns
    zero deliberately, because real friction needs a breakaway torque, a
    viscous coefficient and a gearbox efficiency per joint and none of that
    exists here. The consequence is not neutral: a simulated mechanism swings
    forever, and a real one does not. Nothing here predicts how long a real
    motion takes to die away.

    Joint limits are carried in the model as numbers but are NOT enforced as
    constraints. A simulation will happily drive a joint through its stop.
    Contact is not modelled at all.

The integrators, and what each one does to energy
    Neither is a detail. With no applied torque and no friction the exact
    solution conserves total energy exactly, so how the integrator treats
    energy decides what a long run means.

    rk4
        Classical explicit Runge-Kutta, fourth order accurate per step and NOT
        symplectic. Measured on the two link arm over a fixed window, the
        largest energy deviation falls by 16.0 and then 14.5 as the step is
        halved, against the 16 that fourth order predicts. The coarsest step
        tried sits outside that asymptotic range and does worse.

        The metric there is max |E(t) - E0| rather than E(end) - E0
        deliberately. Endpoint drift depends on where in an oscillation the
        run happens to stop, which makes it a noisy way to measure an order.

        At the small steps a trajectory actually uses, the energy error drops
        to round-off, around 1e-13 J against a total of 0.39 J, so the
        truncation law is not observable there at all. Being non symplectic
        still means the error accumulates in one direction rather than
        cancelling, so a long enough run drifts.

    semi_implicit_euler
        Symplectic, and only first order accurate. Over two seconds of the
        same arm the energy reverses direction 14 times rather than moving
        monotonically, which is the symplectic signature. That is evidence of
        oscillation, NOT a proof that the band stays bounded over an arbitrary
        horizon, and this module does not claim the stronger thing.

        First order is not a detail: at a 2 ms step the band reached 25
        percent of the total energy on that run. It buys qualitative long-run
        behaviour at a real cost in pointwise accuracy.

    Both are explicit, so both need a step below the fastest natural period of
    the mechanism. There is no automatic check for that here: a step that is
    too large produces a divergence that is obvious, not a subtly wrong
    answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.assembly.frames import GRAVITY_DIRECTION, STANDARD_GRAVITY
from core.assembly.model import Assembly
from core.assembly.statics import link_com_positions

from .equations import (coriolis_matrix, friction_torques, gravity_torques,
                        kinetic_energy_j, mass_matrix)

INTEGRATORS = ("rk4", "semi_implicit_euler")


def potential_energy_j(assembly: Assembly, q, density_kg_m3: float,
                       datum_m: float = 0.0) -> float:
    """Gravitational potential energy, along this project's -y gravity.

    The datum is arbitrary and only differences matter, which is why it is a
    parameter rather than a hidden choice.
    """
    coms = link_com_positions(assembly, q)
    up = -np.asarray(GRAVITY_DIRECTION, dtype=np.float64)
    return float(sum(
        link.mass_kg(density_kg_m3) * STANDARD_GRAVITY
        * (float(up @ coms[link.name]) - datum_m)
        for link in assembly.links))


def total_energy_j(assembly: Assembly, q, qd, density_kg_m3: float) -> float:
    """Kinetic plus potential. Constant in exact motion with no applied torque."""
    return (kinetic_energy_j(assembly, q, qd, density_kg_m3)
            + potential_energy_j(assembly, q, density_kg_m3))


def forward_dynamics(assembly: Assembly, q, qd, tau, density_kg_m3: float
                     ) -> np.ndarray:
    """Joint accelerations from applied torques.

    Solves the mass matrix system rather than inverting it. M is symmetric
    positive definite for any physical mechanism, so an explicit inverse would
    be both slower and less accurate for nothing.
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    if not (q.shape == qd.shape == tau.shape):
        raise ValueError(
            f"q, qd and tau must agree in length, got {q.shape}, {qd.shape} "
            f"and {tau.shape}")

    bias = (coriolis_matrix(assembly, q, qd, density_kg_m3) @ qd
            + gravity_torques(assembly, q, density_kg_m3)
            + friction_torques(assembly, qd))
    return np.linalg.solve(mass_matrix(assembly, q, density_kg_m3), tau - bias)


@dataclass(frozen=True)
class Trajectory:
    """A simulated motion, with the integrator that produced it recorded.

    The integrator travels with the result because energy drift is a property
    of the method, not of the mechanism, and a trajectory read without knowing
    which one produced it invites the drift to be read as physics.
    """

    time_s: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    integrator: str
    energy_j: np.ndarray

    def energy_drift(self) -> float:
        """Absolute change in total energy from start to finish."""
        return float(abs(self.energy_j[-1] - self.energy_j[0]))

    def energy_band(self) -> float:
        """Peak-to-peak spread, which is what a symplectic method bounds."""
        return float(self.energy_j.max() - self.energy_j.min())


def simulate(assembly: Assembly, q0, qd0, density_kg_m3: float,
             duration_s: float, dt_s: float,
             torque: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
             | None = None,
             integrator: str = "rk4") -> Trajectory:
    """Integrate the forward dynamics from an initial state.

    `torque` receives (t, q, qd) so a controller can be closed around it. When
    it is None the mechanism is unforced, which is the case the energy tests
    use.
    """
    if integrator not in INTEGRATORS:
        raise ValueError(
            f"unknown integrator {integrator!r}, expected one of {INTEGRATORS}")
    if dt_s <= 0.0 or duration_s <= 0.0:
        raise ValueError("duration and step must be positive")

    q = np.asarray(q0, dtype=np.float64).reshape(-1).copy()
    qd = np.asarray(qd0, dtype=np.float64).reshape(-1).copy()
    zero = np.zeros_like(q)

    def applied(t: float, q_, qd_) -> np.ndarray:
        return zero if torque is None else np.asarray(
            torque(t, q_, qd_), dtype=np.float64).reshape(-1)

    def acceleration(t: float, q_, qd_) -> np.ndarray:
        return forward_dynamics(assembly, q_, qd_, applied(t, q_, qd_),
                                density_kg_m3)

    steps = int(round(duration_s / dt_s))
    times = np.zeros(steps + 1)
    qs = np.zeros((steps + 1, q.size))
    qds = np.zeros((steps + 1, q.size))
    energies = np.zeros(steps + 1)
    qs[0], qds[0] = q, qd
    energies[0] = total_energy_j(assembly, q, qd, density_kg_m3)

    for step in range(steps):
        t = step * dt_s
        if integrator == "semi_implicit_euler":
            qd = qd + dt_s * acceleration(t, q, qd)
            q = q + dt_s * qd
        else:
            k1v = acceleration(t, q, qd)
            k2v = acceleration(t + dt_s / 2, q + dt_s / 2 * qd,
                               qd + dt_s / 2 * k1v)
            k3v = acceleration(t + dt_s / 2, q + dt_s / 2 * (qd + dt_s / 2 * k1v),
                               qd + dt_s / 2 * k2v)
            k4v = acceleration(t + dt_s, q + dt_s * (qd + dt_s / 2 * k2v),
                               qd + dt_s * k3v)
            q = q + dt_s * (qd + dt_s / 6 * (k1v + k2v + k3v))
            qd = qd + dt_s / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        times[step + 1] = t + dt_s
        qs[step + 1], qds[step + 1] = q, qd
        energies[step + 1] = total_energy_j(assembly, q, qd, density_kg_m3)

    return Trajectory(time_s=times, q=qs, qd=qds, integrator=integrator,
                      energy_j=energies)
