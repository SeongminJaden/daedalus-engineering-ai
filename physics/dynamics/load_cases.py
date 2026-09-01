"""physics.dynamics.load_cases: the duty cycle a joint actually has to survive.

A single "required torque" number is not enough to select an actuator. A motor
has a **peak** rating it can hold briefly and a **continuous** rating it can
hold indefinitely, and those differ by a factor of two or three. Sizing to the
peak alone gives an over-specified drive; sizing to the continuous value alone
gives one that overheats on every acceleration. Both are computed here and kept
separate.

The continuous figure is the RMS torque over the duty cycle, which is the right
average because motor heating goes as current squared and current is roughly
proportional to torque.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.assembly.frames import STANDARD_GRAVITY
from core.assembly.model import Assembly
from core.assembly.statics import worst_gravity_pose

from .equations import inverse_dynamics, joint_power_w


@dataclass
class LoadCase:
    """One operating point: a pose, a motion and a payload."""

    name: str
    description: str
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    payload_kg: float
    duty_fraction: float = 0.0     # share of the cycle spent here, for RMS

    def payload_force_n(self) -> np.ndarray:
        return np.array([0.0, -self.payload_kg * STANDARD_GRAVITY, 0.0])


@dataclass
class CaseResult:
    case: LoadCase
    torque_nm: np.ndarray
    power_w: np.ndarray
    static_torque_nm: np.ndarray

    @property
    def dynamic_share(self) -> np.ndarray:
        """How much of the torque is NOT explained by holding the pose.

        This is the number that shows why a motor cannot be chosen from statics
        alone.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            share = 1.0 - np.abs(self.static_torque_nm) / np.abs(self.torque_nm)
        return np.nan_to_num(share, nan=0.0, posinf=0.0, neginf=0.0)

    def as_dict(self) -> dict:
        return {
            "case": self.case.name,
            "torque_nm": self.torque_nm.tolist(),
            "power_w": self.power_w.tolist(),
            "static_torque_nm": self.static_torque_nm.tolist(),
            "dynamic_share": self.dynamic_share.tolist(),
        }


@dataclass
class DutyCycle:
    """Peak and continuous requirements across a set of cases."""

    results: list[CaseResult] = field(default_factory=list)

    def peak_torque_nm(self) -> np.ndarray:
        return np.max([np.abs(r.torque_nm) for r in self.results], axis=0)

    def peak_power_w(self) -> np.ndarray:
        return np.max([np.abs(r.power_w) for r in self.results], axis=0)

    def continuous_torque_nm(self) -> np.ndarray:
        """RMS torque weighted by how long each case is held.

        Cases with no declared duty fraction do not contribute: an emergency
        peak that happens for milliseconds should not raise the continuous
        rating.
        """
        weights = np.array([r.case.duty_fraction for r in self.results])
        total = weights.sum()
        if total <= 0:
            return np.zeros_like(self.peak_torque_nm())
        weights = weights / total
        squared = np.sum(
            [w * np.square(r.torque_nm) for w, r in zip(weights, self.results)],
            axis=0)
        return np.sqrt(squared)

    def peak_to_continuous_ratio(self) -> np.ndarray:
        continuous = self.continuous_torque_nm()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = self.peak_torque_nm() / continuous
        return np.nan_to_num(ratio, nan=0.0, posinf=0.0)


def standard_load_cases(
    assembly: Assembly,
    density_kg_m3: float,
    nominal_payload_kg: float = 2.0,
    max_payload_kg: float = 5.0,
    nominal_accel_rad_s2: float = 2.0,
    max_accel_rad_s2: float = 20.0,
    nominal_speed_rad_s: float = 1.0,
) -> list[LoadCase]:
    """A minimal but honest duty cycle for a serial arm.

    Numbering follows the load-case discipline the project uses elsewhere: a
    nominal operating point, the extremes of payload and acceleration, the worst
    gravity pose, a pure holding case, and the combination of the worst of each.
    """
    n = assembly.dof
    worst_q, _ = worst_gravity_pose(
        assembly, density_kg_m3,
        np.array([0.0, -max_payload_kg * STANDARD_GRAVITY, 0.0]))
    mid_q = np.zeros(n)
    mid_q[0] = np.pi / 4

    return [
        LoadCase("LC1_nominal", "nominal payload, moderate acceleration",
                 mid_q, np.full(n, nominal_speed_rad_s),
                 np.full(n, nominal_accel_rad_s2), nominal_payload_kg, 0.55),
        LoadCase("LC2_max_payload", "maximum payload, holding",
                 mid_q, np.zeros(n), np.zeros(n), max_payload_kg, 0.15),
        LoadCase("LC3_max_acceleration", "nominal payload, peak acceleration",
                 mid_q, np.full(n, nominal_speed_rad_s),
                 np.full(n, max_accel_rad_s2), nominal_payload_kg, 0.05),
        LoadCase("LC4_worst_gravity", "worst gravity pose, holding",
                 worst_q, np.zeros(n), np.zeros(n), nominal_payload_kg, 0.10),
        LoadCase("LC6_holding", "nominal payload, stationary",
                 mid_q, np.zeros(n), np.zeros(n), nominal_payload_kg, 0.15),
        LoadCase("LC7_combined_worst",
                 "maximum payload at peak acceleration in the worst pose",
                 worst_q, np.full(n, nominal_speed_rad_s),
                 np.full(n, max_accel_rad_s2), max_payload_kg, 0.0),
    ]


def evaluate_case(assembly: Assembly, case: LoadCase,
                  density_kg_m3: float,
                  require_reachable: bool = True) -> CaseResult:
    """Torques and power for one load case.

    A case whose pose violates a joint limit is REFUSED rather than evaluated.
    Joint limits used to be recorded and never checked, so a duty cycle could
    size a motor from a pose the mechanism cannot reach; the torque computed
    there is not conservative, it describes a different mechanism. Pass
    `require_reachable=False` only to study a pose deliberately outside the
    limits, which is a legitimate thing to want and should have to be asked
    for.
    """
    if require_reachable:
        violations = assembly.limit_violations(case.q)
        if violations:
            raise ValueError(
                f"load case {case.name!r} is not reachable: "
                + "; ".join(str(v) for v in violations))
    torque = inverse_dynamics(assembly, case.q, case.qd, case.qdd,
                              density_kg_m3, tip_force_n=case.payload_force_n())
    static = inverse_dynamics(assembly, case.q, np.zeros_like(case.qd),
                              np.zeros_like(case.qdd), density_kg_m3,
                              tip_force_n=case.payload_force_n())
    return CaseResult(case=case, torque_nm=torque,
                      power_w=joint_power_w(torque, case.qd),
                      static_torque_nm=static)


def evaluate_duty_cycle(assembly: Assembly, density_kg_m3: float,
                        cases: list[LoadCase] | None = None,
                        **kwargs) -> DutyCycle:
    cases = cases or standard_load_cases(assembly, density_kg_m3, **kwargs)
    return DutyCycle([evaluate_case(assembly, case, density_kg_m3)
                      for case in cases])
