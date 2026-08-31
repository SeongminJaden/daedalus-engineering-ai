"""drivetrain.selection.select: choosing a motor and gearbox from a duty cycle.

Selection is driven by the Phase 11 load cases, not by a single torque number.
A drive that meets the peak can still overheat holding the continuous load, and
one sized to the continuous value can stall on the first acceleration, so both
are checked and both are reported.

Every check states Required against Available with an explicit margin, so a
result can be audited rather than trusted.

WHAT THIS IS NOT: a final component decision. The catalogue is illustrative,
the thermal check is a continuous-torque proxy rather than a real thermal model
(which needs duty profile, ambient temperature and thermal resistance), and
friction and compliance are still zero. This is first-pass screening.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drivetrain.gearboxes.catalog import GearboxSpec, gearboxes
from drivetrain.motors.catalog import MotorSpec, motors

# Guidance figure. A load-to-rotor inertia ratio far above this makes a drive
# hard to tune and sluggish to accelerate; it is reported rather than enforced,
# because the acceptable value depends on the controller.
INERTIA_RATIO_GUIDANCE = 10.0


@dataclass
class Requirement:
    """What one joint has to do, from the Phase 11 duty cycle."""

    joint: str
    continuous_torque_nm: float
    peak_torque_nm: float
    max_speed_rad_s: float
    load_inertia_kg_m2: float = 0.0
    max_backlash_arcmin: float | None = None


@dataclass
class Check:
    """One Required against Available comparison."""

    name: str
    required: float
    available: float
    unit: str
    passes: bool
    note: str = ""

    @property
    def margin(self) -> float:
        """Available divided by required. Below 1.0 fails."""
        if self.required == 0:
            return float("inf")
        return self.available / self.required


@dataclass
class Candidate:
    """A motor and gearbox pairing, with every check evaluated."""

    motor: MotorSpec
    gearbox: GearboxSpec
    requirement: Requirement
    checks: list[Check] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return all(check.passes for check in self.checks)

    @property
    def total_mass_kg(self) -> float:
        return self.motor.mass_kg + self.gearbox.mass_kg

    @property
    def limiting_check(self) -> Check:
        """The check with the least headroom: what actually sizes this drive."""
        return min(self.checks, key=lambda c: c.margin)

    @property
    def reflected_load_inertia_kg_m2(self) -> float:
        """Load inertia seen at the motor shaft: J_load / ratio^2."""
        return self.requirement.load_inertia_kg_m2 / self.gearbox.ratio ** 2

    @property
    def inertia_ratio(self) -> float:
        return self.reflected_load_inertia_kg_m2 / self.motor.rotor_inertia_kg_m2

    @property
    def output_inertia_kg_m2(self) -> float:
        """Total inertia referred to the OUTPUT shaft.

        J_out = (J_rotor + J_gearbox_input) * ratio^2 + J_load
        """
        return ((self.motor.rotor_inertia_kg_m2 + self.gearbox.input_inertia_kg_m2)
                * self.gearbox.ratio ** 2
                + self.requirement.load_inertia_kg_m2)

    def as_dict(self) -> dict:
        return {
            "motor": self.motor.id,
            "gearbox": self.gearbox.id,
            "ratio": self.gearbox.ratio,
            "feasible": self.feasible,
            "total_mass_kg": self.total_mass_kg,
            "inertia_ratio": self.inertia_ratio,
            "limiting": self.limiting_check.name,
            "limiting_margin": self.limiting_check.margin,
            "checks": [
                {"name": c.name, "required": c.required, "available": c.available,
                 "unit": c.unit, "margin": c.margin, "passes": c.passes,
                 "note": c.note}
                for c in self.checks
            ],
        }


def output_torque_nm(motor_torque_nm: float, gearbox: GearboxSpec) -> float:
    """Torque at the output shaft: motor torque times ratio times efficiency."""
    return motor_torque_nm * gearbox.ratio * gearbox.efficiency


def required_motor_speed_rad_s(joint_speed_rad_s: float,
                               gearbox: GearboxSpec) -> float:
    """Motor speed for a given output speed: output speed times ratio."""
    return joint_speed_rad_s * gearbox.ratio


def evaluate_candidate(motor: MotorSpec, gearbox: GearboxSpec,
                       requirement: Requirement,
                       safety_factor: float = 1.0) -> Candidate:
    """Run every check for one pairing."""
    candidate = Candidate(motor=motor, gearbox=gearbox, requirement=requirement)

    continuous_available = output_torque_nm(motor.continuous_torque_nm, gearbox)
    peak_available = output_torque_nm(motor.peak_torque_nm, gearbox)
    required_continuous = requirement.continuous_torque_nm * safety_factor
    required_peak = requirement.peak_torque_nm * safety_factor

    candidate.checks.append(Check(
        "continuous torque", required_continuous, continuous_available, "N m",
        continuous_available >= required_continuous,
        "motor continuous torque through the gearbox; sets thermal duty"))
    candidate.checks.append(Check(
        "peak torque", required_peak, peak_available, "N m",
        peak_available >= required_peak,
        "motor peak torque through the gearbox; brief acceleration only"))
    candidate.checks.append(Check(
        "gearbox rated torque", required_continuous,
        gearbox.rated_output_torque_nm, "N m",
        gearbox.rated_output_torque_nm >= required_continuous,
        "gearbox continuous rating, independent of the motor"))
    candidate.checks.append(Check(
        "gearbox peak torque", required_peak, gearbox.peak_output_torque_nm,
        "N m", gearbox.peak_output_torque_nm >= required_peak,
        "gearbox momentary rating"))

    motor_speed = required_motor_speed_rad_s(requirement.max_speed_rad_s, gearbox)
    candidate.checks.append(Check(
        "motor speed", motor_speed, motor.max_speed_rad_s, "rad/s",
        motor.max_speed_rad_s >= motor_speed,
        "a high ratio buys torque and costs speed"))

    if requirement.max_backlash_arcmin is not None:
        candidate.checks.append(Check(
            "backlash", gearbox.backlash_arcmin,
            requirement.max_backlash_arcmin, "arcmin",
            gearbox.backlash_arcmin <= requirement.max_backlash_arcmin,
            "lower is better; this check is inverted"))

    return candidate


def select_drivetrain(requirement: Requirement, safety_factor: float = 1.0,
                      motor_list: list[MotorSpec] | None = None,
                      gearbox_list: list[GearboxSpec] | None = None
                      ) -> tuple[Candidate | None, list[Candidate]]:
    """Best feasible pairing and the ranked alternatives.

    Ranking is by total mass, lightest first: on a serial arm every kilogram at
    a joint is carried by every joint inboard of it, so mass is the cost that
    compounds. Ties break on the limiting margin, preferring more headroom.

    Returns (None, all_candidates) when nothing is feasible. The caller is told
    what was missing rather than handed the least-bad option dressed up as a
    selection.
    """
    candidates = [
        evaluate_candidate(motor, gearbox, requirement, safety_factor)
        for motor in (motor_list or motors())
        for gearbox in (gearbox_list or gearboxes())
    ]
    feasible = [c for c in candidates if c.feasible]
    feasible.sort(key=lambda c: (c.total_mass_kg, -c.limiting_check.margin))
    return (feasible[0] if feasible else None), feasible


def compare_alternatives(candidates: list[Candidate], count: int = 3
                         ) -> list[tuple[Candidate, str]]:
    """The top options with a stated reason each, for a real trade-off record."""
    if not candidates:
        return []
    winner = candidates[0]
    out: list[tuple[Candidate, str]] = []
    for candidate in candidates[:count]:
        if candidate is winner:
            reason = (
                f"selected: lightest feasible at {candidate.total_mass_kg:.2f} kg; "
                f"limited by {candidate.limiting_check.name} at "
                f"{candidate.limiting_check.margin:.2f}x margin")
        else:
            parts = [f"{candidate.total_mass_kg - winner.total_mass_kg:+.2f} kg"]
            if candidate.gearbox.family is not winner.gearbox.family:
                parts.append(
                    f"{candidate.gearbox.family.value} instead of "
                    f"{winner.gearbox.family.value}")
            if candidate.gearbox.backlash_arcmin < winner.gearbox.backlash_arcmin:
                parts.append(
                    f"less backlash ({candidate.gearbox.backlash_arcmin:.0f}' vs "
                    f"{winner.gearbox.backlash_arcmin:.0f}')")
            if candidate.gearbox.efficiency > winner.gearbox.efficiency:
                parts.append(
                    f"better efficiency ({candidate.gearbox.efficiency:.2f} vs "
                    f"{winner.gearbox.efficiency:.2f})")
            if candidate.inertia_ratio < winner.inertia_ratio:
                parts.append("lower inertia ratio")
            parts.append(
                f"limited by {candidate.limiting_check.name} at "
                f"{candidate.limiting_check.margin:.2f}x")
            reason = "; ".join(parts)
        out.append((candidate, reason))
    return out


def infeasibility_report(requirement: Requirement,
                         candidates: list[Candidate]) -> str:
    """Why nothing worked, in terms of the check that failed most often."""
    if not candidates:
        return "no candidates were evaluated"
    failures: dict[str, int] = {}
    worst: dict[str, float] = {}
    for candidate in candidates:
        for check in candidate.checks:
            if not check.passes:
                failures[check.name] = failures.get(check.name, 0) + 1
                worst[check.name] = max(worst.get(check.name, 0.0),
                                        check.required / max(check.available, 1e-30))
    if not failures:
        return "all candidates passed"
    ordered = sorted(failures.items(), key=lambda kv: -kv[1])
    lines = [f"no feasible drivetrain for joint {requirement.joint!r}:"]
    for name, count in ordered:
        lines.append(
            f"  {name} failed in {count} of {len(candidates)} pairings; "
            f"short by up to {worst[name]:.2f}x")
    return "\n".join(lines)
