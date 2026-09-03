"""Joint friction, and the refusal to invent it.

The equations carry a friction term and it has been zero since it was written,
with a docstring saying the data does not exist. Zero friction is optimistic:
it under-states the torque a motor must supply and over-states the torque
available for the load. This module makes the term usable when someone has
measured it, and refuses to produce a number when nobody has.

THE MODEL
=========
Coulomb plus viscous, per joint, in the joint's own coordinates:

    tau_friction = sign(qd) * coulomb_nm + viscous_nm_s_rad * qd

with a breakaway term applied below a small speed threshold, because a joint
that is not moving needs more torque to start than to keep going, and a model
without it under-states the start of every move. The threshold is a modelling
choice and is stated, not hidden: below it the sign function is replaced by a
linear ramp so the torque is continuous and an integrator does not chatter.

WHY THERE ARE NO DEFAULTS
=========================
Coulomb torque, the viscous coefficient and breakaway all depend on the
bearing, the seal, the lubricant, the preload and the temperature. A plausible
default would put a fabricated number inside a torque that a motor is selected
from, and the selection would then look precise. `JointFriction` therefore has
no defaults, `friction_torques` refuses an assembly with no friction data, and
the caller who has none gets an explicit zero from `frictionless` with the
word optimistic in its docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FrictionDataMissing(ValueError):
    """A friction torque was asked for and no measured parameters exist."""


@dataclass(frozen=True)
class JointFriction:
    """Measured friction parameters for one joint.

    Every field is required. `source` is required too: a friction model whose
    numbers cannot be traced is the thing this module exists to prevent.
    """

    coulomb_nm: float
    viscous_nm_s_rad: float
    breakaway_nm: float
    source: str
    creep_speed_rad_s: float = 1e-3

    def __post_init__(self) -> None:
        for name in ("coulomb_nm", "viscous_nm_s_rad", "breakaway_nm"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative")
        if not self.source.strip():
            raise ValueError(
                "a friction model needs a source: which joint, measured how, "
                "by whom. Without it the number cannot be checked and should "
                "not be in a torque a motor is chosen from")
        if self.creep_speed_rad_s <= 0.0:
            raise ValueError("the creep speed must be positive")

    def torque_nm(self, speed_rad_s: float) -> float:
        """Friction torque opposing the motion at this speed."""
        speed = float(speed_rad_s)
        magnitude = abs(speed)
        if magnitude <= self.creep_speed_rad_s:
            # Linear ramp through zero: continuous, and it reaches the
            # breakaway value at the creep speed rather than at zero.
            ramp = magnitude / self.creep_speed_rad_s
            static = self.breakaway_nm * ramp
            return float(np.sign(speed) * static
                         + self.viscous_nm_s_rad * speed)
        return float(np.sign(speed) * self.coulomb_nm
                     + self.viscous_nm_s_rad * speed)


def friction_torques(frictions, qd) -> np.ndarray:
    """Friction torque per joint from measured parameters.

    `frictions` is one JointFriction per joint. A None entry is refused rather
    than treated as zero, because a chain with one unmeasured joint has an
    unknown friction torque, not a smaller one.
    """
    qd = np.asarray(qd, dtype=float).reshape(-1)
    if frictions is None or len(frictions) != qd.size:
        raise FrictionDataMissing(
            f"friction parameters for {qd.size} joints are needed and "
            f"{0 if frictions is None else len(frictions)} were given")
    for index, friction in enumerate(frictions):
        if friction is None:
            raise FrictionDataMissing(
                f"joint {index} has no measured friction parameters; a chain "
                f"with one unmeasured joint has an unknown friction torque, "
                f"not a smaller one")
    return np.array([f.torque_nm(v) for f, v in zip(frictions, qd)], dtype=float)


def frictionless(dof: int) -> np.ndarray:
    """Zero friction, explicitly.

    Optimistic and labelled as such: a real joint needs torque to start moving
    and to keep moving, so a torque computed with this is a lower bound on
    what a motor must supply.
    """
    return np.zeros(int(dof), dtype=float)
