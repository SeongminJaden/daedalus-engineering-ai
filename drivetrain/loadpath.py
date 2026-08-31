"""From a selected drivetrain to the loads its shaft and bearings carry.

Phase 12 picks a motor and gearbox that can supply a joint's torque. That
torque does not stop there: it goes out through a shaft, and the shaft is held
by bearings, and both have to survive it. This module closes that gap so the
load path runs from the duty cycle to the parts that actually carry it.

The layout modelled is the common one: a shaft on two bearings with the load
hanging off the end. An overhung load is worth modelling because it is the
arrangement that produces the largest bending moment for a given load, and it
is what a robot joint usually looks like: the gearbox output is supported, and
the link reaches out past the support.
"""

from __future__ import annotations

from dataclasses import dataclass

from physics.shaft.design import ShaftLoads

from .selection.select import Candidate, output_torque_nm


@dataclass(frozen=True)
class ShaftLayout:
    """Where the bearings and the load sit along the output shaft.

        A ============ B ------ P
        |<-- span -->|<- over ->|

    `bearing_span_m` separates the two supports and `overhang_m` reaches from
    the near support B to the load. A short span with a long overhang multiplies
    the bearing reaction, which is why both are inputs rather than one.
    """

    bearing_span_m: float
    overhang_m: float
    radial_load_n: float
    axial_load_n: float = 0.0

    def __post_init__(self) -> None:
        if self.bearing_span_m <= 0.0:
            raise ValueError("the bearing span must be positive")
        if self.overhang_m < 0.0:
            raise ValueError("the overhang cannot be negative")


@dataclass(frozen=True)
class LoadPath:
    """What the shaft and each bearing carry, and at what speed."""

    output_torque_nm: float
    bending_moment_nm: float
    near_bearing_load_n: float
    far_bearing_load_n: float
    axial_load_n: float
    speed_rad_s: float

    def shaft_loads(self) -> ShaftLoads:
        """The shaft rotates, so its bending is fully reversed.

        This is the step that turns a steady transverse load into a fatigue
        problem, and it is the reason a shaft cannot be sized on torque alone.
        """
        return ShaftLoads.rotating(bending_moment_nm=self.bending_moment_nm,
                                   torque_nm=self.output_torque_nm,
                                   axial_force_n=self.axial_load_n)


def bending_moment_nm(layout: ShaftLayout) -> float:
    """Maximum bending moment, which is at the near bearing: M = P a."""
    return layout.radial_load_n * layout.overhang_m


def bearing_reactions_n(layout: ShaftLayout) -> tuple[float, float]:
    """(near, far) reaction magnitudes for a load overhung past the near support.

    Moments about the far support A give the near reaction

        R_B = P (L + a) / L

    and vertical equilibrium gives R_A = P a / L, acting DOWNWARD: an overhung
    load lifts the far bearing rather than pressing on it. Magnitudes are
    returned because a bearing does not care which way it is pushed, but the
    consequence is worth stating: the near bearing carries more than the whole
    applied load, not a share of it.
    """
    span, overhang = layout.bearing_span_m, layout.overhang_m
    near = layout.radial_load_n * (span + overhang) / span
    far = layout.radial_load_n * overhang / span
    return near, far


def trace(candidate: Candidate, layout: ShaftLayout,
          use_peak_torque: bool = False) -> LoadPath:
    """Carry a selected drivetrain's torque through to the shaft and bearings.

    `use_peak_torque` selects the peak rather than the continuous rating.
    Fatigue and bearing life are accumulated over the duty, so the continuous
    torque is the right one for them; the peak matters for the static check.
    Which one is being used has to be a decision rather than a default, because
    running a life calculation on a peak torque understates life by the cube of
    the ratio.
    """
    motor_torque = (candidate.motor.peak_torque_nm if use_peak_torque
                    else candidate.motor.continuous_torque_nm)
    torque = output_torque_nm(motor_torque, candidate.gearbox)
    near, far = bearing_reactions_n(layout)
    return LoadPath(
        output_torque_nm=torque,
        bending_moment_nm=bending_moment_nm(layout),
        near_bearing_load_n=near, far_bearing_load_n=far,
        axial_load_n=layout.axial_load_n,
        speed_rad_s=candidate.requirement.max_speed_rad_s)
