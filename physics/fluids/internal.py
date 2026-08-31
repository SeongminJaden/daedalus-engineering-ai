"""Internal flow: Reynolds number, friction factor and pressure drop.

VALIDITY, before the implementation:

* **Laminar and turbulent flow have DIFFERENT friction laws and there is no
  valid formula between them.** Below a Reynolds number of about 2300 the flow
  is laminar and f = 64/Re exactly. Above about 4000 it is turbulent and needs
  a correlation. Between the two the flow is intermittent, the friction factor
  is not a single-valued function of Reynolds number, and any formula
  evaluated there is an interpolation rather than physics. This module marks
  that region rather than silently returning a number from whichever branch it
  happened to take.

* **The turbulent correlation is EXPLICIT and therefore approximate.** Colebrook
  is implicit in f and has to be iterated. Haaland's explicit form is used
  because it is stable and closed, and it is checked against an iterated
  Colebrook solution so the approximation is measured rather than trusted.

* **Relative roughness matters more than the pipe material name.** A drawn tube
  and a corroded steel pipe of the same bore differ by two orders of magnitude
  in roughness, and in fully rough turbulent flow the friction factor depends
  on roughness alone and not on Reynolds number at all.

* **Fully developed flow is assumed.** Entrance lengths of tens of diameters
  are needed to reach it, and a short pipe or one full of fittings is dominated
  by the entrance and the fittings rather than by the wall friction this
  computes.

* **Incompressible, Newtonian, steady, constant-property flow.** Compressible
  gas flow at appreciable Mach number needs a different treatment entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

LAMINAR_LIMIT = 2300.0
TURBULENT_ONSET = 4000.0


class FlowRegime(str, Enum):
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"


def reynolds_number(velocity_m_s: float, diameter_m: float,
                    density_kg_m3: float, viscosity_pa_s: float) -> float:
    """Re = rho V D / mu."""
    if min(diameter_m, density_kg_m3, viscosity_pa_s) <= 0.0:
        raise ValueError("diameter, density and viscosity must be positive")
    return density_kg_m3 * abs(velocity_m_s) * diameter_m / viscosity_pa_s


def flow_regime(reynolds: float) -> FlowRegime:
    if reynolds < LAMINAR_LIMIT:
        return FlowRegime.LAMINAR
    if reynolds < TURBULENT_ONSET:
        return FlowRegime.TRANSITIONAL
    return FlowRegime.TURBULENT


def laminar_friction_factor(reynolds: float) -> float:
    """f = 64 / Re, exact for fully developed laminar pipe flow.

    Not a correlation: it falls out of the analytical velocity profile.
    """
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive")
    return 64.0 / reynolds


def haaland_friction_factor(reynolds: float,
                            relative_roughness: float) -> float:
    """Haaland's explicit approximation to Colebrook.

        1/sqrt(f) = -1.8 log10[ (eps/D / 3.7)^1.11 + 6.9/Re ]
    """
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive")
    if relative_roughness < 0.0:
        raise ValueError("relative roughness cannot be negative")
    inner = (relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds
    return (-1.8 * math.log10(inner)) ** -2.0


def colebrook_friction_factor(reynolds: float, relative_roughness: float,
                              tolerance: float = 1e-12,
                              max_iterations: int = 200) -> float:
    """Colebrook solved by fixed-point iteration, for checking Haaland.

        1/sqrt(f) = -2 log10( eps/D/3.7 + 2.51 / (Re sqrt(f)) )

    Implicit, which is why the explicit forms exist. It is here as the
    reference the approximation is measured against, not as the working path.
    """
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive")
    guess = haaland_friction_factor(reynolds, relative_roughness)
    for _ in range(max_iterations):
        root = 1.0 / math.sqrt(guess)
        updated = -2.0 * math.log10(relative_roughness / 3.7
                                    + 2.51 / (reynolds * math.sqrt(guess)))
        new_guess = updated ** -2.0
        if abs(new_guess - guess) <= tolerance * guess:
            return new_guess
        guess = new_guess
    return guess


@dataclass(frozen=True)
class PipeFlow:
    """A pipe flow solution, with its regime stated."""

    reynolds: float
    regime: FlowRegime
    friction_factor: float
    velocity_m_s: float
    pressure_drop_pa: float
    friction_is_interpolated: bool

    @property
    def regime_is_certain(self) -> bool:
        return not self.friction_is_interpolated


def darcy_weisbach_pa(friction_factor: float, length_m: float,
                      diameter_m: float, density_kg_m3: float,
                      velocity_m_s: float) -> float:
    """dp = f (L/D) (rho V^2 / 2).

    Quadratic in velocity, so halving the flow quarters the pressure drop.
    That is why oversizing a cooling line is so effective and why a slightly
    undersized one is so much worse than it looks.
    """
    if min(length_m, diameter_m) <= 0.0:
        raise ValueError("length and diameter must be positive")
    return (friction_factor * length_m / diameter_m * density_kg_m3
            * velocity_m_s ** 2 / 2.0)


def minor_loss_pa(loss_coefficient: float, density_kg_m3: float,
                  velocity_m_s: float) -> float:
    """dp = K rho V^2 / 2, for a fitting, bend or valve.

    Called MINOR by convention only. In a short run with several fittings the
    minor losses exceed the pipe friction, and calling them minor is how they
    end up omitted.
    """
    if loss_coefficient < 0.0:
        raise ValueError("a loss coefficient cannot be negative")
    return loss_coefficient * density_kg_m3 * velocity_m_s ** 2 / 2.0


def solve_pipe_flow(flow_rate_m3_s: float, diameter_m: float, length_m: float,
                    density_kg_m3: float, viscosity_pa_s: float,
                    roughness_m: float = 0.0,
                    minor_loss_coefficients: float = 0.0) -> PipeFlow:
    """Full pressure drop for a volumetric flow through a round pipe.

    The friction factor is chosen by regime. In the transitional band it is
    taken from the turbulent correlation and flagged as INTERPOLATED, because
    something must be returned and nothing there is trustworthy.
    """
    if diameter_m <= 0.0:
        raise ValueError("diameter must be positive")
    area = math.pi * diameter_m ** 2 / 4.0
    velocity = flow_rate_m3_s / area
    reynolds = reynolds_number(velocity, diameter_m, density_kg_m3,
                               viscosity_pa_s)
    regime = flow_regime(reynolds)
    relative_roughness = roughness_m / diameter_m

    if regime is FlowRegime.LAMINAR:
        friction = laminar_friction_factor(reynolds)
        interpolated = False
    else:
        friction = haaland_friction_factor(reynolds, relative_roughness)
        interpolated = regime is FlowRegime.TRANSITIONAL

    drop = darcy_weisbach_pa(friction, length_m, diameter_m, density_kg_m3,
                             velocity)
    drop += minor_loss_pa(minor_loss_coefficients, density_kg_m3, velocity)
    return PipeFlow(reynolds=reynolds, regime=regime, friction_factor=friction,
                    velocity_m_s=velocity, pressure_drop_pa=drop,
                    friction_is_interpolated=interpolated)
