"""Conduction, convection and radiation as thermal resistances.

Phase 18 modelled a motor with one lumped resistance and said plainly that the
mounting dominates it and that it could be out by a factor of two. This is the
layer that lets that number be built rather than assumed.

VALIDITY, before the implementation:

* **These resistances assume ONE-DIMENSIONAL heat flow.** A plane wall
  resistance L/(kA) is exact only when the heat goes straight through. Real
  parts spread heat laterally, and a small heat source on a large plate is
  helped enormously by spreading that this does not model, so a network built
  this way is CONSERVATIVE for a concentrated source and can be badly wrong for
  the geometry of the spreading itself.

* **The convection coefficient is the dominant uncertainty in almost every
  thermal model, and it is not a material property.** Natural convection in air
  spans roughly 5 to 25 W/m^2K depending on orientation, size and surface
  temperature; forced air spans 10 to 300 depending on velocity. A model whose
  answer rests on h is uncertain by whatever h is uncertain by, which is
  usually a factor of two or more. The correlations here are for specific
  geometries and are stated as such.

* **Radiation is not negligible at temperatures people think it is.** A surface
  at 100 C radiating to a 25 C room transfers a comparable amount to natural
  convection, so omitting it typically OVER-predicts the temperature rise. It
  is also fourth-power in absolute temperature, so linearising it is only valid
  over a modest range and the linearisation point matters.

* **Emissivity is a surface property, not a material one.** Polished aluminium
  is about 0.05 and anodised aluminium about 0.8, a factor of sixteen from the
  same metal. Assuming a material's emissivity without knowing its finish is
  the single easiest way to be wrong here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

STEFAN_BOLTZMANN = 5.670374419e-8      # W m^-2 K^-4, exact by SI definition
ABSOLUTE_ZERO_C = -273.15

# Indicative convection coefficient ranges in W/m^2K. These are the SPREAD, not
# values to use: they are here so a caller can see how wide the uncertainty is
# before picking a number.
CONVECTION_RANGES: dict[str, tuple[float, float]] = {
    "natural_air": (5.0, 25.0),
    "forced_air": (10.0, 300.0),
    "natural_water": (50.0, 1000.0),
    "forced_water": (300.0, 12000.0),
}


def celsius_to_kelvin(temperature_c: float) -> float:
    kelvin = temperature_c - ABSOLUTE_ZERO_C
    if kelvin <= 0.0:
        raise ValueError(f"{temperature_c} C is at or below absolute zero")
    return kelvin


# --- conduction --------------------------------------------------------------

def plane_wall_resistance_k_w(thickness_m: float, area_m2: float,
                              conductivity_w_mk: float) -> float:
    """R = L / (k A)."""
    if min(thickness_m, area_m2, conductivity_w_mk) <= 0.0:
        raise ValueError("thickness, area and conductivity must be positive")
    return thickness_m / (conductivity_w_mk * area_m2)


def cylinder_resistance_k_w(inner_radius_m: float, outer_radius_m: float,
                            length_m: float,
                            conductivity_w_mk: float) -> float:
    """R = ln(ro/ri) / (2 pi k L).

    Logarithmic rather than linear in thickness, which is why insulation on a
    small pipe can INCREASE heat loss: the added conduction resistance is
    outweighed by the larger outer surface available to convect from. That is
    the critical radius effect and this expression is where it comes from.
    """
    if inner_radius_m <= 0.0 or outer_radius_m <= inner_radius_m:
        raise ValueError("the outer radius must exceed the inner one")
    if length_m <= 0.0 or conductivity_w_mk <= 0.0:
        raise ValueError("length and conductivity must be positive")
    return math.log(outer_radius_m / inner_radius_m) / (
        2.0 * math.pi * conductivity_w_mk * length_m)


def sphere_resistance_k_w(inner_radius_m: float, outer_radius_m: float,
                          conductivity_w_mk: float) -> float:
    """R = (1/ri - 1/ro) / (4 pi k)."""
    if inner_radius_m <= 0.0 or outer_radius_m <= inner_radius_m:
        raise ValueError("the outer radius must exceed the inner one")
    if conductivity_w_mk <= 0.0:
        raise ValueError("conductivity must be positive")
    return (1.0 / inner_radius_m - 1.0 / outer_radius_m) / (
        4.0 * math.pi * conductivity_w_mk)


def fourier_heat_w(conductivity_w_mk: float, area_m2: float,
                   temperature_difference_k: float,
                   thickness_m: float) -> float:
    """q = k A dT / L, Fourier's law for a plane wall."""
    return (conductivity_w_mk * area_m2 * temperature_difference_k
            / thickness_m)


# --- convection --------------------------------------------------------------

def convection_resistance_k_w(coefficient_w_m2k: float,
                              area_m2: float) -> float:
    """R = 1 / (h A)."""
    if coefficient_w_m2k <= 0.0 or area_m2 <= 0.0:
        raise ValueError("the coefficient and area must be positive")
    return 1.0 / (coefficient_w_m2k * area_m2)


def newton_cooling_w(coefficient_w_m2k: float, area_m2: float,
                     surface_c: float, ambient_c: float) -> float:
    """q = h A (Ts - Tinf). Positive means the surface is losing heat."""
    return coefficient_w_m2k * area_m2 * (surface_c - ambient_c)


def natural_convection_vertical_plate_w_m2k(surface_c: float, ambient_c: float,
                                            height_m: float) -> float:
    """A simplified correlation for a vertical plate in air.

        h = 1.42 (dT / L)^(1/4)

    VALIDITY: air at around atmospheric pressure and moderate temperatures,
    laminar, for a vertical surface. It is the standard simplified form and
    carries the scatter of natural convection generally, so treat it as an
    order of magnitude rather than a value. A horizontal surface, a different
    fluid or a turbulent boundary layer all need a different correlation.
    """
    if height_m <= 0.0:
        raise ValueError("the plate height must be positive")
    difference = abs(surface_c - ambient_c)
    if difference == 0.0:
        return 0.0
    return 1.42 * (difference / height_m) ** 0.25


# --- radiation ---------------------------------------------------------------

def radiation_heat_w(emissivity: float, area_m2: float, surface_c: float,
                     surroundings_c: float) -> float:
    """q = eps sigma A (Ts^4 - Tsur^4), with temperatures in kelvin.

    Fourth power in ABSOLUTE temperature, so it must be evaluated in kelvin.
    Using celsius here is a classic error and gives a wildly wrong answer
    rather than a slightly wrong one.
    """
    if not 0.0 < emissivity <= 1.0:
        raise ValueError("emissivity lies in (0, 1]")
    if area_m2 <= 0.0:
        raise ValueError("area must be positive")
    surface_k = celsius_to_kelvin(surface_c)
    surroundings_k = celsius_to_kelvin(surroundings_c)
    return (emissivity * STEFAN_BOLTZMANN * area_m2
            * (surface_k ** 4 - surroundings_k ** 4))


def radiation_coefficient_w_m2k(emissivity: float, surface_c: float,
                                surroundings_c: float) -> float:
    """The linearised radiation coefficient.

        h_rad = eps sigma (Ts^2 + Tsur^2)(Ts + Tsur)

    Exact at the two temperatures it is evaluated at and increasingly wrong
    away from them, because the underlying law is quartic. It exists so
    radiation can join a resistance network, which is linear by construction.
    """
    if not 0.0 < emissivity <= 1.0:
        raise ValueError("emissivity lies in (0, 1]")
    surface_k = celsius_to_kelvin(surface_c)
    surroundings_k = celsius_to_kelvin(surroundings_c)
    return (emissivity * STEFAN_BOLTZMANN
            * (surface_k ** 2 + surroundings_k ** 2)
            * (surface_k + surroundings_k))


@dataclass(frozen=True)
class SurfaceLoss:
    """How a surface sheds heat, split by mechanism."""

    convection_w: float
    radiation_w: float
    convection_coefficient_w_m2k: float
    radiation_coefficient_w_m2k: float

    @property
    def total_w(self) -> float:
        return self.convection_w + self.radiation_w

    @property
    def radiation_share(self) -> float:
        return 0.0 if self.total_w == 0.0 else self.radiation_w / self.total_w


def surface_loss(area_m2: float, surface_c: float, ambient_c: float,
                 convection_coefficient_w_m2k: float,
                 emissivity: float) -> SurfaceLoss:
    """Convective and radiative loss side by side.

    Reported separately because the relative size is the useful output:
    omitting radiation is common and at moderate temperatures it is not a small
    correction.
    """
    convection = newton_cooling_w(convection_coefficient_w_m2k, area_m2,
                                  surface_c, ambient_c)
    radiation = radiation_heat_w(emissivity, area_m2, surface_c, ambient_c)
    return SurfaceLoss(
        convection_w=convection, radiation_w=radiation,
        convection_coefficient_w_m2k=convection_coefficient_w_m2k,
        radiation_coefficient_w_m2k=radiation_coefficient_w_m2k(
            emissivity, surface_c, ambient_c))
