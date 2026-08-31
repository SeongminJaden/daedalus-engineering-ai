"""Interference fits: interference to contact pressure to torque capacity.

A press fit holds by friction against a pressure the assembly generates itself,
so the whole calculation hangs off how much interference actually survives
assembly.

VALIDITY, before the implementation:

* **Thick-wall elastic (Lame), axisymmetric, plane stress.** If the hub bore
  yields, the pressure stops following the interference and this over-predicts
  the holding capacity. The hoop stress is returned so that can be checked
  rather than assumed.

* **Surface roughness is NOT deducted, and it always reduces the interference.**
  Pressing flattens the asperities on both surfaces, so the interference that
  ends up producing pressure is smaller than the interference that was
  measured, conventionally by about 1.2 times the sum of the two roughness
  heights. On a small fit that can be a large fraction of the total. Everything
  here is therefore OPTIMISTIC, and a real design deducts it.

* **The friction coefficient is the weakest number in the chain.** Pressed dry
  steel on steel is usually quoted between 0.10 and 0.15, shrink-fitted higher,
  and any oil left on the parts lowers it sharply. Torque capacity is directly
  proportional to it, so a coefficient uncertain by a factor of two makes the
  capacity uncertain by a factor of two.

* **Static holding only.** No fretting, no loosening under reversed torque, and
  no thermal effect: a fit assembled at room temperature and run hot in a hub
  that expands faster than its shaft can lose its interference entirely, and
  nothing here models that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Conventional dry pressed steel value. [ASSUMED] and the dominant uncertainty.
FRICTION_PRESSED_STEEL = 0.12


@dataclass(frozen=True)
class PressFitResult:
    """Contact pressure and what it holds."""

    interference_m: float
    contact_pressure_pa: float
    torque_capacity_nm: float
    axial_capacity_n: float
    hub_hoop_stress_pa: float
    hub_yield_safety_factor: float | None
    friction_coefficient: float

    @property
    def hub_yields(self) -> bool:
        return (self.hub_yield_safety_factor is not None
                and self.hub_yield_safety_factor < 1.0)


def contact_pressure_pa(interference_m: float, shaft_diameter_m: float,
                        hub_outer_diameter_m: float,
                        hub_modulus_pa: float, hub_poisson: float,
                        shaft_modulus_pa: float, shaft_poisson: float,
                        shaft_bore_m: float = 0.0) -> float:
    """Lame contact pressure for a diametral interference.

        delta = p d [ (1/E_h)((D^2+d^2)/(D^2-d^2) + nu_h)
                    + (1/E_s)((d^2+di^2)/(d^2-di^2) - nu_s) ]

    solved for p. For a solid shaft the second bracket becomes (1 - nu_s)/E_s,
    and when both parts are the same material the whole thing collapses to
    p = delta E (D^2 - d^2) / (2 d D^2), which is checked against this.
    """
    d, big = shaft_diameter_m, hub_outer_diameter_m
    if d <= 0.0 or big <= d:
        raise ValueError("the hub outer diameter must exceed the shaft diameter")
    if shaft_bore_m < 0.0 or shaft_bore_m >= d:
        raise ValueError("a hollow shaft's bore must be smaller than its outside")
    if interference_m < 0.0:
        raise ValueError("interference is a positive magnitude")

    hub_term = ((big ** 2 + d ** 2) / (big ** 2 - d ** 2) + hub_poisson) / hub_modulus_pa
    inner = shaft_bore_m
    shaft_term = ((d ** 2 + inner ** 2) / (d ** 2 - inner ** 2)
                  - shaft_poisson) / shaft_modulus_pa
    return interference_m / (d * (hub_term + shaft_term))


def hub_hoop_stress_pa(pressure_pa: float, shaft_diameter_m: float,
                       hub_outer_diameter_m: float) -> float:
    """Tangential stress at the hub bore, where it is largest.

        sigma_theta = p (D^2 + d^2) / (D^2 - d^2)

    Always tensile and always larger than the pressure, which is why a thin hub
    splits before the pressure looks alarming.
    """
    d, big = shaft_diameter_m, hub_outer_diameter_m
    return pressure_pa * (big ** 2 + d ** 2) / (big ** 2 - d ** 2)


def analyze_press_fit(interference_m: float, shaft_diameter_m: float,
                      hub_outer_diameter_m: float, engagement_length_m: float,
                      hub_modulus_pa: float, hub_poisson: float,
                      shaft_modulus_pa: float, shaft_poisson: float,
                      friction: float = FRICTION_PRESSED_STEEL,
                      hub_yield_pa: float | None = None,
                      shaft_bore_m: float = 0.0) -> PressFitResult:
    """Pressure, torque and axial capacity, and the hub's hoop stress.

        T = mu p pi d^2 L / 2        F_axial = mu p pi d L
    """
    if engagement_length_m <= 0.0:
        raise ValueError("engagement length must be positive")
    if friction <= 0.0:
        raise ValueError("the friction coefficient must be positive")

    pressure = contact_pressure_pa(
        interference_m, shaft_diameter_m, hub_outer_diameter_m,
        hub_modulus_pa, hub_poisson, shaft_modulus_pa, shaft_poisson,
        shaft_bore_m)
    hoop = hub_hoop_stress_pa(pressure, shaft_diameter_m, hub_outer_diameter_m)
    return PressFitResult(
        interference_m=interference_m, contact_pressure_pa=pressure,
        torque_capacity_nm=(friction * pressure * math.pi
                            * shaft_diameter_m ** 2 * engagement_length_m / 2.0),
        axial_capacity_n=(friction * pressure * math.pi * shaft_diameter_m
                          * engagement_length_m),
        hub_hoop_stress_pa=hoop,
        hub_yield_safety_factor=(None if hub_yield_pa is None
                                 else hub_yield_pa / hoop if hoop > 0 else math.inf),
        friction_coefficient=friction)
