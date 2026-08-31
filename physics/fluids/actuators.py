"""Fluid power actuators, external drag and cooling flow.

VALIDITY, before the implementation:

* **P times A is the THEORETICAL force and a real cylinder does not deliver
  it.** Seal friction, and the back pressure the exhaust side has to push
  through the return line and valve, both subtract. A real pneumatic cylinder
  delivers roughly 80 to 95 percent of P A depending on seals, lubrication and
  how restricted the return path is. The efficiency is an explicit argument
  defaulting to 1.0, which is the OPTIMISTIC value, so that its absence is
  visible rather than assumed away.

* **Retracting is weaker than extending, always.** The rod occupies part of the
  annulus, so the same pressure acts on a smaller area. The ratio is fixed by
  the rod diameter and is often close to two, which is why a cylinder sized on
  its extend force can be badly undersized on the return stroke.

* **Drag coefficients are strongly Reynolds dependent and a single quoted
  value is only valid over a range.** The familiar 0.47 for a sphere holds
  roughly between Reynolds 1e3 and 2e5, and above that the boundary layer
  turns turbulent and the coefficient DROPS by a factor of three or so. Quoting
  a single number across that transition is wrong in a large and surprising
  way, so the caller supplies the coefficient and this states the range.

* **The cooling energy balance is steady state and assumes the coolant
  actually reaches the surface it is meant to cool.** It gives the flow needed
  to carry a heat load away at a stated temperature rise, which is a necessary
  condition and nowhere near sufficient: whether the heat gets INTO the coolant
  is the thermal resistance problem next door.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Indicative sphere drag coefficient and the Reynolds range it holds over.
SPHERE_CD = 0.47
SPHERE_CD_RANGE = (1.0e3, 2.0e5)


@dataclass(frozen=True)
class CylinderForce:
    """Extend and retract forces for a double-acting cylinder."""

    extend_n: float
    retract_n: float
    extend_area_m2: float
    retract_area_m2: float
    efficiency: float

    @property
    def retract_ratio(self) -> float:
        """How much weaker the return stroke is."""
        return self.retract_n / self.extend_n if self.extend_n else 0.0


def cylinder_force(pressure_pa: float, bore_m: float, rod_m: float,
                   efficiency: float = 1.0) -> CylinderForce:
    """F = eta P A, with the annulus area on the retract stroke.

    `efficiency` defaults to 1.0, which is optimistic and deliberately visible.
    """
    if bore_m <= 0.0:
        raise ValueError("bore must be positive")
    if not 0.0 <= rod_m < bore_m:
        raise ValueError("the rod must be smaller than the bore")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency lies in (0, 1]")
    extend_area = math.pi * bore_m ** 2 / 4.0
    retract_area = math.pi * (bore_m ** 2 - rod_m ** 2) / 4.0
    return CylinderForce(
        extend_n=efficiency * pressure_pa * extend_area,
        retract_n=efficiency * pressure_pa * retract_area,
        extend_area_m2=extend_area, retract_area_m2=retract_area,
        efficiency=efficiency)


def cylinder_flow_m3_s(area_m2: float, speed_m_s: float) -> float:
    """Q = A v, the flow a cylinder swallows at a given speed.

    This is what sizes the valve and the line, and it is usually what limits
    the achievable speed rather than the available force.
    """
    if area_m2 <= 0.0 or speed_m_s < 0.0:
        raise ValueError("area must be positive and speed non-negative")
    return area_m2 * speed_m_s


def drag_force_n(density_kg_m3: float, velocity_m_s: float,
                 drag_coefficient: float, frontal_area_m2: float) -> float:
    """F = 0.5 rho V^2 Cd A.

    Quadratic in velocity, so the power to overcome it goes as the CUBE.
    """
    if density_kg_m3 <= 0.0 or frontal_area_m2 <= 0.0:
        raise ValueError("density and area must be positive")
    if drag_coefficient < 0.0:
        raise ValueError("a drag coefficient cannot be negative")
    return 0.5 * density_kg_m3 * velocity_m_s ** 2 * drag_coefficient \
        * frontal_area_m2


def drag_power_w(density_kg_m3: float, velocity_m_s: float,
                 drag_coefficient: float, frontal_area_m2: float) -> float:
    """P = F V, and therefore cubic in velocity."""
    return velocity_m_s * drag_force_n(density_kg_m3, velocity_m_s,
                                       drag_coefficient, frontal_area_m2)


def sphere_cd_is_in_range(reynolds: float) -> bool:
    """Whether the quoted 0.47 actually applies at this Reynolds number."""
    low, high = SPHERE_CD_RANGE
    return low <= reynolds <= high


@dataclass(frozen=True)
class CoolingRequirement:
    """The coolant flow a heat load needs at a stated temperature rise."""

    heat_w: float
    temperature_rise_k: float
    mass_flow_kg_s: float
    volume_flow_m3_s: float

    @property
    def volume_flow_l_min(self) -> float:
        return self.volume_flow_m3_s * 60000.0


def cooling_flow(heat_w: float, temperature_rise_k: float,
                 specific_heat_j_kgk: float,
                 density_kg_m3: float) -> CoolingRequirement:
    """m_dot = Q / (c dT), from the steady energy balance.

    A NECESSARY condition and nowhere near sufficient: it says the coolant can
    carry the heat away, not that the heat can get into the coolant. That is
    the thermal resistance question, and a loop with ample flow and a poor
    interface runs hot regardless.
    """
    if temperature_rise_k <= 0.0:
        raise ValueError(
            "the allowed temperature rise must be positive; a coolant that "
            "does not warm up carries no heat")
    if specific_heat_j_kgk <= 0.0 or density_kg_m3 <= 0.0:
        raise ValueError("specific heat and density must be positive")
    mass_flow = heat_w / (specific_heat_j_kgk * temperature_rise_k)
    return CoolingRequirement(
        heat_w=heat_w, temperature_rise_k=temperature_rise_k,
        mass_flow_kg_s=mass_flow,
        volume_flow_m3_s=mass_flow / density_kg_m3)
