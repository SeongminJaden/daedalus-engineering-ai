"""One-way coupling: a thermal result feeding a structural check.

VALIDITY, before the implementation, and this is the whole point of the module:

* **The coupling here is ONE-WAY, and real coupling is not.** The thermal
  solution sets a temperature, the temperature produces a thermal stress, and
  the thermal stress joins the mechanical one. Nothing goes back the other way.
  In reality the structure's deformation changes the contact pressure at its
  interfaces, which changes the contact resistance, which changes the
  temperature that caused the deformation. Closing that loop needs iteration to
  convergence and is NOT implemented.

* **One-way coupling is not conservative in a known direction.** It is
  sometimes optimistic and sometimes not. A part that expands into its mounting
  gains contact area and runs cooler than this predicts; one that bows away
  from a heatsink loses contact and runs hotter. Which happens depends on the
  geometry, and nothing here can tell them apart, so the error has no
  reliable sign.

* **Superposition of thermal and mechanical stress requires LINEAR ELASTICITY
  and it is the only reason this is legitimate at all.** Once anything yields,
  the two cannot be added, because the thermal stress relaxes as the material
  flows while the mechanical one does not.

* **A single temperature per part is assumed.** A real gradient produces
  stress with no external restraint whatsoever, and this cannot see that: it
  only sees the restrained expansion of a uniformly heated body.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.materials.db import MaterialSpec
from physics.thermal.network import ThermalPath
from physics.thermal.stress import check_thermal_stress


@dataclass(frozen=True)
class CoupledResult:
    """A thermal solution and the structural consequence it produces."""

    power_w: float
    thermal_resistance_k_w: float
    temperature_rise_k: float
    operating_c: float
    reference_c: float
    thermal_stress_pa: float
    mechanical_stress_pa: float
    combined_stress_pa: float
    safety_factor: float
    governing_contribution: str
    dominant_resistance: str

    @property
    def passes(self) -> bool:
        return self.safety_factor >= 1.0

    @property
    def thermal_worsens_it(self) -> bool:
        """Whether the thermal contribution ADDS to the mechanical stress.

        It does not always. A restrained part that is heated goes into
        compression, which relieves a tensile mechanical stress. Reporting the
        direction matters more than reporting the magnitude.
        """
        return abs(self.combined_stress_pa) > abs(self.mechanical_stress_pa)


def thermal_structural(path: ThermalPath, power_w: float,
                       material: MaterialSpec, ambient_c: float,
                       reference_c: float, mechanical_stress_pa: float,
                       constraint: float = 1.0) -> CoupledResult:
    """Solve the heat path, then feed its temperature into a thermal stress.

    `reference_c` is the temperature at which the part was assembled and is
    therefore free of thermal stress. The stress comes from the difference
    between operating and reference, NOT from the operating temperature alone,
    and using the wrong one is a common way to invent stress that is not there.
    """
    rise = path.temperature_rise_k(power_w)
    operating = ambient_c + rise
    stress = check_thermal_stress(material, delta_t_k=operating - reference_c,
                                  mechanical_stress_pa=mechanical_stress_pa,
                                  constraint=constraint)
    return CoupledResult(
        power_w=power_w, thermal_resistance_k_w=path.total_k_w,
        temperature_rise_k=rise, operating_c=operating,
        reference_c=reference_c,
        thermal_stress_pa=stress.thermal_stress_pa,
        mechanical_stress_pa=mechanical_stress_pa,
        combined_stress_pa=stress.combined_stress_pa,
        safety_factor=stress.safety_factor,
        governing_contribution=stress.governing_contribution,
        dominant_resistance=path.dominant().name)
