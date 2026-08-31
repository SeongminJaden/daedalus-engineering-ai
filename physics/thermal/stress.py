"""Thermal strain and the stress that appears when it is prevented.

Heating a free bar makes it longer and produces no stress at all. Heating a bar
whose ends are held produces no strain and a large stress. The distinction is
entirely about restraint, which is why the constraint factor is a required part
of the model rather than an afterthought: the same temperature change can give
zero stress or yield-level stress depending on how the part is mounted.

The magnitudes are easy to underestimate. Aluminium at 23.6e-6 per K in a
71.7 GPa section develops about 1.69 MPa per kelvin when fully restrained, so a
60 K excursion is over 100 MPa before any mechanical load is applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.materials.db import MaterialSpec


def free_expansion_strain(alpha_1_k: float, delta_t_k: float) -> float:
    """epsilon = alpha dT. The strain a free body actually takes up."""
    return alpha_1_k * delta_t_k


def constrained_thermal_stress_pa(youngs_modulus_pa: float, alpha_1_k: float,
                                  delta_t_k: float,
                                  constraint: float = 1.0) -> float:
    """sigma = -c E alpha dT, negative in compression for a temperature RISE.

    The sign is not decoration. A restrained bar that is heated pushes against
    its restraints and goes into COMPRESSION, and a cooled one goes into
    tension. Which one it is decides whether the stress adds to or relieves a
    mechanical tension, and whether it can drive buckling, so returning a
    magnitude would throw away the useful half of the answer.

    `constraint` runs from 0 for free expansion to 1 for full restraint. Real
    mountings sit between: a part bolted at both ends to a frame that itself
    expands is partly restrained, and treating that as fully restrained
    overstates the stress by however much the frame moves.
    """
    if not 0.0 <= constraint <= 1.0:
        raise ValueError("the constraint factor runs from 0 (free) to 1 (full)")
    return -constraint * youngs_modulus_pa * alpha_1_k * delta_t_k


def stress_per_kelvin_pa(youngs_modulus_pa: float, alpha_1_k: float) -> float:
    """E alpha, how much stress one kelvin buys under full restraint."""
    return youngs_modulus_pa * alpha_1_k


def differential_strain(alpha_a_1_k: float, alpha_b_1_k: float,
                        delta_t_k: float) -> float:
    """(alpha_a - alpha_b) dT, the mismatch between two bonded materials.

    This is what makes a dissimilar-material joint a thermal problem even when
    neither part is externally restrained: each one restrains the other. An
    aluminium bracket bolted to a steel frame carries the difference between
    23.6e-6 and 11.7e-6, which is half of aluminium's own expansion.
    """
    return (alpha_a_1_k - alpha_b_1_k) * delta_t_k


@dataclass(frozen=True)
class ThermalStressResult:
    """A thermal stress, the mechanical one it adds to, and the verdict."""

    delta_t_k: float
    alpha_1_k: float
    constraint: float
    thermal_stress_pa: float
    mechanical_stress_pa: float
    combined_stress_pa: float
    yield_strength_pa: float
    safety_factor: float
    governing_contribution: str

    @property
    def passes(self) -> bool:
        return self.safety_factor >= 1.0


def check_thermal_stress(material: MaterialSpec, delta_t_k: float,
                         mechanical_stress_pa: float = 0.0,
                         constraint: float = 1.0,
                         alpha_1_k: float | None = None) -> ThermalStressResult:
    """Combine a restrained thermal stress with a mechanical one.

    Superposition, which linear elasticity permits: the two stresses are
    computed independently and added with their signs. A compressive thermal
    stress genuinely relieves a mechanical tension, and the model says so
    rather than adding magnitudes, because adding magnitudes would report a
    problem where the physics removes one.

    `alpha_1_k` overrides the material's coefficient, which is how a direction
    is chosen for an orthotropic material. For those the isotropic field holds
    the axis-1 value, and using it across the fibres would be wrong by a factor
    of fifty and the wrong sign.
    """
    coefficient = (alpha_1_k if alpha_1_k is not None
                   else material.thermal_expansion_1_k)
    if coefficient is None:
        raise ValueError(
            f"{material.id} has no thermal expansion coefficient, so its "
            f"thermal stress cannot be computed")

    thermal = constrained_thermal_stress_pa(
        material.youngs_modulus_pa, coefficient, delta_t_k, constraint)
    combined = thermal + mechanical_stress_pa
    magnitude = abs(combined)
    safety = (math.inf if magnitude <= 0.0
              else material.yield_strength_pa / magnitude)
    return ThermalStressResult(
        delta_t_k=delta_t_k, alpha_1_k=coefficient, constraint=constraint,
        thermal_stress_pa=thermal, mechanical_stress_pa=mechanical_stress_pa,
        combined_stress_pa=combined,
        yield_strength_pa=material.yield_strength_pa, safety_factor=safety,
        governing_contribution=("thermal"
                                if abs(thermal) > abs(mechanical_stress_pa)
                                else "mechanical"))
