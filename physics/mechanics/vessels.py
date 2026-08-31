"""Pressure vessels, thin walled and thick walled.

VALIDITY, before the implementation:

* **The thin-wall formula UNDER-predicts the hoop stress, and WHICH RADIUS you
  use changes that by an order of magnitude.** It assumes uniform stress
  through the wall; the real distribution peaks at the bore. Measured against
  the Lame solution:

      r/t     with inner radius     with MEAN radius
        2                23.08%                3.85%
        5                 9.84%                0.82%
       10                 4.98%                0.23%
       20                 2.50%                0.06%

  The familiar guidance of "r/t above 10 for a 5% error" is about the
  INNER-radius form, and reading it as a statement about thin-wall theory in
  general is wrong: the mean-radius form used here is inside 4% even at r/t of
  2. The rule was quoted here before it was measured and the numbers did not
  support it as stated, which is why the table is in the docstring rather than
  the rule.

* **Both are for internal pressure with closed ends.** External pressure is a
  different problem: a vessel under external pressure fails by BUCKLING at a
  pressure far below its material strength, and nothing here checks that.

* **Elastic, axisymmetric, and away from discontinuities.** Nozzles, heads,
  flanges and supports all raise the local stress well above these values,
  which is why real vessel codes are mostly about those details rather than
  about the cylinder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VesselStress:
    hoop_pa: float
    longitudinal_pa: float
    radial_pa: float
    model: str


def thin_wall(pressure_pa: float, mean_radius_m: float,
              thickness_m: float) -> VesselStress:
    """sigma_hoop = p r / t, sigma_long = p r / 2t.

    The hoop stress is twice the longitudinal, which is why a cylindrical
    vessel splits along its length rather than around its circumference.
    """
    if mean_radius_m <= 0.0 or thickness_m <= 0.0:
        raise ValueError("radius and thickness must be positive")
    hoop = pressure_pa * mean_radius_m / thickness_m
    return VesselStress(hoop_pa=hoop, longitudinal_pa=0.5 * hoop,
                        radial_pa=-0.5 * pressure_pa, model="thin_wall")


def thick_wall(pressure_pa: float, inner_radius_m: float,
               outer_radius_m: float,
               at_radius_m: float | None = None) -> VesselStress:
    """Lame stresses for internal pressure, evaluated at a radius.

        sigma_hoop = p ri^2 (ro^2 + r^2) / (r^2 (ro^2 - ri^2))
        sigma_r    = p ri^2 (1 - ro^2/r^2) / (ro^2 - ri^2)

    Defaults to the bore, where the hoop stress is largest. The radial stress
    there equals minus the pressure, which is a useful check on the algebra.
    """
    ri, ro = inner_radius_m, outer_radius_m
    if ri <= 0.0 or ro <= ri:
        raise ValueError("the outer radius must exceed the inner one")
    r = ri if at_radius_m is None else at_radius_m
    if not ri - 1e-15 <= r <= ro + 1e-15:
        raise ValueError("the evaluation radius must lie inside the wall")

    span = ro ** 2 - ri ** 2
    hoop = pressure_pa * ri ** 2 * (ro ** 2 + r ** 2) / (r ** 2 * span)
    radial = pressure_pa * ri ** 2 * (1.0 - ro ** 2 / r ** 2) / span
    # Closed ends carry the end load as a uniform longitudinal stress.
    longitudinal = pressure_pa * ri ** 2 / span
    return VesselStress(hoop_pa=hoop, longitudinal_pa=longitudinal,
                        radial_pa=radial, model="thick_wall")


def thin_wall_error(inner_radius_m: float, thickness_m: float) -> float:
    """How much the thin-wall hoop stress under-predicts the Lame value.

    Returned as a positive fraction, for the MEAN-radius form this module
    uses. Computed rather than quoted, so a caller sees the error at their own
    geometry instead of trusting a remembered rule that turns out to be about
    a different formulation.
    """
    outer = inner_radius_m + thickness_m
    mean = inner_radius_m + 0.5 * thickness_m
    exact = thick_wall(1.0, inner_radius_m, outer).hoop_pa
    approximate = thin_wall(1.0, mean, thickness_m).hoop_pa
    return (exact - approximate) / exact
