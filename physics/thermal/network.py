"""Thermal resistance networks and lumped transient response.

Phase 18 used a single lumped resistance from winding to ambient and said it
could be out by a factor of two because the mounting dominates it. A network
lets that be built from its parts, so the mounting appears as its own
resistance and its influence becomes visible rather than buried.

VALIDITY, before the implementation:

* **A series-parallel network assumes ONE-DIMENSIONAL flow through each
  branch.** It cannot represent lateral spreading, and spreading is exactly
  what makes a small heat source on a large heatsink work. A network built as
  pure series therefore UNDER-estimates the spreading and over-estimates the
  temperature, which is the safe direction but can be very conservative.

* **Lumped transient analysis requires the Biot number below about 0.1**, and
  this is a hard condition rather than a guideline. Bi = h Lc / k compares the
  resistance INSIDE the body to the resistance leaving it. Above 0.1 the body
  has real internal gradients, a single temperature does not describe it, and
  the exponential response is wrong in a way that gets worse as Bi grows. The
  check is performed and reported rather than assumed, because a lumped model
  applied to a thick or poorly conducting body looks perfectly reasonable and
  is not.

* **The time constant is only as good as the resistance in it**, and the
  convection resistance carries the usual factor-of-two uncertainty. A
  predicted thermal time constant should be read as an order of magnitude.

* **Contact resistance between parts is NOT included.** A bolted interface has
  a real thermal resistance that depends on pressure, flatness and whether
  there is grease, and it is often comparable to the conduction resistance of
  the parts it joins. Omitting it under-predicts the temperature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Resistance:
    """One named thermal resistance in kelvin per watt."""

    name: str
    value_k_w: float

    def __post_init__(self) -> None:
        if self.value_k_w <= 0.0:
            raise ValueError(
                f"{self.name}: a thermal resistance must be positive")


def series_resistance_k_w(resistances: "list[Resistance]") -> float:
    """Resistances in series add."""
    if not resistances:
        raise ValueError("a series path needs at least one resistance")
    return sum(r.value_k_w for r in resistances)


def parallel_resistance_k_w(resistances: "list[Resistance]") -> float:
    """Resistances in parallel add as reciprocals.

    Two equal paths halve the resistance, which is the whole reason a second
    heat path helps. The result is always SMALLER than the smallest branch, and
    that is a useful sanity check on any network.
    """
    if not resistances:
        raise ValueError("a parallel path needs at least one resistance")
    return 1.0 / sum(1.0 / r.value_k_w for r in resistances)


@dataclass
class ThermalPath:
    """A named series chain from a source to an ambient."""

    resistances: list[Resistance] = field(default_factory=list)

    def add(self, name: str, value_k_w: float) -> "ThermalPath":
        self.resistances.append(Resistance(name, value_k_w))
        return self

    @property
    def total_k_w(self) -> float:
        return series_resistance_k_w(self.resistances)

    def temperature_rise_k(self, power_w: float) -> float:
        return power_w * self.total_k_w

    def dominant(self) -> Resistance:
        """The largest resistance: the one worth attacking.

        A network's value is mostly in identifying this. Improving anything
        else while the dominant term stands buys almost nothing.
        """
        return max(self.resistances, key=lambda r: r.value_k_w)

    def shares(self) -> "dict[str, float]":
        total = self.total_k_w
        return {r.name: r.value_k_w / total for r in self.resistances}


@dataclass(frozen=True)
class TransientResponse:
    """A lumped first-order thermal response, with its validity verdict."""

    biot_number: float
    time_constant_s: float
    capacitance_j_k: float
    resistance_k_w: float
    lumped_valid: bool

    def temperature_c(self, initial_c: float, ambient_c: float,
                      power_w: float, time_s: float) -> float:
        """First-order approach to the steady state.

            T(t) = T_ss + (T0 - T_ss) exp(-t / tau),  T_ss = Tinf + P R
        """
        if time_s < 0.0:
            raise ValueError("time cannot be negative")
        steady = ambient_c + power_w * self.resistance_k_w
        return steady + (initial_c - steady) * math.exp(
            -time_s / self.time_constant_s)


BIOT_LUMPED_LIMIT = 0.1


def biot_number(coefficient_w_m2k: float, characteristic_length_m: float,
                conductivity_w_mk: float) -> float:
    """Bi = h Lc / k, with Lc the volume over the surface area.

    It compares the resistance to getting heat OUT of the surface against the
    resistance to moving it INSIDE the body. Small means the inside is
    effectively isothermal, which is what a lumped model assumes.
    """
    if min(coefficient_w_m2k, characteristic_length_m,
           conductivity_w_mk) <= 0.0:
        raise ValueError("all three quantities must be positive")
    return coefficient_w_m2k * characteristic_length_m / conductivity_w_mk


def lumped_response(mass_kg: float, specific_heat_j_kgk: float,
                    resistance_k_w: float, coefficient_w_m2k: float,
                    characteristic_length_m: float,
                    conductivity_w_mk: float) -> TransientResponse:
    """Time constant and the Biot validity check together.

        tau = R C = R m c

    The Biot number is returned alongside, and `lumped_valid` is false when it
    exceeds the limit. It is returned rather than raised because the number is
    still informative when the model does not apply, and a caller that ignores
    the flag has been told.
    """
    if mass_kg <= 0.0 or specific_heat_j_kgk <= 0.0:
        raise ValueError("mass and specific heat must be positive")
    if resistance_k_w <= 0.0:
        raise ValueError("thermal resistance must be positive")
    capacitance = mass_kg * specific_heat_j_kgk
    bi = biot_number(coefficient_w_m2k, characteristic_length_m,
                     conductivity_w_mk)
    return TransientResponse(
        biot_number=bi, time_constant_s=resistance_k_w * capacitance,
        capacitance_j_k=capacitance, resistance_k_w=resistance_k_w,
        lumped_valid=bi < BIOT_LUMPED_LIMIT)
