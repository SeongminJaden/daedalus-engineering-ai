"""Ply failure criteria and first-ply failure of a laminate.

VALIDITY, before the implementation:

* **The Tsai-Wu failure INDEX is not a safety factor.** The criterion contains
  linear terms in stress as well as quadratic ones, so an index of 0.5 does not
  mean the load can be doubled. The quantity that means that is the STRENGTH
  RATIO, the multiplier on a proportionally scaled load that brings the
  laminate to failure, and it is what `strength_ratio` returns. Reading the
  index as a reciprocal safety factor is a standard error and it is
  unconservative for the load cases where the linear terms matter, which is
  exactly the ones where the material is asymmetric in tension and compression.

* **Tsai-Wu does not identify the failure MODE.** It is one smooth surface
  fitted through the strength data, so it says whether a ply has failed and not
  whether the fibres broke or the matrix cracked. Those have completely
  different consequences: a cracked matrix ply keeps carrying load along its
  fibres, a broken fibre ply does not. Mode-based criteria such as Hashin or
  Puck make that distinction and are not implemented here, which is also why
  progressive damage cannot be built on this.

* **F12, the interaction term, is ASSUMED.** It cannot be obtained from
  uniaxial tests at all; it needs a biaxial one that is rarely done. The value
  used here is the common default -0.5 sqrt(F11 F22). Different defaults are
  in use and they move the predicted strength under biaxial loading.

* **Maximum stress is non-interactive by construction.** It compares each
  stress component against its own allowable independently, so it cannot see
  that a combined state is worse than either component alone. It is included
  because it says WHICH component governs, which Tsai-Wu cannot.

* **First-ply failure is not ultimate failure.** See the module note in `clt`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.materials.db import MaterialSpec

from .clt import Laminate, PlyState, ply_states


class FailureMode(str, Enum):
    """Which stress component reached its allowable, for the max-stress check."""

    FIBRE_TENSION = "fibre_tension"
    FIBRE_COMPRESSION = "fibre_compression"
    MATRIX_TENSION = "matrix_tension"
    MATRIX_COMPRESSION = "matrix_compression"
    SHEAR = "shear"


@dataclass(frozen=True)
class LaminaStrength:
    """The five strengths a ply criterion needs. All positive magnitudes.

    Compressive strengths are separate values and NOT the tensile ones. A
    unidirectional composite is weaker in compression along its fibres and
    several times stronger across them, so assuming symmetry is wrong in both
    directions and unconservative transversely.
    """

    longitudinal_tension_pa: float       # Xt
    longitudinal_compression_pa: float   # Xc
    transverse_tension_pa: float         # Yt
    transverse_compression_pa: float     # Yc
    shear_pa: float                      # S

    def __post_init__(self) -> None:
        values = (self.longitudinal_tension_pa, self.longitudinal_compression_pa,
                  self.transverse_tension_pa, self.transverse_compression_pa,
                  self.shear_pa)
        if min(values) <= 0.0:
            raise ValueError("every strength is a positive magnitude")

    @classmethod
    def from_material(cls, material: MaterialSpec) -> "LaminaStrength":
        """Read the five strengths, refusing when the compressive ones are absent.

        Refusing rather than assuming symmetry: transverse compressive strength
        is about four times the tensile value for carbon epoxy, so defaulting
        one to the other would overstate transverse tensile capacity fourfold.
        """
        required = {
            "strength_long_pa": "longitudinal tensile",
            "strength_long_compressive_pa": "longitudinal compressive",
            "strength_trans_pa": "transverse tensile",
            "strength_trans_compressive_pa": "transverse compressive",
            "strength_shear_pa": "in-plane shear",
        }
        missing = [label for field, label in required.items()
                   if getattr(material, field) is None]
        if missing:
            raise ValueError(
                f"{material.id} has no {', '.join(missing)} strength. A ply "
                f"criterion needs all five and assuming any of them from the "
                f"others is wrong by several times for a composite")
        return cls(
            longitudinal_tension_pa=material.strength_long_pa,
            longitudinal_compression_pa=material.strength_long_compressive_pa,
            transverse_tension_pa=material.strength_trans_pa,
            transverse_compression_pa=material.strength_trans_compressive_pa,
            shear_pa=material.strength_shear_pa)


def tsai_wu_coefficients(strength: LaminaStrength,
                         f12_factor: float = -0.5) -> dict[str, float]:
    """The Tsai-Wu tensor coefficients.

    `f12_factor` multiplies sqrt(F11 F22) to give F12. The default of -0.5 is
    the usual choice and is [ASSUMED]: F12 is not obtainable from uniaxial
    testing.
    """
    xt = strength.longitudinal_tension_pa
    xc = strength.longitudinal_compression_pa
    yt = strength.transverse_tension_pa
    yc = strength.transverse_compression_pa
    s = strength.shear_pa
    f11 = 1.0 / (xt * xc)
    f22 = 1.0 / (yt * yc)
    return {
        "F1": 1.0 / xt - 1.0 / xc,
        "F2": 1.0 / yt - 1.0 / yc,
        "F11": f11,
        "F22": f22,
        "F66": 1.0 / (s * s),
        "F12": f12_factor * math.sqrt(f11 * f22),
    }


def tsai_wu_index(stress_material: np.ndarray, strength: LaminaStrength,
                  f12_factor: float = -0.5) -> float:
    """The Tsai-Wu failure index. Failure at 1.0.

    NOT a reciprocal safety factor: see the module note. Use `strength_ratio`
    for a load multiplier.
    """
    s1, s2, s6 = np.asarray(stress_material, dtype=float).reshape(3)
    c = tsai_wu_coefficients(strength, f12_factor)
    return float(c["F1"] * s1 + c["F2"] * s2
                 + c["F11"] * s1 ** 2 + c["F22"] * s2 ** 2
                 + c["F66"] * s6 ** 2 + 2.0 * c["F12"] * s1 * s2)


def tsai_wu_strength_ratio(stress_material: np.ndarray,
                           strength: LaminaStrength,
                           f12_factor: float = -0.5) -> float:
    """The factor R by which a proportional load can be scaled before failure.

    Substituting R times the stress into the criterion and setting it equal to
    one gives a R^2 + b R - 1 = 0 with

        a = F11 s1^2 + F22 s2^2 + F66 s6^2 + 2 F12 s1 s2
        b = F1 s1 + F2 s2

    and the positive root is the answer. This IS a safety factor for
    proportional loading, which the index is not.
    """
    s1, s2, s6 = np.asarray(stress_material, dtype=float).reshape(3)
    c = tsai_wu_coefficients(strength, f12_factor)
    a = (c["F11"] * s1 ** 2 + c["F22"] * s2 ** 2 + c["F66"] * s6 ** 2
         + 2.0 * c["F12"] * s1 * s2)
    b = c["F1"] * s1 + c["F2"] * s2
    if a <= 0.0:
        if b <= 0.0:
            return math.inf         # no stress, or purely relieving
        return 1.0 / b
    return float((-b + math.sqrt(b * b + 4.0 * a)) / (2.0 * a))


def max_stress_ratio(stress_material: np.ndarray, strength: LaminaStrength
                     ) -> tuple[float, FailureMode]:
    """Strength ratio and governing component by the maximum stress criterion.

    Non-interactive: each component is compared against its own allowable. That
    is its weakness and also the reason to run it, since it names the component
    that governs where Tsai-Wu returns only a number.
    """
    s1, s2, s6 = np.asarray(stress_material, dtype=float).reshape(3)
    candidates: list[tuple[float, FailureMode]] = []
    if s1 > 0:
        candidates.append((strength.longitudinal_tension_pa / s1,
                           FailureMode.FIBRE_TENSION))
    elif s1 < 0:
        candidates.append((strength.longitudinal_compression_pa / -s1,
                           FailureMode.FIBRE_COMPRESSION))
    if s2 > 0:
        candidates.append((strength.transverse_tension_pa / s2,
                           FailureMode.MATRIX_TENSION))
    elif s2 < 0:
        candidates.append((strength.transverse_compression_pa / -s2,
                           FailureMode.MATRIX_COMPRESSION))
    if s6 != 0:
        candidates.append((strength.shear_pa / abs(s6), FailureMode.SHEAR))
    if not candidates:
        return math.inf, FailureMode.FIBRE_TENSION
    return min(candidates, key=lambda pair: pair[0])


@dataclass(frozen=True)
class FirstPlyFailure:
    """Which ply fails first, at what load multiplier, and how."""

    ply_index: int
    angle_deg: float
    strength_ratio: float
    tsai_wu_index: float
    max_stress_ratio: float
    governing_mode: FailureMode
    ply_stresses_pa: np.ndarray
    states: list[PlyState]

    @property
    def passes(self) -> bool:
        return self.strength_ratio >= 1.0


def first_ply_failure(laminate: Laminate, strength: LaminaStrength,
                      force_resultant: np.ndarray,
                      moment_resultant: np.ndarray | None = None,
                      f12_factor: float = -0.5) -> FirstPlyFailure:
    """The ply with the lowest Tsai-Wu strength ratio, and its mode.

    VALIDITY: this is the load at which the FIRST ply fails. A real laminate
    generally carries more after that, so as an ultimate strength it is
    conservative. The exception is a fibre failure in the primary load
    direction, where first-ply really is the end, and the reported mode is what
    tells the two apart.
    """
    states = ply_states(laminate, force_resultant, moment_resultant)
    ratios = [tsai_wu_strength_ratio(state.stress_material, strength,
                                     f12_factor) for state in states]
    worst = int(np.argmin(ratios))
    state = states[worst]
    _, mode = max_stress_ratio(state.stress_material, strength)
    return FirstPlyFailure(
        ply_index=worst, angle_deg=state.angle_deg,
        strength_ratio=ratios[worst],
        tsai_wu_index=tsai_wu_index(state.stress_material, strength,
                                    f12_factor),
        max_stress_ratio=max_stress_ratio(state.stress_material, strength)[0],
        governing_mode=mode, ply_stresses_pa=state.stress_material,
        states=states)
