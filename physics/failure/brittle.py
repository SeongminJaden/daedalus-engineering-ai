"""A ceramic does not yield, and a yield based safety factor is the wrong check.

The material database stores alumina with its flexural strength in the yield
field so the schema stays usable, and its note says plainly that a yield based
safety factor is the wrong failure criterion. This module is the right one, as
far as the data allows.

WHAT A BRITTLE MATERIAL FAILS BY
================================
The largest TENSILE principal stress, not the von Mises equivalent. Von Mises
is a distortion energy criterion for a material that yields by shear; a
ceramic in hydrostatic compression is fine and the same von Mises value in
tension breaks it. `max_principal_stress` returns the number the criterion
needs and `BrittleLimit` compares it.

AND BY HOW BIG IT IS
====================
Ceramic strength is scatter dominated: the part fails at its worst flaw, so a
bigger part is weaker. Weibull's two parameter form gives the size scaling,
sigma2 = sigma1 (V1/V2)^(1/m), and `size_scaled_strength_pa` applies it. The
modulus m is a material and process property between roughly 5 and 20 for
technical ceramics, and this repository has no measured value for the alumina
in its database, so the function REQUIRES m from the caller and the check
refuses to run without one. That refusal is the honest state of this database.

WHAT THIS IS NOT
================
A proof of survival. A Weibull scaling from a bend bar to a part assumes the
same flaw population and the same stress state, and the effective volume of a
part under a non uniform stress field is an integral this module does not
compute; `effective_volume_ratio` takes the ratio from the caller and says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BrittleDataMissing(ValueError):
    """A brittle check that needs a number nobody measured."""


def principal_stresses(voigt) -> np.ndarray:
    """The three principal stresses from a Voigt stress vector or an array of
    them, sorted with the largest tensile value first."""
    s = np.atleast_2d(np.asarray(voigt, dtype=float))
    if s.shape[1] != 6:
        raise ValueError(f"expected six Voigt components, got {s.shape[1]}")
    xx, yy, zz, xy, yz, zx = (s[:, i] for i in range(6))
    out = np.empty((s.shape[0], 3))
    for i in range(s.shape[0]):
        tensor = np.array([[xx[i], xy[i], zx[i]],
                           [xy[i], yy[i], yz[i]],
                           [zx[i], yz[i], zz[i]]])
        out[i] = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    return out


def max_principal_stress(voigt) -> np.ndarray:
    """The largest principal stress per point, which is what breaks a ceramic."""
    return principal_stresses(voigt)[:, 0]


@dataclass(frozen=True)
class WeibullStrength:
    """A characteristic strength at a stated volume, with its modulus."""

    strength_pa: float
    volume_m3: float
    modulus: float                 # the Weibull modulus m
    source: str

    def __post_init__(self) -> None:
        if self.modulus <= 0.0:
            raise ValueError("the Weibull modulus must be positive")
        if not self.source.strip():
            raise ValueError(
                "a Weibull strength without a source cannot be checked and "
                "must not be used to size a ceramic part")


def size_scaled_strength_pa(reference: WeibullStrength, volume_m3: float,
                            effective_volume_ratio: float = 1.0) -> float:
    """Strength of a part of another size, by the two parameter Weibull rule.

    `effective_volume_ratio` scales the part volume to the volume that is
    actually under tension, which for bending is a small fraction of the
    whole. This module does not compute it; a caller who has integrated the
    stress field passes it, and a caller who has not leaves it at one and
    gets a conservative answer for a bend and an optimistic one for a bar in
    uniform tension.
    """
    if volume_m3 <= 0.0 or effective_volume_ratio <= 0.0:
        raise ValueError("volumes must be positive")
    ratio = reference.volume_m3 / (volume_m3 * effective_volume_ratio)
    return float(reference.strength_pa * ratio ** (1.0 / reference.modulus))


def effective_volume_ratio(kind: str) -> float:
    """The classical effective volume fractions, for the two cases that have
    a closed form, and a refusal for everything else."""
    table = {"uniform_tension": 1.0}
    if kind not in table:
        raise BrittleDataMissing(
            f"the effective volume of a {kind!r} stress field is an integral "
            f"over the part; this module does not compute it and will not "
            f"guess. Pass a ratio you computed, or use uniform_tension")
    return table[kind]


@dataclass
class BrittleLimit:
    """A maximum principal stress check against a size scaled strength."""

    material_id: str
    reference: WeibullStrength | None
    part_volume_m3: float
    effective_volume_ratio: float = 1.0

    def allowable_pa(self) -> float:
        if self.reference is None:
            raise BrittleDataMissing(
                f"{self.material_id} has no Weibull strength in this "
                f"repository: no characteristic strength, no test volume and "
                f"no modulus were measured or sourced. A ceramic part cannot "
                f"be sized from a single handbook number, and this check "
                f"refuses rather than pretending the flexural strength is an "
                f"allowable")
        return size_scaled_strength_pa(self.reference, self.part_volume_m3,
                                       self.effective_volume_ratio)

    def check(self, voigt) -> dict:
        """Largest principal stress against the allowable, with the margin."""
        allowable = self.allowable_pa()
        peak = float(np.max(max_principal_stress(voigt)))
        return {"material_id": self.material_id,
                "max_principal_stress_pa": peak,
                "allowable_pa": allowable,
                "passes": bool(peak <= allowable),
                "margin": float(1.0 - peak / allowable) if allowable else float("nan"),
                "criterion": "maximum principal stress against a Weibull size "
                             "scaled strength",
                "note": "von Mises is not the criterion for a brittle material: "
                        "it is a distortion energy measure for a material that "
                        "yields by shear"}
