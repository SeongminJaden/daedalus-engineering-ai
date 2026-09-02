"""Turning stated form targets into a form score, off the evidence ladder.

DesignReference already carries FormTargets: a named target, a weight, a
source and a confidence, with the confidence capped by provenance. Until now
they were computed and recorded and consumed by nothing; DESIGN.md said so.
This module is the consumer. It maps a small vocabulary of targets onto the
measured ShapeMetrics and produces one number for `DesignEntry.form_score`,
which `MultiDesignReview` ranks admissible designs by and nothing else.

WHAT THE SCORE IS
=================
A weighted sum of per-target scores in [0, 1], each a stated function of one
geometric quantity:

    compact      the isoperimetric quotient itself, 36 pi V^2 / A^3, which is
                 1 for a sphere and pi/6 for a cube
    smooth       exp(-roughness / SMOOTH_SCALE_RAD), roughness being the mean
                 dihedral angle between neighbouring faces
    symmetric    exp(-asymmetry / L), asymmetry the mirror distance and L the
                 characteristic length the caller supplies
    slender      1 - compactness, the complement, for a caller who wants a
                 long thin form

The weight of each target is its stated weight times its confidence, exactly
as DesignReference.aesthetic_weights already computes, so a guess pulls less
than a cited convention. Targets outside the vocabulary contribute nothing
and are listed as unmapped, never silently dropped.

WHAT IT IS NOT
==============
Not evidence. The score has no rung on the ladder, never raises confidence in
a physical claim, and cannot lift a failing design over a passing one because
the review never ranks a failing design at all. It is not a theory of taste
either: the vocabulary above is four words, each attached to one number, and
the functions are conventions written here so they can be argued with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.design_reference import DesignReference, FormTarget

from .metrics import PREFERENCE_IS_NOT_EVIDENCE, ShapeMetrics

#: Dihedral roughness at which the smoothness score has fallen to 1/e.
#: Half a radian is a 29 degree mean fold between neighbouring facets, which
#: is a visibly faceted surface; a convention, stated so it can be changed.
SMOOTH_SCALE_RAD = 0.5

VOCABULARY = ("compact", "smooth", "symmetric", "slender")


@dataclass(frozen=True)
class FormScore:
    score: float
    contributions: dict[str, float]      # target name to weighted score
    weights: dict[str, float]            # target name to weight x confidence
    unmapped: tuple[str, ...]            # targets the vocabulary does not know

    @property
    def note(self) -> str:
        return PREFERENCE_IS_NOT_EVIDENCE


def target_score(target: str, metrics: ShapeMetrics,
                 characteristic_length_m: float) -> float | None:
    """The [0, 1] score for one target, or None when the word is unknown."""
    word = target.strip().lower()
    if word == "compact":
        return max(0.0, min(1.0, metrics.compactness))
    if word == "slender":
        return max(0.0, min(1.0, 1.0 - metrics.compactness))
    if word == "smooth":
        return math.exp(-metrics.dihedral_roughness_rad / SMOOTH_SCALE_RAD)
    if word == "symmetric":
        if characteristic_length_m <= 0.0:
            raise ValueError("symmetry needs a positive characteristic length")
        return math.exp(-metrics.mirror_asymmetry_m / characteristic_length_m)
    return None


def form_score(reference: DesignReference, metrics: ShapeMetrics,
               characteristic_length_m: float) -> FormScore:
    """Score one measured shape against a reference's form targets.

    Weights are the reference's own aesthetic weights (weight x confidence),
    so the result is in [0, sum of weights]. A reference with no form targets
    scores zero and says so through empty contributions.
    """
    weights = reference.aesthetic_weights()
    contributions: dict[str, float] = {}
    unmapped: list[str] = []
    for item in reference.of_type(FormTarget):
        value = target_score(item.target, metrics, characteristic_length_m)  # type: ignore[attr-defined]
        if value is None:
            unmapped.append(item.target)  # type: ignore[attr-defined]
            continue
        contributions[item.name] = weights[item.target] * value  # type: ignore[attr-defined]
    return FormScore(score=sum(contributions.values()),
                     contributions=contributions,
                     weights={k: v for k, v in weights.items()},
                     unmapped=tuple(unmapped))
