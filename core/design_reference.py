"""Design references: outside priors that may bias a search, never gate it.

An external language model can supply useful engineering priors. Typical
proportions for a class of part, the wall thickness a foundry or a machinist
would expect, the fillet radius convention that avoids a stress raiser, the
silhouette a family of products shares. None of that is a measurement, and
some of it is a plausible sentence with nothing behind it.

So this module exists to let such material in WITHOUT letting it decide
anything. The rule is a single sentence and everything else follows from it:

    A reference may move where the search STARTS and what it finds
    beautiful. It may not move what counts as safe.

Concretely, a DesignReference can bias the starting point of an optimisation
and weight an aesthetic objective. It has no path at all to a bound, an
allowable stress, a deflection limit, a safety factor, or any verification
gate. There is deliberately no function here that takes an OptimizationProblem
and returns a modified one: the absence is the mechanism. A prior that could
relax an allowable stress would let a sentence a model generated overrule the
physics, which is the one outcome this whole project is arranged to prevent.

PROVENANCE IS MANDATORY
=======================
Every item carries a source, a confidence and a provenance, and construction
fails without them. An unattributed prior is indistinguishable from an
invented one after it has been written down once, and the entire risk here is
that a fluent guess acquires the authority of a datum by being stored.

Confidence is capped against the project's existing evidence ladder rather
than on a scale invented here:

    ASSUMED   no checkable source, so capped at the UNVERIFIED ceiling, 0.20
    CITED     a source is named but nothing in this project has checked it,
              so capped at the SIMULATED ceiling, 0.60
    MEASURED  derived from something this project computed, same 0.60 cap

Nothing reaches REPEATED or above. Those levels are earned by independent
runs agreeing, and reading a reference is not a run.

VALIDITY DOMAIN
===============
Stated before implementing, per the standing discipline.

    A reference describes a FAMILY of parts, not this part. A proportion that
    is typical of a class says nothing about whether it is right for a
    specific load case, and the optimiser is what decides that. Biasing the
    start can only change which local optimum is found first; on a convex
    problem it changes nothing at all except the iteration count.

    Aesthetic targets are subjective and this module does not pretend
    otherwise. It records what was asked for and who asked. It does not
    contain a theory of what looks good.

    References are inspiration, not reproduction. Storing a proportion range
    observed across a product class is not the same as copying a design, and
    nothing here retains or reproduces a source artefact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from brain.semantic.evidence import LEVEL_CONFIDENCE_CEILING, EvidenceLevel


class Provenance(str, Enum):
    """Where an item came from, which is what caps its confidence."""

    ASSUMED = "assumed"
    CITED = "cited"
    MEASURED = "measured"


#: No reference item can be trusted beyond the simulated ceiling, because
#: none of this has been measured on a physical part.
PROVENANCE_CEILING = {
    Provenance.ASSUMED: LEVEL_CONFIDENCE_CEILING[EvidenceLevel.UNVERIFIED],
    Provenance.CITED: LEVEL_CONFIDENCE_CEILING[EvidenceLevel.SIMULATED],
    Provenance.MEASURED: LEVEL_CONFIDENCE_CEILING[EvidenceLevel.SIMULATED],
}


@dataclass(frozen=True)
class ReferenceItem:
    """One prior, inseparable from where it came from.

    The source and confidence are not optional metadata to be filled in
    later. A prior that has lost its provenance reads exactly like a measured
    fact, and there is no way to tell them apart afterwards.
    """

    name: str
    source: str
    confidence: float
    provenance: Provenance

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a reference item needs a name")
        if not self.source.strip():
            raise ValueError(
                f"reference item {self.name!r} has no source; an unattributed "
                f"prior cannot be told apart from an invented one once stored")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(
                f"reference item {self.name!r} has confidence "
                f"{self.confidence}, which must lie in (0, 1]")
        ceiling = PROVENANCE_CEILING[self.provenance]
        if self.confidence > ceiling:
            raise ValueError(
                f"reference item {self.name!r} claims confidence "
                f"{self.confidence} but its provenance is "
                f"{self.provenance.value}, which caps at {ceiling}")

    @property
    def is_assumed(self) -> bool:
        return self.provenance is Provenance.ASSUMED

    def label(self) -> str:
        """How the item should appear anywhere a human reads it."""
        tag = "[ASSUMED] " if self.is_assumed else ""
        return (f"{tag}{self.name} (confidence {self.confidence:.2f}, "
                f"source: {self.source})")


@dataclass(frozen=True)
class RangePrior(ReferenceItem):
    """A prior on a scalar, expressed as the range a class of parts occupies.

    A range rather than a value, because a reference describes a family. The
    midpoint is what biases a starting point; the width is what says how
    little the reference actually pins down.
    """

    minimum: float = 0.0
    maximum: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.minimum < self.maximum:
            raise ValueError(
                f"prior {self.name!r} has an empty range "
                f"[{self.minimum}, {self.maximum}]")

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.minimum + self.maximum)


@dataclass(frozen=True)
class FormTarget(ReferenceItem):
    """A shaping goal: a symmetry, a silhouette, a continuity class.

    Deliberately a free-text target with a weight rather than a structured
    geometric constraint. This module does not contain a theory of what looks
    good, and encoding one here would give a subjective preference the
    appearance of a specification.
    """

    target: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.target.strip():
            raise ValueError(f"form target {self.name!r} states no target")
        if self.weight < 0.0:
            raise ValueError(
                f"form target {self.name!r} has a negative weight")


@dataclass(frozen=True)
class LoadPathHint(ReferenceItem):
    """A suggestion about where material should go, for seeding topology.

    A hint seeds an initial density field. It cannot constrain the result: the
    stress constraint and the physics decide what survives, and a hint that
    disagrees with them loses.
    """

    description: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.description.strip():
            raise ValueError(f"load path hint {self.name!r} describes nothing")


@dataclass(frozen=True)
class DesignReference:
    """A bundle of priors, and the only thing that ever consumes them.

    It exposes exactly two effects, both optional and both harmless: a
    starting point for the search, and weights for an aesthetic objective.
    There is no third method, and no method that touches an
    OptimizationProblem. The physics is unreachable from here BY CONSTRUCTION
    rather than by a check that could be forgotten.
    """

    name: str
    items: tuple[ReferenceItem, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a design reference needs a name")
        seen: set[str] = set()
        for item in self.items:
            if item.name in seen:
                raise ValueError(
                    f"design reference {self.name!r} carries two items called "
                    f"{item.name!r}; a later one would silently win")
            seen.add(item.name)

    def of_type(self, kind: type) -> tuple[ReferenceItem, ...]:
        return tuple(item for item in self.items if isinstance(item, kind))

    def prior(self, name: str) -> RangePrior | None:
        for item in self.of_type(RangePrior):
            if item.name == name:
                return item                                    # type: ignore
        return None

    @property
    def assumed_items(self) -> tuple[ReferenceItem, ...]:
        return tuple(item for item in self.items if item.is_assumed)

    def weakest_confidence(self) -> float:
        """The confidence of the least supported item, or 0 when empty.

        Reported as the weakest rather than the average, because a bundle is
        only as trustworthy as the item someone acts on, and averaging lets
        one solid citation launder several guesses.
        """
        return min((item.confidence for item in self.items), default=0.0)

    def provenance_report(self) -> tuple[str, ...]:
        """Every item, labelled, for a log or a console. Assumed ones flagged."""
        return tuple(item.label() for item in self.items)

    # --- the only two effects ------------------------------------------- #

    def starting_point(self, problem, fallback: np.ndarray) -> np.ndarray:
        """Bias where the search starts, clipped into the existing bounds.

        `problem` supplies the bounds and is READ ONLY here. Anything the
        reference asks for outside them is clipped away rather than widening
        them, so a prior can never reach a design the bounds excluded. The
        priors are read by variable name: outer_width_m, outer_height_m and
        wall_thickness_m.
        """
        start = np.asarray(fallback, dtype=float).copy()
        for index, variable in enumerate(("outer_width_m", "outer_height_m",
                                          "wall_thickness_m")):
            prior = self.prior(variable)
            if prior is not None:
                start[index] = prior.midpoint
        start = problem.clip_to_bounds(start)
        if not problem.is_geometrically_valid(start):
            # A reference can suggest an impossible section. Fall back rather
            # than hand the optimiser a start it cannot evaluate.
            return np.asarray(fallback, dtype=float)
        return start

    def aesthetic_weights(self) -> dict[str, float]:
        """Form targets and their weights, for an aesthetic objective.

        Weights are scaled by confidence, so a guess pulls less hard than a
        cited convention. Nothing here enters a constraint.
        """
        return {item.target: item.weight * item.confidence      # type: ignore
                for item in self.of_type(FormTarget)}


    # --- what the Brain stores ------------------------------------------ #

    def as_dict(self) -> dict:
        """The form the episodic store keeps, in `record_run(meta=...)`.

        No schema change is needed: the run's meta field already exists. What
        matters is WHAT is stored. Every item keeps its source, its confidence
        and its provenance, so a later reader can ask which priors were in
        play when a design turned out well WITHOUT being able to mistake a
        guess for a finding. The evidence ceiling is stored alongside, because
        a number read back out of a database has lost the context that capped
        it.
        """
        return {
            "reference": self.name,
            "evidence_ceiling": LEVEL_CONFIDENCE_CEILING[
                EvidenceLevel.SIMULATED],
            "physically_validated": False,
            "weakest_confidence": self.weakest_confidence(),
            "assumed_item_count": len(self.assumed_items),
            "items": [
                {"name": item.name, "source": item.source,
                 "confidence": item.confidence,
                 "provenance": item.provenance.value,
                 "assumed": item.is_assumed}
                for item in self.items
            ],
        }


def describe_influence() -> tuple[str, ...]:
    """Exactly what a reference can and cannot reach, for a report or a doc.

    Written as data rather than prose in a docstring so a test can assert the
    list has not quietly grown a third entry.
    """
    return (
        "may bias the starting point of a search, within the existing bounds",
        "may weight an aesthetic objective, scaled by its own confidence",
    )


def forbidden_influence() -> tuple[str, ...]:
    """What a reference must never reach. Asserted by the tests."""
    return (
        "design variable bounds",
        "allowable stress",
        "maximum deflection",
        "safety factor",
        "any verification gate or evidence level",
    )
