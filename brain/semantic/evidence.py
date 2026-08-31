"""brain.semantic.evidence - evidence levels and the confidence function.

This is the safety-critical part of the Brain. Everything the system "knows"
came out of a simulation whose fidelity is beam theory (see physics/README.md),
so the store must never let simulated agreement masquerade as physical fact.

The one inviolable rule: **EXPERIMENTALLY_VALIDATED is reachable only with
physical-test evidence.** No amount of simulation, repetition or agreement can
promote an item to it. That is what makes this a store of *evidence-graded
experience* rather than a store of facts.

Independence is counted by distinct **run**, not by episode. Five episodes
inside one optimizer run are five samples of one search, not five independent
observations - counting them as independent is exactly how a Brain talks
itself into false confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvidenceLevel(str, Enum):
    """How far a statement has earned trust. Ordered."""

    UNVERIFIED = "unverified"
    SIMULATED = "simulated"
    REPEATED = "repeated"
    HIGH_CONFIDENCE = "high_confidence"
    EXPERIMENTALLY_VALIDATED = "experimentally_validated"

    @property
    def rank(self) -> int:
        return LEVEL_ORDER.index(self)

    def __lt__(self, other):  # type: ignore[override]
        if not isinstance(other, EvidenceLevel):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other):  # type: ignore[override]
        if not isinstance(other, EvidenceLevel):
            return NotImplemented
        return self.rank <= other.rank


LEVEL_ORDER = [
    EvidenceLevel.UNVERIFIED,
    EvidenceLevel.SIMULATED,
    EvidenceLevel.REPEATED,
    EvidenceLevel.HIGH_CONFIDENCE,
    EvidenceLevel.EXPERIMENTALLY_VALIDATED,
]

# Confidence can never exceed the ceiling of the level that has been earned.
LEVEL_CONFIDENCE_CEILING = {
    EvidenceLevel.UNVERIFIED: 0.20,
    EvidenceLevel.SIMULATED: 0.60,
    EvidenceLevel.REPEATED: 0.80,
    EvidenceLevel.HIGH_CONFIDENCE: 0.95,
    EvidenceLevel.EXPERIMENTALLY_VALIDATED: 0.99,
}


class EvidenceKind(str, Enum):
    SIMULATION = "simulation"        # a solver run - beam-theory fidelity
    TEST_SUITE = "test_suite"        # a passing verification test
    ANALYTICAL = "analytical"        # closed-form derivation
    PHYSICAL_TEST = "physical_test"  # a real part, measured. The only gate to
                                     # EXPERIMENTALLY_VALIDATED.


@dataclass(frozen=True)
class Evidence:
    """One piece of support for a statement."""

    kind: EvidenceKind
    ref: str                     # episode id, test name, report id
    run_id: str | None = None    # independence is counted by this
    note: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind.value, "ref": self.ref,
                "run_id": self.run_id, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        return cls(kind=EvidenceKind(d["kind"]), ref=d["ref"],
                   run_id=d.get("run_id"), note=d.get("note", ""))


@dataclass(frozen=True)
class Counterexample:
    """An observation that contradicts the statement."""

    ref: str
    description: str
    resolved: bool = False       # resolved = explained away, no longer counts

    def as_dict(self) -> dict:
        return {"ref": self.ref, "description": self.description,
                "resolved": self.resolved}

    @classmethod
    def from_dict(cls, d: dict) -> "Counterexample":
        return cls(ref=d["ref"], description=d["description"],
                   resolved=bool(d.get("resolved", False)))


# --- transition thresholds (parameterized, not hard-coded into the logic) --- #
@dataclass(frozen=True)
class PromotionPolicy:
    """Thresholds for the evidence-level state machine."""

    repeat_independent_runs: int = 3   # distinct runs needed for REPEATED
    high_confidence_evidence: int = 8  # total supporting evidence items
    high_confidence_runs: int = 5      # distinct runs for HIGH_CONFIDENCE
    saturation: float = 4.0            # k in n/(n+k); higher = slower growth


DEFAULT_POLICY = PromotionPolicy()


def independent_runs(evidence: list[Evidence]) -> int:
    """Distinct runs represented. Evidence with no run_id counts once each,
    since a test or a derivation is its own independent source."""
    runs = {e.run_id for e in evidence if e.run_id is not None}
    standalone = sum(1 for e in evidence if e.run_id is None)
    return len(runs) + standalone


def unresolved(counterexamples: list[Counterexample]) -> int:
    return sum(1 for c in counterexamples if not c.resolved)


def derive_level(
    evidence: list[Evidence],
    counterexamples: list[Counterexample],
    policy: PromotionPolicy = DEFAULT_POLICY,
) -> EvidenceLevel:
    """The state machine. Pure function of the evidence on hand.

    Rules, in order:
      * no evidence                                   -> UNVERIFIED
      * physical-test evidence (and nothing unresolved
        contradicting it)                             -> EXPERIMENTALLY_VALIDATED
      * enough evidence from enough independent runs,
        with zero unresolved counterexamples          -> HIGH_CONFIDENCE
      * consistent across >= repeat_independent_runs  -> REPEATED
      * anything else with support                    -> SIMULATED

    An unresolved counterexample caps the level at REPEATED: a statement with a
    standing contradiction is not high-confidence, whatever else supports it.
    """
    if not evidence:
        return EvidenceLevel.UNVERIFIED

    open_counters = unresolved(counterexamples)
    has_physical = any(e.kind is EvidenceKind.PHYSICAL_TEST for e in evidence)
    runs = independent_runs(evidence)

    # The one gate that simulation can never open.
    if has_physical and open_counters == 0:
        return EvidenceLevel.EXPERIMENTALLY_VALIDATED

    if open_counters == 0 and (
        len(evidence) >= policy.high_confidence_evidence
        and runs >= policy.high_confidence_runs
    ):
        return EvidenceLevel.HIGH_CONFIDENCE

    if runs >= policy.repeat_independent_runs:
        return EvidenceLevel.REPEATED

    return EvidenceLevel.SIMULATED


def compute_confidence(
    evidence: list[Evidence],
    counterexamples: list[Counterexample],
    level: EvidenceLevel | None = None,
    policy: PromotionPolicy = DEFAULT_POLICY,
) -> float:
    """An explicit, bounded, monotone function of the evidence. Not invented.

        support   = n / (n + k)              in [0, 1), increasing in n
        penalty   = 1 / (1 + unresolved)     in (0, 1], decreasing in c
        result    = min(support * penalty, ceiling(level))

    Adding evidence never lowers confidence; adding an unresolved
    counterexample always lowers it; the level ceiling caps it.
    """
    if level is None:
        level = derive_level(evidence, counterexamples, policy)

    n = len(evidence)
    support = n / (n + policy.saturation) if n else 0.0
    penalty = 1.0 / (1.0 + unresolved(counterexamples))
    return float(min(support * penalty, LEVEL_CONFIDENCE_CEILING[level]))
