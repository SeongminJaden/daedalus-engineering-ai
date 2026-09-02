"""Classifying a part into a family: rules first, a learned model as a check.

Two classifiers, deliberately, and they are not equals.

The RULE classifier reads the topology the descriptor vector already holds:
how many faces of which kind, how many holes, what the Euler characteristic
says about through-cavities. Each rule states what it matches and why, and
when nothing matches it says UNKNOWN with the reasons rather than picking the
nearest family. Its answer is a reading of geometry, graded SIMULATED like any
other computed label.

The LEARNED classifier is a nearest-neighbour vote over standardised
descriptors, with open-set rejection: a part farther from every training
example than the training set is from itself is UNKNOWN. Its answer is graded
SURROGATE. It is here to be compared against the rules, and later against the
embeddings, not to decide anything. A learned model that agrees with nothing
is not evidence, and one that agrees with the rules on five synthetic families
has shown that it can learn what the rules already knew.

WHY NEAREST NEIGHBOUR AND NOT A NETWORK
=======================================
Five families and a twenty two number vector do not need a network, and a
method whose every decision can be traced to a named training part is worth
more here than a percentage point. It also adds no dependency: numpy is
already present, and a classifier is not a reason to add scikit-learn.

VALIDITY DOMAIN
===============
    The five synthetic families. A part outside them is UNKNOWN when the
    classifier is working and misclassified when it is not, and the
    open-set threshold is what separates those; it is calibrated on the
    training set and reported, not assumed. The fixtures authored in Fusion
    are the one set of parts outside the generator this is measured on, and
    the measurement is written in the tests rather than here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from brain.semantic.evidence import EvidenceKind, EvidenceLevel

from .descriptors import DESCRIPTOR_NAMES, ShapeDescriptor

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Classification:
    family: str
    method: str
    evidence: EvidenceLevel
    reason: str
    #: Fraction of the vote for the winner (learned) or 1.0 (rules).
    confidence: float = 1.0
    #: Standardised distance to the nearest training part (learned only).
    distance: float | None = None

    @property
    def decided(self) -> bool:
        return self.family != UNKNOWN


# ------------------------------------------------------------------ rules

def _all_planes(d: ShapeDescriptor) -> bool:
    return d["frac_plane"] == 1.0


def rule_classify(d: ShapeDescriptor) -> Classification:
    """Topology rules, each stated. UNKNOWN when none holds."""
    faces, euler = int(d["faces"]), int(d["euler"])
    holes, fillets = int(d["hole_count"]), int(d["fillet_count"])
    planes = round(d["frac_plane"] * faces)
    cylinders = round(d["frac_cylinder"] * faces)
    reasons: list[str] = []

    def verdict(family: str, why: str) -> Classification:
        return Classification(family=family, method="topology_rules",
                              evidence=EvidenceLevel.SIMULATED, reason=why)

    if _all_planes(d) and holes == 0 and fillets == 0:
        if faces == 6 and euler == 2:
            return verdict("box", "6 planar faces, genus 0")
        if faces == 10 and euler == 0:
            return verdict("hollow_rect",
                           "10 planar faces, genus 1: one through cavity")
        if faces == 8 and euler == 2:
            return verdict("l_bracket", "8 planar faces, genus 0: one "
                                        "reentrant corner")
        reasons.append(f"all planar with {faces} faces and euler {euler}, "
                       f"which is none of box (6, 2), hollow_rect (10, 0) or "
                       f"l_bracket (8, 2)")
    else:
        reasons.append("not all planar" if not _all_planes(d)
                       else f"{holes} holes and {fillets} fillets")

    if holes in (2, 4) and fillets == 4 and planes == 6 and euler == 2 - 2 * holes \
            and cylinders == holes + 4:
        return verdict("plate_with_holes",
                       f"6 planar faces, 4 cylindrical fillets, {holes} through "
                       f"holes, genus {holes}")
    reasons.append(f"plate needs 2 or 4 holes with 4 fillets on 6 planes; "
                   f"saw {holes} holes, {fillets} fillets, {planes} planes")

    if faces == 5 and planes == 3 and cylinders == 2 and holes == 0 \
            and fillets == 0 and euler == 2:
        return verdict("stepped_shaft",
                       "2 convex cylinders and 3 planes: a shoulder")
    reasons.append(f"shaft needs 2 cylinders and 3 planes in 5 faces; saw "
                   f"{cylinders} cylinders, {planes} planes, {faces} faces")

    return Classification(family=UNKNOWN, method="topology_rules",
                          evidence=EvidenceLevel.SIMULATED,
                          reason="; ".join(reasons), confidence=0.0)


# ------------------------------------------------------- nearest neighbour

@dataclass
class NearestNeighbourClassifier:
    """Standardised k nearest neighbours with open-set rejection."""

    k: int = 5
    #: Multiple of the training set's own 99th percentile nearest-neighbour
    #: distance beyond which a query is UNKNOWN. Measured at 1.0, 1.25 and
    #: 1.5 against the Fusion fixtures: at 1.25 and above a part containing a
    #: cone was accepted as a box. A false reject of a family member costs a
    #: look at the rules; a false accept costs a wrong answer. So 1.0.
    rejection_factor: float = 1.0
    mean_: np.ndarray = field(default=None, repr=False)
    scale_: np.ndarray = field(default=None, repr=False)
    x_: np.ndarray = field(default=None, repr=False)
    y_: list[str] = field(default_factory=list, repr=False)
    threshold_: float = float("inf")

    def fit(self, descriptors: Sequence[ShapeDescriptor],
            families: Sequence[str]) -> "NearestNeighbourClassifier":
        x = np.array([d.vector() for d in descriptors], dtype=float)
        if len(x) < self.k + 1:
            raise ValueError(f"need more than k={self.k} training parts, "
                             f"got {len(x)}")
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        # a constant column carries no information and must not divide by zero
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        self.x_ = (x - self.mean_) / self.scale_
        self.y_ = list(families)
        # leave-one-out nearest distance within the training set
        d = np.linalg.norm(self.x_[:, None, :] - self.x_[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        self.threshold_ = float(np.percentile(nearest, 99)) * self.rejection_factor
        return self

    def classify(self, descriptor: ShapeDescriptor) -> Classification:
        if self.x_ is None:
            raise RuntimeError("classifier has not been fitted")
        q = (descriptor.vector() - self.mean_) / self.scale_
        d = np.linalg.norm(self.x_ - q, axis=1)
        order = np.argsort(d)[:self.k]
        nearest = float(d[order[0]])
        votes: dict[str, int] = {}
        for i in order:
            votes[self.y_[i]] = votes.get(self.y_[i], 0) + 1
        winner = max(votes, key=votes.get)
        confidence = votes[winner] / len(order)
        if nearest > self.threshold_:
            return Classification(
                family=UNKNOWN, method="knn_descriptors",
                evidence=EvidenceLevel.SURROGATE,
                reason=f"nearest training part is {nearest:.2f} standardised "
                       f"units away, beyond the {self.threshold_:.2f} the "
                       f"training set spans; nearest vote would have been "
                       f"{winner}",
                confidence=0.0, distance=nearest)
        return Classification(
            family=winner, method="knn_descriptors",
            evidence=EvidenceLevel.SURROGATE,
            reason=f"{votes[winner]} of {len(order)} nearest training parts "
                   f"are {winner}; a suggestion, not a verdict",
            confidence=confidence, distance=nearest)


# ------------------------------------------------------------- evaluation

@dataclass(frozen=True)
class Evaluation:
    n: int
    rule_accuracy: float
    knn_accuracy: float
    rule_unknown: int
    knn_unknown: int
    disagreements: tuple[tuple[str, str, str], ...]   # (truth, rule, knn)


def evaluate(classifier: NearestNeighbourClassifier,
             descriptors: Sequence[ShapeDescriptor],
             truth: Sequence[str]) -> Evaluation:
    rule_hits = knn_hits = rule_unknown = knn_unknown = 0
    disagreements = []
    for d, t in zip(descriptors, truth):
        r = rule_classify(d)
        k = classifier.classify(d)
        rule_hits += r.family == t
        knn_hits += k.family == t
        rule_unknown += r.family == UNKNOWN
        knn_unknown += k.family == UNKNOWN
        if r.family != t or k.family != t:
            disagreements.append((t, r.family, k.family))
    n = max(len(truth), 1)
    return Evaluation(n=len(truth), rule_accuracy=rule_hits / n,
                      knn_accuracy=knn_hits / n, rule_unknown=rule_unknown,
                      knn_unknown=knn_unknown,
                      disagreements=tuple(disagreements))


def classification_label(result: Classification) -> dict:
    """The classification as a dataset label, graded by what produced it."""
    from .schema import label

    kind = (EvidenceKind.SURROGATE if result.evidence is EvidenceLevel.SURROGATE
            else EvidenceKind.ANALYTICAL)
    item = label(result.confidence, "vote_fraction", kind, result.method,
                 note=result.reason, family=result.family)
    return item
