"""brain.semantic.knowledge - evidence-graded knowledge items.

Every item carries its provenance: what supports it, what contradicts it, what
it assumes, and how far up the evidence ladder it has actually climbed. The
level and the confidence are always *derived* from that record - never set by
hand - so a statement cannot be talked up without new evidence appearing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from brain.db import BrainDB, dumps, loads

from .evidence import (
    DEFAULT_POLICY,
    Counterexample,
    Evidence,
    EvidenceLevel,
    PromotionPolicy,
    compute_confidence,
    derive_level,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Knowledge:
    """One statement plus everything that decides how far to trust it."""

    statement: str
    domain: str
    source: str
    # Stable identity of the claim. Defaults to the statement for one-off
    # items; generalizers pass an explicit key so that re-running them
    # consolidates evidence instead of creating near-duplicate rows.
    claim_key: str = ""
    knowledge_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    evidence: list[Evidence] = field(default_factory=list)
    counterexamples: list[Counterexample] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    policy: PromotionPolicy = DEFAULT_POLICY

    def __post_init__(self):
        if not self.claim_key:
            self.claim_key = self.statement

    @property
    def evidence_level(self) -> EvidenceLevel:
        return derive_level(self.evidence, self.counterexamples, self.policy)

    @property
    def confidence(self) -> float:
        return compute_confidence(
            self.evidence, self.counterexamples, self.evidence_level, self.policy)

    def add_evidence(self, item: Evidence) -> "Knowledge":
        self.evidence.append(item)
        self.updated_at = _now()
        return self

    def add_counterexample(self, item: Counterexample) -> "Knowledge":
        self.counterexamples.append(item)
        self.updated_at = _now()
        return self

    def as_dict(self) -> dict:
        return {
            "knowledge_id": self.knowledge_id,
            "claim_key": self.claim_key,
            "statement": self.statement,
            "domain": self.domain,
            "source": self.source,
            "evidence": [e.as_dict() for e in self.evidence],
            "counterexamples": [c.as_dict() for c in self.counterexamples],
            "assumptions": list(self.assumptions),
            "confidence": self.confidence,
            "evidence_level": self.evidence_level.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SemanticMemory:
    """Persistence for knowledge items."""

    def __init__(self, db: BrainDB):
        self.db = db

    def store(self, k: Knowledge) -> str:
        self.db.execute(
            """INSERT OR REPLACE INTO knowledge
               (knowledge_id, claim_key, statement, domain, source, evidence,
                counterexamples, assumptions, confidence, evidence_level,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (k.knowledge_id, k.claim_key, k.statement, k.domain, k.source,
             dumps([e.as_dict() for e in k.evidence]),
             dumps([c.as_dict() for c in k.counterexamples]),
             dumps(list(k.assumptions)),
             k.confidence, k.evidence_level.value, k.created_at, _now()),
        )
        return k.knowledge_id

    @staticmethod
    def _row(row) -> Knowledge:
        return Knowledge(
            knowledge_id=row["knowledge_id"],
            claim_key=row["claim_key"],
            statement=row["statement"],
            domain=row["domain"],
            source=row["source"],
            evidence=[Evidence.from_dict(d) for d in loads(row["evidence"], [])],
            counterexamples=[
                Counterexample.from_dict(d)
                for d in loads(row["counterexamples"], [])
            ],
            assumptions=loads(row["assumptions"], []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, knowledge_id: str) -> Knowledge | None:
        row = self.db.query_one(
            "SELECT * FROM knowledge WHERE knowledge_id = ?", (knowledge_id,))
        return None if row is None else self._row(row)

    def by_domain(self, domain: str | None = None,
                  min_level: EvidenceLevel | None = None) -> list[Knowledge]:
        if domain is None:
            rows = self.db.query("SELECT * FROM knowledge ORDER BY confidence DESC")
        else:
            rows = self.db.query(
                "SELECT * FROM knowledge WHERE domain = ? ORDER BY confidence DESC",
                (domain,))
        items = [self._row(r) for r in rows]
        if min_level is not None:
            items = [k for k in items if k.evidence_level.rank >= min_level.rank]
        return items

    def find_by_claim(self, claim_key: str) -> Knowledge | None:
        row = self.db.query_one(
            "SELECT * FROM knowledge WHERE claim_key = ?", (claim_key,))
        return None if row is None else self._row(row)

    def find_by_statement(self, statement: str) -> Knowledge | None:
        row = self.db.query_one(
            "SELECT * FROM knowledge WHERE statement = ?", (statement,))
        return None if row is None else self._row(row)

    def upsert_by_claim(self, k: Knowledge) -> Knowledge:
        """Merge evidence into the existing item for this claim.

        This is how a claim climbs the ladder across runs: a second run
        contributing evidence raises the independent-run count instead of
        creating a second, separately-weak item. The statement text is
        refreshed from the incoming version, because generalizers restate it
        over the whole corpus each pass ("active in 9/9" supersedes "3/3").
        """
        existing = self.find_by_claim(k.claim_key)
        if existing is None:
            self.store(k)
            return k
        # Key on run_id too: two runs can produce evidence with the same
        # local ref, and merging those would undercount independence -
        # which is exactly what the promotion rules depend on.
        seen = {(e.kind, e.ref, e.run_id) for e in existing.evidence}
        for e in k.evidence:
            if (e.kind, e.ref, e.run_id) not in seen:
                existing.add_evidence(e)
        seen_c = {c.ref for c in existing.counterexamples}
        for c in k.counterexamples:
            if c.ref not in seen_c:
                existing.add_counterexample(c)
        for a in k.assumptions:
            if a not in existing.assumptions:
                existing.assumptions.append(a)
        existing.statement = k.statement
        self.store(existing)
        return existing


    # Back-compat alias: identity defaults to the statement.
    def upsert_by_statement(self, k: Knowledge) -> Knowledge:
        return self.upsert_by_claim(k)
