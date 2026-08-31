"""brain.strategy.strategies - solution strategies earned from data.

A strategy is a *reusable move* ("when X binds, push Y"), promoted only once
the data supports it. Nothing here is hand-written engineering advice: the one
real strategy implemented below is derived from measured sensitivities, and it
carries the samples it was derived from as evidence.

The Brain stays model-independent: sensitivities are passed in by the caller,
who owns the physics. This package never imports the solver.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from brain.db import BrainDB, dumps, loads
from brain.semantic.evidence import (
    DEFAULT_POLICY,
    Evidence,
    EvidenceKind,
    EvidenceLevel,
    PromotionPolicy,
    compute_confidence,
    derive_level,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# A move must win in at least this share of samples before it is a strategy.
PROMOTION_THRESHOLD = 0.7


@dataclass
class Strategy:
    """A promoted move, with the evidence that promoted it."""

    name: str
    statement: str
    context: dict = field(default_factory=dict)
    strategy_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    evidence: list[Evidence] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    policy: PromotionPolicy = DEFAULT_POLICY

    @property
    def evidence_level(self) -> EvidenceLevel:
        return derive_level(self.evidence, [], self.policy)

    @property
    def confidence(self) -> float:
        return compute_confidence(self.evidence, [], self.evidence_level, self.policy)

    def as_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "statement": self.statement,
            "context": dict(self.context),
            "evidence": [e.as_dict() for e in self.evidence],
            "confidence": self.confidence,
            "evidence_level": self.evidence_level.value,
        }


class StrategyStore:
    def __init__(self, db: BrainDB):
        self.db = db

    def store(self, s: Strategy) -> str:
        self.db.execute(
            """INSERT INTO strategies (strategy_id, name, statement, context,
                                       evidence, confidence, evidence_level,
                                       created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                   statement=excluded.statement,
                   context=excluded.context,
                   evidence=excluded.evidence,
                   confidence=excluded.confidence,
                   evidence_level=excluded.evidence_level,
                   updated_at=excluded.updated_at""",
            (s.strategy_id, s.name, s.statement, dumps(s.context),
             dumps([e.as_dict() for e in s.evidence]), s.confidence,
             s.evidence_level.value, s.created_at, _now()))
        return s.strategy_id

    @staticmethod
    def _row(row) -> Strategy:
        return Strategy(
            strategy_id=row["strategy_id"],
            name=row["name"],
            statement=row["statement"],
            context=loads(row["context"], {}),
            evidence=[Evidence.from_dict(d) for d in loads(row["evidence"], [])],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def all(self) -> list[Strategy]:
        return [self._row(r) for r in
                self.db.query("SELECT * FROM strategies ORDER BY confidence DESC")]

    def get(self, name: str) -> Strategy | None:
        row = self.db.query_one("SELECT * FROM strategies WHERE name = ?", (name,))
        return None if row is None else self._row(row)

    def applicable(self, context: dict, min_confidence: float = 0.0) -> list[Strategy]:
        """Strategies whose stored context is a subset of the query context."""
        out = []
        for s in self.all():
            if s.confidence < min_confidence:
                continue
            if all(context.get(k) == v for k, v in s.context.items()):
                out.append(s)
        return out


def derive_stiffness_strategy(
    samples: list[dict],
    store: StrategyStore | None = None,
    threshold: float = PROMOTION_THRESHOLD,
) -> Strategy | None:
    """Which design variable buys the most stiffness per kilogram?

    Each sample is one design's measured sensitivities:

        {"ref": <episode/design id>, "run_id": <run>,
         "d_deflection": {"outer_width_m": ..., "outer_height_m": ...,
                          "wall_thickness_m": ...},
         "d_mass":       {... same keys ...}}

    Efficiency per variable is |d(deflection)/dx| / (d(mass)/dx): deflection
    removed per kilogram added. The winner is whichever variable maximizes it.
    If one variable wins in at least `threshold` of the samples, that becomes a
    strategy - otherwise nothing is promoted, because there is no consistent
    move to recommend.

    Sensitivities come from the caller (Phase 2 autodiff), so this module stays
    free of any physics dependency.
    """
    if not samples:
        return None

    wins: dict[str, list[dict]] = {}
    for sample in samples:
        d_defl = sample.get("d_deflection", {})
        d_mass = sample.get("d_mass", {})
        efficiency = {}
        for var, dd in d_defl.items():
            dm = d_mass.get(var)
            if dm is None or dm <= 0:
                continue          # a variable that does not add mass tells us nothing
            efficiency[var] = abs(float(dd)) / float(dm)
        if not efficiency:
            continue
        winner = max(efficiency, key=lambda v: efficiency[v])
        wins.setdefault(winner, []).append(sample)

    if not wins:
        return None

    total = sum(len(v) for v in wins.values())
    best_var, supporters = max(wins.items(), key=lambda kv: len(kv[1]))
    share = len(supporters) / total
    if share < threshold:
        return None

    strategy = Strategy(
        name=f"stiffness-per-mass:{best_var}",
        statement=(
            f"When tip deflection is the binding constraint, increasing "
            f"'{best_var}' removes the most deflection per kilogram added "
            f"(best in {len(supporters)}/{total} sampled designs). For a "
            f"rectangular section this follows I ~ h^3 in the bending "
            f"direction: height buys stiffness far faster than width or wall."
        ),
        context={"binding_constraint": "deflection"},
        evidence=[
            Evidence(
                kind=EvidenceKind.SIMULATION,
                ref=str(s.get("ref", "unknown")),
                run_id=s.get("run_id"),
                note=f"most efficient variable: {best_var}",
            )
            for s in supporters
        ],
    )
    if store is not None:
        store.store(strategy)
    return strategy
