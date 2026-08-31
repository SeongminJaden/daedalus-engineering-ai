"""brain.knowledge_graph.graph - typed concept graph.

Concepts and typed edges, e.g.

    al_7075_t6 --is_a--> material
    link       --loaded_in--> bending
    bending    --causes--> deflection

Edges are only created from actual observations and each carries its evidence,
so the graph cannot fill up with plausible-sounding relations nobody measured.
"""

from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime, timezone

from brain.db import BrainDB, dumps, loads
from brain.semantic.evidence import Evidence, compute_confidence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeGraph:
    def __init__(self, db: BrainDB):
        self.db = db

    # --- concepts ------------------------------------------------------- #
    def add_concept(self, name: str, kind: str = "concept") -> str:
        existing = self.db.query_one(
            "SELECT concept_id FROM concepts WHERE name = ?", (name,))
        if existing:
            return existing["concept_id"]
        concept_id = uuid.uuid4().hex[:12]
        self.db.execute(
            "INSERT INTO concepts (concept_id, name, kind, created_at) "
            "VALUES (?, ?, ?, ?)",
            (concept_id, name, kind, _now()))
        return concept_id

    def concepts(self) -> list[dict]:
        return [dict(r) for r in self.db.query("SELECT * FROM concepts ORDER BY name")]

    def has_concept(self, name: str) -> bool:
        return self.db.query_one(
            "SELECT 1 FROM concepts WHERE name = ?", (name,)) is not None

    # --- edges ---------------------------------------------------------- #
    def add_edge(self, source: str, relation: str, target: str,
                 evidence: list[Evidence] | None = None) -> str:
        """Assert a relation. Requires evidence - an unsupported edge is an
        opinion, and this graph does not store opinions."""
        evidence = evidence or []
        if not evidence:
            raise ValueError(
                f"edge {source} -{relation}-> {target} needs at least one "
                "evidence item; unsupported relations are not stored"
            )
        self.add_concept(source)
        self.add_concept(target)

        existing = self.db.query_one(
            "SELECT * FROM edges WHERE source=? AND relation=? AND target=?",
            (source, relation, target))
        if existing:
            merged = [Evidence.from_dict(d) for d in loads(existing["evidence"], [])]
            seen = {(e.kind, e.ref, e.run_id) for e in merged}
            merged += [e for e in evidence
                       if (e.kind, e.ref, e.run_id) not in seen]
            self.db.execute(
                "UPDATE edges SET evidence=?, confidence=? WHERE edge_id=?",
                (dumps([e.as_dict() for e in merged]),
                 compute_confidence(merged, []), existing["edge_id"]))
            return existing["edge_id"]

        edge_id = uuid.uuid4().hex[:12]
        self.db.execute(
            """INSERT INTO edges (edge_id, source, relation, target, evidence,
                                  confidence, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (edge_id, source, relation, target,
             dumps([e.as_dict() for e in evidence]),
             compute_confidence(evidence, []), _now()))
        return edge_id

    @staticmethod
    def _edge_row(row) -> dict:
        out = dict(row)
        out["evidence"] = loads(out["evidence"], [])
        return out

    def edges(self) -> list[dict]:
        return [self._edge_row(r) for r in
                self.db.query("SELECT * FROM edges ORDER BY created_at")]

    def neighbors(self, name: str, relation: str | None = None,
                  direction: str = "out") -> list[dict]:
        """Adjacent concepts. direction: out | in | both."""
        if direction not in ("out", "in", "both"):
            raise ValueError("direction must be out, in or both")
        rows: list = []
        if direction in ("out", "both"):
            sql = "SELECT * FROM edges WHERE source = ?"
            params = [name]
            if relation:
                sql += " AND relation = ?"
                params.append(relation)
            rows += self.db.query(sql, tuple(params))
        if direction in ("in", "both"):
            sql = "SELECT * FROM edges WHERE target = ?"
            params = [name]
            if relation:
                sql += " AND relation = ?"
                params.append(relation)
            rows += self.db.query(sql, tuple(params))
        return [self._edge_row(r) for r in rows]

    def path(self, source: str, target: str, max_depth: int = 6) -> list[dict] | None:
        """Shortest directed path as a list of edges, or None.

        Breadth-first, so the first path found is a shortest one.
        """
        if source == target:
            return []
        queue = deque([(source, [])])
        seen = {source}
        while queue:
            node, trail = queue.popleft()
            if len(trail) >= max_depth:
                continue
            for edge in self.neighbors(node, direction="out"):
                nxt = edge["target"]
                if nxt == target:
                    return trail + [edge]
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, trail + [edge]))
        return None
