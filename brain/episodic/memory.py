"""brain.episodic.memory - episodes, designs and runs.

The literal record of what was tried. Phase 4 writes JSONL as it goes (crash
safe); this ingests that into queryable structure. The Design Repository is the
`designs` table: every evaluated genome with its metrics, keyed so episodes and
retrieval can both point at it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from brain.db import BrainDB, dumps, loads


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodicMemory:
    def __init__(self, db: BrainDB):
        self.db = db

    # --- runs ----------------------------------------------------------- #
    def record_run(self, run_id: str, problem_name: str, termination: str | None = None,
                   iterations: int = 0, best_mass_kg: float | None = None,
                   meta: dict | None = None) -> str:
        self.db.execute(
            """INSERT INTO runs (run_id, problem_name, created_at, termination,
                                 iterations, best_mass_kg, meta)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET
                   termination=excluded.termination,
                   iterations=excluded.iterations,
                   best_mass_kg=excluded.best_mass_kg,
                   meta=excluded.meta""",
            (run_id, problem_name, _now(), termination, iterations,
             best_mass_kg, dumps(meta or {})),
        )
        return run_id

    def get_run(self, run_id: str) -> dict | None:
        row = self.db.query_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        out = dict(row)
        out["meta"] = loads(out["meta"], {})
        return out

    def runs(self) -> list[dict]:
        return [self.get_run(r["run_id"]) for r in
                self.db.query("SELECT run_id FROM runs ORDER BY created_at")]

    # --- designs -------------------------------------------------------- #
    def record_design(self, genome: dict, metrics: dict, feasible: bool,
                      active_constraints: list[str] | None = None,
                      run_id: str | None = None,
                      design_id: str | None = None) -> str:
        design_id = design_id or uuid.uuid4().hex[:12]
        self.db.execute(
            """INSERT OR REPLACE INTO designs
               (design_id, run_id, material_id, genome, metrics, feasible,
                active_constraints, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (design_id, run_id, genome.get("material_id", "unknown"),
             dumps(genome), dumps(metrics), int(bool(feasible)),
             dumps(active_constraints or []), _now()),
        )
        return design_id

    def get_design(self, design_id: str) -> dict | None:
        row = self.db.query_one(
            "SELECT * FROM designs WHERE design_id = ?", (design_id,))
        return None if row is None else self._design_row(row)

    @staticmethod
    def _design_row(row) -> dict:
        out = dict(row)
        out["genome"] = loads(out["genome"], {})
        out["metrics"] = loads(out["metrics"], {})
        out["active_constraints"] = loads(out["active_constraints"], [])
        out["feasible"] = bool(out["feasible"])
        return out

    def designs(self, run_id: str | None = None,
                feasible_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM designs"
        clauses, params = [], []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if feasible_only:
            clauses.append("feasible = 1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at"
        return [self._design_row(r) for r in self.db.query(sql, tuple(params))]

    def best_design(self, run_id: str | None = None) -> dict | None:
        """Lightest feasible design on record."""
        candidates = [
            d for d in self.designs(run_id=run_id, feasible_only=True)
            if d["metrics"].get("mass_kg") is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda d: d["metrics"]["mass_kg"])

    # --- episodes ------------------------------------------------------- #
    def record_episode(self, episode, run_id: str,
                       design_id: str | None = None) -> str:
        """Store one Phase 4 Episode (pydantic model or dict)."""
        e = episode if isinstance(episode, dict) else episode.model_dump()
        self.db.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, run_id, design_id, parent_design_id, iteration,
                timestamp, hypothesis, action, strategy_used, conclusion,
                confidence, feasible, is_new_best, evaluations, seconds,
                observation, constraint_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (e["id"], run_id, design_id, e.get("parent_design_id"),
             e["iteration"], e["timestamp"], e["hypothesis"], e["action"],
             e["strategy_used"], e["conclusion"], float(e["confidence"]),
             int(bool(e.get("feasible", False))),
             int(bool(e.get("is_new_best", False))),
             int(e.get("evaluations", 0)), float(e.get("seconds", 0.0)),
             dumps(e.get("observation", {})),
             dumps(e.get("constraint_status", {}))),
        )
        return e["id"]

    @staticmethod
    def _episode_row(row) -> dict:
        out = dict(row)
        out["observation"] = loads(out["observation"], {})
        out["constraint_status"] = loads(out["constraint_status"], {})
        out["feasible"] = bool(out["feasible"])
        out["is_new_best"] = bool(out["is_new_best"])
        return out

    def get_episode(self, episode_id: str) -> dict | None:
        row = self.db.query_one(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,))
        return None if row is None else self._episode_row(row)

    def episodes(self, run_id: str | None = None) -> list[dict]:
        if run_id is None:
            rows = self.db.query("SELECT * FROM episodes ORDER BY timestamp, iteration")
        else:
            rows = self.db.query(
                "SELECT * FROM episodes WHERE run_id = ? ORDER BY iteration",
                (run_id,))
        return [self._episode_row(r) for r in rows]

    # --- ingestion ------------------------------------------------------ #
    def ingest_jsonl(self, path: str | Path, run_id: str,
                     problem_name: str = "unknown") -> int:
        """Load a Phase 4 episode log, creating a design row per episode.

        Idempotent by episode id, so re-ingesting the same log does not
        duplicate history.
        """
        from agent.experiment_manager import EpisodeLog

        episodes = EpisodeLog.read(path)
        self.record_run(run_id, problem_name, iterations=len(episodes))
        for e in episodes:
            design_id = self.record_design(
                genome=e.design_genome,
                metrics=e.observation,
                feasible=e.feasible,
                active_constraints=e.constraint_status.get("active", []),
                run_id=run_id,
                design_id=f"d-{e.id}",
            )
            self.record_episode(e, run_id=run_id, design_id=design_id)
        return len(episodes)
