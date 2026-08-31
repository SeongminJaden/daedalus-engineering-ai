"""brain.db - the Brain's SQLite schema and connection.

Standard-library sqlite3 only: the Brain must be loadable with no ML stack, no
service and no model (MD's model/brain separation). A run writes it, and any
later process - including one with no reasoner at all - can open the file and
query it.

One file, several tables, because episodes, designs, knowledge and the concept
graph are all views of the same accumulated experience and are queried together.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("runs") / "brain.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    problem_name  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    termination   TEXT,
    iterations    INTEGER DEFAULT 0,
    best_mass_kg  REAL,
    meta          TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS designs (
    design_id          TEXT PRIMARY KEY,
    run_id             TEXT,
    material_id        TEXT NOT NULL,
    genome             TEXT NOT NULL,   -- JSON: the design variables
    metrics            TEXT NOT NULL,   -- JSON: evaluated metrics
    feasible           INTEGER NOT NULL DEFAULT 0,
    active_constraints TEXT DEFAULT '[]',
    created_at         TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id        TEXT PRIMARY KEY,
    run_id            TEXT,
    design_id         TEXT,
    parent_design_id  TEXT,
    iteration         INTEGER NOT NULL,
    timestamp         TEXT NOT NULL,
    hypothesis        TEXT NOT NULL,
    action            TEXT NOT NULL,
    strategy_used     TEXT NOT NULL,
    conclusion        TEXT NOT NULL,
    confidence        REAL NOT NULL,
    feasible          INTEGER NOT NULL DEFAULT 0,
    is_new_best       INTEGER NOT NULL DEFAULT 0,
    evaluations       INTEGER DEFAULT 0,
    seconds           REAL DEFAULT 0.0,
    observation       TEXT DEFAULT '{}',
    constraint_status TEXT DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (design_id) REFERENCES designs(design_id)
);

CREATE TABLE IF NOT EXISTS knowledge (
    knowledge_id    TEXT PRIMARY KEY,
    -- Stable identity for "the same claim". The statement text embeds live
    -- counts ("active in 9/9 episodes"), so keying on it would file every
    -- re-generalization as a brand new, separately-weak item instead of
    -- consolidating the evidence.
    claim_key       TEXT NOT NULL UNIQUE,
    statement       TEXT NOT NULL,
    domain          TEXT NOT NULL,
    source          TEXT NOT NULL,
    evidence        TEXT NOT NULL DEFAULT '[]',
    counterexamples TEXT NOT NULL DEFAULT '[]',
    assumptions     TEXT NOT NULL DEFAULT '[]',
    confidence      REAL NOT NULL DEFAULT 0.0,
    evidence_level  TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id    TEXT PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    statement      TEXT NOT NULL,
    context        TEXT NOT NULL DEFAULT '{}',
    evidence       TEXT NOT NULL DEFAULT '[]',
    confidence     REAL NOT NULL DEFAULT 0.0,
    evidence_level TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    kind       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id    TEXT PRIMARY KEY,
    source     TEXT NOT NULL,
    relation   TEXT NOT NULL,
    target     TEXT NOT NULL,
    evidence   TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    UNIQUE (source, relation, target)
);

CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);
CREATE INDEX IF NOT EXISTS idx_designs_run ON designs(run_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge(domain);
CREATE INDEX IF NOT EXISTS idx_knowledge_claim ON knowledge(claim_key);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
"""


class BrainDB:
    """Thin connection wrapper. Owns the schema, nothing else."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # --- helpers -------------------------------------------------------- #
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "BrainDB":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def table_names(self) -> list[str]:
        return sorted(
            r["name"] for r in self.query(
                "SELECT name FROM sqlite_master WHERE type='table'")
        )

    def counts(self) -> dict[str, int]:
        return {
            t: self.query_one(f"SELECT COUNT(*) AS n FROM {t}")["n"]
            for t in self.table_names()
        }


def dumps(value) -> str:
    return json.dumps(value, default=str)


def loads(text: str | None, default=None):
    if not text:
        return default if default is not None else {}
    return json.loads(text)
