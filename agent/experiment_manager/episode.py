"""agent.experiment_manager.episode - the episode record, and its JSONL store.

One episode per loop iteration. This is the seed of the Phase 5 Brain: the
schema is chosen now so that structured memory, retrieval and strategy
generalization later have something real to read, instead of having to
back-fill provenance that was never captured.

Deliberately recorded per episode:
  * `hypothesis`  - why the action was taken, not just what happened
  * `parent_design_id` - lineage, so a run reads as a search tree
  * `confidence`  - how much the conclusion should be trusted
  * `strategy_used` - which policy produced the action, for later credit
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field


class Episode(BaseModel):
    """One iteration of the autonomous loop."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    parent_design_id: str | None = None
    iteration: int = Field(ge=0)
    timestamp: str = Field(min_length=1)

    hypothesis: str = Field(min_length=1)
    action: str = Field(min_length=1)
    strategy_used: str = Field(min_length=1)

    design_genome: dict = Field(default_factory=dict)
    observation: dict = Field(default_factory=dict)
    constraint_status: dict = Field(default_factory=dict)

    conclusion: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    feasible: bool = False
    is_new_best: bool = False
    evaluations: int = Field(default=0, ge=0)
    seconds: float = Field(default=0.0, ge=0.0)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


class EpisodeLog:
    """Append-only JSONL store, one object per line.

    JSONL rather than a single JSON array so a run that is killed mid-flight
    still leaves a readable log - every complete line is a valid episode.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._episodes: list[Episode] = []

    def append(self, episode: Episode) -> Episode:
        self._episodes.append(episode)
        with self.path.open("a") as fh:
            fh.write(episode.model_dump_json() + "\n")
        return episode

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes)

    @property
    def episodes(self) -> list[Episode]:
        return list(self._episodes)

    @classmethod
    def read(cls, path: str | Path) -> list[Episode]:
        """Load and validate every episode from a JSONL file."""
        out: list[Episode] = []
        with Path(path).open() as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Episode.model_validate(json.loads(line)))
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        f"{path}:{line_no} is not a valid episode: {exc}"
                    ) from exc
        return out
