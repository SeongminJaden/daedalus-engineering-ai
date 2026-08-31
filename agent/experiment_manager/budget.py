"""agent.experiment_manager.budget - compute budget tracking.

An autonomous loop with no budget is a hang. Limits come from the active GPU
profile (`budget.max_evaluations`, `budget.max_seconds`) so a laptop and a
datacenter card get proportionate ones, and either can be overridden per run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.profile import load_profile


@dataclass
class ComputeBudget:
    """Counts what a run has spent and says when it is out."""

    max_evaluations: int
    max_seconds: float
    evaluations: int = 0
    _start: float = field(default_factory=time.monotonic)

    @classmethod
    def from_profile(
        cls,
        profile: str | None = None,
        max_evaluations: int | None = None,
        max_seconds: float | None = None,
    ) -> "ComputeBudget":
        cfg = load_profile(profile).get("budget", {})
        return cls(
            max_evaluations=int(
                max_evaluations if max_evaluations is not None
                else cfg.get("max_evaluations", 20000)
            ),
            max_seconds=float(
                max_seconds if max_seconds is not None
                else cfg.get("max_seconds", 300)
            ),
        )

    def spend(self, evaluations: int) -> None:
        if evaluations < 0:
            raise ValueError("evaluations must be >= 0")
        self.evaluations += evaluations

    @property
    def seconds(self) -> float:
        return time.monotonic() - self._start

    def exceeded(self) -> bool:
        return (self.evaluations >= self.max_evaluations
                or self.seconds >= self.max_seconds)

    def reason(self) -> str | None:
        if self.evaluations >= self.max_evaluations:
            return (f"evaluation budget spent: {self.evaluations} >= "
                    f"{self.max_evaluations}")
        if self.seconds >= self.max_seconds:
            return f"time budget spent: {self.seconds:.1f}s >= {self.max_seconds}s"
        return None

    def as_dict(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "max_evaluations": self.max_evaluations,
            "seconds": round(self.seconds, 3),
            "max_seconds": self.max_seconds,
        }
