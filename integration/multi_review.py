"""Comparing several complete designs across every check at once.

A single verdict says whether one design passes. A review of several says which
to build, and that is a different question: the answer depends on what is being
traded, and the designs will not agree about which check governs them.

VALIDITY, before the implementation:

* **A ranking needs a stated criterion and there is no neutral one.** Ranking by
  governing safety factor prefers the most conservative design; ranking by mass
  prefers the lightest, which is usually the one closest to failing. The
  criterion is an argument, and the report shows the alternatives alongside so
  a different choice is visible rather than hidden.

* **Designs are only comparable on checks they BOTH ran.** One design assessed
  against fatigue and another not assessed against it are not two data points
  on the same axis, and the count of unassessed modes travels with each design
  for that reason.

* **A failing design is not ranked against passing ones.** It is reported as
  failing. Sorting it into a list by margin would put it in a position
  implying it is merely worse, and it is not admissible at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .checks import AssemblyStatus, AssemblyVerdict


class RankBy(str, Enum):
    """What a ranking optimises. There is no neutral choice."""

    GOVERNING_MARGIN = "governing_margin"    # most conservative first
    MASS = "mass"                            # lightest first
    COST = "cost"                            # cheapest first
    FEWEST_GAPS = "fewest_gaps"              # best understood first


@dataclass(frozen=True)
class DesignEntry:
    """One candidate design and its verdict."""

    name: str
    verdict: AssemblyVerdict
    mass_kg: float = 0.0
    cost_usd: float = 0.0

    @property
    def admissible(self) -> bool:
        """Whether it may be ranked at all: a failing design may not."""
        return self.verdict.status is not AssemblyStatus.FAILED

    @property
    def governing_margin(self) -> float:
        factor = self.verdict.governing_safety_factor
        return 0.0 if factor is None else factor

    @property
    def gap_count(self) -> int:
        return len(self.verdict.unassessed())

    @property
    def governing_check(self) -> str:
        governing = self.verdict.governing()
        return ("nothing assessed" if governing is None
                else f"{governing.component}/{governing.failure_mode}")


@dataclass
class MultiDesignReview:
    """Several designs, ranked by a stated criterion, with the rejects named."""

    entries: list[DesignEntry]
    rank_by: RankBy = RankBy.GOVERNING_MARGIN

    @property
    def admissible(self) -> list[DesignEntry]:
        return [e for e in self.entries if e.admissible]

    @property
    def rejected(self) -> list[DesignEntry]:
        """Failing designs, reported rather than ranked."""
        return [e for e in self.entries if not e.admissible]

    def ranked(self, rank_by: RankBy | None = None) -> list[DesignEntry]:
        """Admissible designs in order. Ties break on name for reproducibility."""
        criterion = rank_by or self.rank_by
        keys = {
            RankBy.GOVERNING_MARGIN: lambda e: (-e.governing_margin, e.name),
            RankBy.MASS: lambda e: (e.mass_kg, e.name),
            RankBy.COST: lambda e: (e.cost_usd, e.name),
            RankBy.FEWEST_GAPS: lambda e: (e.gap_count, e.name),
        }
        return sorted(self.admissible, key=keys[criterion])

    def best(self, rank_by: RankBy | None = None) -> DesignEntry | None:
        order = self.ranked(rank_by)
        return order[0] if order else None

    def disagreement(self) -> "dict[RankBy, str]":
        """Which design each criterion would pick.

        The useful output when they differ: it shows that the choice is a
        judgement rather than a computation, and names what each judgement
        costs.
        """
        return {criterion: (self.best(criterion).name
                            if self.best(criterion) else "none")
                for criterion in RankBy}

    def criteria_agree(self) -> bool:
        return len(set(self.disagreement().values())) == 1

    def comparable_checks(self) -> set[str]:
        """Failure modes every admissible design actually assessed.

        Designs are only comparable on these. A mode one design checked and
        another did not is not a shared axis.
        """
        if not self.admissible:
            return set()
        sets = [{f"{r.component}/{r.failure_mode}"
                 for r in entry.verdict.verdicts()}
                for entry in self.admissible]
        return set.intersection(*sets)
