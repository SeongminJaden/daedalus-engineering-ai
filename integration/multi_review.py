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

* **A shape preference is not evidence and cannot rescue a design.** Form
  ranks admissible candidates against one another. Since a failing design is
  never admissible, no preference value can lift it above a passing one; the
  guard is structural rather than a comparison someone has to remember to
  write.

* **Dominance is a stronger statement than a ranking, and a weaker one than a
  choice.** A design no better than another on ANY axis and worse on some can
  be discarded without arguing about criteria. What survives is the set worth
  arguing over, and that set usually has more than one member. Saying which of
  them to build is still a judgement and this does not make it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from optimization.multi_objective.pareto import non_dominated_mask

from .checks import AssemblyStatus, AssemblyVerdict


class RankBy(str, Enum):
    """What a ranking optimises. There is no neutral choice."""

    GOVERNING_MARGIN = "governing_margin"    # most conservative first
    MASS = "mass"                            # lightest first
    COST = "cost"                            # cheapest first
    FEWEST_GAPS = "fewest_gaps"              # best understood first
    FORM = "form"                            # a stated shape preference


@dataclass(frozen=True)
class DesignEntry:
    """One candidate design and its verdict."""

    name: str
    verdict: AssemblyVerdict
    mass_kg: float = 0.0
    cost_usd: float = 0.0
    #: A shape preference supplied by the caller, higher meaning more
    #: preferred. It is NOT evidence and carries no confidence: it ranks
    #: admissible designs against each other and nothing else. A design that
    #: fails a check is never ranked at all, so no value here can rescue one.
    form_score: float = 0.0

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
            RankBy.FORM: lambda e: (-e.form_score, e.name),
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

    def _objective_matrix(self, entries) -> np.ndarray:
        """Every axis as a MINIMISED column, which is what the filter expects.

        The governing margin is maximised, so it is negated here. Passing it
        raw would select exactly the wrong designs, which is the failure the
        shared filter's own docstring warns about.

        Mass and cost default to zero. When a caller leaves them unset they are
        constant across the entries and therefore do not discriminate, which is
        correct: an axis carrying no information should not decide anything.
        """
        return np.array([[e.mass_kg, e.cost_usd, -e.governing_margin,
                          float(e.gap_count), -e.form_score]
                         for e in entries], dtype=float)

    def non_dominated(self) -> list[DesignEntry]:
        """Admissible designs that nothing else beats outright.

        Uses the project's existing non-dominated filter rather than a second
        implementation of dominance, so there is one definition to be right
        about.
        """
        entries = self.admissible
        if not entries:
            return []
        mask = non_dominated_mask(self._objective_matrix(entries))
        return [entry for entry, keep in zip(entries, mask) if keep]

    def dominated(self) -> list[DesignEntry]:
        """Designs that can be discarded without choosing a criterion."""
        surviving = {id(e) for e in self.non_dominated()}
        return [e for e in self.admissible if id(e) not in surviving]

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
