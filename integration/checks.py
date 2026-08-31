"""One verification check, and the vocabulary for aggregating many of them.

VALIDITY OF THE AGGREGATION ITSELF, which is the new thing this layer adds and
the thing most likely to be misread:

* **An assembly verdict is only as trustworthy as its weakest method.** Every
  check here carries idealisations, and the aggregate inherits all of them at
  once. The reported minimum safety factor is therefore an UPPER bound on the
  real margin, not an estimate of it. Each result carries the single most
  optimistic assumption behind it so the review can say which one is load
  bearing.

* **A minimum over safety factors is not a system safety factor.** Several
  independent modes each sitting at 1.1 give a system that fails more often
  than one mode at 1.1, because there are more ways to lose. Nothing here
  computes a system reliability, and the minimum must not be read as one.

* **Coverage is over REGISTERED methods, not over reality.** A failure mode
  nobody has implemented cannot be checked and cannot even be listed unless
  someone named it. So `assessed` does not mean `safe`; it means a method ran.
  The known-but-unimplemented modes are enumerated explicitly for that reason,
  and the unknown ones are, by construction, not.

* **NOT_ASSESSED is not a pass.** A component whose failure mode has no
  applicable method must say so. Treating an absent check as a satisfied one
  is the single most dangerous thing an integration layer can do, because it
  converts ignorance into confidence at exactly the point where a human stops
  looking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# A safety factor may sit this far below 1.0 and still count as passing.
#
# Not slack, and not a fudge. An optimiser drives its active constraints
# EXACTLY to their limits, so the design it returns has a ratio of 1.0 to
# within its own convergence tolerance, landing on either side of it by
# floating point alone. An aggregation layer comparing strictly against 1.0
# rejects the very optimum it was handed, at random. This matches the
# feasibility tolerance the constraint evaluator already uses, so the two
# layers agree about what "satisfied" means.
FEASIBILITY_TOLERANCE = 1e-4


def satisfies(safety_factor: float) -> bool:
    """Whether a factor counts as passing, at the shared tolerance."""
    return safety_factor >= 1.0 - FEASIBILITY_TOLERANCE


class CheckStatus(str, Enum):
    """What happened to one check."""

    PASSED = "passed"
    FAILED = "failed"
    # The duty makes this mode impossible, so there is nothing to check. A
    # tension member cannot buckle.
    NOT_APPLICABLE = "not_applicable"
    # The mode IS possible here and no registered method can evaluate it. This
    # is a gap, not a pass.
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class CheckResult:
    """One failure mode, on one component, with its verdict and its caveat."""

    component: str
    failure_mode: str
    status: CheckStatus
    method: str | None = None
    safety_factor: float | None = None
    detail: str = ""
    # The single most optimistic idealisation behind THIS number. Carried per
    # check rather than per phase, because the review needs the assumption
    # belonging to whichever check turns out to govern.
    optimistic_assumption: str = ""

    def __post_init__(self) -> None:
        if self.status in (CheckStatus.PASSED, CheckStatus.FAILED):
            if self.method is None:
                raise ValueError(
                    f"{self.component}/{self.failure_mode} has a verdict but "
                    f"names no method; a verdict with no method behind it "
                    f"cannot be audited")
            if self.safety_factor is None:
                raise ValueError(
                    f"{self.component}/{self.failure_mode} has a verdict but "
                    f"no safety factor")
        if self.status is CheckStatus.NOT_ASSESSED and not self.detail:
            raise ValueError(
                f"{self.component}/{self.failure_mode} is unassessed and says "
                f"nothing about why; an unexplained gap is indistinguishable "
                f"from an oversight")

    @property
    def is_verdict(self) -> bool:
        return self.status in (CheckStatus.PASSED, CheckStatus.FAILED)


class AssemblyStatus(str, Enum):
    """The assembly-level outcome. Three states, deliberately.

    PASSED_WITH_GAPS exists because collapsing it into PASSED would hide the
    difference between "everything applicable was checked and held" and
    "everything we could check held, and some things we could not check were
    not checked".
    """

    PASSED = "passed"
    PASSED_WITH_GAPS = "passed_with_gaps"
    FAILED = "failed"


@dataclass
class AssemblyVerdict:
    """Every check on every component, aggregated conjunctively."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    # --- slices --------------------------------------------------------------

    def verdicts(self) -> list[CheckResult]:
        return [r for r in self.results if r.is_verdict]

    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is CheckStatus.FAILED]

    def unassessed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is CheckStatus.NOT_ASSESSED]

    def components(self) -> list[str]:
        seen: list[str] = []
        for result in self.results:
            if result.component not in seen:
                seen.append(result.component)
        return seen

    # --- the conjunctive rule ------------------------------------------------

    @property
    def status(self) -> AssemblyStatus:
        """PASS requires EVERY applicable check to pass, and none to be missing.

        Conjunctive: one failure anywhere fails the assembly. There is no
        averaging and no weighting, because a joint whose bearing outlasts the
        machine and whose bolt separates on the first cycle is not a
        three-quarters-good joint.
        """
        if self.failures():
            return AssemblyStatus.FAILED
        if self.unassessed():
            return AssemblyStatus.PASSED_WITH_GAPS
        return AssemblyStatus.PASSED

    @property
    def passes(self) -> bool:
        """True only for a fully assessed, fully passing assembly.

        PASSED_WITH_GAPS is deliberately NOT true here. A caller that wants to
        proceed despite gaps has to look at them.
        """
        return self.status is AssemblyStatus.PASSED

    def governing(self) -> CheckResult | None:
        """The check closest to failing: the one that actually sizes the design.

        Ties break on component then failure mode so the answer is stable
        across runs rather than depending on evaluation order.
        """
        candidates = [r for r in self.verdicts() if r.safety_factor is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda r: (r.safety_factor, r.component,
                                              r.failure_mode))

    @property
    def governing_safety_factor(self) -> float | None:
        governing = self.governing()
        return None if governing is None else governing.safety_factor

    def summary(self) -> str:
        governing = self.governing()
        if governing is None:
            return f"{self.status.value}: nothing was assessed"
        return (f"{self.status.value}: {governing.component} "
                f"{governing.failure_mode} governs at "
                f"{governing.safety_factor:.3f}")
