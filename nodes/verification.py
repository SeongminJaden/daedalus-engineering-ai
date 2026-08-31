"""Routing a design to external verification, and what to record when there is none.

The funnel wants an answer to one question: has this design been checked by
anything other than our own solver? Today the answer is no, because the
external verification node is not available. The job of this module is to make
that answer explicit and carry it forward, rather than letting the absence of
an external check look like the absence of a problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from .descriptor import CapabilityUnavailable
from .fusion_node import (FUSION_CAPABILITY, FusionVerificationReport,
                          FusionVerificationRequest, verify)
from .registry import CapabilityRegistry

SELF_FEM_ONLY = "self_fem_only"
EXTERNALLY_VERIFIED = "externally_verified"


@dataclass(frozen=True)
class VerificationStatus:
    """What verification a design has actually had.

    `report` is None whenever `status` is not EXTERNALLY_VERIFIED, and there is
    no other way to construct this: a status object cannot claim an external
    check and carry nothing, and it cannot carry numbers while admitting there
    was no external check.
    """

    design_id: str
    status: str
    reason: str = ""
    report: FusionVerificationReport | None = None

    def __post_init__(self) -> None:
        if self.status == EXTERNALLY_VERIFIED and self.report is None:
            raise ValueError(
                "a design cannot be marked externally verified without the "
                "report that verified it")
        if self.status != EXTERNALLY_VERIFIED and self.report is not None:
            raise ValueError(
                f"status {self.status!r} carries a verification report; that "
                f"would let unverified results travel as verified ones")
        if self.status != EXTERNALLY_VERIFIED and not self.reason:
            raise ValueError("a design without external verification must say "
                             "why it does not have any")

    @property
    def is_externally_verified(self) -> bool:
        return self.status == EXTERNALLY_VERIFIED

    def as_dict(self) -> dict:
        """The form the episode log and the Brain store.

        The reason travels with the status. A record saying only
        "self_fem_only" would leave a later reader guessing whether the check
        was skipped, failed, or never possible.
        """
        return {"design_id": self.design_id, "status": self.status,
                "reason": self.reason,
                "external_source": self.report.source if self.report else None}


def request_external_verification(
        request: FusionVerificationRequest,
        registry: CapabilityRegistry | None = None) -> VerificationStatus:
    """Try the external verification node; record honestly when it is not there.

    An unavailable node is an expected outcome here, not an error to propagate:
    the pipeline has to keep running and produce designs, it just may not call
    them externally verified. What it must never do is continue as though the
    check happened, so the unavailability is converted into a status the rest
    of the system carries, not swallowed.
    """
    try:
        if registry is not None:
            registry.require(FUSION_CAPABILITY)
        report = verify(request)
    except CapabilityUnavailable as unavailable:
        return VerificationStatus(design_id=request.design_id,
                                  status=SELF_FEM_ONLY,
                                  reason=unavailable.reason)
    return VerificationStatus(design_id=request.design_id,
                              status=EXTERNALLY_VERIFIED, report=report)
