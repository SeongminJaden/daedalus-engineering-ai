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
# Agreed with an independently written solver. This is a real strengthening of
# the evidence and it is NOT physical validation: see the note below.
CROSS_VALIDATED = "cross_validated"
EXTERNALLY_VERIFIED = "externally_verified"

# Statuses in increasing order of evidence. None of them is a physical test.
#
# A NOTE ON THE NAMES, because EXTERNALLY_VERIFIED promises more than it
# delivers. The Fusion node it was written for runs FEA, which is a simulation
# too. Neither that status nor CROSS_VALIDATED means anything has been built or
# measured, and no combination of solvers agreeing with each other can reach
# EXPERIMENTALLY_VALIDATED on the Brain's evidence ladder, which only physical
# test evidence opens. The ladder here is about how many independent
# implementations agree, not about contact with reality.
STATUS_ORDER = (SELF_FEM_ONLY, CROSS_VALIDATED, EXTERNALLY_VERIFIED)


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
    cross_validation: "CrossValidation | FlowCrossValidation | None" = None

    def __post_init__(self) -> None:
        if self.status == EXTERNALLY_VERIFIED and self.report is None:
            raise ValueError(
                "a design cannot be marked externally verified without the "
                "report that verified it")
        if self.status != EXTERNALLY_VERIFIED and self.report is not None:
            raise ValueError(
                f"status {self.status!r} carries a verification report; that "
                f"would let unverified results travel as verified ones")
        if self.status == CROSS_VALIDATED and self.cross_validation is None:
            raise ValueError(
                "a design cannot be marked cross validated without the "
                "measured agreement that cross validated it")
        if self.status != CROSS_VALIDATED and self.cross_validation is not None:
            raise ValueError(
                f"status {self.status!r} carries a cross-validation record; a "
                f"measurement must travel with the claim it supports")
        if self.status == SELF_FEM_ONLY and not self.reason:
            raise ValueError("a design without external verification must say "
                             "why it does not have any")

    @property
    def is_externally_verified(self) -> bool:
        return self.status == EXTERNALLY_VERIFIED

    @property
    def is_cross_validated(self) -> bool:
        return self.status == CROSS_VALIDATED

    @property
    def is_physically_validated(self) -> bool:
        """Always False, and it is a property so the answer is explicit.

        No status this module can produce means anything was built or measured.
        A caller looking for physical validation gets a definite no rather than
        an absence they have to notice.
        """
        return False

    def as_dict(self) -> dict:
        """The form the episode log and the Brain store.

        The reason travels with the status. A record saying only
        "self_fem_only" would leave a later reader guessing whether the check
        was skipped, failed, or never possible.
        """
        return {"design_id": self.design_id, "status": self.status,
                "reason": self.reason,
                "external_source": self.report.source if self.report else None,
                "cross_validated_against":
                    self.cross_validation.solver if self.cross_validation
                    else None,
                "physically_validated": False}


@dataclass(frozen=True)
class CrossValidation:
    """A measured agreement with an independently written solver.

    The numbers travel with the claim. A status saying "cross validated" and
    carrying no measurement would be an assertion, and the whole point of
    running a second solver is to have a figure rather than a belief.
    """

    solver: str
    solver_version: str
    displacement_relative_error: float
    stress_relative_error: float
    tolerance: float
    element_type: str

    @property
    def agrees(self) -> bool:
        return (self.displacement_relative_error <= self.tolerance
                and self.stress_relative_error <= self.tolerance)


@dataclass(frozen=True)
class FlowCrossValidation:
    """A measured agreement with an independently written CFD solver.

    Kept separate from CrossValidation rather than widened into it. A flow
    comparison has no displacement and no element type, and a record carrying
    fields that do not apply to it is an invitation to fill them with
    something that was never measured.
    """

    solver: str
    solver_version: str
    mean_velocity_relative_error: float
    profile_relative_error: float
    tolerance: float
    discretisation: str

    @property
    def agrees(self) -> bool:
        return (self.mean_velocity_relative_error <= self.tolerance
                and self.profile_relative_error <= self.tolerance)


def flow_cross_validated_status(design_id: str,
                                validation: FlowCrossValidation
                                ) -> VerificationStatus:
    """Promote a flow result to cross validated, or explain why it cannot be.

    The same rule as the structural case: disagreement carries the measured
    error rather than quietly reverting to the weaker status.
    """
    if not validation.agrees:
        return VerificationStatus(
            design_id=design_id, status=SELF_FEM_ONLY,
            reason=(f"{validation.solver} disagreed: mean velocity error "
                    f"{validation.mean_velocity_relative_error:.3e}, profile "
                    f"error {validation.profile_relative_error:.3e}, against a "
                    f"tolerance of {validation.tolerance:.3e}, on "
                    f"{validation.discretisation}"))
    return VerificationStatus(design_id=design_id, status=CROSS_VALIDATED,
                              cross_validation=validation)


def cross_validated_status(design_id: str, validation: CrossValidation
                           ) -> VerificationStatus:
    """Promote a design to cross validated, or explain why it cannot be.

    Disagreement does not silently downgrade: the reason carries the measured
    error so a caller sees what failed rather than only that something did.
    """
    if not validation.agrees:
        return VerificationStatus(
            design_id=design_id, status=SELF_FEM_ONLY,
            reason=(f"{validation.solver} disagreed: displacement error "
                    f"{validation.displacement_relative_error:.3e}, stress "
                    f"error {validation.stress_relative_error:.3e}, against a "
                    f"tolerance of {validation.tolerance:.3e}"))
    return VerificationStatus(design_id=design_id, status=CROSS_VALIDATED,
                              cross_validation=validation)


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
