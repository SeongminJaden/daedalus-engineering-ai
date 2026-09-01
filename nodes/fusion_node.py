"""The Fusion verification node: contract and workflow wired, implementation empty.

Fusion is a paid product and the entitlement has not been bought yet. What
exists here is everything except the call itself: the capability declaration,
the request and report contracts, the registration slot, and the path a design
takes to reach this node and come back. Filling it in later means replacing one
function body with an MCP call, and nothing around it has to move.

**The stub never returns a number.** This is the whole reason it is written the
way it is. A verification node that returned zeros, empty fields, or defaults
on unavailability would put values into the pipeline that no solver produced,
and downstream those are indistinguishable from results. It raises
`CapabilityUnavailable` instead, and a test asserts that no call to it can
yield anything numeric. A design that has not been through it stays marked as
verified by our own FEM only, and that mark is what the funnel and the Brain
record.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.registry import Category, Condition, Cost, Fidelity, Method

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

FUSION_NODE_NAME = "fusion.verification"
FUSION_CAPABILITY = "analysis.fea.fusion"

# Stated once, used everywhere it needs explaining.
FUSION_UNAVAILABLE_REASON = (
    "unavailable: requires Fusion paid entitlement, and Fusion does not run "
    "on Linux where this node lives. The parameters to CAD half runs on a "
    "Windows host instead; the parametric backend here is build123d, which "
    "is OpenCASCADE and native. Being a different kernel from Fusion is "
    "useful rather than a compromise: a STEP file from each is a second "
    "opinion on the analyzer reading the other")

# The analyses this node is contracted to provide once it exists. Listed so the
# contract is reviewable now, not so anything can claim to have run them.
FUSION_ANALYSES = ("stress", "deflection", "buckling", "thermal", "modal")


@dataclass(frozen=True)
class FusionVerificationRequest:
    """What a design must supply to be verified externally.

    `geometry_step_path` and not a density field: this node verifies clean
    B-rep geometry, which is also why it is the planned answer to the grey
    designs that density-based topology optimisation produces.
    """

    design_id: str
    geometry_step_path: str
    material_id: str
    load_cases: tuple[dict, ...] = ()
    analyses: tuple[str, ...] = FUSION_ANALYSES

    def __post_init__(self) -> None:
        unknown = set(self.analyses) - set(FUSION_ANALYSES)
        if unknown:
            raise ValueError(
                f"this node does not contract to provide {sorted(unknown)}; "
                f"it provides {list(FUSION_ANALYSES)}")


@dataclass(frozen=True)
class FusionVerificationReport:
    """The shape a real Fusion result will have.

    Defined so the consuming code can be written and reviewed against a fixed
    contract. **Nothing in this module ever constructs one.** It becomes
    reachable when the entitlement is bought and `verify` stops raising.
    """

    design_id: str
    source: str
    max_von_mises_pa: float
    max_deflection_m: float
    buckling_load_factor: float
    max_temperature_k: float
    first_natural_frequency_hz: float
    notes: str = ""


def fusion_descriptor(available: bool = False,
                      address: str = "http://localhost:8931/mcp"
                      ) -> NodeDescriptor:
    """The node as the registry sees it: declared, addressed, and not available.

    It is registered rather than omitted so the router can say the capability
    exists and why it cannot be used. An absent node and a blocked one look the
    same to a caller otherwise, and they are not the same situation.
    """
    return NodeDescriptor(
        name=FUSION_NODE_NAME, transport=Transport.HTTP, address=address,
        available=available,
        unavailable_reason="" if available else FUSION_UNAVAILABLE_REASON)


def fusion_capability_method() -> Method:
    """The capability declaration, in the same schema as every local method."""
    return Method(
        name=FUSION_CAPABILITY,
        category=Category.ANALYSIS,
        summary="External FEA verification of clean B-rep geometry in Fusion.",
        inputs=("geometry_step_path", "material_id", "load_cases"),
        outputs=FUSION_ANALYSES,
        fidelity=Fidelity.FEM3D,
        cost=Cost.HEAVY,
        conditions=(
            Condition("the design has exportable B-rep geometry, not a "
                      "density field",
                      lambda c: c.supports("brep")),
        ),
        implementation="nodes.fusion_node.verify",
        evidence="UNVERIFIED",
        notes="Not implemented. The entitlement is not bought, so this node "
              "raises rather than returning results. It is the planned route "
              "to clean geometry for topology output, which SIMP leaves grey.")


def verify(request: FusionVerificationRequest) -> FusionVerificationReport:
    """Would run the external verification. Raises, because it cannot.

    The return annotation is the contract, and it is deliberately never
    honoured while the node is unavailable. There is no partial result, no
    empty report and no zero-filled placeholder: every one of those would enter
    the pipeline looking like a measurement.
    """
    raise CapabilityUnavailable(
        capability=FUSION_CAPABILITY, node=FUSION_NODE_NAME,
        reason=FUSION_UNAVAILABLE_REASON)
