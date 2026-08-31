"""Nodes, capability routing across their boundaries, and the Fusion stub.

Two things carry the weight here.

`test_the_fusion_stub_never_returns_a_number` is the honesty gate. A
verification node that returned zeros or an empty report when it cannot run
would put values into the pipeline that no solver produced, and downstream
those are indistinguishable from measurements.

`test_a_capability_runs_over_a_real_mcp_round_trip` is the demonstration that
this is a protocol boundary and not a wrapper: the call goes out through MCP
and the numbers that come back match the in-process ones.
"""

import asyncio

import numpy as np
import pytest

from core.registry import (DEFAULT_REGISTRY, Category, Cost, Fidelity, Method,
                           ProblemContext)
from nodes import (EXTERNALLY_VERIFIED, FUSION_ANALYSES, FUSION_CAPABILITY,
                   FUSION_NODE_NAME, SELF_FEM_ONLY, CapabilityRegistry,
                   CapabilityUnavailable, DuplicateCapability,
                   FusionVerificationReport, FusionVerificationRequest,
                   NodeDescriptor, Transport, UnknownCapability,
                   VerificationStatus, build_roster, evaluate_beam_section,
                   fusion_descriptor, request_external_verification, route,
                   verify)

MCP_AVAILABLE = True
try:  # the node layer is an optional dependency
    from mcp import Client  # noqa: F401

    from nodes.engine_node import build_engine_server
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    MCP_AVAILABLE = False

requires_mcp = pytest.mark.skipif(not MCP_AVAILABLE,
                                  reason="the MCP SDK is not installed")


# --- descriptors -------------------------------------------------------------

def test_an_unavailable_node_must_say_why():
    """An unexplained absence is not something a caller can act on."""
    with pytest.raises(ValueError, match="reason"):
        NodeDescriptor(name="x", transport=Transport.HTTP,
                       address="http://localhost:1", available=False)


def test_a_remote_node_needs_an_address():
    with pytest.raises(ValueError, match="address"):
        NodeDescriptor(name="x", transport=Transport.HTTP)


# --- routing across nodes ----------------------------------------------------

def beam_context(slenderness: float) -> ProblemContext:
    return ProblemContext(geometry="prismatic_beam",
                          representations=("prismatic_beam",),
                          slenderness=slenderness, needs_stress_field=False)


def test_the_roster_carries_unavailable_nodes_rather_than_hiding_them():
    """An unbuilt capability and a blocked one call for different responses."""
    registry = build_roster()
    names = [c.name for c in registry.all()]
    assert FUSION_CAPABILITY in names
    assert FUSION_NODE_NAME in [n.name for n in registry.nodes()]


def test_an_unavailable_node_excludes_its_capabilities_with_a_reason():
    registry = build_roster()
    context = ProblemContext(geometry="brep", representations=("brep",),
                             slenderness=30.0, needs_stress_field=True)
    candidates = registry.query(context, Category.ANALYSIS)
    assert FUSION_CAPABILITY not in candidates.names()
    reason, = candidates.reason(FUSION_CAPABILITY)
    assert "entitlement" in reason


def test_local_and_remote_capabilities_obey_the_same_applicability_rule():
    """One rule, whatever side of a process boundary the method is on.

    Two selection paths would mean the declarations that keep local routing
    honest do not cover remote routing.
    """
    registry = build_roster()
    candidates = registry.query(beam_context(6.0), Category.ANALYSIS)
    # The Phase 14b exclusion still applies once the method is a node capability.
    assert "beam_eb" not in candidates.names()
    assert "slenderness" in candidates.reason("beam_eb")[0]
    assert "beam_timoshenko" in candidates.names()


def test_a_capability_can_fail_for_both_reasons_at_once():
    """Making the node reachable must not leave a still-invalid candidate."""
    registry = CapabilityRegistry()
    method = Method(name="picky", category=Category.ANALYSIS, summary="",
                    inputs=(), outputs=(), fidelity=Fidelity.BEAM,
                    cost=Cost.CHEAP,
                    conditions=(DEFAULT_REGISTRY.get("beam_eb").conditions[0],))
    registry.register(method, fusion_descriptor())
    reasons = registry.query(ProblemContext()).reason("picky")
    assert len(reasons) == 2


def test_two_providers_cannot_share_a_name():
    registry = CapabilityRegistry()
    method = DEFAULT_REGISTRY.get("fem3d")
    registry.register(method, fusion_descriptor(available=True))
    with pytest.raises(DuplicateCapability):
        registry.register(method, fusion_descriptor(available=True))


def test_an_unknown_capability_reports_what_is_known():
    with pytest.raises(UnknownCapability):
        build_roster().get("nothing_like_this")


def test_requiring_an_unavailable_capability_raises_at_the_point_of_use():
    registry = build_roster()
    with pytest.raises(CapabilityUnavailable) as excinfo:
        registry.require(FUSION_CAPABILITY)
    assert excinfo.value.capability == FUSION_CAPABILITY
    assert "entitlement" in excinfo.value.reason


def test_routing_is_deterministic():
    context = beam_context(30.0)
    assert (build_roster().query(context).names()
            == build_roster().query(context).names())


# --- the honesty gate --------------------------------------------------------

def test_the_fusion_stub_never_returns_a_number():
    """The gate. There is no call to this node that yields a result.

    Not a zero, not an empty report, not a partial one. Every one of those
    would travel downstream looking like a measurement from a solver that never
    ran.
    """
    request = FusionVerificationRequest(
        design_id="d1", geometry_step_path="/tmp/x.step",
        material_id="al_7075_t6")
    with pytest.raises(CapabilityUnavailable) as excinfo:
        verify(request)
    assert "Fusion paid entitlement" in excinfo.value.reason
    # And it is not available by any argument the caller can pass.
    for analyses in ((), ("stress",), FUSION_ANALYSES):
        with pytest.raises(CapabilityUnavailable):
            verify(FusionVerificationRequest(
                design_id="d", geometry_step_path="p", material_id="m",
                analyses=analyses))


def test_the_stub_refuses_analyses_it_does_not_contract_to_provide():
    with pytest.raises(ValueError, match="does not contract"):
        FusionVerificationRequest(design_id="d", geometry_step_path="p",
                                  material_id="m", analyses=("telepathy",))


def test_a_design_without_external_verification_is_marked_self_fem_only():
    """The funnel keeps running; it just may not call the design verified."""
    request = FusionVerificationRequest(
        design_id="d1", geometry_step_path="/tmp/x.step", material_id="al")
    status = request_external_verification(request, build_roster())
    assert status.status == SELF_FEM_ONLY
    assert not status.is_externally_verified
    assert "entitlement" in status.reason
    assert status.report is None
    record = status.as_dict()
    assert record["external_source"] is None
    assert record["reason"]


def test_a_status_cannot_claim_verification_it_does_not_have():
    with pytest.raises(ValueError, match="without the report"):
        VerificationStatus(design_id="d", status=EXTERNALLY_VERIFIED)


def test_a_status_cannot_carry_results_it_says_it_does_not_have():
    """Otherwise unverified numbers travel as verified ones."""
    report = FusionVerificationReport(
        design_id="d", source="fusion", max_von_mises_pa=1.0,
        max_deflection_m=1.0, buckling_load_factor=1.0,
        max_temperature_k=300.0, first_natural_frequency_hz=1.0)
    with pytest.raises(ValueError, match="unverified"):
        VerificationStatus(design_id="d", status=SELF_FEM_ONLY,
                           reason="not available", report=report)


def test_a_design_without_verification_must_say_why():
    with pytest.raises(ValueError, match="why"):
        VerificationStatus(design_id="d", status=SELF_FEM_ONLY)


# --- a real MCP round trip ---------------------------------------------------

@requires_mcp
def test_a_capability_runs_over_a_real_mcp_round_trip():
    """Not a wrapper: the call and the result cross the protocol.

    The engine's beam analysis is invoked through an MCP client against an MCP
    server, and the numbers are required to match the in-process call exactly.
    A transport that quietly changed a result would be worse than no transport.
    """
    arguments = dict(width_m=0.05, height_m=0.08, wall_thickness_m=0.003,
                     length_m=0.5, tip_load_n=196.2,
                     youngs_modulus_pa=71.7e9, density_kg_m3=2810.0,
                     yield_strength_pa=503e6, shear_deformation=True)

    async def call():
        server = build_engine_server()
        async with Client(server) as client:
            listing = await client.list_tools()
            names = {t.name for t in listing.tools}
            assert {"list_capabilities", "route", "evaluate_beam_section"} <= names
            result = await client.call_tool("evaluate_beam_section", arguments)
            assert not result.is_error
            return result.structured_content

    over_mcp = asyncio.run(call())
    in_process = evaluate_beam_section(**arguments)

    assert over_mcp["model"] == "timoshenko"
    for key in ("mass_kg", "tip_deflection_m", "max_bending_stress_pa"):
        assert over_mcp[key] == pytest.approx(in_process[key], rel=1e-12)
    assert over_mcp["mass_kg"] > 0.0


@requires_mcp
def test_routing_over_mcp_carries_the_exclusions_too():
    """A caller that only learns what it may use cannot tell why."""
    async def call():
        server = build_engine_server()
        async with Client(server) as client:
            result = await client.call_tool(
                "route", {"geometry": "prismatic_beam", "slenderness": 6.0,
                          "category": "analysis"})
            assert not result.is_error
            return result.structured_content

    payload = asyncio.run(call())
    applicable = {row["name"] for row in payload["applicable"]}
    excluded = {row["name"]: row["reasons"] for row in payload["excluded"]}
    assert "beam_timoshenko" in applicable
    assert "beam_eb" not in applicable
    assert any("slenderness" in reason for reason in excluded["beam_eb"])


def test_route_agrees_in_process_and_over_the_wire():
    """The transport must not change a routing decision."""
    direct = route(geometry="prismatic_beam", slenderness=6.0,
                   category="analysis")
    assert {row["name"] for row in direct["applicable"]} == {
        "fem3d", "beam_timoshenko", "surrogate_screen"}
