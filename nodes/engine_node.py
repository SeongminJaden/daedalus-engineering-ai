"""The Daedalus engine exposed as an MCP node.

The capabilities here are the ones another node most needs: what this engine
can do for a given problem, which method it would route to and what it ruled
out, and one real analysis so the transport is carrying engineering results
rather than a health check.

`mcp` is an optional dependency. Importing this module without it raises with
an instruction rather than failing obscurely, and nothing in the analysis stack
imports it, so a user who never talks to another node never installs it.
"""

from __future__ import annotations

from typing import Any

from core.registry import DEFAULT_REGISTRY, Category, ProblemContext

from .descriptor import NodeDescriptor, Transport
from .registry import CapabilityRegistry

ENGINE_NODE_NAME = "daedalus.engine"

_MISSING_MCP = (
    "the MCP SDK is not installed. The node layer is optional:\n"
    "  env -u PYTHONPATH .venv/bin/pip install -r requirements-nodes.txt")


def require_mcp():
    """Import the MCP server class, or explain how to get it."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ModuleNotFoundError as missing:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(_MISSING_MCP) from missing
    return MCPServer


def _context_from(geometry: str | None, slenderness: float | None,
                  needs_stress_field: bool, has_stress_constraint: bool | None,
                  ) -> ProblemContext:
    return ProblemContext(
        geometry=geometry,
        representations=(geometry,) if geometry else None,
        slenderness=slenderness,
        needs_stress_field=needs_stress_field,
        has_stress_constraint=has_stress_constraint)


def route(geometry: str | None = None, slenderness: float | None = None,
          category: str | None = None, needs_stress_field: bool = False,
          has_stress_constraint: bool | None = None,
          registry: CapabilityRegistry | None = None) -> dict[str, Any]:
    """Which capabilities serve this problem, and why the rest do not.

    This is the routing decision itself, made available to other nodes. The
    exclusions travel with it: a caller that only learns what it may use cannot
    tell the difference between a method that does not exist and one that was
    ruled out, and those call for different responses.
    """
    if registry is None:
        registry = default_capability_registry()
    chosen = Category(category) if category else None
    candidates = registry.query(
        _context_from(geometry, slenderness, needs_stress_field,
                      has_stress_constraint), chosen)
    return {
        "applicable": [
            {"name": c.name, "node": c.node.name,
             "transport": c.node.transport.value,
             "fidelity": c.method.fidelity.name.lower(),
             "cost": c.method.cost.name.lower(),
             "summary": c.method.summary}
            for c in candidates.applicable],
        "excluded": [
            {"name": e.capability.name, "node": e.capability.node.name,
             "reasons": list(e.failed)}
            for e in candidates.excluded],
    }


def evaluate_beam_section(width_m: float, height_m: float,
                          wall_thickness_m: float, length_m: float,
                          tip_load_n: float, youngs_modulus_pa: float,
                          density_kg_m3: float, yield_strength_pa: float,
                          shear_deformation: bool = True) -> dict[str, float]:
    """Evaluate one hollow rectangular cantilever section.

    Real physics on the real kernel, not a stand-in: this is the same code path
    the local stack uses, reached through the node interface.

    The result carries `model` so the caller knows which beam theory produced
    it. A deflection from the Euler-Bernoulli path and one from the Timoshenko
    path are different numbers, and a consumer that cannot tell them apart
    would be comparing across models without knowing it. That is why the return
    type is not a plain mapping of floats: the metrics travel with the name of
    the model that produced them, and the MCP schema has to allow that.
    """
    import numpy as np

    from physics.structural.beam import (SHEAR_WEB_FACTOR, BeamLoadCase,
                                         evaluate_beam_case)

    case = BeamLoadCase(
        length_m=length_m, tip_load_n=tip_load_n,
        youngs_modulus_pa=youngs_modulus_pa, density_kg_m3=density_kg_m3,
        yield_strength_pa=yield_strength_pa,
        shear_factor=SHEAR_WEB_FACTOR if shear_deformation else 0.0)
    metrics = evaluate_beam_case(np.array([width_m]), np.array([height_m]),
                                 np.array([wall_thickness_m]), case)
    result = metrics.candidate(0)
    result["model"] = "timoshenko" if shear_deformation else "euler_bernoulli"
    result["slenderness"] = length_m / height_m if height_m else float("nan")
    return result


def default_capability_registry() -> CapabilityRegistry:
    """The in-process methods, registered as this node's capabilities."""
    registry = CapabilityRegistry()
    registry.adopt(DEFAULT_REGISTRY)
    return registry


def engine_descriptor(transport: Transport = Transport.IN_PROCESS,
                      address: str = "") -> NodeDescriptor:
    return NodeDescriptor(name=ENGINE_NODE_NAME, transport=transport,
                          address=address)


def build_engine_server(registry: CapabilityRegistry | None = None):
    """An MCP server exposing this engine's capabilities.

    Returned rather than run, so the same object serves a test over an
    in-memory transport and a deployment over stdio or HTTP. The transport is
    chosen by the caller; nothing in the tools depends on it.
    """
    MCPServer = require_mcp()
    registry = registry if registry is not None else default_capability_registry()
    server = MCPServer(name=ENGINE_NODE_NAME,
                       instructions="Daedalus engineering engine: capability "
                                    "routing and structural analysis.")

    @server.tool(name="list_capabilities",
                 description="Every capability this node provides.")
    def list_capabilities() -> list[dict[str, Any]]:
        return [
            {"name": c.name, "category": c.method.category.value,
             "fidelity": c.method.fidelity.name.lower(),
             "cost": c.method.cost.name.lower(),
             "summary": c.method.summary,
             "available": c.available}
            for c in registry.all()]

    @server.tool(name="route",
                 description="Which capabilities serve a problem, and why the "
                             "others were excluded.")
    def route_tool(geometry: str | None = None,
                   slenderness: float | None = None,
                   category: str | None = None,
                   needs_stress_field: bool = False,
                   has_stress_constraint: bool | None = None
                   ) -> dict[str, Any]:
        return route(geometry, slenderness, category, needs_stress_field,
                     has_stress_constraint, registry=registry)

    @server.tool(name="evaluate_beam_section",
                 description="Evaluate a hollow rectangular cantilever "
                             "section: mass, stress, deflection, safety factor.")
    def evaluate_tool(width_m: float, height_m: float, wall_thickness_m: float,
                      length_m: float, tip_load_n: float,
                      youngs_modulus_pa: float, density_kg_m3: float,
                      yield_strength_pa: float,
                      shear_deformation: bool = True) -> dict[str, Any]:
        return evaluate_beam_section(
            width_m, height_m, wall_thickness_m, length_m, tip_load_n,
            youngs_modulus_pa, density_kg_m3, yield_strength_pa,
            shear_deformation)

    return server
