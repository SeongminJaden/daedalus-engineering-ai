"""Capability routing across node boundaries.

This extends the Phase 14b method registry rather than replacing it. A
capability is a registered method plus the node that provides it, and the
router applies one rule to all of them: a capability is a candidate when the
method applies to the problem AND the node providing it is available. Whether
the method runs in this process, in a subprocess over stdio, or on another
machine over HTTP does not enter the decision.

That uniformity is the point. Without it there would be one selection path for
local methods and another for remote ones, and the applicability declarations
that keep the local path honest would not cover the remote one.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.registry import (Category, Method, MethodRegistry, ProblemContext)

from .descriptor import CapabilityUnavailable, NodeDescriptor, Transport

IN_PROCESS_NODE = NodeDescriptor(name="daedalus.local",
                                 transport=Transport.IN_PROCESS)


@dataclass(frozen=True)
class Capability:
    """A method, and where it runs."""

    method: Method
    node: NodeDescriptor

    @property
    def name(self) -> str:
        return self.method.name

    @property
    def available(self) -> bool:
        return self.node.available


@dataclass(frozen=True)
class CapabilityExclusion:
    """A capability that was ruled out, and every reason it was."""

    capability: Capability
    failed: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityCandidates:
    """What may run for a problem, and what may not, with reasons."""

    applicable: tuple[Capability, ...]
    excluded: tuple[CapabilityExclusion, ...]

    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.applicable)

    def excluded_names(self) -> tuple[str, ...]:
        return tuple(e.capability.name for e in self.excluded)

    def reason(self, name: str) -> tuple[str, ...]:
        for exclusion in self.excluded:
            if exclusion.capability.name == name:
                return exclusion.failed
        return ()


class DuplicateCapability(ValueError):
    """Two nodes claimed the same capability name."""


class UnknownCapability(KeyError):
    """A capability name that was never registered."""


class CapabilityRegistry:
    """Methods and the nodes that provide them, queried as one collection."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._nodes: dict[str, NodeDescriptor] = {}

    def __len__(self) -> int:
        return len(self._capabilities)

    def __contains__(self, name: object) -> bool:
        return name in self._capabilities

    # --- registration --------------------------------------------------------

    def register(self, method: Method, node: NodeDescriptor) -> Capability:
        if method.name in self._capabilities:
            existing = self._capabilities[method.name].node.name
            raise DuplicateCapability(
                f"{method.name!r} is already provided by node {existing!r}; "
                f"two providers under one name cannot be routed "
                f"deterministically")
        capability = Capability(method=method, node=node)
        self._capabilities[method.name] = capability
        self._nodes.setdefault(node.name, node)
        return capability

    def adopt(self, registry: MethodRegistry,
              node: NodeDescriptor = IN_PROCESS_NODE) -> None:
        """Register every method of an in-process registry as a capability.

        The Phase 14b registry becomes the in-process node's capability set
        unchanged, so nothing about local routing changes when the node layer
        is added.
        """
        for method in registry.all():
            self.register(method, node)

    def nodes(self) -> tuple[NodeDescriptor, ...]:
        return tuple(self._nodes[k] for k in sorted(self._nodes))

    def node(self, name: str) -> NodeDescriptor:
        try:
            return self._nodes[name]
        except KeyError:
            raise UnknownCapability(
                f"no node named {name!r}. Known: "
                f"{', '.join(sorted(self._nodes))}") from None

    def all(self) -> tuple[Capability, ...]:
        """Every capability, ordered by name so iteration is deterministic."""
        return tuple(self._capabilities[k] for k in sorted(self._capabilities))

    def get(self, name: str) -> Capability:
        try:
            return self._capabilities[name]
        except KeyError:
            raise UnknownCapability(
                f"{name!r} is not registered. Known: "
                f"{', '.join(sorted(self._capabilities))}") from None

    # --- routing -------------------------------------------------------------

    def query(self, context: ProblemContext,
              category: Category | None = None) -> CapabilityCandidates:
        """Split capabilities into what can serve this problem and what cannot.

        A capability fails for either reason, and both are reported. A method
        that is out of range and hosted on an unreachable node lists both, so
        making the node reachable does not silently produce a still-invalid
        candidate.
        """
        applicable: list[Capability] = []
        excluded: list[CapabilityExclusion] = []
        for capability in self.all():
            if category is not None and capability.method.category is not category:
                continue
            failed = list(capability.method.applicability(context).failed)
            if not capability.node.available:
                failed.append(
                    f"node {capability.node.name!r} is unavailable: "
                    f"{capability.node.unavailable_reason}")
            if failed:
                excluded.append(CapabilityExclusion(capability=capability,
                                                    failed=tuple(failed)))
            else:
                applicable.append(capability)
        applicable.sort(key=lambda c: (-int(c.method.fidelity),
                                       int(c.method.cost), c.name))
        return CapabilityCandidates(applicable=tuple(applicable),
                                    excluded=tuple(excluded))

    def require(self, name: str) -> Capability:
        """Fetch a capability, refusing if its node cannot serve it.

        Callers that are about to invoke something go through this, so an
        unavailable node produces an exception at the point of use rather than
        a result that came from nowhere.
        """
        capability = self.get(name)
        if not capability.node.available:
            raise CapabilityUnavailable(
                capability=name, node=capability.node.name,
                reason=capability.node.unavailable_reason)
        return capability
