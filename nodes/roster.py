"""The node roster: every node this system knows about, available or not.

Unavailable nodes are in the roster. That is deliberate. A router that only
knows about reachable nodes cannot distinguish a capability nobody has built
from one that exists and is blocked, and those lead to different decisions:
the first needs implementing, the second needs an entitlement.
"""

from __future__ import annotations

from core.registry import DEFAULT_REGISTRY, MethodRegistry

from .engine_node import engine_descriptor
from .fusion_node import fusion_capability_method, fusion_descriptor
from .reasoning_node import reasoning_capability_method, reasoning_descriptor
from .registry import CapabilityRegistry


def build_roster(methods: MethodRegistry | None = None,
                 fusion_available: bool = False,
                 reasoning_available: bool = False) -> CapabilityRegistry:
    """Assemble the full capability registry across all known nodes.

    `fusion_available` exists so the consuming code can be exercised against
    the available branch without pretending the entitlement is bought. Tests
    use it to check that a real report would be accepted; nothing in the
    shipped path sets it.
    """
    registry = CapabilityRegistry()
    registry.adopt(methods if methods is not None else DEFAULT_REGISTRY,
                   engine_descriptor())
    registry.register(fusion_capability_method(),
                      fusion_descriptor(available=fusion_available))
    registry.register(reasoning_capability_method(),
                      reasoning_descriptor(available=reasoning_available))
    return registry
