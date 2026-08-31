"""What a node is, and what it means for one to be unavailable.

A node is somewhere a capability can run: this process, a local subprocess
speaking MCP over stdio, or a remote service speaking MCP over HTTP. The
router does not care which. It cares whether the capability applies to the
problem and whether the node that provides it can actually be reached.

**Unavailability is declared, not discovered by failing.** A node that needs a
paid entitlement nobody has bought yet says so up front, and the router
excludes its capabilities the same way it excludes a beam model outside its
slenderness range: with a stated reason. That is the whole point of routing on
declarations. Finding out by calling and getting an error is worse, and
finding out by getting a plausible number back is worst of all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Transport(str, Enum):
    """How a node is reached. The router treats all three identically."""

    IN_PROCESS = "in_process"   # this process, no protocol involved
    STDIO = "stdio"             # a local subprocess speaking MCP
    HTTP = "http"               # a remote service speaking MCP


@dataclass(frozen=True)
class NodeDescriptor:
    """A node's identity, address and availability."""

    name: str
    transport: Transport
    address: str = ""
    available: bool = True
    # Why not, in words a caller can act on. Required when unavailable, so a
    # node cannot be quietly absent.
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        if not self.available and not self.unavailable_reason:
            raise ValueError(
                f"node {self.name!r} is unavailable but gives no reason; an "
                f"unexplained absence is not something a caller can act on")
        if self.transport is not Transport.IN_PROCESS and not self.address:
            raise ValueError(
                f"node {self.name!r} uses {self.transport.value} and needs an "
                f"address")

    @property
    def is_local(self) -> bool:
        return self.transport in (Transport.IN_PROCESS, Transport.STDIO)


class CapabilityUnavailable(RuntimeError):
    """A capability was invoked on a node that cannot serve it.

    Raised instead of returning anything. A caller that wanted a stress number
    must not receive a value that did not come from a solver, and must not
    receive a zero, an empty result or a default that reads as one.
    """

    def __init__(self, capability: str, node: str, reason: str):
        super().__init__(f"{capability!r} is unavailable on node {node!r}: "
                         f"{reason}")
        self.capability = capability
        self.node = node
        self.reason = reason
