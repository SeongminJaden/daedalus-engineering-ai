"""The reasoning node: an interface, filled today by a Claude session.

There is no server here on purpose. The reasoning that drives this project
currently comes from a language model session operating the CLI, which is a
node in the architectural sense (it consumes capabilities and decides what to
do next) without being one in the protocol sense. Writing a server that wraps a
session would be inventing a boundary that nothing is on the other side of.

What is declared is the contract, so that replacing the session with a local
model is a matter of providing this capability rather than redesigning the
routing. The Phase 14b selector is the seam on the inside; this is the same
seam drawn at the node boundary.
"""

from __future__ import annotations

from core.registry import Category, Cost, Fidelity, Method

from .descriptor import NodeDescriptor, Transport

REASONING_NODE_NAME = "reasoning.external"
REASONING_CAPABILITY = "reasoning.llm"

REASONING_UNAVAILABLE_REASON = (
    "unavailable as a protocol node: reasoning is currently supplied by an "
    "operator session driving the CLI, not by a served capability")


def reasoning_descriptor(available: bool = False,
                         transport: Transport = Transport.STDIO,
                         address: str = "operator-session") -> NodeDescriptor:
    return NodeDescriptor(
        name=REASONING_NODE_NAME, transport=transport, address=address,
        available=available,
        unavailable_reason="" if available else REASONING_UNAVAILABLE_REASON)


def reasoning_capability_method() -> Method:
    """The declaration a local model would have to satisfy to take this over."""
    return Method(
        name=REASONING_CAPABILITY,
        category=Category.SELECTION,
        summary="Propose the next design action given the run state.",
        inputs=("run_state", "episode_history", "capability_listing"),
        outputs=("action", "hypothesis"),
        fidelity=Fidelity.ANALYTICAL,
        cost=Cost.MODERATE,
        conditions=(),
        implementation="nodes.reasoning_node (contract only)",
        evidence="UNVERIFIED",
        notes="No served implementation. The heuristic selector in "
              "agent.strategy covers this in process and does not need a "
              "node; a served model would replace the ordering rule while "
              "routing over the same registry.")
