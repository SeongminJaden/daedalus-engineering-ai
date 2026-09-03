"""Dispatching a registry method name to the code that runs it.

This is what closes the gap between routing and execution. Before it, the
selector could recommend a topology method and the loop would carry on running
the parametric one, recording the recommendation as unmet. With it, a
recommendation the loop can dispatch is actually executed and the episode
records the method that genuinely produced the design.

A name with no executor is refused rather than defaulted. Falling back to the
parametric solver would put a design in the log under another method's name.
"""

from __future__ import annotations

from typing import Callable

from . import cad, freeform, parametric, topology
from .outcome import DesignOutcome


class NoExecutor(KeyError):
    """A registry method that this loop has no implementation to run."""


EXECUTORS: dict[str, Callable[..., DesignOutcome]] = {
    parametric.METHOD: parametric.run,
    topology.COMPLIANCE_METHOD:
        lambda op, **kw: topology.run(op, method=topology.COMPLIANCE_METHOD, **kw),
    topology.STRESS_METHOD:
        lambda op, **kw: topology.run(op, method=topology.STRESS_METHOD, **kw),
    cad.METHOD: cad.run,
    freeform.METHOD: freeform.run,
}


def executable_methods() -> frozenset[str]:
    """Registry names this loop can actually run."""
    return frozenset(EXECUTORS)


def execute(method: str, op, **kwargs) -> DesignOutcome:
    """Run a method and return its outcome, tagged with the method that ran.

    The returned outcome's `method` is checked against the requested name. An
    executor that reported a different method would make the episode log wrong
    in exactly the way this layer exists to prevent.
    """
    try:
        runner = EXECUTORS[method]
    except KeyError:
        raise NoExecutor(
            f"no executor for {method!r}; this loop can run "
            f"{', '.join(sorted(EXECUTORS))}") from None
    outcome = runner(op, **kwargs)
    if outcome.method != method:
        raise RuntimeError(
            f"executor for {method!r} reported {outcome.method!r}; the episode "
            f"log would misattribute the design")
    return outcome
