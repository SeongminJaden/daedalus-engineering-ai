"""The problem features that applicability conditions are written against.

Every field defaults to None, meaning "not stated". A condition that needs a
feature the caller did not state must fail rather than pass: an unstated
slenderness is not evidence that a beam model applies. `require` exists so
conditions express that without each one re-implementing the check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProblemContext:
    """A declarative description of the problem being routed."""

    # Geometry and kinematics
    geometry: str | None = None            # the problem's natural representation
    # Every representation the problem can be posed in. One physical design
    # task is often reachable by more than one: a bracket can be attacked as a
    # parametric section or as a density field over its design domain, and a
    # selector that had to pick a single representation up front could never
    # escalate between them. When unset, only `geometry` counts.
    representations: tuple[str, ...] | None = None
    slenderness: float | None = None       # length over section height, L/h
    # Materials
    material_class: str | None = None      # isotropic, orthotropic
    # What is being asked for
    objective: str | None = None           # mass, compliance, stress
    has_stress_constraint: bool | None = None
    # Whether the duty involves repeated loading, and whether any member
    # carries compression. Both decide whether a failure mode is even possible,
    # so both gate a method rather than merely informing it.
    has_cyclic_load: bool | None = None
    has_compressive_load: bool | None = None
    needs_stress_field: bool | None = None
    needs_gradients: bool | None = None
    # Scale
    n_design_variables: int | None = None
    n_elements: int | None = None

    def supports(self, representation: str) -> bool:
        """Whether the problem can be posed in this representation.

        Falls back to `geometry` when `representations` is unset, so a context
        that names a single geometry still routes. An unstated geometry fails,
        via `require`.
        """
        if self.representations is not None:
            return representation in self.representations
        return self.require("geometry") == representation

    def require(self, field_name: str) -> Any:
        """Read a field, treating "not stated" as a failed requirement.

        Raises `Unstated`, which `Condition.holds` turns into a failure. The
        alternative, defaulting an unstated feature to something permissive,
        is how a beam model ends up running on a problem nobody characterised.
        """
        value = getattr(self, field_name)
        if value is None:
            raise Unstated(field_name)
        return value


class Unstated(Exception):
    """A condition needed a problem feature that was not stated."""

    def __init__(self, field_name: str):
        super().__init__(f"problem context does not state {field_name!r}")
        self.field_name = field_name
