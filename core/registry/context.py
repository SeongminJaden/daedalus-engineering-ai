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
    # Whether a member transmits torque while rotating, and whether a rotating
    # support carries load. Both decide whether a drivetrain check exists at
    # all, so both gate a method.
    transmits_torque: bool | None = None
    has_rotating_support: bool | None = None
    # Whether the problem states a duty cycle to heat a motor with, and whether
    # the part sees a temperature change. Neither check exists without them.
    has_duty_cycle: bool | None = None
    has_temperature_change: bool | None = None
    # Whether the assembly has a preloaded bolted connection, and whether a
    # gear mesh transmits torque. Neither check exists without them.
    has_bolted_joint: bool | None = None
    has_gear_mesh: bool | None = None
    # How many objectives the problem states. A single objective does not need
    # a Pareto front, and saying so is what keeps the expensive method off a
    # problem that has no trade-off in it.
    n_objectives: int | None = None
    # Whether any design variable is discrete or categorical. This gates the
    # continuous operators rather than letting them produce a fractional
    # material and round it somewhere out of sight.
    has_discrete_variables: bool | None = None
    # Whether the part is a laminate with a stated stacking sequence. An
    # orthotropic material alone is not enough: CLT designs the STACK, and a
    # single ply or an isotropic part has no stack to design.
    has_layup: bool | None = None
    # Shaft-to-hub torque transfer (a key, spline or interference fit), a
    # welded joint, and whether the design needs dimensional tolerances. Each
    # gates a Phase 23 method.
    has_shaft_hub_connection: bool | None = None
    has_welded_joint: bool | None = None
    requires_tolerances: bool | None = None
    # A multiaxial stress state to resolve, a non-circular section in torsion,
    # a pressurised vessel, and concentrated contact between curved bodies.
    has_multiaxial_stress: bool | None = None
    has_noncircular_torsion: bool | None = None
    has_internal_pressure: bool | None = None
    has_concentrated_contact: bool | None = None
    # A heat path to build as a network, and a transient thermal question.
    has_heat_path: bool | None = None
    has_thermal_transient: bool | None = None
    # Fluid carried through a conduit, a body moving through a fluid, and
    # fluid power actuation.
    # Whether the problem is an articulated chain of rigid links. The
    # multibody methods exist only for a mechanism; a single bracket has no
    # mass matrix, and routing one there would return a 0 by 0 answer rather
    # than declining.
    has_articulated_chain: bool | None = None
    has_internal_flow: bool | None = None
    # Whether the internal flow is laminar. The closed-form correlations cover
    # both regimes, but a CFD run configured with a laminar closure covers only
    # one, and running it on a turbulent duct would return a confident wrong
    # answer rather than failing. So the regime gates that method.
    flow_is_laminar: bool | None = None
    has_external_flow: bool | None = None
    has_fluid_actuator: bool | None = None
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
