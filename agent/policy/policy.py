"""The policy seam, and the validation every proposal passes through."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from core.engineering_ir.schema import (BoundaryCondition, Constraints,
                                        EngineeringProblem, Geometry, Load,
                                        LoadApplication, LoadType, Objective,
                                        ObjectiveQuantity, ObjectiveSense,
                                        SectionType, Vec3)
from core.materials import load_materials
from core.registry import DEFAULT_REGISTRY

#: A goal sentence longer than this is not parsed. A policy that accepts
#: unbounded input is a policy that can be steered by a wall of text.
MAX_GOAL_CHARACTERS = 2000

#: Physical bounds a proposal must sit inside. These are not opinions about
#: good design; they are the range in which this project's models were
#: measured, and a proposal outside them is extrapolation nobody checked.
BOUNDS = {
    "length_m": (0.01, 5.0),
    "max_width_m": (0.002, 2.0),
    "max_height_m": (0.002, 2.0),
    "magnitude_n": (0.0, 1e6),
    "max_deflection_m": (1e-6, 1.0),
    "max_stress_pa": (1e5, 5e9),
}


class InvalidProposal(ValueError):
    """A proposal that did not survive validation."""


@dataclass
class PolicyProposal:
    """What a policy returns: a proposal, its reasoning, and its origin.

    `verified` is always False. A proposal becomes a design when a solver
    labels it, and this dataclass exists so that the distinction is carried in
    the type rather than in a comment.
    """

    problem: EngineeringProblem
    rationale: str
    origin: str
    verified: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class StrategyChoice:
    method: str
    rationale: str
    origin: str


@dataclass
class RetrySuggestion:
    """What to change after a failure. Advice, never an override."""

    action: str
    rationale: str
    origin: str
    parameters: dict = field(default_factory=dict)


class Policy(Protocol):
    """What the loop may ask a policy for."""

    def propose_problem(self, goal: str) -> PolicyProposal: ...

    def choose_strategy(self, problem: EngineeringProblem,
                        available: set[str]) -> StrategyChoice: ...

    def suggest_retry(self, failure: str,
                      problem: EngineeringProblem) -> RetrySuggestion: ...


# ------------------------------------------------------------- validation

def validate_problem(problem: EngineeringProblem) -> list[str]:
    """Refuse a proposal this project cannot honestly work on.

    Returns warnings for things that are odd but allowed. Raises for things
    that are wrong: a material that is not in the database, a dimension
    outside the range the models were measured in, a load with no direction, a
    limit that no part can meet.
    """
    warnings: list[str] = []
    known = {m.id for m in load_materials().materials}
    if problem.material_id not in known:
        raise InvalidProposal(
            f"material {problem.material_id!r} is not in the database; the "
            f"policy may not invent one. Known materials: "
            f"{', '.join(sorted(known))}")

    geometry = problem.geometry
    for name in ("length_m", "max_width_m", "max_height_m"):
        value = getattr(geometry, name)
        if value is None:
            continue
        low, high = BOUNDS[name]
        if not low <= value <= high:
            raise InvalidProposal(
                f"{name} = {value} is outside the range this project's models "
                f"were measured in ({low} to {high} m)")

    for index, load in enumerate(problem.loads):
        low, high = BOUNDS["magnitude_n"]
        if not low <= load.magnitude_n <= high:
            raise InvalidProposal(
                f"load {index} magnitude {load.magnitude_n} N is outside "
                f"{low} to {high} N")
        direction = (load.direction.x, load.direction.y, load.direction.z)
        if not any(direction):
            raise InvalidProposal(
                f"load {index} has no direction; a magnitude without a "
                f"direction is not a load")
        if load.magnitude_n == 0.0:
            warnings.append(f"load {index} is zero, which checks nothing")

    constraints = problem.constraints
    for name in ("max_deflection_m", "max_stress_pa"):
        value = getattr(constraints, name)
        if value is None:
            continue
        low, high = BOUNDS[name]
        if not low <= value <= high:
            raise InvalidProposal(
                f"{name} = {value} is outside {low} to {high}")
    if constraints.max_deflection_m is not None and geometry.length_m:
        ratio = constraints.max_deflection_m / geometry.length_m
        if ratio > 0.5:
            warnings.append(
                f"the deflection limit is {ratio:.0%} of the length, which "
                f"constrains nothing in practice")
    if not problem.objectives:
        raise InvalidProposal("a problem with no objective is not a problem")
    if (constraints.max_stress_pa is None
            and constraints.min_safety_factor is None):
        raise InvalidProposal(
            "the problem constrains neither a stress limit nor a safety "
            "factor, so minimising mass is unbounded and the optimisation "
            "layer refuses it. This is caught here rather than three calls "
            "later, and the policy may not pick a safety factor for you")
    return warnings


def validate_strategy(method: str, available: set[str]) -> None:
    if method not in available:
        raise InvalidProposal(
            f"strategy {method!r} is not executable here; the loop can run "
            f"{', '.join(sorted(available))}")
    if DEFAULT_REGISTRY.get(method) is None:
        raise InvalidProposal(f"strategy {method!r} is not in the registry")


# ------------------------------------------------------- the rule policy

#: What a sentence may say, and the unit it says it in.
_NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
PATTERNS = {
    "length_mm": re.compile(_NUMBER + r"\s*mm\s+long", re.I),
    "length_m": re.compile(_NUMBER + r"\s*m\s+long", re.I),
    "load_n": re.compile(_NUMBER + r"\s*N\b", re.I),
    "load_kg": re.compile(_NUMBER + r"\s*kg\b", re.I),
    "deflection_mm": re.compile(r"(?:deflect\w*|deflection)[^0-9]{0,20}"
                                + _NUMBER + r"\s*mm", re.I),
    "width_mm": re.compile(_NUMBER + r"\s*mm\s+wide", re.I),
    "height_mm": re.compile(_NUMBER + r"\s*mm\s+(?:tall|high)", re.I),
    "safety_factor": re.compile(r"(?:safety factor|factor of safety|\bSF\b)"
                                r"[^0-9]{0,10}" + _NUMBER, re.I),
    "stress_mpa": re.compile(r"(?:stress|allowable)[^0-9]{0,20}" + _NUMBER
                             + r"\s*MPa", re.I),
}


class RuleBasedPolicy:
    """A deterministic policy. No network, no model, and it says so.

    It reads the quantities the IR needs out of a sentence with named
    patterns, and refuses when a required one is missing rather than choosing
    a default that the caller would then own without knowing it.
    """

    origin = "rule_based_policy"

    def __init__(self, default_material: str = "al_7075_t6") -> None:
        self.default_material = default_material

    def propose_problem(self, goal: str) -> PolicyProposal:
        if len(goal) > MAX_GOAL_CHARACTERS:
            raise InvalidProposal(
                f"the goal is {len(goal)} characters; this policy reads at "
                f"most {MAX_GOAL_CHARACTERS}")
        text = goal.strip()
        if not text:
            raise InvalidProposal("an empty goal states no problem")

        length_m = None
        if (match := PATTERNS["length_mm"].search(text)):
            length_m = float(match.group(1)) / 1000.0
        elif (match := PATTERNS["length_m"].search(text)):
            length_m = float(match.group(1))
        if length_m is None:
            raise InvalidProposal(
                "the goal does not state a length; this policy will not "
                "choose one for you")

        load_n = None
        if (match := PATTERNS["load_n"].search(text)):
            load_n = float(match.group(1))
        elif (match := PATTERNS["load_kg"].search(text)):
            load_n = float(match.group(1)) * 9.81
        if load_n is None:
            raise InvalidProposal("the goal does not state a load")

        deflection = None
        if (match := PATTERNS["deflection_mm"].search(text)):
            deflection = float(match.group(1)) / 1000.0

        safety_factor = None
        if (match := PATTERNS["safety_factor"].search(text)):
            safety_factor = float(match.group(1))
        stress_limit = None
        if (match := PATTERNS["stress_mpa"].search(text)):
            stress_limit = float(match.group(1)) * 1e6
        if safety_factor is None and stress_limit is None:
            raise InvalidProposal(
                "the goal states neither a safety factor nor a stress limit; "
                "without one, minimising mass is unbounded, and this policy "
                "will not choose a safety factor on your behalf")

        material = self.default_material
        known = {m.id for m in load_materials().materials}
        for candidate in known:
            if candidate.replace("_", " ") in text.lower() or candidate in text:
                material = candidate
                break

        width = height = None
        if (match := PATTERNS["width_mm"].search(text)):
            width = float(match.group(1)) / 1000.0
        if (match := PATTERNS["height_mm"].search(text)):
            height = float(match.group(1)) / 1000.0

        problem = EngineeringProblem(
            name="policy_proposal",
            geometry=Geometry(length_m=length_m, max_width_m=width,
                              max_height_m=height,
                              section_type=SectionType.HOLLOW_RECTANGLE),
            material_id=material,
            loads=[Load(type=LoadType.POINT_FORCE, magnitude_n=load_n,
                        direction=Vec3(x=0.0, y=-1.0, z=0.0),
                        application=LoadApplication.TIP)],
            boundary_conditions=[BoundaryCondition()],
            constraints=Constraints(max_deflection_m=deflection,
                                    max_stress_pa=stress_limit,
                                    min_safety_factor=safety_factor),
            objectives=[Objective(sense=ObjectiveSense.MINIMIZE,
                                  quantity=ObjectiveQuantity.MASS)])
        warnings = validate_problem(problem)
        rationale = (f"read a length of {length_m} m and a load of {load_n} N "
                     f"from the goal"
                     + (f", a deflection limit of {deflection} m" if deflection
                        else ", and no deflection limit was stated")
                     + (f", a safety factor of {safety_factor}"
                        if safety_factor else "")
                     + (f", a stress limit of {stress_limit / 1e6:.0f} MPa"
                        if stress_limit else "")
                     + f"; material {material}"
                     + (" (the default, because the goal named none)"
                        if material == self.default_material else " (named in the goal)"))
        return PolicyProposal(problem=problem, rationale=rationale,
                              origin=self.origin, warnings=warnings)

    def choose_strategy(self, problem: EngineeringProblem,
                        available: set[str]) -> StrategyChoice:
        """Prefer the cheapest strategy that can express the problem.

        A stated envelope is what the topology and free form strategies need;
        without one they cannot run at all, which is a fact about the problem
        rather than a preference.
        """
        has_envelope = (problem.geometry.max_height_m is not None
                        and problem.geometry.max_width_m is not None)
        if not has_envelope:
            method = "parametric_section"
            why = ("no design envelope is stated, so only the parametric "
                   "section can run")
        elif problem.constraints.max_deflection_m is not None:
            method = "generative_cad" if "generative_cad" in available else "parametric_section"
            why = ("a deflection limit with an envelope: the family search "
                   "returns the lightest solver verified part under it")
        else:
            method = ("freeform_topology" if "freeform_topology" in available
                      else "topology_compliance")
            why = ("an envelope with no deflection limit is a stiffness "
                   "problem at a volume, which is what topology answers")
        validate_strategy(method, available)
        return StrategyChoice(method=method, rationale=why, origin=self.origin)

    def suggest_retry(self, failure: str,
                      problem: EngineeringProblem) -> RetrySuggestion:
        """Map a failure to a direction, from the failures this project has
        actually produced."""
        text = failure.lower()
        if "returned nothing" in text or "jacobian" in text:
            return RetrySuggestion(
                action="refine_mesh",
                rationale=("the solver rejected the mesh, which on curved "
                           "parts is a nonpositive Jacobian; the measured "
                           "remedy is a finer mesh, not a different element"),
                origin=self.origin, parameters={"factor": 0.7})
        if "disconnected" in text or "load path is cut" in text:
            return RetrySuggestion(
                action="raise_volume_fraction",
                rationale=("the extracted structure lost its load path, which "
                           "the volume fraction search bottoms out on"),
                origin=self.origin, parameters={"increment": 0.05})
        if "no admissible candidate" in text or "envelope" in text:
            return RetrySuggestion(
                action="widen_envelope",
                rationale="no candidate fits the stated envelope",
                origin=self.origin)
        return RetrySuggestion(
            action="stop",
            rationale=(f"this policy has no measured remedy for {failure!r}; "
                       f"guessing one would waste solver time"),
            origin=self.origin)


# -------------------------------------------------- the language model seam

class LanguageModelPolicy:
    """A policy backed by a text completion callable.

    The callable is supplied by the caller, so no key, endpoint or vendor
    appears here. The model is asked for JSON and its answer goes through the
    same `validate_problem` as everything else; a model that returns prose, a
    material that does not exist, or a length of a kilometre is refused with
    the reason. There is no repair step: a proposal that does not validate is
    not silently fixed into one that does.
    """

    origin = "language_model_policy"

    PROMPT = (
        "Return ONLY a JSON object with these keys and SI units: "
        "length_m, max_width_m, max_height_m, material_id, load_n, "
        "load_direction (three numbers), max_deflection_m (or null), "
        "and one of max_stress_pa or min_safety_factor. "
        "Do not add commentary. The goal is: {goal}")

    def __init__(self, complete: Callable[[str], str] | None = None,
                 fallback: Policy | None = None) -> None:
        if complete is None:
            raise InvalidProposal(
                "a language model policy needs a completion callable; without "
                "one this class would be a rule based policy wearing the "
                "wrong name")
        self.complete = complete
        self.fallback = fallback or RuleBasedPolicy()

    def propose_problem(self, goal: str) -> PolicyProposal:
        raw = self.complete(self.PROMPT.format(goal=goal))
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise InvalidProposal(
                f"the model did not return JSON: {str(exc)[:80]}") from None
        if not isinstance(data, dict):
            raise InvalidProposal("the model returned JSON that is not an object")
        required = ("length_m", "material_id", "load_n")
        missing = [key for key in required if key not in data]
        if missing:
            raise InvalidProposal(
                f"the model left out {', '.join(missing)}, which the problem "
                f"cannot be built without")
        direction = data.get("load_direction") or [0.0, -1.0, 0.0]
        try:
            problem = EngineeringProblem(
                name=str(data.get("name", "policy_proposal"))[:60],
                geometry=Geometry(length_m=float(data["length_m"]),
                                  max_width_m=(None if data.get("max_width_m") is None
                                               else float(data["max_width_m"])),
                                  max_height_m=(None if data.get("max_height_m") is None
                                                else float(data["max_height_m"]))),
                material_id=str(data["material_id"]),
                loads=[Load(magnitude_n=float(data["load_n"]),
                            direction=Vec3(x=float(direction[0]),
                                           y=float(direction[1]),
                                           z=float(direction[2])))],
                boundary_conditions=[BoundaryCondition()],
                constraints=Constraints(
                    max_deflection_m=(None if data.get("max_deflection_m") is None
                                      else float(data["max_deflection_m"])),
                    max_stress_pa=(None if data.get("max_stress_pa") is None
                                   else float(data["max_stress_pa"])),
                    min_safety_factor=(None if data.get("min_safety_factor") is None
                                       else float(data["min_safety_factor"]))),
                objectives=[Objective(sense=ObjectiveSense.MINIMIZE,
                                      quantity=ObjectiveQuantity.MASS)])
        except Exception as exc:                      # pydantic or float errors
            raise InvalidProposal(
                f"the model's values do not form a problem: {str(exc)[:200]}") from None
        warnings = validate_problem(problem)
        return PolicyProposal(problem=problem,
                              rationale="proposed by a language model and "
                                        "validated here; not verified",
                              origin=self.origin, warnings=warnings)

    def choose_strategy(self, problem: EngineeringProblem,
                        available: set[str]) -> StrategyChoice:
        raw = self.complete(
            "Return ONLY the name of one strategy from this list: "
            + ", ".join(sorted(available)))
        method = raw.strip().strip('"').split()[0] if raw.strip() else ""
        validate_strategy(method, available)
        return StrategyChoice(method=method,
                              rationale="chosen by a language model, checked "
                                        "against the executable set",
                              origin=self.origin)

    def suggest_retry(self, failure: str,
                      problem: EngineeringProblem) -> RetrySuggestion:
        """Deliberately not the model's job.

        A retry direction is a statement about which failures this project has
        measured a remedy for, and a model has no access to that. It is
        delegated to the rule policy and the origin says so.
        """
        return self.fallback.suggest_retry(failure, problem)
