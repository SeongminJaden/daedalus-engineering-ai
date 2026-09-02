"""Design intent as a claim that a solver can check, not a sentence to store.

A designer says a feature is FOR something: the wall thickness is what makes
the link stiff, the holes are clearance and carry nothing, the length is set
by the reach and everything else follows. Geometry alone cannot recover any
of that; the roadmap says so, and it is why a feature recogniser reports what
a shape IS and never what it is FOR. This module takes the opposite road. It
lets intent be STATED, with provenance, as a `DesignReference` item, and then
turns each statement into an ablation the real solver runs: change the one
thing the claim is about, label both parts through Gmsh and CalculiX, and see
whether the quantity moved the way the claim said.

WHAT AN ABLATION CAN AND CANNOT SAY
===================================
    SUPPORTED     the quantity moved in the claimed direction by more than
                  the mesh noise, and by the claimed amount where one was
                  given, or by at least the amount the claim calls
                  meaningful where none was.
    REFUTED       it moved the other way, or by an amount outside the
                  claimed one, again by more than the noise.
    INCONCLUSIVE  the effect is inside the mesh sensitivity of the labels
                  themselves. Two solves cannot tell it from noise, and the
                  honest answer is that nothing was learned.

A SUPPORTED claim is one solver run on one part and grades SIMULATED like any
other; it is recorded in the Brain as evidence for the claim, and a REFUTED
one as a counterexample, so that the same claim tested on many parts climbs
or falls by the ordinary rules of the evidence ladder and never past
SIMULATED without independent runs. Nothing here asserts intent. It measures
whether a stated intent survives contact with the solver.

VALIDITY DOMAIN
===============
    One family, one parameter, one load case per claim. A claim about a
    parameter the family does not have, or an ablation that leaves the
    admissible region, is refused rather than approximated. The noise floor
    is the labeller's mesh sensitivity, which is a two-mesh estimate and not
    a bound; an effect just above it is weak evidence and the ratio is
    reported so a reader can judge. Everything SIMULATED.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from brain.semantic.evidence import Counterexample, Evidence, EvidenceKind
from brain.semantic.knowledge import Knowledge, SemanticMemory
from core.design_reference import DesignReference, ReferenceItem
from core.materials.db import MaterialSpec, get_material

from .engine import make_part
from .families import FAMILIES
from .labeller import LoadCase
from .schema import PartRecord


class Direction(str, Enum):
    UP = "up"          # scaling the parameter by the factor raises the quantity
    DOWN = "down"      # lowers it
    NONE = "none"      # leaves it within the tolerance


class Outcome(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


#: An effect has to exceed this many times the larger mesh sensitivity of the
#: two labels before it counts as an effect at all.
NOISE_MARGIN = 2.0


@dataclass(frozen=True)
class IntentClaim(ReferenceItem):
    """What a parameter is for, stated as a prediction the solver can test.

    Inherits the mandatory source, confidence and provenance of every
    reference item, so an intent without an author cannot be constructed,
    and its confidence is capped by where it came from exactly as a
    proportion prior's is.
    """

    family: str = ""
    parameter: str = ""
    role: str = ""                      # the designer's words
    quantity: str = "tip_deflection_m"  # the label the effect is measured on
    direction: Direction = Direction.DOWN
    factor: float = 2.0                 # the ablation scales the parameter by this
    expected_ratio: float | None = None  # quantity_ablated / quantity_base, if claimed
    #: Relative. On an expected ratio: how far off still counts. On NONE: the
    #: largest change that still counts as no effect. On a direction with no
    #: ratio: the SMALLEST change that counts as the claimed role, because a
    #: parameter that moves the quantity by one percent is not "what makes
    #: the link stiff" however real that percent is.
    tolerance: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family not in FAMILIES:
            raise ValueError(f"claim {self.name!r} is about family "
                             f"{self.family!r}, which does not exist")
        fam = FAMILIES[self.family]
        if self.parameter not in fam.bounds:
            raise ValueError(f"claim {self.name!r} is about parameter "
                             f"{self.parameter!r}, which {self.family} does not "
                             f"have; it has {sorted(fam.bounds)}")
        if self.factor <= 0.0 or self.factor == 1.0:
            raise ValueError("the ablation factor must be positive and not 1")
        if not self.role.strip():
            raise ValueError(f"claim {self.name!r} states no role; an intent "
                             f"without a reason is a parameter change")
        if self.expected_ratio is not None and self.expected_ratio <= 0.0:
            raise ValueError("an expected ratio must be positive")
        if not 0.0 < self.tolerance < 1.0:
            raise ValueError("tolerance is relative and must lie in (0, 1)")

    def statement(self) -> str:
        magnitude = (f" by a factor near {self.expected_ratio:g}"
                     if self.expected_ratio is not None else "")
        return (f"{self.family}: {self.parameter} is {self.role}; scaling it by "
                f"{self.factor:g} moves {self.quantity} {self.direction.value}"
                f"{magnitude}")

    def claim_key(self) -> str:
        return (f"intent:{self.family}:{self.parameter}:{self.quantity}:"
                f"{self.direction.value}:{self.factor:g}")


def intent_claims(reference: DesignReference) -> tuple[IntentClaim, ...]:
    return reference.of_type(IntentClaim)          # type: ignore[return-value]


@dataclass(frozen=True)
class AblationResult:
    claim: IntentClaim
    base: PartRecord
    ablated: PartRecord
    base_value: float
    ablated_value: float
    ratio: float
    noise_floor: float          # relative, from the labels' mesh sensitivity
    outcome: Outcome
    reason: str

    @property
    def effect(self) -> float:
        """Relative change, signed."""
        return self.ratio - 1.0


def _label_value(record: PartRecord, quantity: str) -> tuple[float, float]:
    item = record.labels[quantity]
    return abs(float(item["value"])), float(item.get("mesh_sensitivity", 0.0))


def _judge(claim: IntentClaim, ratio: float, noise: float) -> tuple[Outcome, str]:
    effect = ratio - 1.0
    floor = NOISE_MARGIN * noise
    if claim.direction is Direction.NONE:
        if noise > claim.tolerance:
            return Outcome.INCONCLUSIVE, (
                f"mesh sensitivity {noise:.1%} exceeds the {claim.tolerance:.0%} "
                f"tolerance the claim is stated to, so two solves cannot bound "
                f"the effect")
        if abs(effect) <= claim.tolerance:
            return Outcome.SUPPORTED, (
                f"{claim.quantity} changed by {effect:+.1%}, inside the "
                f"{claim.tolerance:.0%} tolerance")
        return Outcome.REFUTED, (
            f"{claim.quantity} changed by {effect:+.1%}, outside the "
            f"{claim.tolerance:.0%} tolerance for a parameter claimed to have "
            f"no effect")
    if abs(effect) <= floor:
        return Outcome.INCONCLUSIVE, (
            f"{claim.quantity} changed by {effect:+.1%}, inside {NOISE_MARGIN:g} "
            f"times the mesh sensitivity of {noise:.1%}; nothing was learned")
    went_up = effect > 0.0
    if went_up != (claim.direction is Direction.UP):
        return Outcome.REFUTED, (
            f"{claim.quantity} went {'up' if went_up else 'down'} by "
            f"{abs(effect):.1%} when the claim said {claim.direction.value}")
    if claim.expected_ratio is not None:
        miss = abs(ratio / claim.expected_ratio - 1.0)
        if miss > claim.tolerance:
            return Outcome.REFUTED, (
                f"direction right, magnitude wrong: ratio {ratio:.3g} against "
                f"the claimed {claim.expected_ratio:g}, off by {miss:.0%} with "
                f"{claim.tolerance:.0%} allowed")
        return Outcome.SUPPORTED, (
            f"ratio {ratio:.3g} against the claimed {claim.expected_ratio:g}, "
            f"off by {miss:.0%}; effect {effect:+.1%} against noise {noise:.1%}")
    if abs(effect) < claim.tolerance:
        return Outcome.REFUTED, (
            f"{claim.quantity} went {claim.direction.value}, but by only "
            f"{abs(effect):.1%}, less than the {claim.tolerance:.0%} the claim "
            f"calls meaningful; the direction is real and the role is not")
    return Outcome.SUPPORTED, (
        f"{claim.quantity} went {claim.direction.value} by {abs(effect):.1%} "
        f"against a mesh sensitivity of {noise:.1%}")


def ablated_parameters(claim: IntentClaim, base: dict[str, float]
                       ) -> dict[str, float]:
    fam = FAMILIES[claim.family]
    changed = dict(base)
    value = base[claim.parameter] * claim.factor
    if claim.parameter in fam.integer_parameters:
        value = float(round(value))
    changed[claim.parameter] = value
    if not fam.admissible(changed):
        raise ValueError(
            f"scaling {claim.parameter} by {claim.factor:g} from "
            f"{base[claim.parameter]:g} leaves {claim.family}'s admissible "
            f"region; choose a base part with room to move")
    return changed


def ablate(claim: IntentClaim, base_params: dict[str, float],
           step_dir: str | Path, material: MaterialSpec | None = None,
           total_load_n: float = -100.0) -> AblationResult:
    """Build and label the base part and the ablated part, and judge."""
    fam = FAMILIES[claim.family]
    material = material or get_material("al_7075_t6")
    case = LoadCase(total_load_n=total_load_n, direction=fam.load_direction)
    changed = ablated_parameters(claim, base_params)
    base, _ = make_part(fam, base_params, Path(step_dir), material, case)
    ablated, _ = make_part(fam, changed, Path(step_dir), material, case)
    base_value, base_noise = _label_value(base, claim.quantity)
    ablated_value, ablated_noise = _label_value(ablated, claim.quantity)
    ratio = ablated_value / base_value if base_value else float("inf")
    noise = max(base_noise, ablated_noise)
    outcome, reason = _judge(claim, ratio, noise)
    return AblationResult(claim=claim, base=base, ablated=ablated,
                          base_value=base_value, ablated_value=ablated_value,
                          ratio=ratio, noise_floor=noise, outcome=outcome,
                          reason=reason)


def record_in_brain(result: AblationResult, memory: SemanticMemory,
                    run_id: str, domain: str = "design_intent") -> Knowledge | None:
    """A supported ablation becomes evidence for the claim, a refuted one a
    counterexample, an inconclusive one nothing. The level is derived by the
    ladder from what accumulates, and one run is SIMULATED at most."""
    if result.outcome is Outcome.INCONCLUSIVE:
        return None
    claim = result.claim
    key = claim.claim_key()
    existing = memory.find_by_claim(key)
    knowledge = existing if existing is not None else Knowledge(
        statement=claim.statement(), domain=domain, source=claim.source,
        claim_key=key,
        assumptions=[f"provenance {claim.provenance.value}, stated confidence "
                     f"{claim.confidence:.2f}",
                     "one cantilever load case, CalculiX C3D10 labels"])
    ref = f"{result.base.part_id} vs {result.ablated.part_id}"
    if result.outcome is Outcome.SUPPORTED:
        knowledge.add_evidence(Evidence(kind=EvidenceKind.SIMULATION, ref=ref,
                                        run_id=run_id, note=result.reason))
    else:
        knowledge.add_counterexample(Counterexample(ref=ref,
                                                    description=result.reason))
    memory.store(knowledge)
    return knowledge
