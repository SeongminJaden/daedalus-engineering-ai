"""What a method declares about itself.

The point of the declaration is the **applicability range**. A method library
whose entries do not say when they stop being valid is a library that will
happily hand a reasoner a wrong answer, and the reasoner has no way to know.
Phase 7 is the worked example: a cheap Euler-Bernoulli model was used on a link
of slenderness about 6, it omitted shear, and the resulting optimum failed the
3D FEM gate. Nothing in the code was broken. The model was used outside its
range, and no part of the system was in a position to say so.

Declaring the range as data means the selector excludes such a method before it
runs, and can say which condition excluded it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable

from .context import ProblemContext, Unstated


class Category(str, Enum):
    """What kind of job a method does."""

    DESIGN_GENERATION = "design_generation"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    SELECTION = "selection"


class Fidelity(IntEnum):
    """Relative faithfulness to the physics. Ordered, and only ordered.

    These rank methods against each other; they are not error bars. A higher
    number does not promise a smaller error on any particular problem, only
    that the model carries more of the physics. What the numbers actually cost
    and buy is measured per method, not inferred from this scale.
    """

    ANALYTICAL = 1
    BEAM = 2
    TIMOSHENKO = 3
    FEM3D = 4


class Cost(IntEnum):
    """Relative expense, on the same footing as Fidelity: ordered, not absolute."""

    TRIVIAL = 1
    CHEAP = 2
    MODERATE = 3
    HEAVY = 4


@dataclass(frozen=True)
class Condition:
    """One named applicability requirement.

    The description is not decoration. When a method is excluded the selector
    reports this text, so an exclusion is explainable rather than a silent
    absence from a candidate list.
    """

    description: str
    predicate: Callable[[ProblemContext], bool]

    def evaluate(self, context: ProblemContext) -> str | None:
        """None if the requirement is met, otherwise why it was not.

        A condition that asks for a feature the caller never stated fails. It
        does not pass by default and it does not raise: an uncharacterised
        problem is exactly the case where a method must not be assumed valid.
        That failure reports differently from a stated value that is out of
        range, because they call for different fixes: one needs the problem
        characterised, the other needs a different method.
        """
        try:
            return None if self.predicate(context) else self.description
        except Unstated as unstated:
            return (f"{self.description} (unknown: the problem "
                    f"does not state {unstated.field_name!r})")

    def holds(self, context: ProblemContext) -> bool:
        return self.evaluate(context) is None


@dataclass(frozen=True)
class Applicability:
    """Whether a method may run, and if not, which condition stopped it."""

    applicable: bool
    failed: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.applicable


@dataclass(frozen=True)
class Method:
    """A registered method and everything the selector routes on."""

    name: str
    category: Category
    summary: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    fidelity: Fidelity
    cost: Cost
    conditions: tuple[Condition, ...] = ()
    # Where the method actually lives, so a registry entry can be traced to
    # code rather than being a claim about code.
    implementation: str = ""
    evidence: str = ""
    notes: str = ""

    def applicability(self, context: ProblemContext) -> Applicability:
        failed = tuple(reason for reason in
                       (c.evaluate(context) for c in self.conditions)
                       if reason is not None)
        return Applicability(applicable=not failed, failed=failed)

    def applies_to(self, context: ProblemContext) -> bool:
        return self.applicability(context).applicable
