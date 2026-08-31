"""brain.skills.skills - reusable procedures. Stub.

A *strategy* says which move to make ("push height when deflection binds"). A
*skill* is the procedure that carries a move out - a parameterized recipe the
agent can invoke, with preconditions and expected effects.

Phase 5 ships the shape only. Nothing here is populated from data yet, and
deliberately so: inventing skills before there are enough episodes to earn them
would put unsupported procedures in a store whose whole point is that
everything in it carries evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Skill:
    """A named, reusable procedure. Shape only in this phase."""

    name: str
    description: str
    preconditions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    expected_effect: str = ""

    def is_applicable(self, context: dict) -> bool:
        """Every precondition must be a truthy key in the context."""
        return all(bool(context.get(p)) for p in self.preconditions)


class SkillLibrary:
    """In-memory for now; persists alongside strategies when it earns rows."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> Skill:
        self._skills[skill.name] = skill
        return skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def applicable(self, context: dict) -> list[Skill]:
        return [s for s in self._skills.values() if s.is_applicable(context)]

    def __len__(self) -> int:
        return len(self._skills)
