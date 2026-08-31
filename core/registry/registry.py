"""The registry itself: methods as data, queried by problem.

A selector asks this what it may use. The intelligence of anything built on
top is bounded by the breadth and the accuracy of what is registered here,
which is why methods are declared rather than hardcoded into a switch: adding
a method is adding a row, not editing the reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context import ProblemContext
from .method import Applicability, Category, Cost, Fidelity, Method


class DuplicateMethod(ValueError):
    """Two methods claimed the same name."""


class UnknownMethod(KeyError):
    """A name that was never registered."""


@dataclass(frozen=True)
class Exclusion:
    """A method that was ruled out, and why."""

    method: Method
    failed: tuple[str, ...]


@dataclass(frozen=True)
class Candidates:
    """The result of a query: what may run, and what may not, with reasons.

    The excluded list is returned rather than discarded. A selector that can
    only see what survived cannot explain itself, and an exclusion is often
    the most informative thing about a routing decision.
    """

    applicable: tuple[Method, ...]
    excluded: tuple[Exclusion, ...]

    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.applicable)

    def excluded_names(self) -> tuple[str, ...]:
        return tuple(e.method.name for e in self.excluded)

    def reason(self, name: str) -> tuple[str, ...]:
        """Why `name` was excluded. Empty if it was not."""
        for exclusion in self.excluded:
            if exclusion.method.name == name:
                return exclusion.failed
        return ()


class MethodRegistry:
    """A name-keyed collection of methods, queried by problem context."""

    def __init__(self) -> None:
        self._methods: dict[str, Method] = {}

    def __len__(self) -> int:
        return len(self._methods)

    def __contains__(self, name: object) -> bool:
        return name in self._methods

    def register(self, method: Method) -> Method:
        if method.name in self._methods:
            raise DuplicateMethod(
                f"{method.name!r} is already registered; a registry with two "
                f"methods under one name cannot be routed deterministically")
        self._methods[method.name] = method
        return method

    def get(self, name: str) -> Method:
        try:
            return self._methods[name]
        except KeyError:
            raise UnknownMethod(
                f"{name!r} is not registered. Known: "
                f"{', '.join(sorted(self._methods))}") from None

    def all(self) -> tuple[Method, ...]:
        """Every method, ordered by name so iteration is deterministic."""
        return tuple(self._methods[k] for k in sorted(self._methods))

    def by_category(self, category: Category) -> tuple[Method, ...]:
        return tuple(m for m in self.all() if m.category is category)

    def applicability(self, name: str, context: ProblemContext) -> Applicability:
        return self.get(name).applicability(context)

    def query(self, context: ProblemContext,
              category: Category | None = None) -> Candidates:
        """Split the registry into what applies to this problem and what does not.

        Applicable methods come back ordered by descending fidelity, then
        ascending cost, then name. The name breaks ties so that two methods of
        equal fidelity and cost always come back in the same order: a selector
        that picks the first candidate must not depend on dict ordering.
        """
        pool = self.by_category(category) if category is not None else self.all()
        applicable: list[Method] = []
        excluded: list[Exclusion] = []
        for method in pool:
            verdict = method.applicability(context)
            if verdict.applicable:
                applicable.append(method)
            else:
                excluded.append(Exclusion(method=method, failed=verdict.failed))
        applicable.sort(key=lambda m: (-int(m.fidelity), int(m.cost), m.name))
        return Candidates(applicable=tuple(applicable), excluded=tuple(excluded))

    def cheapest_applicable(self, context: ProblemContext,
                            category: Category | None = None) -> Method | None:
        """The least expensive method that is still valid here.

        This is the routing rule the loop wants most of the time: spend the
        least that is defensible, and let the verification gate decide whether
        it was enough. Ties break on higher fidelity, then name.
        """
        candidates = self.query(context, category).applicable
        if not candidates:
            return None
        return min(candidates, key=lambda m: (int(m.cost), -int(m.fidelity), m.name))
