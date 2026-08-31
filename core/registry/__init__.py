"""core.registry: methods declared as data, routed on their applicability.

The reasoner's reach is bounded by what is registered here. Adding a method is
adding a row, not editing a selector, and every row has to say when it stops
being valid.
"""

from .catalog import (
    EULER_BERNOULLI_SLENDERNESS,
    TIMOSHENKO_SLENDERNESS,
    DEFAULT_REGISTRY,
    build_default_registry,
)
from .context import ProblemContext, Unstated
from .method import (
    Applicability,
    Category,
    Condition,
    Cost,
    Fidelity,
    Method,
)
from .registry import (
    Candidates,
    DuplicateMethod,
    Exclusion,
    MethodRegistry,
    UnknownMethod,
)

__all__ = [
    "Applicability", "Candidates", "Category", "Condition", "Cost",
    "DEFAULT_REGISTRY", "DuplicateMethod", "EULER_BERNOULLI_SLENDERNESS",
    "Exclusion", "Fidelity", "Method", "MethodRegistry", "ProblemContext",
    "TIMOSHENKO_SLENDERNESS", "UnknownMethod", "Unstated",
    "build_default_registry",
]
