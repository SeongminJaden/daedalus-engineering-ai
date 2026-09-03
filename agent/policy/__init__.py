"""agent.policy: turning a sentence into a problem, and choosing what to run.

THE RULE THIS PACKAGE EXISTS TO ENFORCE
=======================================
A policy proposes. It never decides. Everything it produces is a proposal that
has to survive validation against the schema and the databases this project
already has, and then be verified by a solver before it is worth anything. A
policy that returns a material that does not exist, a load with no direction,
a deflection limit of zero or a strategy the registry does not hold is refused
with the reason, not corrected quietly.

That is the same discipline the surrogate lives under, for the same reason: a
prediction that cannot be checked is not evidence, and a fluent sentence is
less checkable than a number.

WHAT IS HERE
============
RuleBasedPolicy
    Deterministic, local, no network. Reads a goal sentence for the quantities
    this project's IR needs, picks a strategy from the registry's own
    conditions, and suggests a retry direction from what actually failed. It
    is not a language model and does not pretend to be one.

LanguageModelPolicy
    The seam. It takes a `complete(prompt) -> str` callable, asks for JSON,
    and puts the answer through exactly the same validation. No API key is
    stored or needed here; the caller supplies the callable. Without one the
    class refuses to be constructed rather than falling back to something that
    looks like a model and is not.

validate_problem
    The gate both of them pass through.
"""

from .policy import (InvalidProposal, LanguageModelPolicy, Policy,
                     PolicyProposal, RuleBasedPolicy, StrategyChoice,
                     RetrySuggestion, validate_problem, validate_strategy)

__all__ = ["InvalidProposal", "LanguageModelPolicy", "Policy", "PolicyProposal",
           "RetrySuggestion", "RuleBasedPolicy", "StrategyChoice",
           "validate_problem", "validate_strategy"]
