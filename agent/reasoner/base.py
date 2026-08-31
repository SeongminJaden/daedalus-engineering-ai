"""agent.reasoner.base - the decision interface of the autonomous loop.

WHAT THIS IS, PLAINLY: the reasoner shipped in Phase 4 is a **deterministic
rule-based heuristic**. It is not a language model, and it does not "reason"
in any sense beyond following the explore/exploit schedule written into it.
Calling it AI reasoning would be an overclaim.

What it *is* is the seam. `Reasoner.decide()` is the extension point where an
LLM-backed policy can be dropped in later without touching the loop engine -
the loop only ever sees `decide(state, history) -> Action`. In the wider
system the language model is the outer orchestrator (a session driving this
engine), not this class.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class ActionKind(str, Enum):
    """What the loop should do next."""

    EXPLOIT = "exploit"     # locally refine the incumbent best
    EXPLORE = "explore"     # restart local search from a fresh random point
    STOP = "stop"           # the reasoner itself wants to end the run


@dataclass
class Action:
    """A decision, with the hypothesis that motivated it.

    `hypothesis` is recorded in the episode log so a run can be read back as a
    sequence of intentions rather than just numbers.
    """

    kind: ActionKind
    hypothesis: str
    strategy: str
    start_x: np.ndarray | None = None

    def __post_init__(self):
        if self.start_x is not None:
            self.start_x = np.asarray(self.start_x, dtype=float)


@dataclass
class ReasonerState:
    """What the reasoner is allowed to see - deliberately small."""

    iteration: int
    best_x: np.ndarray | None
    best_mass_kg: float | None
    best_feasible: bool
    iterations_without_improvement: int
    evaluations_used: int
    seconds_used: float
    lower: np.ndarray = field(default_factory=lambda: np.zeros(3))
    upper: np.ndarray = field(default_factory=lambda: np.ones(3))


class Reasoner(abc.ABC):
    """Chooses the next action. Swap the implementation, keep the loop."""

    name: str = "abstract"

    @abc.abstractmethod
    def decide(self, state: ReasonerState, history: Sequence) -> Action:
        """Return the next action given the run state and past episodes."""
        raise NotImplementedError
