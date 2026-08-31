"""agent.reasoner.heuristic - the rule-based policy used in Phase 4.

Not a language model. A schedule.

The policy makes the loop a **multi-start global search** around the Phase 3
local optimizer, which is the actual value it adds over calling that optimizer
once: SLSQP converges to the nearest KKT point, and with two design variables
pinned to their bounds the interesting question is whether a different basin
does better. Explore answers that; exploit polishes the winner.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from .base import Action, ActionKind, Reasoner, ReasonerState


class HeuristicReasoner(Reasoner):
    """Deterministic explore/exploit schedule.

    Rules, in order:
      1. First iteration: exploit from the given starting point.
      2. After `patience` iterations with no improvement: explore, to escape a
         basin the local optimizer cannot leave on its own.
      3. Otherwise explore with probability `explore_probability`, else exploit.

    Seeded, so a whole run reproduces exactly.
    """

    name = "heuristic-explore-exploit"

    def __init__(
        self,
        seed: int = 0,
        explore_probability: float = 0.4,
        patience: int = 2,
        jitter: float = 0.15,
    ):
        if not 0.0 <= explore_probability <= 1.0:
            raise ValueError("explore_probability must be in [0, 1]")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.rng = random.Random(seed)
        self.explore_probability = explore_probability
        self.patience = patience
        self.jitter = jitter

    def _random_start(self, state: ReasonerState) -> np.ndarray:
        """Uniform point in the box, with the wall pulled inside the cavity."""
        x = np.array([
            self.rng.uniform(lo, hi)
            for lo, hi in zip(state.lower, state.upper)
        ])
        ceiling = 0.45 * min(x[0], x[1])
        x[2] = min(x[2], max(state.lower[2], ceiling))
        return x

    def _jittered_best(self, state: ReasonerState) -> np.ndarray:
        """Perturb the incumbent so exploit does not re-run an identical solve."""
        x = np.array(state.best_x, dtype=float)
        for i in range(len(x)):
            x[i] *= 1.0 + self.rng.gauss(0.0, self.jitter)
        x = np.clip(x, state.lower, state.upper)
        ceiling = 0.45 * min(x[0], x[1])
        x[2] = min(x[2], max(state.lower[2], ceiling))
        return x

    def decide(self, state: ReasonerState, history: Sequence) -> Action:
        if state.best_x is None:
            return Action(
                kind=ActionKind.EXPLOIT,
                hypothesis=(
                    "No incumbent yet; a local solve from the nominal start "
                    "should establish a feasible baseline design."
                ),
                strategy="initial-exploit",
                start_x=None,
            )

        if state.iterations_without_improvement >= self.patience:
            return Action(
                kind=ActionKind.EXPLORE,
                hypothesis=(
                    f"No improvement for {state.iterations_without_improvement} "
                    "iterations; the local optimizer is stuck in one basin, so a "
                    "fresh random start may find a better one."
                ),
                strategy="explore-on-stall",
                start_x=self._random_start(state),
            )

        if self.rng.random() < self.explore_probability:
            return Action(
                kind=ActionKind.EXPLORE,
                hypothesis=(
                    "Scheduled exploration: sample a new basin to test whether "
                    f"the incumbent {state.best_mass_kg:.4f} kg is global."
                ),
                strategy="explore-scheduled",
                start_x=self._random_start(state),
            )

        return Action(
            kind=ActionKind.EXPLOIT,
            hypothesis=(
                f"Refine around the incumbent ({state.best_mass_kg:.4f} kg) "
                "from a jittered start to confirm it is locally optimal."
            ),
            strategy="exploit-jittered",
            start_x=self._jittered_best(state),
        )
