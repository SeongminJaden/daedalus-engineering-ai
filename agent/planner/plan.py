"""agent.planner.plan - turn an Action into a concrete experiment.

Thin by design in Phase 4: the reasoner decides *what kind* of move to make,
the planner decides *how to run it*. Keeping them apart means an LLM reasoner
can later emit richer actions without the loop learning how to drive scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agent.reasoner import Action, ActionKind


@dataclass
class ExperimentPlan:
    """A fully specified local search to run."""

    action: Action
    start_x: np.ndarray | None
    max_iter: int
    label: str

    @property
    def kind(self) -> ActionKind:
        return self.action.kind


def plan_experiment(action: Action, local_max_iter: int = 200) -> ExperimentPlan:
    """Bind an action to the optimizer settings that will execute it.

    Exploration gets a shorter budget than exploitation: a random start is a
    cheap probe of a basin, and spending a full solve on every probe would
    burn the compute budget on basins that are obviously worse.
    """
    if action.kind is ActionKind.STOP:
        raise ValueError("STOP actions are handled by the loop, not planned")

    if action.kind is ActionKind.EXPLORE:
        return ExperimentPlan(
            action=action,
            start_x=action.start_x,
            max_iter=max(20, local_max_iter // 2),
            label="explore:local-solve-from-random-start",
        )

    return ExperimentPlan(
        action=action,
        start_x=action.start_x,
        max_iter=local_max_iter,
        label="exploit:local-solve-from-incumbent",
    )
