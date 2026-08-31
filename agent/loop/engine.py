"""agent.loop.engine - the autonomous design loop state machine.

Implements the cycle:

    OBSERVE -> REASON -> PLAN -> DESIGN -> SIMULATE -> EVALUATE
            -> LEARN -> UPDATE_BRAIN -> (back to REASON)

One pass through those states is one iteration and produces exactly one
episode. The loop is a multi-start orchestrator around the Phase 3 local
optimizer: that is what it adds over a single solve, since SLSQP only ever
finds the nearest KKT point.

Scope honesty: the "reasoning" here is a rule-based heuristic policy
(agent.reasoner.heuristic), not a language model. The LLM sits *outside* this
engine as the session driving it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from agent.evaluator import judge
from agent.experiment_manager import ComputeBudget, Episode, EpisodeLog
from agent.planner import plan_experiment
from agent.reasoner import Action, ActionKind, HeuristicReasoner, Reasoner, ReasonerState
from agent.execution import OutcomeVerdict, execute, executable_methods
from agent.execution.parametric import METHOD as PARAMETRIC_METHOD
from optimization.constraints import OptimizationProblem
from optimization.gradient import default_start


class LoopPhase(str, Enum):
    """MD state machine. Recorded so a live view can show where the loop is."""

    OBSERVE = "observe"
    REASON = "reason"
    PLAN = "plan"
    DESIGN = "design"
    SIMULATE = "simulate"
    EVALUATE = "evaluate"
    LEARN = "learn"
    UPDATE_BRAIN = "update_brain"
    DONE = "done"


PHASE_CYCLE = (
    LoopPhase.OBSERVE, LoopPhase.REASON, LoopPhase.PLAN, LoopPhase.DESIGN,
    LoopPhase.SIMULATE, LoopPhase.EVALUATE, LoopPhase.LEARN,
    LoopPhase.UPDATE_BRAIN,
)


class TerminationReason(str, Enum):
    USER_STOP = "user_stop"
    TARGET_REACHED = "target_reached"
    CONVERGED = "converged"
    COMPUTE_BUDGET_EXCEEDED = "compute_budget_exceeded"
    CONSTRAINTS_UNSATISFIABLE = "constraints_unsatisfiable"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class LoopConfig:
    """Run settings. Every termination rule is parameterized, none hard-coded."""

    max_iterations: int = 8
    seed: int = 0

    # Convergence: stop after this many consecutive iterations whose relative
    # improvement stayed under `convergence_epsilon`. The MD rule of "50
    # iterations under 0.1%" is this with (50, 1e-3).
    convergence_patience: int = 4
    convergence_epsilon: float = 1e-3

    target_mass_kg: float | None = None

    # Declare the constraints unsatisfiable only after this many independent
    # starts have all failed to find any feasible design. One failure is a bad
    # start; several from different basins is evidence about the problem.
    unsatisfiable_after: int = 3

    local_max_iter: int = 200

    # Settings for the topology strategies. Their defaults are small because
    # every bisection step inside one is a full topology solve, which is why
    # the registry rates those methods HEAVY.
    topology_options: dict = field(default_factory=lambda: {
        "iterations": 25, "bisection_steps": 3})
    max_evaluations: int | None = None
    max_seconds: float | None = None
    profile: str | None = None


@dataclass
class LoopResult:
    """Outcome of a whole run."""

    best_x: np.ndarray | None
    best_evaluation: object | None
    termination: TerminationReason
    termination_detail: str
    iterations: int
    episodes: list[Episode]
    budget: dict
    run_id: str
    episode_log_path: Path | None = None

    @property
    def best_mass_kg(self) -> float | None:
        return None if self.best_evaluation is None else self.best_evaluation.mass_kg


@dataclass
class _RunState:
    best_x: np.ndarray | None = None
    best_evaluation: object | None = None
    # Which method produced the incumbent, and the incumbent itself. The
    # incumbent can come from any strategy, while `best_x` stays the best
    # parametric vector because that is the space the explore/exploit schedule
    # jitters in. Conflating the two would have the reasoner perturbing a
    # design vector that no longer describes the best design.
    best_method: str | None = None
    best_outcome: object | None = None
    iterations_without_improvement: int = 0
    small_improvement_streak: int = 0
    consecutive_infeasible: int = 0
    phase: LoopPhase = LoopPhase.OBSERVE
    last_design_id: str | None = None
    episodes: list[Episode] = field(default_factory=list)


class DesignLoop:
    """Runs the state machine until a termination condition fires."""

    def __init__(
        self,
        op: OptimizationProblem,
        config: LoopConfig | None = None,
        reasoner: Reasoner | None = None,
        episode_log: EpisodeLog | None = None,
        dashboard=None,
        stop_flag=None,
        brain=None,
        problem_name: str | None = None,
    ):
        self.op = op
        self.config = config or LoopConfig()
        self.reasoner = reasoner or HeuristicReasoner(seed=self.config.seed)
        self.episode_log = episode_log
        self.dashboard = dashboard
        # Any zero-arg callable returning True ends the run - that is how an
        # operator (or a test) requests USER_STOP without killing the process.
        self.stop_flag = stop_flag
        self.budget = ComputeBudget.from_profile(
            self.config.profile,
            max_evaluations=self.config.max_evaluations,
            max_seconds=self.config.max_seconds,
        )
        self.run_id = uuid.uuid4().hex[:12]
        self.state = _RunState()
        self.brain = brain
        self.problem_name = problem_name or getattr(op.problem, "name", "unknown")

        # Closing the loop: start from the best design any previous run left in
        # the Brain instead of rediscovering it. None on a cold Brain, which is
        # the normal first case and must stay handled.
        self.warm_start_x = None
        if brain is not None:
            candidate = brain.warm_start()
            if candidate is not None and op.is_geometrically_valid(candidate):
                self.warm_start_x = op.clip_to_bounds(candidate)

    # --- state machine ---------------------------------------------------- #
    def _set_phase(self, phase: LoopPhase) -> None:
        self.state.phase = phase
        self._refresh_dashboard()

    def _refresh_dashboard(self) -> None:
        if self.dashboard is None:
            return
        s = self.state
        self.dashboard.update(
            status=s.phase.value,
            generation=len(s.episodes),
            generations_total=self.config.max_iterations,
            best_objective=(
                None if s.best_evaluation is None else s.best_evaluation.mass_kg
            ),
            evaluated=self.budget.evaluations,
            active_constraint=(
                ", ".join(s.best_evaluation.active_constraints()) or "none"
                if s.best_evaluation is not None else "-"
            ),
            budget=f"{self.budget.evaluations}/{self.budget.max_evaluations} evals, "
                   f"{self.budget.seconds:.0f}/{self.budget.max_seconds:.0f}s",
        )

    def _observe(self) -> ReasonerState:
        self._set_phase(LoopPhase.OBSERVE)
        s = self.state
        return ReasonerState(
            iteration=len(s.episodes),
            best_x=s.best_x,
            best_mass_kg=(
                None if s.best_evaluation is None else s.best_evaluation.mass_kg
            ),
            best_feasible=(
                s.best_evaluation is not None and s.best_evaluation.is_feasible()
            ),
            iterations_without_improvement=s.iterations_without_improvement,
            evaluations_used=self.budget.evaluations,
            seconds_used=self.budget.seconds,
            lower=self.op.lower,
            upper=self.op.upper,
        )

    def _terminated_before_iteration(self) -> tuple[TerminationReason, str] | None:
        """Conditions checked before spending any more compute."""
        if self.stop_flag is not None and self.stop_flag():
            return TerminationReason.USER_STOP, "stop requested by operator"

        if self.budget.exceeded():
            return (TerminationReason.COMPUTE_BUDGET_EXCEEDED,
                    self.budget.reason() or "budget exceeded")

        ev = self.state.best_evaluation
        if (self.config.target_mass_kg is not None and ev is not None
                and ev.is_feasible() and ev.mass_kg <= self.config.target_mass_kg):
            return (TerminationReason.TARGET_REACHED,
                    f"feasible design at {ev.mass_kg:.6f} kg reached the target "
                    f"{self.config.target_mass_kg:.6f} kg")

        if self.state.small_improvement_streak >= self.config.convergence_patience:
            return (TerminationReason.CONVERGED,
                    f"{self.state.small_improvement_streak} consecutive iterations "
                    f"improved by less than {self.config.convergence_epsilon:.3%}")

        if (self.state.best_evaluation is None
                and self.state.consecutive_infeasible >= self.config.unsatisfiable_after):
            return (TerminationReason.CONSTRAINTS_UNSATISFIABLE,
                    f"no feasible design found after "
                    f"{self.state.consecutive_infeasible} independent starts")
        return None

    def _run_experiment(self, plan, method: str):
        """DESIGN + SIMULATE: run the selected method, then read its outcome.

        The method comes from the reasoner, which took it from the registry.
        Before this dispatch existed the loop ran the parametric solver
        whatever was selected, and a topology recommendation could only be
        recorded as unmet.
        """
        self._set_phase(LoopPhase.DESIGN)
        start = plan.start_x
        if start is None:
            start = (self.warm_start_x if self.warm_start_x is not None
                     else default_start(self.op))

        self._set_phase(LoopPhase.SIMULATE)
        if method == PARAMETRIC_METHOD:
            kwargs = {"start_x": start, "max_iter": plan.max_iter}
        else:
            kwargs = dict(self.config.topology_options)
        outcome = execute(method, self.op, **kwargs)
        self.budget.spend(outcome.evaluations)
        return outcome

    @staticmethod
    def _method_of(action: Action) -> str:
        """The registry method named in the action's strategy.

        The routing reasoner writes "method:inner-strategy"; the Phase 4
        heuristic writes a bare strategy name and means the parametric solver,
        which is all it ever ran.
        """
        head, separator, _ = action.strategy.partition(":")
        if separator and head in executable_methods():
            return head
        return PARAMETRIC_METHOD

    def step(self) -> Episode:
        """One full pass of the state machine. Produces exactly one episode."""
        observation = self._observe()

        self._set_phase(LoopPhase.REASON)
        action: Action = self.reasoner.decide(observation, self.state.episodes)
        if action.kind is ActionKind.STOP:
            raise StopIteration("reasoner requested stop")

        self._set_phase(LoopPhase.PLAN)
        plan = plan_experiment(action, local_max_iter=self.config.local_max_iter)

        method = self._method_of(action)
        outcome = self._run_experiment(plan, method)
        elapsed = outcome.seconds

        self._set_phase(LoopPhase.EVALUATE)
        evaluation = OutcomeVerdict(outcome)

        self._set_phase(LoopPhase.LEARN)
        incumbent = (
            None if self.state.best_evaluation is None
            else self.state.best_evaluation.mass_kg
        )
        verdict = judge(evaluation, incumbent, outcome.converged)

        # --- fold the verdict into the run state ---
        relative = (
            None if incumbent is None
            else (incumbent - evaluation.mass_kg) / incumbent
        )
        if verdict.is_new_best:
            if outcome.design_vector is not None:
                self.state.best_x = np.array(outcome.design_vector, dtype=float)
            self.state.best_evaluation = evaluation
            self.state.best_method = outcome.method
            self.state.best_outcome = outcome
            self.state.iterations_without_improvement = 0
        else:
            self.state.iterations_without_improvement += 1

        if verdict.feasible:
            self.state.consecutive_infeasible = 0
        else:
            self.state.consecutive_infeasible += 1

        # Convergence counts iterations whose improvement was below epsilon,
        # including iterations that improved nothing at all.
        if relative is not None and relative < self.config.convergence_epsilon:
            self.state.small_improvement_streak += 1
        elif relative is not None:
            self.state.small_improvement_streak = 0

        self._set_phase(LoopPhase.UPDATE_BRAIN)
        episode = Episode(
            id=uuid.uuid4().hex[:12],
            parent_design_id=self.state.last_design_id,
            iteration=len(self.state.episodes),
            timestamp=Episode.now_iso(),
            hypothesis=action.hypothesis,
            action=action.kind.value,
            strategy_used=action.strategy,
            design_genome=self._genome_of(outcome),
            observation={"mass_kg": outcome.mass_kg, **outcome.detail},
            constraint_status={
                **{k: float(v) for k, v in evaluation.constraints.items()},
                "active": verdict.active_constraints,
                "optimizer_success": bool(outcome.converged),
            },
            conclusion=verdict.conclusion,
            confidence=verdict.confidence,
            feasible=verdict.feasible,
            is_new_best=verdict.is_new_best,
            evaluations=outcome.evaluations,
            seconds=round(elapsed, 4),
        )
        self.state.episodes.append(episode)
        self.state.last_design_id = episode.id
        if self.episode_log is not None:
            self.episode_log.append(episode)
        if self.brain is not None:
            self.brain.record_episode(
                episode, run_id=self.run_id, problem_name=self.problem_name)
        self._refresh_dashboard()
        return episode

    def _genome_of(self, outcome) -> dict:
        """The design, described in the terms the method that made it uses.

        A topology design has no outer width or wall thickness. Writing those
        fields anyway, from a vector that produced a different design, would
        make the episode log say a shape was built that was not.
        """
        common = {"method": outcome.method,
                  "representation": outcome.representation,
                  "material_id": self.op.problem.material_id}
        if outcome.design_vector is not None:
            x = outcome.design_vector
            return {**common,
                    "outer_width_m": float(x[0]),
                    "outer_height_m": float(x[1]),
                    "wall_thickness_m": float(x[2])}
        density = outcome.density_field
        return {**common,
                "n_elements": int(density.size),
                "volume_fraction": float(density.mean())}

    def run(self) -> LoopResult:
        """Iterate until a termination condition fires."""
        termination = TerminationReason.MAX_ITERATIONS
        detail = f"reached max_iterations={self.config.max_iterations}"

        while len(self.state.episodes) < self.config.max_iterations:
            stop = self._terminated_before_iteration()
            if stop is not None:
                termination, detail = stop
                break
            try:
                self.step()
            except StopIteration as exc:
                termination = TerminationReason.USER_STOP
                detail = str(exc)
                break
        else:
            # Loop finished on iteration count; a terminal condition may still
            # have become true on the final iteration, and it is more
            # informative than "ran out of iterations".
            stop = self._terminated_before_iteration()
            if stop is not None:
                termination, detail = stop

        self._set_phase(LoopPhase.DONE)
        if self.brain is not None:
            self.brain.episodic.record_run(
                run_id=self.run_id,
                problem_name=self.problem_name,
                termination=termination.value,
                iterations=len(self.state.episodes),
                best_mass_kg=(
                    None if self.state.best_evaluation is None
                    else self.state.best_evaluation.mass_kg
                ),
                meta={"budget": self.budget.as_dict(),
                      "warm_started": self.warm_start_x is not None},
            )
        return LoopResult(
            best_x=self.state.best_x,
            best_evaluation=self.state.best_evaluation,
            termination=termination,
            termination_detail=detail,
            iterations=len(self.state.episodes),
            episodes=list(self.state.episodes),
            budget=self.budget.as_dict(),
            run_id=self.run_id,
            episode_log_path=(
                None if self.episode_log is None else self.episode_log.path
            ),
        )
