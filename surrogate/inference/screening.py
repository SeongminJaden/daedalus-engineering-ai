"""surrogate.inference.screening - two-stage screen and verify.

THE RULE THIS MODULE EXISTS TO ENFORCE: a surrogate never decides anything on
its own. It ranks; the solver decides.

    1. predict every candidate with the surrogate (cheap, approximate)
    2. keep the top_k by predicted objective, preferring predicted-feasible
    3. re-evaluate ONLY those k with the real Phase 2 solver
    4. choose the winner from the SOLVER's numbers

So the reported design is always solver-verified. `ScreeningResult.verified` is
True only because step 3 actually ran; the surrogate's own prediction for the
winner is kept alongside purely so the two can be compared.

The screening step can be wrong - it may drop a good candidate it mis-ranked.
That is a recall risk, and it is the price of the speedup. What it cannot do is
put an unverified design in front of a user as a result.

The same rule is now written into the evidence ladder rather than only into
this module's control flow. A result whose numbers came from the solver grades
SIMULATED; one that never reached the solver grades SURROGATE, and the verdict
layer refuses to build a pass or a fail on it. `screened_check` is the only
shape a bare prediction can take in an assembly verdict, and that shape is a
gap, not a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.semantic.evidence import Evidence, EvidenceKind, EvidenceLevel
from integration.checks import CheckResult, CheckStatus
from optimization.constraints import OptimizationProblem, evaluate_batch
from physics.structural import load_case_from_problem

from .predict import Prediction, SurrogatePredictor, build_inputs


@dataclass
class ScreeningResult:
    """Outcome of a screen-and-verify pass."""

    best_x: np.ndarray | None
    best_mass_kg: float | None
    best_constraints: dict[str, float] = field(default_factory=dict)
    verified: bool = False
    n_screened: int = 0
    n_verified: int = 0
    predicted_best_mass_kg: float | None = None
    surrogate_error_on_winner: float | None = None
    expected_relative_error: dict[str, float] = field(default_factory=dict)

    def is_feasible(self, tol: float = 1e-4) -> bool:
        return bool(self.best_constraints) and all(
            v >= -tol for v in self.best_constraints.values())

    # --- where this sits on the evidence ladder --------------------------------

    @property
    def evidence_kind(self) -> EvidenceKind:
        """SIMULATION when the reported design was solver-evaluated, SURROGATE
        otherwise. `verified` is the only thing that decides this."""
        return (EvidenceKind.SIMULATION if self.verified
                else EvidenceKind.SURROGATE)

    @property
    def evidence_level(self) -> EvidenceLevel:
        return (EvidenceLevel.SIMULATED if self.verified
                else EvidenceLevel.SURROGATE)

    def as_evidence(self, ref: str, run_id: str | None = None) -> Evidence:
        """This screening pass as a Brain evidence item, graded by whether the
        solver actually ran on the winner."""
        if self.verified:
            note = (f"solver-verified winner from {self.n_verified} of "
                    f"{self.n_screened} screened")
        else:
            note = (f"screening only; nothing verified out of "
                    f"{self.n_screened} screened")
        return Evidence(kind=self.evidence_kind, ref=ref, run_id=run_id,
                        note=note)


def screened_check(component: str, failure_mode: str, prediction: Prediction,
                   i: int, method: str = "surrogate_mlp") -> CheckResult:
    """The only CheckResult a bare prediction may become: SCREENED, a gap.

    The predicted safety factor and its expected error go into `detail` as
    text. They are deliberately NOT placed in `safety_factor`, where the
    review would read them back as a solved number and let them govern.
    """
    sf = float(prediction.values["safety_factor"][i])
    err = prediction.expected_relative_error.get("safety_factor", 0.0)
    return CheckResult(
        component=component, failure_mode=failure_mode,
        status=CheckStatus.SCREENED, method=method,
        detail=(f"surrogate predicts safety factor {sf:.3f} with p95 held-out "
                f"error {err:.1%}; not a verdict, run the solver"),
        evidence_kind=prediction.evidence_kind,
    )


def screen_and_verify(
    predictor: SurrogatePredictor,
    op: OptimizationProblem,
    candidates: np.ndarray,
    top_k: int = 16,
    device: str | None = None,
) -> ScreeningResult:
    """Screen many candidates with the surrogate, verify a few with the solver."""
    candidates = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    if candidates.shape[1] != 3:
        raise ValueError("candidates must have columns (b, h, t)")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    valid = np.array([op.is_geometrically_valid(row) for row in candidates])
    if not valid.any():
        return ScreeningResult(best_x=None, best_mass_kg=None, verified=False,
                               n_screened=int(candidates.shape[0]))

    workable = candidates[valid]
    case = load_case_from_problem(op.problem)

    # --- stage 1: predict everything ---
    prediction = predictor.predict(
        build_inputs(workable[:, 0], workable[:, 1], workable[:, 2], case))
    pred_mass = prediction.values["mass_kg"]
    pred_stress = prediction.values["max_bending_stress_pa"]
    pred_defl = prediction.values["tip_deflection_m"]

    predicted_feasible = pred_stress <= op.allowable_stress_pa
    if op.max_deflection_m is not None:
        predicted_feasible &= pred_defl <= op.max_deflection_m

    # Feasible-looking candidates first, lightest first within each group. Any
    # infeasible-looking ones are still eligible if there are not enough
    # feasible ones - the surrogate's feasibility call is itself approximate,
    # so it must not hard-filter.
    order = sorted(
        range(workable.shape[0]),
        key=lambda i: (not bool(predicted_feasible[i]), float(pred_mass[i])),
    )
    chosen = order[:min(top_k, len(order))]

    # --- stage 2: verify the shortlist with the real solver ---
    shortlist = workable[chosen]
    mass, constraints, ok = evaluate_batch(op, shortlist)

    best_i, best_mass = None, np.inf
    for i in range(shortlist.shape[0]):
        if not ok[i] or not np.isfinite(mass[i]):
            continue
        feasible = all(
            float(np.ravel(v)[i]) >= -1e-4 for v in constraints.values())
        if feasible and mass[i] < best_mass:
            best_i, best_mass = i, float(mass[i])

    if best_i is None:      # nothing on the shortlist verified as feasible
        return ScreeningResult(
            best_x=None, best_mass_kg=None, verified=False,
            n_screened=int(candidates.shape[0]),
            n_verified=int(shortlist.shape[0]),
            expected_relative_error=prediction.expected_relative_error)

    predicted_for_winner = float(pred_mass[chosen[best_i]])
    return ScreeningResult(
        best_x=shortlist[best_i],
        best_mass_kg=best_mass,                       # SOLVER value, not predicted
        best_constraints={k: float(np.ravel(v)[best_i])
                          for k, v in constraints.items()},
        verified=True,
        n_screened=int(candidates.shape[0]),
        n_verified=int(shortlist.shape[0]),
        predicted_best_mass_kg=predicted_for_winner,
        surrogate_error_on_winner=abs(predicted_for_winner - best_mass) / best_mass,
        expected_relative_error=prediction.expected_relative_error,
    )


def brute_force_best(op: OptimizationProblem, candidates: np.ndarray):
    """Solver-evaluate every candidate. The reference screening is judged
    against - and what screening exists to avoid paying for."""
    candidates = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    valid = np.array([op.is_geometrically_valid(row) for row in candidates])
    if not valid.any():
        return None, None
    workable = candidates[valid]
    mass, constraints, ok = evaluate_batch(op, workable)

    best_i, best_mass = None, np.inf
    for i in range(workable.shape[0]):
        if not ok[i] or not np.isfinite(mass[i]):
            continue
        if all(float(np.ravel(v)[i]) >= -1e-4 for v in constraints.values()):
            if mass[i] < best_mass:
                best_i, best_mass = i, float(mass[i])
    if best_i is None:
        return None, None
    return workable[best_i], best_mass
