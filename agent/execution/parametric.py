"""Executing the parametric section strategy: the Phase 3 local solve."""

from __future__ import annotations

import time

import numpy as np

from optimization.constraints.evaluate import evaluate_design
from optimization.gradient.slsqp import default_start, optimize_slsqp

from .outcome import DesignOutcome

METHOD = "parametric_section"


def run(op, start_x: np.ndarray | None = None, max_iter: int = 60,
        **_: object) -> DesignOutcome:
    """Refine the section parameters from a starting point."""
    start = start_x if start_x is not None else default_start(op)
    began = time.monotonic()
    result = optimize_slsqp(op, x0=start, max_iter=max_iter)
    elapsed = time.monotonic() - began

    evaluation = evaluate_design(op, result.x)
    return DesignOutcome(
        method=METHOD,
        mass_kg=float(evaluation.mass_kg),
        feasible=bool(evaluation.is_feasible()),
        constraints=dict(evaluation.constraints),
        evaluations=int(result.n_evaluations),
        seconds=elapsed,
        converged=bool(result.success),
        design_vector=np.asarray(result.x, dtype=float),
        detail={
            "max_bending_stress_pa": float(evaluation.max_bending_stress_pa),
            "tip_deflection_m": float(evaluation.tip_deflection_m),
            "safety_factor": float(evaluation.safety_factor),
            "first_natural_frequency_hz":
                float(evaluation.first_natural_frequency_hz)})
