"""Phase 3 verification: constrained mass minimization, two independent methods.

The load-bearing check is the cross-verification: SLSQP (local, gradient-based,
exact Warp derivatives) and differential evolution (global, gradient-free,
GPU-batched population) share only the problem definition. Their algorithms,
their search behaviour and their constraint handling are different, so their
agreeing on the same optimum is real evidence rather than a restatement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.design_genome import DesignBounds  # noqa: E402
from core.materials import get_material  # noqa: E402
from optimization.constraints import (  # noqa: E402
    FEASIBILITY_TOL, build_optimization_problem, evaluate_batch, evaluate_design,
)
from optimization.evolutionary import optimize_differential_evolution  # noqa: E402
from optimization.gradient import optimize_slsqp  # noqa: E402
from optimization.multi_objective import non_dominated_mask, pareto_front  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402

CROSS_METHOD_TOLERANCE = 0.01     # 1%, as specified
INITIAL_MASS_KG = 1.686           # the Phase 2 hand-checked over-designed link


@pytest.fixture(scope="module")
def op():
    return build_optimization_problem(build_mvp_problem())


@pytest.fixture(scope="module")
def slsqp(op):
    return optimize_slsqp(op)


@pytest.fixture(scope="module")
def de(op):
    return optimize_differential_evolution(op, seed=0)


# --------------------------------------------------------------------------- #
# problem setup
# --------------------------------------------------------------------------- #
def test_allowable_stress_is_the_tighter_limit(op):
    """min(explicit ceiling, yield/SF) = min(120, 503/2 = 251.5) MPa."""
    material = get_material(op.problem.material_id)
    assert op.allowable_stress_pa == 120.0e6
    assert op.allowable_stress_pa < material.allowable_stress_pa(2.0)


def test_envelope_caps_outer_dimensions(op):
    assert op.upper[0] <= op.problem.geometry.max_width_m
    assert op.upper[1] <= op.problem.geometry.max_height_m


def test_minimum_wall_thickness_is_the_assumed_one(op):
    """t_min is an ASSUMED manufacturability limit and the optimum sits on it,
    so a silent change here would silently change every optimized mass."""
    assert op.lower[2] == 0.001
    assert DesignBounds().wall_thickness_m.min == 0.001


def test_unconstrained_problem_is_rejected():
    problem = build_mvp_problem()
    problem.constraints.max_stress_pa = None
    problem.constraints.min_safety_factor = None
    with pytest.raises(ValueError, match="unbounded"):
        build_optimization_problem(problem)


# --------------------------------------------------------------------------- #
# 1. feasibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["slsqp", "de"])
def test_optimum_is_feasible(method, slsqp, de, op):
    result = slsqp if method == "slsqp" else de
    ev = result.evaluation
    assert ev.is_feasible(), (
        f"{result.method} infeasible: {ev.constraints}"
    )
    assert ev.max_bending_stress_pa <= op.allowable_stress_pa * (1 + FEASIBILITY_TOL)
    assert ev.tip_deflection_m <= op.max_deflection_m * (1 + FEASIBILITY_TOL)
    assert op.is_geometrically_valid(result.x)


@pytest.mark.parametrize("method", ["slsqp", "de"])
def test_optimum_respects_bounds(method, slsqp, de, op):
    x = (slsqp if method == "slsqp" else de).x
    assert np.all(x >= op.lower - 1e-12)
    assert np.all(x <= op.upper + 1e-12)


@pytest.mark.parametrize("method", ["slsqp", "de"])
def test_optimizer_reports_success(method, slsqp, de):
    assert (slsqp if method == "slsqp" else de).success


# --------------------------------------------------------------------------- #
# 2. cross-verification - the core check
# --------------------------------------------------------------------------- #
def test_two_methods_agree_on_optimal_mass(slsqp, de):
    rel = abs(slsqp.mass_kg - de.mass_kg) / slsqp.mass_kg
    assert rel < CROSS_METHOD_TOLERANCE, (
        f"SLSQP {slsqp.mass_kg:.6f} kg vs DE {de.mass_kg:.6f} kg, "
        f"relative difference {rel:.3e}"
    )


def test_two_methods_agree_on_design_variables(slsqp, de):
    """Not just the objective - the actual geometry must agree too, otherwise
    the match could be two different designs of coincidentally equal mass."""
    for i, name in enumerate(("b", "h", "t")):
        rel = abs(slsqp.x[i] - de.x[i]) / max(abs(slsqp.x[i]), 1e-12)
        assert rel < 0.02, f"{name}: SLSQP {slsqp.x[i]:.6g} vs DE {de.x[i]:.6g}"


def test_optimum_beats_the_initial_design(slsqp):
    assert slsqp.mass_kg < INITIAL_MASS_KG
    reduction = 1.0 - slsqp.mass_kg / INITIAL_MASS_KG
    assert reduction > 0.5, f"only {reduction:.1%} lighter"


# --------------------------------------------------------------------------- #
# 3. local optimality
# --------------------------------------------------------------------------- #
def test_no_feasible_perturbation_reduces_mass(slsqp, op):
    """Perturb each variable both ways; any perturbation that stays feasible
    and in-bounds must not be lighter than the optimum."""
    best = slsqp.mass_kg
    for i in range(3):
        for rel_step in (1e-3, 5e-3, 2e-2):
            for sign in (-1.0, +1.0):
                x = np.array(slsqp.x, dtype=float)
                x[i] *= 1.0 + sign * rel_step
                if np.any(x < op.lower) or np.any(x > op.upper):
                    continue
                if not op.is_geometrically_valid(x):
                    continue
                ev = evaluate_design(op, x)
                if not ev.is_feasible():
                    continue
                assert ev.mass_kg >= best * (1.0 - 1e-4), (
                    f"variable {i} step {sign * rel_step:+.3g} gives "
                    f"{ev.mass_kg:.8f} kg < optimum {best:.8f} kg"
                )


def test_relaxing_the_binding_constraint_allows_less_mass(op):
    """If deflection is what binds, doubling the deflection budget must let the
    optimizer find a lighter design. A constraint that is truly active has to
    behave this way; a slack one would not."""
    relaxed_problem = build_mvp_problem()
    relaxed_problem.constraints.max_deflection_m *= 2.0
    relaxed = build_optimization_problem(relaxed_problem)
    assert optimize_slsqp(relaxed).mass_kg < optimize_slsqp(op).mass_kg


# --------------------------------------------------------------------------- #
# 4. which constraint binds
# --------------------------------------------------------------------------- #
def test_deflection_is_the_binding_constraint(slsqp):
    """Stiffness, not strength, limits this link: the stress constraint is far
    from active while deflection sits on its limit."""
    active = slsqp.evaluation.active_constraints()
    assert "deflection" in active, f"active constraints: {active}"
    assert "stress" not in active
    assert slsqp.evaluation.constraints["stress"] > 0.5   # >50% margin left


def test_safety_factor_far_exceeds_the_minimum(slsqp):
    assert slsqp.evaluation.safety_factor > 2.0


# --------------------------------------------------------------------------- #
# 5. determinism
# --------------------------------------------------------------------------- #
def test_de_is_deterministic_for_a_seed(op):
    a = optimize_differential_evolution(op, seed=7, popsize=10, max_iter=40)
    b = optimize_differential_evolution(op, seed=7, popsize=10, max_iter=40)
    assert np.array_equal(a.x, b.x)
    assert a.mass_kg == b.mass_kg


def test_slsqp_is_deterministic(op):
    a, b = optimize_slsqp(op), optimize_slsqp(op)
    assert np.array_equal(a.x, b.x)


# --------------------------------------------------------------------------- #
# batch evaluation used by the evolutionary driver
# --------------------------------------------------------------------------- #
def test_batch_marks_invalid_geometry_without_evaluating_it(op):
    X = np.array([
        [0.05, 0.08, 0.005],    # valid
        [0.02, 0.02, 0.015],    # t >= min(b,h)/2 -> impossible
    ])
    mass, cons, ok = evaluate_batch(op, X)
    assert ok.tolist() == [True, False]
    assert np.isfinite(mass[0]) and np.isnan(mass[1])
    assert cons["cavity_b"][1] < 0        # flagged by the analytic term
    assert cons["cavity_b"][0] > 0


def test_batch_matches_single_evaluation(op):
    X = np.array([[0.05, 0.08, 0.005], [0.03, 0.06, 0.002]])
    mass, _, _ = evaluate_batch(op, X)
    for i, row in enumerate(X):
        assert mass[i] == pytest.approx(evaluate_design(op, row).mass_kg, rel=1e-12)


def test_penalized_objective_is_finite_everywhere(op):
    """Including deep in the geometrically impossible region, where the metrics
    are NaN - a NaN objective would blind the search."""
    from optimization.evolutionary import penalized_objective
    X = np.array([
        [0.010, 0.010, 0.020],
        [0.100, 0.100, 0.001],
        [0.010, 0.100, 0.020],
        [0.050, 0.080, 0.005],
    ]).T
    cost = penalized_objective(op, X)
    assert np.all(np.isfinite(cost)), cost
    assert cost[3] < cost[0]     # a valid design must beat an impossible one


# --------------------------------------------------------------------------- #
# multi-objective stub
# --------------------------------------------------------------------------- #
def test_non_dominated_mask():
    f = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 1.0], [4.0, 6.0], [2.0, 3.0]])
    mask = non_dominated_mask(f)
    assert mask[0] and mask[2]
    assert not mask[3]                       # dominated by every other row
    assert pareto_front(f).shape[1] == 2


def test_scalarize_rejects_wrong_weight_count():
    from optimization.multi_objective import scalarize
    with pytest.raises(ValueError):
        scalarize(np.array([[1.0, 2.0]]), np.array([1.0]))
