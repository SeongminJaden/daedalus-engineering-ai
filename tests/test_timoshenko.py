"""Phase 7.5 verification: the shear correction, and closing the fidelity loop.

Phase 7 found that the Phase 3 optimum passed beam theory and then failed 3D
FEM, because Euler-Bernoulli omits shear deformation and the MVP link is not
slender. This file checks the repair: the cheap model gains a shear term, is
calibrated against the high-fidelity model, and the design re-optimized under it
now passes the gate.

The shear factor is [ASSUMED] (thin-walled box: the two vertical webs carry the
shear). It is not derived here. It is measured against 3D FEM below, and the
agreement is asserted rather than taken on faith.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.design_genome import DesignGenome, HollowRectangleSection  # noqa: E402
from optimization.constraints import build_optimization_problem, evaluate_design  # noqa: E402
from optimization.gradient import optimize_slsqp  # noqa: E402
from physics.fem import high_fidelity_verify  # noqa: E402
from physics.fem.mesh import hollow_rect_mesh  # noqa: E402
from physics.fem.solver import solve_linear_elasticity  # noqa: E402
from physics.structural import SHEAR_WEB_FACTOR, evaluate_beam  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402

E, NU, LOAD = 71.7e9, 0.33, 196.2
SECTION = (0.010, 0.040, 0.001)          # b, h, t


def analytic(length, shear: bool):
    b, h, t = SECTION
    inertia = (b * h ** 3 - (b - 2 * t) * (h - 2 * t) ** 3) / 12.0
    delta = LOAD * length ** 3 / (3.0 * E * inertia)
    if shear:
        g = E / (2.0 * (1.0 + NU))
        delta += LOAD * length / (g * SHEAR_WEB_FACTOR * 2.0 * t * h)
    return delta


def fem_deflection(length):
    b, h, t = SECTION
    nx = max(12, min(60, int(round(length / (2 * t)))))
    mesh = hollow_rect_mesh(length, h, b, t, nx=nx, elements_through_wall=1)
    solution = solve_linear_elasticity(
        mesh, E, NU, mesh.nodes_at_x(0.0), mesh.nodes_at_x(length), -LOAD, 1)
    assert solution.report.converged
    return abs(solution.tip_deflection())


@pytest.fixture(scope="module")
def calibration():
    """FEM vs both 1D models across slenderness."""
    out = []
    for ratio in (4, 6, 8, 12, 20):
        length = ratio * SECTION[1]
        out.append((ratio, fem_deflection(length),
                    analytic(length, False), analytic(length, True)))
    return out


# =========================================================================== #
# the shear correction is a real improvement, measured
# =========================================================================== #
def test_timoshenko_beats_euler_bernoulli_against_fem(calibration):
    eb = np.mean([abs(e - f) / f for _, f, e, _ in calibration])
    ti = np.mean([abs(t - f) / f for _, f, _, t in calibration])
    assert ti < eb / 3.0, (
        f"shear term should cut the error several-fold: "
        f"Euler-Bernoulli {eb:.3%}, Timoshenko {ti:.3%}")
    assert ti < 0.01, f"Timoshenko mean error {ti:.3%} vs 3D FEM"


def test_shear_matters_most_for_a_stubby_beam(calibration):
    """At low L/h Euler-Bernoulli is badly wrong and the correction rescues it."""
    ratio, fem, eb, ti = calibration[0]
    assert ratio == 4
    assert eb / fem < 0.96, f"expected Euler-Bernoulli to under-predict, got {eb/fem:.4f}"
    assert ti / fem == pytest.approx(1.0, abs=0.02)


def test_slender_beam_is_where_the_two_models_agree(calibration):
    ratio, fem, eb, ti = calibration[-1]
    assert ratio == 20
    assert abs(eb - ti) / ti < 0.01     # shear share is negligible by here


def test_every_slenderness_is_within_tolerance(calibration):
    for ratio, fem, _, ti in calibration:
        assert ti / fem == pytest.approx(1.0, abs=0.02), f"L/h={ratio}"


# =========================================================================== #
# closing the loop: re-optimize, then pass the high fidelity gate
# =========================================================================== #
@pytest.fixture(scope="module")
def problem():
    return build_mvp_problem()


@pytest.fixture(scope="module")
def reoptimized(problem):
    return optimize_slsqp(build_optimization_problem(problem))


def test_reoptimized_design_is_heavier_than_the_euler_bernoulli_optimum(reoptimized):
    """Correcting the model costs mass: the shear term needs real stiffness that
    Euler-Bernoulli was getting for free."""
    assert reoptimized.mass_kg > 0.249977
    assert reoptimized.mass_kg < 0.249977 * 1.05    # a correction, not a redesign


def test_reoptimized_design_is_still_deflection_limited(reoptimized):
    assert "deflection" in reoptimized.evaluation.active_constraints()


def test_old_optimum_now_fails_the_corrected_model(problem):
    """The design Phase 3 produced is infeasible once shear is accounted for.
    This is the failure the high fidelity gate exposed, now visible cheaply."""
    op = build_optimization_problem(problem)
    old = evaluate_design(op, np.array([0.010, 0.0809596, 0.001]))
    assert old.tip_deflection_m > op.max_deflection_m
    assert not old.is_feasible()


@pytest.mark.slow
def test_reoptimized_design_passes_3d_fem(problem, reoptimized):
    """The point of the whole exercise: the corrected optimum clears the gate
    that the previous one failed."""
    op = build_optimization_problem(problem)
    genome = DesignGenome(
        section=HollowRectangleSection(
            outer_width_m=reoptimized.x[0], outer_height_m=reoptimized.x[1],
            wall_thickness_m=reoptimized.x[2]),
        material_id=problem.material_id)
    result = high_fidelity_verify(genome, problem, elements_through_wall=1)
    assert result.converged
    assert result.tip_deflection_m <= op.max_deflection_m, (
        f"3D FEM deflection {result.tip_deflection_m * 1e3:.5f} mm exceeds the "
        f"{op.max_deflection_m * 1e3:.3f} mm limit")
    # The corrected 1D model should now track FEM closely and stay on the
    # conservative side, which is why optimizing to the limit still passes.
    model = evaluate_design(op, reoptimized.x)
    assert result.tip_deflection_m / model.tip_deflection_m == pytest.approx(
        1.0, abs=0.02)
    assert result.tip_deflection_m <= model.tip_deflection_m * 1.001


@pytest.mark.slow
def test_old_optimum_still_fails_3d_fem(problem):
    genome = DesignGenome(
        section=HollowRectangleSection(outer_width_m=0.010,
                                       outer_height_m=0.0809596,
                                       wall_thickness_m=0.001),
        material_id=problem.material_id)
    result = high_fidelity_verify(genome, problem, elements_through_wall=1)
    op = build_optimization_problem(problem)
    assert result.tip_deflection_m > op.max_deflection_m


# =========================================================================== #
# the correction stays differentiable
# =========================================================================== #
def test_gradients_still_flow_through_the_shear_term(problem):
    from physics.structural import beam_gradients
    genome = [DesignGenome(
        section=HollowRectangleSection(outer_width_m=0.05, outer_height_m=0.08,
                                       wall_thickness_m=0.005),
        material_id=problem.material_id)]
    grads = beam_gradients(genome, problem, "tip_deflection_m")
    for name, values in grads.items():
        assert np.all(np.isfinite(values)), name
        assert values[0] < 0, f"d(deflection)/d{name} should be negative"


def test_shear_term_changes_the_gradient(problem):
    from physics.structural import beam_gradients
    genome = [DesignGenome(
        section=HollowRectangleSection(outer_width_m=0.05, outer_height_m=0.08,
                                       wall_thickness_m=0.005),
        material_id=problem.material_id)]
    with_shear = beam_gradients(genome, problem, "tip_deflection_m")
    without = beam_gradients(genome, problem, "tip_deflection_m",
                             shear_deformation=False)
    # Wall thickness carries the shear area, so its sensitivity must change.
    assert abs(with_shear["wall_thickness_m"][0]) > abs(
        without["wall_thickness_m"][0])


# =========================================================================== #
# the lesson is recorded, and stays simulation-grade
# =========================================================================== #
def test_fidelity_lesson_is_recorded_as_simulation_only():
    from brain import Brain
    from brain.semantic import EvidenceLevel, record_fidelity_lesson

    with Brain(":memory:") as memory:
        lesson = record_fidelity_lesson(
            memory.semantic,
            cheap_model="Euler-Bernoulli beam theory",
            corrected_model="Timoshenko",
            reference_model="3D FEM",
            mean_error_before=0.0207,
            mean_error_after=0.0035,
            slenderness_range=(4, 20),
            evidence_refs=[f"fem-vs-beam-Lh{r}" for r in (4, 6, 8, 12, 20)],
            run_id="phase7.5",
        )
        assert "shear deformation" in lesson.statement
        assert lesson.evidence_level is not EvidenceLevel.EXPERIMENTALLY_VALIDATED
        # Five comparisons from ONE study is one independent run, so SIMULATED.
        assert lesson.evidence_level is EvidenceLevel.SIMULATED
        assert any("ASSUMED" in a for a in lesson.assumptions)
        assert len(memory.knowledge(lesson.domain)) == 1
