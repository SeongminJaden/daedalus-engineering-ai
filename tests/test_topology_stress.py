"""Stress-constrained topology optimisation.

The gate that matters here is the adjoint sensitivity. Everything else in the
method is arithmetic on top of it, and a wrong gradient produces a design that
looks plausible and is not an optimum of anything.
"""

import numpy as np
import pytest

from optimization.topology.simp import (SimpProblem, compliance_and_sensitivity,
                                        optimize, stiffness_scale)
from optimization.topology.stress import (MIN_DENSITY, StressProblem,
                                          _project_to_volume, evaluate,
                                          optimize_constrained, optimize_stress,
                                          p_norm_sensitivity, p_norm_stress,
                                          relaxed_stress)
from physics.fem.mesh import l_bracket_mesh
from physics.fem.solver import solve_linear_elasticity

E, NU, LOAD, SIZE = 71.7e9, 0.33, 300.0, 0.10


def make_problem(n=8, nz=1, volume_fraction=0.5, limit_pa=60e6, p=8.0,
                 tol=1e-8):
    # The exact arm thickness is irrelevant to a topology benchmark, so the
    # rounded arm is accepted deliberately rather than constraining n.
    mesh = l_bracket_mesh(SIZE, 0.4, 0.01, n, nz=nz, allow_snapping=True)
    top = mesh.nodes_where(np.abs(mesh.node_coords[:, 1] - SIZE) < 1e-9)
    tip = mesh.nodes_where(np.abs(mesh.node_coords[:, 0] - SIZE) < 1e-9)
    base = SimpProblem(mesh=mesh, youngs_modulus_pa=E, poisson_ratio=NU,
                       fixed_nodes=top, load_nodes=tip, total_load_n=-LOAD,
                       load_direction=1, volume_fraction=volume_fraction,
                       filter_radius_elements=1.5)
    return StressProblem(base=base, stress_limit_pa=limit_pa, p_norm=p,
                         solver_tolerance=tol)


def test_adjoint_sensitivity_matches_finite_difference():
    """The core gate: d(sigma_PN)/dx from the adjoint against a difference quotient.

    The tolerance is on the error scaled by the largest sensitivity, not on the
    relative error of each entry. Entries near zero have a relative error
    dominated by how precisely the two solves converged, and judging them by
    their own magnitude fails a gradient that is correct.
    """
    problem = make_problem(n=8, nz=1, tol=1e-12)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.4, 0.9, problem.n_elements())

    pn0, analytic, _, _ = p_norm_sensitivity(problem, x)
    scale = np.abs(analytic).max()
    assert scale > 0

    step = 1e-6
    probes = rng.choice(problem.n_elements(), size=12, replace=False)
    worst = 0.0
    for e in probes:
        plus = x.copy(); plus[e] += step
        minus = x.copy(); minus[e] -= step
        numeric = (evaluate(problem, plus).p_norm
                   - evaluate(problem, minus).p_norm) / (2.0 * step)
        worst = max(worst, abs(numeric - analytic[e]) / scale)
    assert worst < 1e-4, f"worst scaled adjoint error {worst:.2e}"


def test_p_norm_over_estimates_the_true_maximum():
    """The aggregate is conservative, and that is the safe direction.

    The plain sum form used here sums over every element, so it can only exceed
    its largest term. A design that satisfies the aggregate therefore satisfies
    the real elementwise limit. The averaged form of the P-norm does the
    opposite and under-reports the maximum; this asserts which one is in use,
    because getting it backwards means shipping overstressed parts.
    """
    problem = make_problem(n=8, nz=1)
    x = np.full(problem.n_elements(), 0.6)
    ev = evaluate(problem, x)
    relaxed, _ = relaxed_stress(problem, x, ev.displacements)
    true_max = relaxed.max()

    previous = None
    for p in (2.0, 4.0, 8.0, 16.0, 32.0):
        problem.p_norm = p
        aggregate = p_norm_stress(problem, relaxed) * problem.stress_limit_pa
        assert aggregate >= true_max, f"P={p} under-reports the maximum"
        if previous is not None:
            assert aggregate < previous, f"P={p} did not tighten"
        previous = aggregate
    # Converging from above: by P=32 it is within a few percent.
    assert previous / true_max < 1.1


def test_relaxation_removes_the_stress_singularity():
    """Emptying an element must drive its stress measure to zero.

    Without relaxation it does not. The unrelaxed stress of a vanishing element
    RISES and settles on a finite value, because the strain grows as fast as
    the stiffness falls. A constraint on that quantity forbids removing
    material exactly where material most needs removing, which is the
    singularity the qp relaxation exists to break.
    """
    problem = make_problem(n=8, nz=1)
    base = problem.base
    x = np.full(problem.n_elements(), 1.0)
    ev = evaluate(problem, x)
    relaxed, _ = relaxed_stress(problem, x, ev.displacements)
    hot = int(np.argmax(relaxed))

    micro_trace, relaxed_trace = [], []
    for d in (1.0, 0.1, 0.01, MIN_DENSITY):
        probe = x.copy(); probe[hot] = d
        solution = solve_linear_elasticity(
            base.mesh, E, NU, fixed_nodes=base.fixed_nodes,
            load_nodes=base.load_nodes, total_load_n=base.total_load_n,
            load_direction=base.load_direction,
            element_scale=stiffness_scale(probe, base.penalty))
        r, m = relaxed_stress(problem, probe, solution.displacements)
        micro_trace.append(m[hot]); relaxed_trace.append(r[hot])

    # Unrelaxed: does not vanish, and is worse than at full density.
    assert micro_trace[-1] > micro_trace[0]
    assert micro_trace[-1] > 0.5 * micro_trace[0]
    # Relaxed: monotonically to nothing.
    assert all(b < a for a, b in zip(relaxed_trace, relaxed_trace[1:]))
    assert relaxed_trace[-1] < 1e-4 * relaxed_trace[0]


def test_projection_holds_the_volume_exactly():
    """The volume is a constraint, not something the step drifts away from."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0.2, 0.8, 200)
    direction = rng.uniform(-1.0, 1.0, 200)
    for target in (0.35, 0.5, 0.65):
        stepped = _project_to_volume(x, direction, 0.1, target)
        assert stepped.mean() == pytest.approx(target, abs=1e-6)
        assert stepped.min() >= MIN_DENSITY - 1e-12
        assert stepped.max() <= 1.0 + 1e-12


def test_projection_saturates_when_the_target_is_out_of_reach():
    """A move limit can put the target volume beyond one step.

    Starting from a mean of 0.5, a move limit of 0.1 cannot reach 0.1 in a
    single step. The step must land on the closest reachable volume rather than
    overshoot the move limit to satisfy the constraint, and the caller reaches
    the target over several iterations instead.
    """
    rng = np.random.default_rng(1)
    x = rng.uniform(0.2, 0.8, 200)
    direction = rng.uniform(-1.0, 1.0, 200)
    stepped = _project_to_volume(x, direction, 0.1, 0.1)
    assert stepped.mean() > 0.1
    assert np.abs(stepped - x).max() <= 2.0 * 0.1 + 1e-12
    # It went as far down as the move limit allows.
    floor = np.maximum(x - 2.0 * 0.1, MIN_DENSITY)
    assert stepped.mean() == pytest.approx(floor.mean(), abs=1e-6)


def test_projection_respects_the_move_limit():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.3, 0.7, 200)
    direction = rng.uniform(-1.0, 1.0, 200)
    stepped = _project_to_volume(x, direction, 0.05, float(x.mean()))
    assert np.abs(stepped - x).max() <= 2.0 * 0.05 + 1e-12


def test_stress_minimisation_lowers_the_p_norm():
    problem = make_problem(n=10, nz=1, volume_fraction=0.5)
    result = optimize_stress(problem, max_iterations=12, move_limit=0.1)
    assert result.final_p_norm < result.p_norm_history[0]
    assert result.volume_fraction == pytest.approx(0.5, abs=1e-6)


def test_stress_constraint_lowers_peak_stress_against_compliance():
    """The point of the method, stated as a comparison at equal volume.

    Compliance minimisation ignores the re-entrant corner and piles material
    against it. Constraining the stress pulls material off the corner and drops
    the peak, and it pays for that in compliance. All three of those have to
    hold at once, or the constraint is not doing what it claims.

    What it does NOT do is round the corner into a clean fillet. On the
    20x20x2 bracket the constrained design is markedly greyer than the
    compliance one (grey fraction 0.59 against 0.80). That is not asserted
    here, because at the mesh size this test can afford the compliance design
    is itself 95% grey and the comparison carries no signal; it is recorded in
    the module documentation with the mesh it was measured on.
    """
    problem = make_problem(n=10, nz=1, volume_fraction=0.5, p=8.0)
    base = problem.base

    unconstrained = optimize(base, max_iterations=25)
    loose = evaluate(problem, unconstrained.density)
    # Set the limit below what compliance minimisation achieves, so the
    # constraint actually binds. A constraint that is already satisfied proves
    # nothing.
    # 0.9 and not lower: on a 10x10x1 mesh the design has too little freedom
    # to shed 20%, and raising the penalty does not change that (the p-norm
    # plateaus near 1.10 whatever the weight). Demanding it would test the mesh
    # resolution, not the constraint.
    problem.stress_limit_pa = 0.9 * loose.max_relaxed_stress_pa

    constrained = optimize_constrained(problem, max_iterations=30,
                                       move_limit=0.1, penalty_weight=10.0,
                                       penalty_growth=1.4)
    assert constrained.found_feasible, "no feasible iterate was reached"
    design = constrained.best_feasible_density

    tight = evaluate(problem, design)
    violating = evaluate(problem, unconstrained.density)
    assert violating.p_norm > 1.0, "the limit did not bind on the compliance design"
    assert tight.p_norm <= 1.0
    assert tight.max_relaxed_stress_pa < violating.max_relaxed_stress_pa

    # It is paid for in compliance.
    c_free, _, _ = compliance_and_sensitivity(base, unconstrained.density)
    c_held, _, _ = compliance_and_sensitivity(base, design)
    assert c_held > c_free


def test_reported_history_describes_the_returned_design():
    """A penalty method can finish outside the feasible set.

    The final history entry has to belong to the density that is returned, or
    the reported constraint value was never checked on the reported design.
    """
    problem = make_problem(n=8, nz=1, volume_fraction=0.5)
    result = optimize_constrained(problem, max_iterations=6, move_limit=0.1)
    assert len(result.p_norm_history) == result.iterations + 1
    final = evaluate(problem, result.density)
    assert final.p_norm == pytest.approx(result.final_p_norm, rel=1e-6)
