"""Three-field SIMP: Heaviside projection and the chain rule through it.

The gate is `test_chain_rule_matches_finite_difference`. Phases 13 and 14a
verified adjoints that give the derivative with respect to the physical
density; with a filter and a projection in front, that is no longer the
derivative the optimizer needs. An unchecked chain rule would descend a
gradient belonging to a different problem and still return a plausible shape.
"""

import numpy as np
import pytest

from optimization.topology.export import grey_fraction
from optimization.topology.projection import (DEFAULT_ETA, BetaSchedule,
                                              DesignTransform,
                                              build_density_filter, project,
                                              projection_derivative)
from optimization.topology.simp import (MIN_DENSITY, SimpProblem,
                                        compliance_and_sensitivity, optimize)
from optimization.topology.threefield import optimize_projected
from physics.fem.mesh import solid_box_mesh

E, NU, LOAD = 71.7e9, 0.33, 200.0
LENGTH, HEIGHT, WIDTH = 0.16, 0.05, 0.02


def make_problem(nx=10, ny=4, nz=2, volume_fraction=0.4):
    mesh = solid_box_mesh(LENGTH, HEIGHT, WIDTH, nx, ny, nz)
    return SimpProblem(
        mesh=mesh, youngs_modulus_pa=E, poisson_ratio=NU,
        fixed_nodes=mesh.nodes_at_x(0.0), load_nodes=mesh.nodes_at_x(LENGTH),
        total_load_n=-LOAD, load_direction=1,
        volume_fraction=volume_fraction, filter_radius_elements=1.5)


# --- the gate ----------------------------------------------------------------

def test_chain_rule_matches_finite_difference():
    """d/d(design) of compliance, through the filter and the projection."""
    problem = make_problem()
    rng = np.random.default_rng(0)
    design = rng.uniform(0.3, 0.8, problem.mesh.n_elements)

    for beta in (1.0, 4.0, 16.0):
        transform = DesignTransform.for_mesh(problem.mesh, 1.5, beta=beta)

        def compliance_of(x):
            return compliance_and_sensitivity(problem, transform.physical(x))[0]

        _, d_physical, _ = compliance_and_sensitivity(
            problem, transform.physical(design))
        analytic = transform.chain(d_physical, design)
        scale = np.abs(analytic).max()
        assert scale > 0

        step = 1e-6
        worst = 0.0
        for e in rng.choice(problem.mesh.n_elements, size=8, replace=False):
            plus = design.copy(); plus[e] += step
            minus = design.copy(); minus[e] -= step
            numeric = (compliance_of(plus) - compliance_of(minus)) / (2 * step)
            worst = max(worst, abs(numeric - analytic[e]) / scale)
        assert worst < 1e-4, f"beta={beta}: worst scaled error {worst:.2e}"


# --- the projection itself ---------------------------------------------------

def test_projection_pushes_towards_solid_and_void():
    x = np.linspace(0.0, 1.0, 21)
    gentle = project(x, 1.0)
    sharp = project(x, 32.0)
    # Endpoints and the threshold are fixed points for any beta.
    for beta in (1.0, 8.0, 32.0):
        projected = project(x, beta)
        assert projected[0] == pytest.approx(0.0, abs=1e-12)
        assert projected[-1] == pytest.approx(1.0, abs=1e-12)
        assert project(np.array([DEFAULT_ETA]), beta)[0] == pytest.approx(0.5)
    # Sharper means further from the middle on both sides.
    below, above = x < DEFAULT_ETA, x > DEFAULT_ETA
    assert np.all(sharp[below] <= gentle[below] + 1e-12)
    assert np.all(sharp[above] >= gentle[above] - 1e-12)


def test_projection_is_monotone_and_bounded():
    x = np.linspace(0.0, 1.0, 101)
    for beta in (1.0, 8.0, 64.0):
        projected = project(x, beta)
        assert np.all(np.diff(projected) >= -1e-12)
        assert projected.min() >= -1e-12 and projected.max() <= 1.0 + 1e-12


def test_projection_derivative_matches_finite_difference():
    x = np.linspace(0.02, 0.98, 49)
    for beta in (1.0, 8.0, 32.0):
        step = 1e-7
        numeric = (project(x + step, beta) - project(x - step, beta)) / (2 * step)
        analytic = projection_derivative(x, beta)
        assert np.abs(numeric - analytic).max() / analytic.max() < 1e-5


def test_a_tiny_beta_is_the_identity():
    """The formula divides by tanh terms that vanish as beta does."""
    x = np.linspace(0.0, 1.0, 11)
    assert project(x, 0.0) == pytest.approx(x)
    assert projection_derivative(x, 0.0) == pytest.approx(np.ones_like(x))


# --- the filter --------------------------------------------------------------

def test_the_density_filter_is_an_average():
    """Row-normalised, so a uniform field passes through unchanged."""
    mesh = solid_box_mesh(LENGTH, HEIGHT, WIDTH, 8, 4, 2)
    matrix = build_density_filter(mesh, 1.5)
    ones = np.ones(mesh.n_elements)
    assert matrix @ ones == pytest.approx(ones)
    assert matrix.min() >= 0.0


def test_the_chain_uses_the_exact_filter_transpose():
    """<H x, y> == <x, H^T y>, or the forward and backward passes disagree."""
    mesh = solid_box_mesh(LENGTH, HEIGHT, WIDTH, 8, 4, 2)
    matrix = build_density_filter(mesh, 1.5)
    rng = np.random.default_rng(3)
    x = rng.normal(size=mesh.n_elements)
    y = rng.normal(size=mesh.n_elements)
    assert float((matrix @ x) @ y) == pytest.approx(float(x @ (matrix.T @ y)))


# --- continuation ------------------------------------------------------------

def test_beta_continuation_rises_and_stops():
    schedule = BetaSchedule(start=1.0, maximum=8.0, every=10, factor=2.0)
    assert schedule.beta_at(1) == 1.0
    assert schedule.beta_at(10) == 1.0
    assert schedule.beta_at(11) == 2.0
    assert schedule.beta_at(21) == 4.0
    assert schedule.beta_at(31) == 8.0
    assert schedule.beta_at(101) == 8.0      # capped, not runaway


# --- what it buys ------------------------------------------------------------

def test_projection_produces_a_blacker_design_than_plain_simp():
    """The point of the formulation, as a measured comparison.

    Both at the same volume on the same mesh. The projected run must be less
    grey; it is allowed to be worse in compliance, because a design that can
    actually be manufactured is a different and harder thing to ask for.
    """
    problem = make_problem(nx=16, ny=6, nz=1)
    plain = optimize(problem, max_iterations=60)
    projected = optimize_projected(
        problem, max_iterations=90, move_limit=0.1,
        schedule=BetaSchedule(start=1.0, maximum=32.0, every=15))

    assert grey_fraction(projected.density) < grey_fraction(plain.density)
    assert projected.volume_fraction == pytest.approx(
        problem.volume_fraction, abs=1e-3)


def test_the_volume_constraint_is_enforced_on_the_physical_field():
    """Holding the design variable's mean fixed would mean nothing.

    The solver sees the projected field, so that is the field whose volume has
    to match the target.
    """
    problem = make_problem(nx=12, ny=4, nz=1, volume_fraction=0.35)
    result = optimize_projected(problem, max_iterations=25, move_limit=0.1)
    assert result.density.mean() == pytest.approx(0.35, abs=1e-3)
    for volume in result.volume_history:
        assert volume == pytest.approx(0.35, abs=1e-2)


def test_the_physical_density_never_reaches_zero():
    """A true zero makes the stiffness matrix singular, so the solve fails."""
    problem = make_problem(nx=10, ny=4, nz=1)
    result = optimize_projected(
        problem, max_iterations=40,
        schedule=BetaSchedule(start=1.0, maximum=64.0, every=10))
    assert result.density.min() >= MIN_DENSITY - 1e-12


def test_a_small_filter_radius_fragments_the_projected_design():
    """The failure a grey fraction alone would hide.

    Sharpening the design lowers the grey fraction and, at a filter radius of
    1.5 elements, also breaks it into pieces. Material not connected to the
    load path carries nothing, so that design is not an improvement on the grey
    one. Widening the filter is what holds it together, and it costs
    compliance. Both halves are asserted so neither can be quietly dropped.
    """
    from optimization.topology.export import connected_fraction

    mesh_problem = make_problem(nx=24, ny=8, nz=2)
    schedule = lambda: BetaSchedule(start=1.0, maximum=64.0, every=20)

    narrow = make_problem(nx=24, ny=8, nz=2)
    narrow.filter_radius_elements = 1.5
    wide = make_problem(nx=24, ny=8, nz=2)
    wide.filter_radius_elements = 3.0

    thin = optimize_projected(narrow, max_iterations=200, move_limit=0.1,
                              schedule=schedule())
    thick = optimize_projected(wide, max_iterations=200, move_limit=0.1,
                               schedule=schedule())

    # Both are black and white.
    assert grey_fraction(thin.density) < 0.2
    assert grey_fraction(thick.density) < 0.2
    # Only the wider filter keeps the design in one piece.
    assert connected_fraction(mesh_problem.mesh, thin.density) < 0.95
    assert connected_fraction(mesh_problem.mesh, thick.density) > 0.99
    # And it pays for that in compliance.
    assert thick.final_compliance > thin.final_compliance


def test_connected_fraction_reads_one_for_a_solid_block():
    from optimization.topology.export import connected_fraction

    problem = make_problem(nx=8, ny=4, nz=2)
    assert connected_fraction(problem.mesh,
                              np.ones(problem.mesh.n_elements)) == 1.0
    assert connected_fraction(problem.mesh,
                              np.zeros(problem.mesh.n_elements)) == 0.0
