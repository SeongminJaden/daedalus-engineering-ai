"""Phase 13 verification: SIMP topology optimization.

The gate is the sensitivity. Everything the optimizer does follows from
dc/dx_e, and a wrong derivative produces a plausible-looking shape that is not
an optimum of anything. It is checked against finite differences of the actual
objective, element by element.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from optimization.topology import (  # noqa: E402
    MIN_DENSITY, PENALTY, SimpProblem, VOID_STIFFNESS_RATIO,
    apply_sensitivity_filter, build_filter_weights, checkerboard_metric,
    compliance_and_sensitivity, export_stl, grey_fraction,
    largest_connected_component, oc_update, optimize, solve, stiffness_scale,
    stiffness_scale_derivative, voxel_surface,
)
from physics.fem.mesh import solid_box_mesh  # noqa: E402
from physics.fem.solver import solve_linear_elasticity  # noqa: E402

E, NU, LOAD = 71.7e9, 0.33, 200.0


def make_problem(nx=12, ny=4, nz=2, length=0.12, height=0.04, width=0.02,
                 volume_fraction=0.4):
    mesh = solid_box_mesh(length, height, width, nx, ny, nz)
    return SimpProblem(
        mesh=mesh, youngs_modulus_pa=E, poisson_ratio=NU,
        fixed_nodes=mesh.nodes_at_x(0.0), load_nodes=mesh.nodes_at_x(length),
        total_load_n=-LOAD, load_direction=1,
        volume_fraction=volume_fraction, filter_radius_elements=1.5)


# =========================================================================== #
# the SIMP interpolation
# =========================================================================== #
def test_stiffness_scale_endpoints():
    assert stiffness_scale(np.array([1.0]))[0] == pytest.approx(1.0)
    assert stiffness_scale(np.array([0.0]))[0] == pytest.approx(
        VOID_STIFFNESS_RATIO)


def test_void_stiffness_is_small_but_never_zero():
    """A truly zero stiffness makes K singular and the solve fails rather than
    returning a soft region."""
    assert 0 < VOID_STIFFNESS_RATIO < 1e-6
    assert np.all(stiffness_scale(np.zeros(10)) > 0)


def test_penalty_makes_intermediate_density_inefficient():
    """The point of p=3: half density buys only an eighth of the stiffness, so
    grey is a bad deal and the optimizer is pushed toward solid or void."""
    assert PENALTY == 3.0
    half = stiffness_scale(np.array([0.5]))[0]
    assert half == pytest.approx(0.5 ** 3, abs=1e-6)
    assert half < 0.5


def test_scale_derivative_matches_finite_difference():
    x = np.array([0.2, 0.5, 0.8])
    delta = 1e-7
    numeric = (stiffness_scale(x + delta) - stiffness_scale(x - delta)) / (2 * delta)
    assert np.allclose(stiffness_scale_derivative(x), numeric, rtol=1e-5)


# =========================================================================== #
# 1. THE GATE: sensitivity against finite differences
# =========================================================================== #
def test_sensitivity_matches_finite_differences():
    """dc/dx_e from the self-adjoint formula, against a re-solve either side.

    This is the whole optimizer in one check: if the derivative is wrong the
    result is a shape that optimises nothing.
    """
    problem = make_problem()
    rng = np.random.default_rng(0)
    x = rng.uniform(0.3, 0.9, problem.n_elements())
    _, sensitivity, _ = compliance_and_sensitivity(problem, x)

    delta = 1e-6
    worst = 0.0
    for element in rng.choice(problem.n_elements(), size=6, replace=False):
        up, down = x.copy(), x.copy()
        up[element] += delta
        down[element] -= delta
        c_up, _, _ = compliance_and_sensitivity(problem, up)
        c_down, _, _ = compliance_and_sensitivity(problem, down)
        numeric = (c_up - c_down) / (2 * delta)
        worst = max(worst, abs(sensitivity[element] - numeric)
                    / max(abs(numeric), 1e-30))
    assert worst < 1e-3, f"worst relative sensitivity error {worst:.3e}"


def test_sensitivity_is_negative_everywhere():
    """Adding material can only reduce compliance."""
    problem = make_problem()
    x = np.full(problem.n_elements(), 0.5)
    _, sensitivity, _ = compliance_and_sensitivity(problem, x)
    assert np.all(sensitivity <= 0)


def test_compliance_computed_two_ways_agrees():
    """c = F.U from the solver, against sum(scale * u_e^T Ke0 u_e). Two
    different routes to the same number."""
    problem = make_problem()
    x = np.full(problem.n_elements(), 0.6)
    compliance, _, solution = compliance_and_sensitivity(problem, x)
    assert solution.compliance() == pytest.approx(compliance, rel=1e-9)


# =========================================================================== #
# 2. the solid regression bridge
# =========================================================================== #
def test_full_density_reproduces_the_phase_7_solid_solve():
    """x = 1 everywhere must give exactly the plain FEM result, or the scaled
    path has drifted from the verified one."""
    problem = make_problem()
    mesh = problem.mesh
    scaled = solve(problem, np.ones(problem.n_elements()))
    plain = solve_linear_elasticity(
        mesh, E, NU, mesh.nodes_at_x(0.0), mesh.nodes_at_x(mesh.nx * mesh.dx),
        -LOAD, 1)
    assert scaled.tip_deflection() == pytest.approx(plain.tip_deflection(),
                                                    rel=1e-9)
    # Absolute tolerance scaled to the solution: the out-of-plane components of
    # this symmetric problem are numerical zero (order 1e-18) and comparing
    # those against each other measures round-off, not agreement.
    magnitude = np.abs(plain.displacements).max()
    assert np.allclose(scaled.displacements, plain.displacements, rtol=1e-9,
                       atol=1e-12 * magnitude)


def test_lower_density_is_more_compliant():
    problem = make_problem()
    n = problem.n_elements()
    stiff, _, _ = compliance_and_sensitivity(problem, np.ones(n))
    soft, _, _ = compliance_and_sensitivity(problem, np.full(n, 0.5))
    assert soft > stiff


def test_zero_scale_is_rejected_by_the_solver():
    problem = make_problem()
    mesh = problem.mesh
    with pytest.raises(ValueError, match="strictly positive"):
        solve_linear_elasticity(
            mesh, E, NU, mesh.nodes_at_x(0.0),
            mesh.nodes_at_x(mesh.nx * mesh.dx), -LOAD, 1,
            element_scale=np.zeros(mesh.n_elements))


# =========================================================================== #
# 3 & 4. optimization: it improves, and it respects the volume constraint
# =========================================================================== #
@pytest.fixture(scope="module")
def optimized():
    problem = make_problem(nx=16, ny=6, nz=2, length=0.12, height=0.04)
    return problem, optimize(problem, max_iterations=40)


def test_compliance_improves_substantially(optimized):
    _, result = optimized
    history = result.compliance_history
    assert history[-1] < history[0] * 0.6, (
        f"compliance only went {history[0]:.4e} -> {history[-1]:.4e}")


def test_compliance_trend_is_downward(optimized):
    """OC with move limits is not strictly monotone, but the trend must be
    clearly down: comparing halves of the history is the honest test."""
    _, result = optimized
    history = np.array(result.compliance_history)
    first_half = history[: len(history) // 2].mean()
    second_half = history[len(history) // 2:].mean()
    assert second_half < first_half


def test_volume_constraint_is_met(optimized):
    problem, result = optimized
    assert result.volume_fraction == pytest.approx(problem.volume_fraction,
                                                   abs=1e-6)
    for volume in result.volume_history:
        assert volume == pytest.approx(problem.volume_fraction, abs=1e-6)


def test_densities_stay_within_bounds(optimized):
    _, result = optimized
    assert result.density.min() >= MIN_DENSITY - 1e-12
    assert result.density.max() <= 1.0 + 1e-12


def test_oc_update_hits_the_volume_target():
    """The bisection on the Lagrange multiplier must land on the target.

    Targets are kept inside one move limit of the starting mean (0.5): a single
    move-limited step cannot reach an arbitrary volume, and expecting it to
    would be testing the wrong thing.
    """
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(0.2, 0.8, n)
    sensitivity = -rng.uniform(0.1, 10.0, n)
    for target in (0.35, 0.5, 0.65):
        updated = oc_update(x, sensitivity, target)
        assert updated.mean() == pytest.approx(target, abs=1e-6)


def test_oc_reaches_a_distant_target_over_several_steps():
    """A far target is reached by iterating, one move limit at a time."""
    rng = np.random.default_rng(4)
    x = rng.uniform(0.4, 0.6, 200)
    sensitivity = -rng.uniform(0.1, 10.0, 200)
    for _ in range(6):
        x = oc_update(x, sensitivity, 0.15)
    assert x.mean() == pytest.approx(0.15, abs=1e-6)


def test_oc_respects_the_move_limit():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.3, 0.7, 100)
    sensitivity = -rng.uniform(0.1, 100.0, 100)
    updated = oc_update(x, sensitivity, 0.4, move_limit=0.05)
    assert np.abs(updated - x).max() <= 0.05 + 1e-12


# =========================================================================== #
# 5. the filter suppresses checkerboarding
# =========================================================================== #
def test_filter_reduces_checkerboarding():
    """Without a filter the solution alternates solid and void, which is a
    numerical artifact rather than a structure. The metric makes the difference
    measurable instead of a matter of opinion.
    """
    problem = make_problem(nx=16, ny=6, nz=2)
    filtered = optimize(problem, max_iterations=25, use_filter=True)
    unfiltered = optimize(problem, max_iterations=25, use_filter=False)
    metric_filtered = checkerboard_metric(problem.mesh, filtered.density)
    metric_unfiltered = checkerboard_metric(problem.mesh, unfiltered.density)
    assert metric_filtered < metric_unfiltered, (
        f"filtered {metric_filtered:.4f} vs unfiltered {metric_unfiltered:.4f}")


def test_filter_weights_are_local_and_symmetric():
    mesh = solid_box_mesh(0.06, 0.04, 0.02, 6, 4, 2)
    rows, weights = build_filter_weights(mesh, 1.5)
    assert len(rows) == mesh.n_elements
    for neighbours, w in zip(rows, weights):
        assert len(neighbours) == len(w)
        assert np.all(w > 0)
        assert len(neighbours) <= 27


def test_sensitivity_filter_preserves_sign():
    mesh = solid_box_mesh(0.06, 0.04, 0.02, 6, 4, 2)
    rows, weights = build_filter_weights(mesh, 1.5)
    rng = np.random.default_rng(3)
    sensitivity = -rng.uniform(0.1, 5.0, mesh.n_elements)
    density = rng.uniform(0.2, 1.0, mesh.n_elements)
    filtered = apply_sensitivity_filter(sensitivity, density, rows, weights)
    assert np.all(filtered <= 0)


def test_filter_radius_must_be_positive():
    mesh = solid_box_mesh(0.06, 0.04, 0.02, 6, 4, 2)
    with pytest.raises(ValueError, match="radius"):
        build_filter_weights(mesh, 0.0)


# =========================================================================== #
# 6. the load path is physically sensible
# =========================================================================== #
def test_material_migrates_away_from_the_neutral_axis():
    """A bent cantilever wants material at the top and bottom, where the
    bending stress is highest, and not in the middle. That is the physics the
    optimizer should rediscover without being told."""
    problem = make_problem(nx=20, ny=8, nz=2, length=0.12, height=0.04)
    result = optimize(problem, max_iterations=40)
    grid = result.density.reshape(problem.mesh.nx, problem.mesh.ny,
                                  problem.mesh.nz)
    by_height = grid.mean(axis=(0, 2))
    middle = by_height[len(by_height) // 2 - 1: len(by_height) // 2 + 1].mean()
    extremes = (by_height[0] + by_height[-1]) / 2
    assert extremes > middle, (
        f"expected material at the extremes, got extremes {extremes:.3f} vs "
        f"middle {middle:.3f}")


def test_material_concentrates_near_the_support():
    """The bending moment is largest at the root, so material should be too."""
    problem = make_problem(nx=20, ny=8, nz=2, length=0.12, height=0.04)
    result = optimize(problem, max_iterations=40)
    grid = result.density.reshape(problem.mesh.nx, problem.mesh.ny,
                                  problem.mesh.nz)
    by_length = grid.mean(axis=(1, 2))
    assert by_length[:5].mean() > by_length[-5:].mean()


# =========================================================================== #
# export, and the honest limits of it
# =========================================================================== #
def test_voxel_surface_volume_matches_the_retained_voxels(tmp_path):
    problem = make_problem(nx=10, ny=4, nz=2)
    density = np.zeros(problem.n_elements())
    density[: problem.n_elements() // 2] = 1.0
    report = export_stl(problem.mesh, density, tmp_path / "t.stl",
                        largest_component_only=False)
    expected = report.retained_elements * problem.mesh.element_volume
    assert report.volume_m3 == pytest.approx(expected, rel=1e-6)
    assert (tmp_path / "t.stl").stat().st_size > 0


def test_export_refuses_an_empty_threshold(tmp_path):
    problem = make_problem(nx=6, ny=4, nz=2)
    with pytest.raises(ValueError, match="no element survives"):
        export_stl(problem.mesh, np.zeros(problem.n_elements()),
                   tmp_path / "empty.stl")


def test_disconnected_material_is_dropped():
    """Material attached only through an edge or corner carries no load and
    makes the surface non-manifold, so it is removed."""
    mesh = solid_box_mesh(0.06, 0.04, 0.02, 6, 4, 2)
    density = np.zeros(mesh.n_elements)
    cell = np.array([mesh.dx, mesh.dy, mesh.dz])
    index = np.round(mesh.element_centroids() / cell - 0.5).astype(np.int64)
    lookup = {tuple(row): e for e, row in enumerate(index)}
    # A connected block, plus one island touching nothing.
    for i in range(3):
        for j in range(2):
            density[lookup[(i, j, 0)]] = 1.0
            density[lookup[(i, j, 1)]] = 1.0
    density[lookup[(5, 3, 0)]] = 1.0
    cleaned = largest_connected_component(mesh, density)
    assert cleaned[lookup[(5, 3, 0)]] == 0.0
    assert cleaned[lookup[(0, 0, 0)]] == 1.0


def test_grey_fraction_reports_intermediate_density():
    """SIMP leaves grey, and no material is 40% present. The number has to be
    visible so the gap between the field and the thresholded shape is known."""
    assert grey_fraction(np.array([0.0, 1.0, 1.0, 0.0])) == 0.0
    assert grey_fraction(np.array([0.5, 0.5, 0.0, 1.0])) == 0.5


def test_voxel_surface_is_blocky_by_construction():
    """Every triangle is axis aligned. This output is a voxel model and looks
    like one; smoothing it is a separate step."""
    mesh = solid_box_mesh(0.04, 0.04, 0.02, 4, 4, 2)
    density = np.ones(mesh.n_elements)
    vertices, faces, retained = voxel_surface(mesh, density)
    assert retained == mesh.n_elements
    normals = []
    for tri in faces:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        normal = np.cross(b - a, c - a)
        normals.append(normal / max(np.linalg.norm(normal), 1e-30))
    normals = np.array(normals)
    # Axis aligned means each normal has one non-zero component.
    assert np.all(np.isclose(np.abs(normals).max(axis=1), 1.0, atol=1e-9))
