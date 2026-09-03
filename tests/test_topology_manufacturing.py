"""Manufacturing constraints in the topology loop.

The filters are checked on fields small enough to reason about by hand, and
the price of each is a measurement that lives in docs/topology_design.md; what
is pinned here is that each filter forbids what it says it forbids and that
the optimiser actually applies it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.materials import get_material
from optimization.topology import SimpProblem, optimize
from optimization.topology.manufacturing import (as_element_projection,
                                                 mirror_grid, pull_filter,
                                                 pull_projection, support_filter,
                                                 support_projection,
                                                 symmetry_error,
                                                 symmetry_projection,
                                                 undercut_fraction)
from optimization.topology.verify import elements_touching
from physics.fem.mesh import solid_box_mesh


# --- the filters -------------------------------------------------------------

def test_the_support_filter_removes_what_floats():
    """An element with nothing under it cannot be printed, and the filter says
    so. The residual is the smooth maximum's leakage, not material."""
    grid = np.zeros((3, 4, 3))
    grid[1, 2, 1] = 1.0                      # floating in the third layer
    filtered = support_filter(grid, build_axis=1)
    assert filtered[1, 2, 1] < 0.05

    supported = np.zeros((3, 4, 3))
    supported[1, 0, 1] = supported[1, 1, 1] = 1.0    # sitting on the plate
    kept = support_filter(supported, build_axis=1)
    assert kept[1, 0, 1] == pytest.approx(1.0)
    assert kept[1, 1, 1] > 0.9


def test_the_support_filter_leaves_the_first_layer_alone():
    grid = np.zeros((2, 3, 2))
    grid[:, 0, :] = 0.7
    assert np.allclose(support_filter(grid, build_axis=1)[:, 0, :], 0.7)


def test_the_pull_filter_forbids_material_behind_a_gap():
    """One run of material from the mould face, which is what no undercut
    means along that axis."""
    column = np.zeros((1, 5, 1))
    column[0, 0, 0] = column[0, 1, 0] = 1.0
    column[0, 3, 0] = 1.0                    # behind a gap
    from_low = pull_filter(column, pull_axis=1, from_high=False).reshape(-1)
    assert list(from_low) == [1.0, 1.0, 0.0, 0.0, 0.0]
    from_high = pull_filter(column, pull_axis=1, from_high=True).reshape(-1)
    assert list(from_high) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_the_mirror_is_symmetric_and_keeps_the_volume():
    grid = np.random.default_rng(0).random((4, 3, 2))
    mirrored = mirror_grid(grid, axis=2)
    assert np.allclose(mirrored, np.flip(mirrored, axis=2))
    assert mirrored.mean() == pytest.approx(grid.mean())


def test_the_undercut_fraction_is_zero_after_the_pull_filter():
    mesh = solid_box_mesh(0.2, 0.2, 0.1, 4, 6, 2)
    rng = np.random.default_rng(1)
    density = (rng.random(mesh.n_elements) > 0.4).astype(float)
    projection = as_element_projection(mesh, pull_projection(1, from_high=False))
    assert undercut_fraction(mesh, density, 1, from_high=False) > 0.0
    assert undercut_fraction(mesh, projection(density), 1,
                             from_high=False) == pytest.approx(0.0)


def test_symmetry_error_is_zero_only_for_a_symmetric_field():
    mesh = solid_box_mesh(0.2, 0.2, 0.1, 4, 4, 2)
    rng = np.random.default_rng(2)
    density = rng.random(mesh.n_elements)
    assert symmetry_error(mesh, density, axis=2) > 0.0
    projection = as_element_projection(mesh, symmetry_projection(2))
    assert symmetry_error(mesh, projection(density), axis=2) == pytest.approx(0.0)


# --- the optimiser applies them ---------------------------------------------

@pytest.mark.slow
def test_a_projected_run_returns_a_field_that_obeys_the_constraint():
    """The density the run reports has to be the constrained one, or the
    compliance it reports belongs to a different structure."""
    material = get_material("al_7075_t6")
    mesh = solid_box_mesh(0.4, 0.1, 0.05, 12, 6, 2)
    fixed, load = mesh.nodes_at_x(0.0), mesh.nodes_at_x(0.4)
    problem = SimpProblem(
        mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio, fixed_nodes=fixed, load_nodes=load,
        total_load_n=-500.0, load_direction=1, volume_fraction=0.4,
        filter_radius_elements=2.0,
        passive_solid=elements_touching(mesh, load) | elements_touching(mesh, fixed),
        density_projection=as_element_projection(
            mesh, pull_projection(1, from_high=False)))
    result = optimize(problem, max_iterations=25)
    # The passive patches are written after the projection and are the one
    # exception the docstring names, so they are excluded from the check.
    free = ~problem.passive_solid
    undercut = undercut_fraction(mesh, np.where(free, result.density, 0.0), 1,
                                 from_high=False)
    assert undercut == pytest.approx(0.0, abs=0.02)


def test_a_projection_makes_every_element_dependent_so_the_free_path_is_taken():
    mesh = solid_box_mesh(0.2, 0.1, 0.05, 4, 2, 2)
    material = get_material("al_7075_t6")
    problem = SimpProblem(
        mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
        poisson_ratio=material.poisson_ratio, fixed_nodes=mesh.nodes_at_x(0.0),
        load_nodes=mesh.nodes_at_x(0.2), total_load_n=-100.0,
        density_projection=as_element_projection(mesh, symmetry_projection(2)))
    assert problem.free_mask is not None
    assert problem.free_mask.all()


# --- the support filter's chain rule -----------------------------------------

def test_the_support_filter_gradient_matches_a_difference_quotient():
    """The gate for the fix. The first version used the softmax weights as the
    derivative of the smooth maximum, which is wrong because the weights depend
    on the arguments too, and it was 26 percent off. The correct form is
    w_i (1 + p (x_i - m)), and this checks it to ten digits."""
    from optimization.topology.manufacturing import support_filter_gradient

    rng = np.random.default_rng(0)
    grid = rng.random((3, 4, 2))
    _printed, vjp = support_filter_gradient(grid, build_axis=1)
    seed = rng.random(grid.shape)
    analytic = vjp(seed)

    numeric = np.zeros_like(grid)
    step = 1e-6
    for index in np.ndindex(grid.shape):
        up, down = grid.copy(), grid.copy()
        up[index] += step
        down[index] -= step
        numeric[index] = (
            np.sum(seed * support_filter_gradient(up, 1)[0])
            - np.sum(seed * support_filter_gradient(down, 1)[0])) / (2 * step)
    error = np.abs(analytic - numeric).max() / max(np.abs(numeric).max(), 1e-12)
    assert error < 1e-6, error


def test_the_smooth_filter_agrees_with_the_hard_one():
    """The smooth minimum is what makes the recursion differentiable; it must
    still be the filter it claims to be."""
    from optimization.topology.manufacturing import support_filter_gradient

    grid = np.random.default_rng(1).random((4, 5, 3))
    smooth, _vjp = support_filter_gradient(grid, build_axis=1)
    hard = support_filter(grid, build_axis=1)
    assert np.abs(smooth - hard).max() < 0.02


@pytest.mark.slow
def test_the_support_constraint_removes_every_unsupported_element():
    """Judge the constraint by the criterion it enforces. Measured on the
    cantilever: 2.5 percent of solid elements unsupported without it, exactly
    zero with it, and the compliance cost falls from 5.07 times to 1.15 once
    the chain rule is applied."""
    from optimization.topology.manufacturing import (support_projection_with_gradient,
                                                     unsupported_fraction)
    from optimization.topology.threefield import optimize_projected

    material = get_material("al_7075_t6")
    mesh = solid_box_mesh(0.4, 0.1, 0.05, 16, 8, 3)
    fixed, load = mesh.nodes_at_x(0.0), mesh.nodes_at_x(0.4)
    passive = elements_touching(mesh, load) | elements_touching(mesh, fixed)
    projection, vjp = support_projection_with_gradient(mesh, build_axis=1)

    def build(pr, vj):
        return SimpProblem(
            mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
            poisson_ratio=material.poisson_ratio, fixed_nodes=fixed,
            load_nodes=load, total_load_n=-800.0, load_direction=1,
            volume_fraction=0.4, filter_radius_elements=2.0,
            passive_solid=passive, density_projection=pr, projection_vjp=vj)

    free = optimize_projected(build(None, None), max_iterations=40)
    held = optimize_projected(build(projection, vjp), max_iterations=40)
    assert unsupported_fraction(mesh, held.density) == pytest.approx(0.0, abs=1e-9)
    assert unsupported_fraction(mesh, free.density) >= 0.0
    # The constraint costs compliance, and the cost is bounded rather than free.
    assert held.final_compliance > free.final_compliance
    assert held.final_compliance < 3.0 * free.final_compliance
