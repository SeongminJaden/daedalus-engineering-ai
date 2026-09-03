"""Designing for several load cases, and the cost of designing for one.

The measurement that matters is the cross evaluation: a structure optimised
for one load, evaluated under the others. On the cantilever the single case
designs come out between 1.9 and 9.3 times worse under a case they were not
designed for, and the equal weight design is never worse than 1.86 times the
best design for any case. Those numbers are in docs/topology_design.md; what
is pinned here is the machinery that produced them.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.materials import get_material
from optimization.topology import SimpProblem, optimize
from optimization.topology.multiload import (LoadCase, case_compliance,
                                             couple_force_vector,
                                             cross_evaluation,
                                             format_cross_table,
                                             optimize_multiload,
                                             weighted_compliance)
from optimization.topology.verify import elements_touching
from physics.fem.mesh import solid_box_mesh

MATERIAL = get_material("al_7075_t6")


def small_problem():
    mesh = solid_box_mesh(0.4, 0.1, 0.05, 12, 6, 2)
    fixed, tip = mesh.nodes_at_x(0.0), mesh.nodes_at_x(0.4)
    return SimpProblem(
        mesh=mesh, youngs_modulus_pa=MATERIAL.youngs_modulus_pa,
        poisson_ratio=MATERIAL.poisson_ratio, fixed_nodes=fixed, load_nodes=tip,
        total_load_n=-500.0, load_direction=1, volume_fraction=0.4,
        filter_radius_elements=2.0,
        passive_solid=elements_touching(mesh, tip) | elements_touching(mesh, fixed))


# --- a couple is a couple ----------------------------------------------------

def test_a_couple_has_no_net_force_and_the_torque_asked_for():
    problem = small_problem()
    mesh = problem.mesh
    nodes = mesh.nodes_at_x(0.4)
    vector = couple_force_vector(mesh, nodes, torque_nm=25.0, axis=0)
    forces = vector.reshape(-1, 3)[nodes]
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-9)

    coords = np.asarray(mesh.node_coords)[nodes]
    radius = coords - coords.mean(axis=0)
    moment = np.cross(radius, forces).sum(axis=0)
    assert moment[0] == pytest.approx(25.0, rel=1e-9)
    assert np.allclose(moment[1:], 0.0, atol=1e-9)
    # Nothing outside the loaded face carries a force.
    other = np.setdiff1d(np.arange(mesh.n_nodes), nodes)
    assert np.allclose(vector.reshape(-1, 3)[other], 0.0)


def test_a_face_with_no_lever_arm_is_refused():
    problem = small_problem()
    single = np.array([0])
    with pytest.raises(ValueError, match="lever arm"):
        couple_force_vector(problem.mesh, single, 10.0, axis=0)


# --- the objective is the weighted sum ---------------------------------------

def test_the_objective_is_exactly_the_weighted_sum_of_the_cases():
    problem = small_problem()
    density = np.full(problem.n_elements(), 0.5)
    tip = problem.mesh.nodes_at_x(0.4)
    cases = [LoadCase("y", tip, total_load_n=-500.0, load_direction=1, weight=3.0),
             LoadCase("z", tip, total_load_n=-500.0, load_direction=2, weight=1.0)]
    total, gradient, per_case = weighted_compliance(problem, density, cases)
    expected = 0.75 * per_case["y"] + 0.25 * per_case["z"]
    assert total == pytest.approx(expected, rel=1e-12)
    assert gradient.shape == (problem.n_elements(),)
    assert np.all(gradient <= 0.0)      # more material never costs compliance


def test_one_case_reproduces_the_single_load_objective():
    problem = small_problem()
    density = np.full(problem.n_elements(), 0.5)
    case = LoadCase("y", problem.mesh.nodes_at_x(0.4), total_load_n=-500.0,
                    load_direction=1)
    total, _gradient, per_case = weighted_compliance(problem, density, [case])
    alone, _sensitivity = case_compliance(problem, density, case)
    assert total == pytest.approx(alone, rel=1e-12)
    assert per_case["y"] == pytest.approx(alone, rel=1e-12)


def test_weights_must_be_usable():
    problem = small_problem()
    density = np.full(problem.n_elements(), 0.5)
    tip = problem.mesh.nodes_at_x(0.4)
    with pytest.raises(ValueError, match="non-negative"):
        weighted_compliance(problem, density,
                            [LoadCase("y", tip, total_load_n=-1.0, weight=0.0)])
    with pytest.raises(ValueError, match="at least one"):
        optimize_multiload(problem, [], max_iterations=1)


# --- the trap, measured ------------------------------------------------------

@pytest.mark.slow
def test_a_single_case_design_is_worse_under_another_case():
    """The reason this module exists. Both designs are legitimate optima of
    what they were given, and each is poor under the other's load."""
    problem = small_problem()
    tip = problem.mesh.nodes_at_x(0.4)
    y_case = LoadCase("y", tip, total_load_n=-500.0, load_direction=1)
    z_case = LoadCase("z", tip, total_load_n=-500.0, load_direction=2)

    y_design = optimize_multiload(problem, [y_case], max_iterations=30).density
    z_design = optimize_multiload(problem, [z_case], max_iterations=30).density
    both = optimize_multiload(problem, [y_case, z_case], max_iterations=30).density

    rows = cross_evaluation(problem, {"y only": y_design, "z only": z_design,
                                      "both": both}, [y_case, z_case])
    table = {r["design"]: r for r in rows}
    print("\n" + format_cross_table(rows))

    assert table["y only"]["y"] < table["z only"]["y"]
    assert table["z only"]["z"] < table["y only"]["z"]
    # The compromise is not the best at either and is not the worst at either.
    for case in ("y", "z"):
        best = min(r[case] for r in rows)
        worst = max(r[case] for r in rows)
        assert best < table["both"][case] < worst


@pytest.mark.slow
def test_a_multi_case_run_reaches_the_volume_and_reports_every_case():
    problem = small_problem()
    tip = problem.mesh.nodes_at_x(0.4)
    cases = [LoadCase("y", tip, total_load_n=-500.0, load_direction=1),
             LoadCase("torsion", tip,
                      force_vector=couple_force_vector(problem.mesh, tip, 20.0))]
    result = optimize_multiload(problem, cases, max_iterations=20)
    assert set(result.final_per_case) == {"y", "torsion"}
    assert result.volume_fraction == pytest.approx(problem.volume_fraction, abs=0.02)
    assert result.volume_history[0] == pytest.approx(problem.volume_fraction,
                                                     abs=0.005)
    assert result.objective_history[-1] < result.objective_history[0]
    assert np.all(result.density[problem.passive_solid] == 1.0)


def test_a_single_case_multiload_run_matches_the_plain_optimiser():
    """The same problem through both paths, so the new objective is not a
    different objective wearing the same name."""
    problem = small_problem()
    case = LoadCase("y", problem.mesh.nodes_at_x(0.4), total_load_n=-500.0,
                    load_direction=1)
    plain = optimize(problem, max_iterations=15)
    multi = optimize_multiload(problem, [case], max_iterations=15)
    assert multi.objective_history[-1] == pytest.approx(
        plain.compliance_history[-1], rel=0.05)
