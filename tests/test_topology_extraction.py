"""What a topology run actually hands over, measured in another solver.

SIMP reports the compliance of a density field. The part is that field
thresholded. The two are different structures, and the tests here measure the
difference rather than describing it: the thresholded voxels are written as
their own hexahedral mesh and solved by CalculiX, which is an independent
solver from this project's matrix-free FEM, so the comparison is also a cross
validation.

The first measurement made on this path changed the code. On a plain SIMP run
the elements carrying the point load stayed grey (0.39 at the tip), so every
threshold from 0.2 up severed the load path and there was no part to solve at
all. That is why SimpProblem now carries passive regions, and why these tests
pin both the refusal and the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.materials import get_material
from nodes import calculix as ccx
from optimization.topology import SimpProblem, optimize
from optimization.topology.threefield import optimize_projected
from optimization.topology.verify import (DisconnectedAtThreshold, compliance_from,
                                          elements_touching, format_table,
                                          retained_submesh, threshold_table,
                                          verify_extracted)
from physics.fem.mesh import solid_box_mesh

pytestmark = pytest.mark.slow
requires_ccx = pytest.mark.skipif(not ccx.is_available(),
                                  reason="CalculiX is required")

MATERIAL = get_material("al_7075_t6")


def cantilever(divisions=(20, 8, 3), length=0.4, height=0.08, width=0.04,
               volume_fraction=0.4, radius=2.0, passive=True) -> SimpProblem:
    mesh = solid_box_mesh(length, height, width, *divisions)
    fixed, load = mesh.nodes_at_x(0.0), mesh.nodes_at_x(length)
    patches = None
    if passive:
        patches = elements_touching(mesh, load) | elements_touching(mesh, fixed)
    return SimpProblem(mesh=mesh, youngs_modulus_pa=MATERIAL.youngs_modulus_pa,
                       poisson_ratio=MATERIAL.poisson_ratio, fixed_nodes=fixed,
                       load_nodes=load, total_load_n=-800.0, load_direction=1,
                       volume_fraction=volume_fraction,
                       filter_radius_elements=radius, passive_solid=patches)


@pytest.fixture(scope="module")
def projected():
    problem = cantilever()
    return problem, optimize_projected(problem, max_iterations=60)


@pytest.fixture(scope="module")
def coarse():
    problem = cantilever(divisions=(16, 6, 3))
    return problem, optimize_projected(problem, max_iterations=40)


# --- passive regions ---------------------------------------------------------

def test_passive_elements_stay_solid_and_the_volume_still_lands(projected):
    """The load and support patches are held at one, and the volume fraction
    over the whole domain is still the requested one, which is what the free
    fraction correction is for."""
    problem, result = projected
    assert problem.passive_solid.sum() > 0
    assert np.all(result.density[problem.passive_solid] == 1.0)
    assert result.volume_fraction == pytest.approx(problem.volume_fraction, abs=0.02)


def test_without_passive_regions_the_load_patch_goes_grey_and_the_part_is_cut():
    """The measurement that put passive regions in: on this problem the plain
    run leaves the loaded elements below every useful threshold, so the
    extracted part has no load path. The refusal names the counts."""
    problem = cantilever(passive=False)
    result = optimize(problem, max_iterations=40)
    loaded = elements_touching(problem.mesh, problem.load_nodes)
    assert result.density[loaded].max() < 0.5
    with pytest.raises(DisconnectedAtThreshold, match="load path is cut"):
        retained_submesh(problem.mesh, result.density, 0.5,
                         problem.fixed_nodes, problem.load_nodes)


def test_a_forced_element_is_reported_not_hidden(projected):
    """Extraction may hold the patches solid, and then it has to say how many
    elements it added that the field did not have."""
    problem, result = projected
    density = result.density.copy()
    density[problem.passive_solid] = 0.0          # pretend the field dropped them
    _sub, report = retained_submesh(
        problem.mesh, density, 0.5, problem.fixed_nodes, problem.load_nodes,
        keep_elements=problem.passive_solid)
    assert report["elements_forced_solid"] == int(problem.passive_solid.sum())


# --- the submesh -------------------------------------------------------------

def test_the_submesh_is_exactly_the_retained_voxels(projected):
    problem, result = projected
    sub, report = retained_submesh(problem.mesh, result.density, 0.5,
                                   problem.fixed_nodes, problem.load_nodes)
    kept = int((result.density >= 0.5).sum())
    assert report["elements_above_threshold"] == kept
    assert sub.n_elements + report["island_elements_dropped"] == kept
    assert sub.connectivity.shape[1] == 8
    assert sub.connectivity.max() == sub.n_nodes - 1
    assert sub.n_nodes < problem.mesh.n_nodes
    # Every local node maps back to a parent node with the same coordinates.
    parent = np.asarray(problem.mesh.node_coords)[sub.parent_nodes]
    assert np.allclose(sub.node_coords, parent)


def test_a_threshold_that_keeps_nothing_is_refused(projected):
    problem, result = projected
    with pytest.raises(DisconnectedAtThreshold, match="keeps no element"):
        retained_submesh(problem.mesh, result.density, 1.01,
                         problem.fixed_nodes, problem.load_nodes)


# --- the measurement itself --------------------------------------------------

@requires_ccx
def test_the_extracted_part_is_not_the_field_and_the_gap_is_reported(projected):
    """The number this module exists for. The part is heavier than the field
    at a low threshold (grey elements round up to solid) and lighter at a high
    one, and its compliance is not the field's at any threshold."""
    problem, result = projected
    rows = threshold_table(problem, result.density, result.final_compliance,
                           MATERIAL.density_kg_m3)
    solved = [r for r in rows if "refused" not in r]
    assert len(solved) >= 2, rows
    masses = [r["mass_kg"] for r in solved]
    assert masses == sorted(masses, reverse=True), "a higher threshold kept more"
    for r in solved:
        assert r["part_compliance_j"] > 0.0
        assert np.isfinite(r["compliance_ratio"])
        assert abs(r["compliance_ratio"] - 1.0) > 0.01, (
            "the extracted part matched the field to one percent, which would "
            "mean the threshold changed nothing")
    print("\n" + format_table(rows))


@requires_ccx
def test_less_grey_means_a_part_closer_to_the_field(projected, coarse):
    """Why the gap exists, as a comparison rather than an assertion of theory:
    the run with less intermediate density extracts to a part whose compliance
    is nearer the field's."""
    fine_problem, fine = projected
    coarse_problem, rough = coarse

    def grey(d):
        return float(np.mean((d > 0.1) & (d < 0.9)))

    def ratio_at(problem, result, threshold):
        return verify_extracted(problem, result.density, threshold,
                                result.final_compliance,
                                MATERIAL.density_kg_m3).compliance_ratio

    assert grey(fine.density) < grey(rough.density)
    fine_gap = abs(ratio_at(fine_problem, fine, 0.3) - 1.0)
    rough_gap = abs(ratio_at(coarse_problem, rough, 0.3) - 1.0)
    print(f"\ngrey {grey(rough.density):.2f} gap {rough_gap:.2f}; "
          f"grey {grey(fine.density):.2f} gap {fine_gap:.2f}")
    assert fine_gap < rough_gap


@requires_ccx
def test_the_compliance_is_the_work_of_the_load(projected):
    """The comparison is only meaningful if both numbers are the same
    functional. This checks the CalculiX side against its own displacements."""
    problem, result = projected
    check = verify_extracted(problem, result.density, 0.3,
                             result.final_compliance, MATERIAL.density_kg_m3)
    sub, _ = retained_submesh(problem.mesh, result.density, 0.3,
                              problem.fixed_nodes, problem.load_nodes)
    loaded = sub.local_nodes(problem.load_nodes)
    solution = ccx.solve(sub, problem.youngs_modulus_pa, problem.poisson_ratio,
                         sub.local_nodes(problem.fixed_nodes), loaded,
                         total_load_n=problem.total_load_n,
                         load_direction=problem.load_direction,
                         element_type=ccx.ElementType.C3D8I)
    again = compliance_from(solution.displacements, loaded,
                            problem.total_load_n, problem.load_direction)
    assert again == pytest.approx(check.part_compliance_j, rel=1e-9)


# --- the stress constraint, and what it does not promise ---------------------

@requires_ccx
def test_a_stress_constrained_design_reports_both_numbers_and_claims_neither():
    """The p-norm constraint acts on the design's own relaxed measure. The
    part's peak is a re-entrant corner singularity, so the check reports what
    each number is and refuses to call any of them the peak stress.

    Measured on the L bracket at 1152 elements: p-norm 0.998, design relaxed
    peak 37.5 MPa against a 60 MPa limit, the voxel part reads 38.2 MPa, and
    the smoothed part meshed with linear tetrahedra reads 81.0, 66.2, 70.9 and
    75.6 MPa at 12, 8, 5 and 3.5 mm while its displacement converges
    monotonically from 7.40e-4 to 6.82e-4 m. Displacement converges; the peak
    does not.
    """
    from optimization.topology.stress import StressProblem, optimize_constrained
    from optimization.topology.verify import stress_check
    from physics.fem.mesh import l_bracket_mesh

    size = 0.4
    mesh = l_bracket_mesh(size, 0.4, 0.01, 20, nz=2, allow_snapping=True)
    top = mesh.nodes_where(np.abs(mesh.node_coords[:, 1] - size) < 1e-9)
    tip = mesh.nodes_where(np.abs(mesh.node_coords[:, 0] - size) < 1e-9)
    base = SimpProblem(mesh=mesh, youngs_modulus_pa=MATERIAL.youngs_modulus_pa,
                       poisson_ratio=MATERIAL.poisson_ratio, fixed_nodes=top,
                       load_nodes=tip, total_load_n=-3000.0, load_direction=1,
                       volume_fraction=0.4, filter_radius_elements=2.0,
                       passive_solid=(elements_touching(mesh, tip)
                                      | elements_touching(mesh, top)))
    problem = StressProblem(base=base, stress_limit_pa=60e6, p_norm=8.0)
    result = optimize_constrained(problem, max_iterations=30)

    check = stress_check(problem, result, MATERIAL.density_kg_m3)
    assert check.physical_peak_verified is False
    assert check.extracted_peak_pa, "no threshold produced a solvable part"
    assert "does not converge" in check.summary()
    assert check.design_max_relaxed_pa > 0.0
    print("\n" + check.summary())
