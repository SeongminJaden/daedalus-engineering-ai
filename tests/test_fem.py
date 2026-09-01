"""Phase 7 verification: 3D linear-elastic FEM.

Order matters here. The patch test gates everything: an element that cannot
reproduce a constant strain state exactly is not an element, and no downstream
result from it means anything. Only after that do the convergence and stress
tests carry weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.design_genome import DesignGenome, HollowRectangleSection  # noqa: E402
from physics.fem import (  # noqa: E402
    elasticity_matrix, element_stiffness, hollow_rect_mesh, solid_box_mesh,
    solve_linear_elasticity, von_mises,
)
from physics.fem.element import shape_derivatives, strain_displacement  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402

E, NU = 70.0e9, 0.3


def assemble(mesh, youngs=E, nu=NU):
    """Dense assembly. Only for the small patch-test meshes."""
    ke = element_stiffness(mesh.dx, mesh.dy, mesh.dz, youngs, nu)
    k = np.zeros((mesh.n_dofs, mesh.n_dofs))
    for e in range(mesh.n_elements):
        dofs = (mesh.connectivity[e][:, None] * 3 + np.arange(3)).reshape(-1)
        k[np.ix_(dofs, dofs)] += ke
    return k


def centre_b(mesh):
    scale = np.array([2 / mesh.dx, 2 / mesh.dy, 2 / mesh.dz])
    return strain_displacement(shape_derivatives(0.0, 0.0, 0.0) * scale)


# =========================================================================== #
# element sanity
# =========================================================================== #
def test_element_matrix_is_symmetric_with_six_rigid_body_modes():
    ke = element_stiffness(1e-3, 2e-3, 3e-3, E, NU)
    assert np.allclose(ke, ke.T)
    w = np.linalg.eigvalsh(ke)
    assert int((w < 1e-6 * w.max()).sum()) == 6
    assert w.min() > -1e-6 * w.max()


def test_elasticity_matrix_rejects_bad_inputs():
    with pytest.raises(ValueError):
        elasticity_matrix(70e9, 0.5)
    with pytest.raises(ValueError):
        elasticity_matrix(-1.0, 0.3)


def test_von_mises_of_uniaxial_stress_equals_that_stress():
    assert von_mises([[1e8, 0, 0, 0, 0, 0]])[0] == pytest.approx(1e8)


def test_von_mises_of_hydrostatic_stress_is_zero():
    assert von_mises([[5e7, 5e7, 5e7, 0, 0, 0]])[0] == pytest.approx(0.0, abs=1e-3)


# =========================================================================== #
# 1. THE PATCH TEST - the gate
# =========================================================================== #
@pytest.fixture(scope="module")
def patch():
    mesh = solid_box_mesh(0.03, 0.03, 0.03, 3, 3, 3)
    return mesh, assemble(mesh)


def test_patch_reproduces_a_constant_strain_field_exactly(patch):
    """Prescribe u = A x on the boundary; the interior must come out exactly on
    that linear field, to machine precision."""
    mesh, k = patch
    x = mesh.node_coords
    a = np.array([[1e-4, 2e-5, 3e-5], [2e-5, -5e-5, 1e-5], [3e-5, 1e-5, 7e-5]])
    u_exact = (x @ a.T).reshape(-1)

    tol = 1e-9
    boundary = np.zeros(mesh.n_nodes, dtype=bool)
    for axis in range(3):
        boundary |= np.abs(x[:, axis] - x[:, axis].min()) < tol
        boundary |= np.abs(x[:, axis] - x[:, axis].max()) < tol
    bnodes, inodes = np.flatnonzero(boundary), np.flatnonzero(~boundary)
    assert len(inodes) > 0, "patch has no interior nodes to test"

    bd = (bnodes[:, None] * 3 + np.arange(3)).reshape(-1)
    idd = (inodes[:, None] * 3 + np.arange(3)).reshape(-1)
    u = np.zeros(mesh.n_dofs)
    u[bd] = u_exact[bd]
    u[idd] = np.linalg.solve(k[np.ix_(idd, idd)], -k[np.ix_(idd, bd)] @ u_exact[bd])

    rel = np.abs(u[idd] - u_exact[idd]).max() / np.abs(u_exact).max()
    assert rel < 1e-12, f"patch test failed: interior rel err {rel:.3e}"


def test_patch_reproduces_the_exact_constant_stress(patch):
    mesh, k = patch
    x = mesh.node_coords
    a = np.array([[1e-4, 2e-5, 3e-5], [2e-5, -5e-5, 1e-5], [3e-5, 1e-5, 7e-5]])
    u = (x @ a.T).reshape(-1)

    d = elasticity_matrix(E, NU)
    exact = d @ np.array([a[0, 0], a[1, 1], a[2, 2],
                          2 * a[0, 1], 2 * a[1, 2], 2 * a[0, 2]])
    b0 = centre_b(mesh)
    for e in range(mesh.n_elements):
        dofs = (mesh.connectivity[e][:, None] * 3 + np.arange(3)).reshape(-1)
        rel = np.abs(d @ (b0 @ u[dofs]) - exact).max() / np.abs(exact).max()
        assert rel < 1e-12, f"element {e} stress rel err {rel:.3e}"


@pytest.mark.parametrize("kind", ["translation", "rotation"])
def test_rigid_body_motion_produces_no_stress_and_no_force(patch, kind):
    mesh, k = patch
    x = mesh.node_coords
    if kind == "translation":
        u = np.tile([1e-3, -2e-3, 5e-4], mesh.n_nodes)
    else:
        spin = np.array([[0, -1e-4, 0], [1e-4, 0, 0], [0, 0, 0]])
        u = (x @ spin.T).reshape(-1)

    forces = k @ u
    scale = np.abs(k).max() * np.abs(u).max()
    assert np.abs(forces).max() < 1e-9 * scale

    d = elasticity_matrix(E, NU)
    b0 = centre_b(mesh)
    for e in range(mesh.n_elements):
        dofs = (mesh.connectivity[e][:, None] * 3 + np.arange(3)).reshape(-1)
        assert np.abs(d @ (b0 @ u[dofs])).max() < 1.0    # Pa, vs GPa moduli


# =========================================================================== #
# 2. slender-beam limit: FEM must meet Euler-Bernoulli
# =========================================================================== #
def cantilever(nx, ny, nz, length=1.0, height=0.05, width=0.05, load=100.0):
    mesh = solid_box_mesh(length, height, width, nx, ny, nz)
    solution = solve_linear_elasticity(
        mesh, E, NU, mesh.nodes_at_x(0.0), mesh.nodes_at_x(length), -load, 1)
    return mesh, solution


def beam_tip_deflection(load=100.0, length=1.0, height=0.05, width=0.05):
    inertia = width * height ** 3 / 12.0
    return load * length ** 3 / (3.0 * E * inertia)


@pytest.fixture(scope="module")
def refinement_study():
    out = []
    for n in (2, 3, 4):
        mesh, solution = cantilever(20 * n, 2 * n, 2 * n)
        out.append((mesh, solution, abs(solution.tip_deflection())))
    return out


def test_slender_beam_converges_to_euler_bernoulli(refinement_study):
    """The shear-free limit: a slender FEM cantilever must agree with beam
    theory. This is the evidence that the element solves elasticity correctly."""
    exact = beam_tip_deflection()
    finest = refinement_study[-1][2]
    ratio = finest / exact
    assert 0.99 < ratio < 1.02, f"finest mesh gives {ratio:.4f} of beam theory"


def test_deflection_improves_monotonically_with_refinement(refinement_study):
    ratios = [d / beam_tip_deflection() for _, _, d in refinement_study]
    for coarse, fine in zip(ratios, ratios[1:]):
        assert fine >= coarse - 1e-6, f"refinement made it worse: {ratios}"


def test_incompatible_modes_are_what_make_this_work():
    """Without the bubble modes the same mesh shear-locks badly. Pinning this
    stops anyone quietly turning them off."""
    plain = element_stiffness(0.05, 0.0125, 0.0125, E, NU, incompatible_modes=False)
    enhanced = element_stiffness(0.05, 0.0125, 0.0125, E, NU, incompatible_modes=True)
    # Enhanced assumed strain can only remove energy, never add it.
    assert np.trace(enhanced) < np.trace(plain)


# =========================================================================== #
# 4. mesh convergence (Cauchy)
# =========================================================================== #
def test_tip_deflection_is_cauchy_convergent(refinement_study):
    d = [x[2] for x in refinement_study]
    step1, step2 = abs(d[1] - d[0]), abs(d[2] - d[1])
    assert step2 < step1, f"successive changes not shrinking: {d}"


def test_solver_reports_convergence(refinement_study):
    for mesh, solution, _ in refinement_study:
        assert solution.report.converged, (
            f"CG stalled at residual {solution.report.residual:.2e}")


# =========================================================================== #
# 3 & the honesty point: the clamped singularity
# =========================================================================== #
def test_peak_stress_does_not_converge_but_deflection_does(refinement_study):
    """The property that forces an honest stress measure.

    A perfectly clamped face is a stress singularity: the reported peak keeps
    climbing with refinement and never settles, while the deflection converges.
    If this test ever starts failing because the peak DID converge, the mesh is
    too coarse to see the singularity, not the singularity gone.
    """
    peaks = [float(s.element_von_mises.max()) for _, s, _ in refinement_study]
    deflections = [d for _, _, d in refinement_study]

    peak_growth = (peaks[-1] - peaks[0]) / peaks[0]
    defl_change = abs(deflections[-1] - deflections[0]) / deflections[0]

    assert peak_growth > 0.05, (
        f"peak stress barely moved ({peak_growth:.1%}); expected it to keep "
        f"rising with refinement: {peaks}")
    assert defl_change < 0.01, (
        f"deflection should be nearly converged, changed {defl_change:.2%}")
    assert peak_growth > 5 * defl_change


def test_gauge_stress_matches_beam_theory_at_the_same_fibre():
    """Away from the support the FEM must reproduce the beam bending field.

    The comparison is made at the gauge element's own station and fibre. A naive
    comparison against the extreme-fibre value would show a 20-30% "error" that
    is only the element-centre sampling position.
    """
    length, height, width, load = 0.5, 0.05, 0.05, 200.0
    inertia = width * height ** 3 / 12.0
    mesh, solution = cantilever(80, 8, 8, length, height, width, load)

    centroids = mesh.element_centroids()
    vm = solution.element_von_mises
    offset = height
    band = (centroids[:, 0] >= offset) & (centroids[:, 0] < offset + mesh.dx * 1.001)
    idx = np.flatnonzero(band)
    j = idx[int(np.argmax(vm[idx]))]

    moment = load * (length - centroids[j, 0])
    nominal = abs(moment * (centroids[j, 1] - height / 2) / inertia)
    assert vm[j] == pytest.approx(nominal, rel=0.02), (
        f"gauge {vm[j]:.4g} vs beam {nominal:.4g}")


def test_stress_concentrates_toward_the_support():
    """Peak stress lives at the clamped end, not out along the span."""
    mesh, solution = cantilever(60, 6, 6, 0.5, 0.05, 0.05, 200.0)
    centroids = mesh.element_centroids()
    peak_x = centroids[int(np.argmax(solution.element_von_mises)), 0]
    assert peak_x < 0.1 * 0.5


# =========================================================================== #
# 5. physical sanity
# =========================================================================== #
def test_deflection_follows_the_load_direction():
    mesh, solution = cantilever(40, 4, 4)
    assert solution.tip_deflection(direction=1) < 0        # load is -y


def test_thicker_wall_is_stiffer():
    problem = build_mvp_problem()
    from core.materials import get_material
    material = get_material(problem.material_id)
    out = []
    for t in (0.0015, 0.003):
        mesh = hollow_rect_mesh(0.3, 0.04, 0.04, t, nx=12, elements_through_wall=1)
        s = solve_linear_elasticity(
            mesh, material.youngs_modulus_pa, material.poisson_ratio,
            mesh.nodes_at_x(0.0), mesh.nodes_at_x(0.3), -200.0, 1)
        out.append(abs(s.tip_deflection()))
    assert out[1] < out[0], f"thicker wall deflected more: {out}"


def test_doubling_the_load_doubles_the_deflection():
    """Linear elasticity, so this must hold exactly."""
    mesh = solid_box_mesh(0.5, 0.05, 0.05, 40, 4, 4)
    a = solve_linear_elasticity(mesh, E, NU, mesh.nodes_at_x(0.0),
                                mesh.nodes_at_x(0.5), -100.0, 1)
    b = solve_linear_elasticity(mesh, E, NU, mesh.nodes_at_x(0.0),
                                mesh.nodes_at_x(0.5), -200.0, 1)
    assert abs(b.tip_deflection()) == pytest.approx(
        2 * abs(a.tip_deflection()), rel=1e-4)


# =========================================================================== #
# mesh construction
# =========================================================================== #
def test_hollow_mesh_removes_the_cavity():
    mesh = hollow_rect_mesh(0.2, 0.04, 0.04, 0.005, nx=8, elements_through_wall=2)
    assert mesh.n_elements < mesh.nx * mesh.ny * mesh.nz
    centroids = mesh.element_centroids()
    inside = ((centroids[:, 1] > 0.005) & (centroids[:, 1] < 0.035)
              & (centroids[:, 2] > 0.005) & (centroids[:, 2] < 0.035))
    assert not inside.any(), "cavity cells were not removed"


def test_hollow_mesh_rejects_impossible_wall():
    with pytest.raises(ValueError):
        hollow_rect_mesh(0.2, 0.04, 0.04, 0.02, nx=8)


def test_solid_mesh_node_count():
    mesh = solid_box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
    assert mesh.n_elements == 8
    assert mesh.n_nodes == 27
    assert mesh.n_dofs == 81


# =========================================================================== #
# the funnel, and what higher fidelity is still not
# =========================================================================== #
def test_high_fidelity_verify_reports_both_stress_measures():
    from physics.fem import high_fidelity_verify
    problem = build_mvp_problem()
    genome = DesignGenome(
        section=HollowRectangleSection(outer_width_m=0.02, outer_height_m=0.04,
                                       wall_thickness_m=0.002),
        material_id=problem.material_id)
    result = high_fidelity_verify(genome, problem, elements_through_wall=1,
                                  max_dofs=60000)
    assert result.fidelity == "fem3d"
    assert result.peak_von_mises_pa > 0
    assert result.gauge_von_mises_pa > 0
    # The gauge sits in the smooth bending field, so it is below the peak.
    assert result.gauge_von_mises_pa < result.peak_von_mises_pa
    assert result.gauge_agreement == pytest.approx(1.0, abs=0.05)
    assert any("mesh dependent" in w for w in result.warnings)


def test_high_fidelity_rejects_invalid_geometry():
    from physics.fem import high_fidelity_verify
    problem = build_mvp_problem()
    bad = DesignGenome(
        section=HollowRectangleSection(outer_width_m=0.02, outer_height_m=0.02,
                                       wall_thickness_m=0.02),
        material_id=problem.material_id)
    with pytest.raises(ValueError, match="invalid"):
        high_fidelity_verify(bad, problem)


def test_fem_evidence_is_still_only_simulation():
    """Higher fidelity does NOT move a claim up the evidence ladder past
    simulation. Only a physical test opens that gate, and 3D FEM is not one."""
    from brain.semantic import (
        Evidence, EvidenceKind, EvidenceLevel, derive_level,
    )
    fem = [Evidence(EvidenceKind.SIMULATION, f"fem3d-{i}", run_id=f"r{i}",
                    note="fidelity=fem3d")
           for i in range(50)]
    level = derive_level(fem, [])
    assert level is not EvidenceLevel.EXPERIMENTALLY_VALIDATED
    assert level.rank <= EvidenceLevel.HIGH_CONFIDENCE.rank


# --- the L bracket arm must land on a cell boundary -------------------------

def test_an_unbuildable_arm_thickness_is_refused_not_rounded():
    """Silently meshing a different bracket is what this prevents.

    Found by measuring against an independent CAD volume: n=10 with a 0.25
    fraction used to build a 0.020 arm instead of 0.025, a 17.7 percent volume
    error, and a solver comparison meshed from here cannot see it because both
    solvers then agree on the wrong solid.
    """
    from physics.fem.mesh import l_bracket_mesh

    with pytest.raises(ValueError, match="allow_snapping"):
        l_bracket_mesh(0.10, 0.25, 0.01, n=10, nz=2)


def test_the_refusal_names_grids_that_would_work():
    """An error that only says no costs the caller a guessing game."""
    from physics.fem.mesh import l_bracket_mesh

    with pytest.raises(ValueError) as excinfo:
        l_bracket_mesh(0.10, 0.25, 0.01, n=10, nz=2)
    message = str(excinfo.value)
    assert "n=[" in message
    for suggested in (4, 8, 12, 16, 20):
        assert str(suggested) in message
    # And every suggestion must actually build.
    for suggested in (4, 8, 12, 16, 20):
        l_bracket_mesh(0.10, 0.25, 0.01, n=suggested, nz=2)


def test_snapping_stays_available_when_it_is_asked_for_explicitly():
    """Rounding is fine when it is chosen, which is the whole difference."""
    from physics.fem.mesh import l_bracket_mesh, realised_arm_thickness

    mesh = l_bracket_mesh(0.10, 0.25, 0.01, n=10, nz=2, allow_snapping=True)
    assert mesh.n_elements > 0
    assert realised_arm_thickness(0.10, 0.25, 10) == pytest.approx(0.020)


def test_a_buildable_request_is_unaffected():
    from physics.fem.mesh import l_bracket_mesh, realised_arm_thickness

    l_bracket_mesh(0.10, 0.4, 0.01, n=20, nz=2)
    assert realised_arm_thickness(0.10, 0.4, 20) == pytest.approx(0.040)
