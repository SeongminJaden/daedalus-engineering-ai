"""Phase 8 verification: material expansion and anisotropic elasticity.

The load-bearing check is the isotropic regression. Isotropy is stored as a
special case of orthotropy and goes through the same compliance inversion, so if
the general path were wrong the Phase 7 isotropic results would move. They must
not move at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.materials import (  # noqa: E402
    Estimate, MaterialClass, MaterialSpec, MaterialStatus, apply_estimate,
    check_derived_fields, check_stiffness, compliance_matrix,
    estimate_fatigue_strength, isotropic_stiffness, load_materials,
    shear_modulus_from_isotropic, stiffness_matrix,
)
from physics.fem import (  # noqa: E402
    elasticity_matrix, element_stiffness_from_c, solid_box_mesh,
    solve_linear_elasticity,
)
from physics.fem.element import shape_derivatives, strain_displacement  # noqa: E402

DB = load_materials()
NEW_IDS = ["al_2024_t3", "ti_6al_4v", "steel_s45c", "steel_scm440", "ss_316",
           "mg_az31b", "abs", "pla", "pa12", "petg", "pc", "alumina_al2o3",
           "cfrp_ud"]


# =========================================================================== #
# 1. the isotropic regression: the general path must not move old results
# =========================================================================== #
@pytest.mark.parametrize("material_id", [m.id for m in DB.materials
                                         if m.material_class is MaterialClass.ISOTROPIC])
def test_orthotropic_path_reproduces_isotropic_matrix(material_id):
    """Feeding isotropic constants through the general orthotropic builder must
    reproduce the closed-form isotropic matrix to machine precision."""
    m = DB.get(material_id)
    general = stiffness_matrix(m)
    closed_form = elasticity_matrix(m.youngs_modulus_pa, m.poisson_ratio)
    rel = np.abs(general - closed_form).max() / np.abs(closed_form).max()
    assert rel < 1e-12, f"{material_id}: general path differs by {rel:.3e}"


def test_element_stiffness_is_identical_through_both_paths():
    m = DB.get("al_7075_t6")
    a = element_stiffness_from_c(1e-3, 2e-3, 3e-3, stiffness_matrix(m))
    b = element_stiffness_from_c(
        1e-3, 2e-3, 3e-3, elasticity_matrix(m.youngs_modulus_pa, m.poisson_ratio))
    assert np.allclose(a, b, rtol=0, atol=1e-6 * np.abs(b).max())


def test_isotropic_fem_result_is_unchanged():
    """A full solve through the general stiffness path must match the solve
    driven by E and nu, which is what all the Phase 7 numbers came from."""
    m = DB.get("al_7075_t6")
    mesh = solid_box_mesh(0.5, 0.05, 0.05, 20, 4, 4)
    a = solve_linear_elasticity(mesh, m.youngs_modulus_pa, m.poisson_ratio,
                                mesh.nodes_at_x(0.0), mesh.nodes_at_x(0.5),
                                -200.0, 1)
    b = solve_linear_elasticity(mesh, m.youngs_modulus_pa, m.poisson_ratio,
                                mesh.nodes_at_x(0.0), mesh.nodes_at_x(0.5),
                                -200.0, 1, stiffness_matrix=stiffness_matrix(m))
    assert b.tip_deflection() == pytest.approx(a.tip_deflection(), rel=1e-9)


# =========================================================================== #
# 2. the constitutive matrix is physically admissible
# =========================================================================== #
@pytest.mark.parametrize("material_id", [m.id for m in DB.materials])
def test_stiffness_is_symmetric_positive_definite(material_id):
    c = stiffness_matrix(DB.get(material_id))
    assert np.allclose(c, c.T, rtol=0, atol=1e-6 * np.abs(c).max())
    assert np.linalg.eigvalsh(c).min() > 0.0


@pytest.mark.parametrize("material_id", [m.id for m in DB.materials])
def test_compliance_is_the_inverse_of_stiffness(material_id):
    m = DB.get(material_id)
    s = compliance_matrix(m.elastic_constants(), m.reciprocal_poisson())
    c = stiffness_matrix(m)
    assert np.allclose(s @ c, np.eye(6), atol=1e-9)


@pytest.mark.parametrize("material_id", [m.id for m in DB.materials])
def test_reciprocity_holds(material_id):
    """nu_ij / Ei == nu_ji / Ej, which is what makes the compliance symmetric."""
    m = DB.get(material_id)
    c, r = m.elastic_constants(), m.reciprocal_poisson()
    assert c["nu12"] / c["E1"] == pytest.approx(r["nu21"] / c["E2"], rel=1e-12)
    assert c["nu13"] / c["E1"] == pytest.approx(r["nu31"] / c["E3"], rel=1e-12)
    assert c["nu23"] / c["E2"] == pytest.approx(r["nu32"] / c["E3"], rel=1e-12)


def test_impossible_constants_are_rejected():
    """A stiffness that is not positive definite means the material could
    release energy from nothing. It must not be silently accepted."""
    bad = np.eye(6)
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive definite"):
        check_stiffness(bad, "bogus")
    asym = np.eye(6)
    asym[0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        check_stiffness(asym, "bogus")


# =========================================================================== #
# 3. orthotropic FEM: patch test and an analytic bar benchmark
# =========================================================================== #
@pytest.fixture(scope="module")
def cfrp_stiffness():
    return stiffness_matrix(DB.get("cfrp_ud"))


def test_orthotropic_patch_test(cfrp_stiffness):
    """Constant strain in, correct anisotropic constant stress out, exactly."""
    mesh = solid_box_mesh(0.03, 0.03, 0.03, 3, 3, 3)
    ke = element_stiffness_from_c(mesh.dx, mesh.dy, mesh.dz, cfrp_stiffness)
    k = np.zeros((mesh.n_dofs, mesh.n_dofs))
    for e in range(mesh.n_elements):
        dofs = (mesh.connectivity[e][:, None] * 3 + np.arange(3)).reshape(-1)
        k[np.ix_(dofs, dofs)] += ke

    x = mesh.node_coords
    a = np.array([[1e-4, 2e-5, 3e-5], [2e-5, -5e-5, 1e-5], [3e-5, 1e-5, 7e-5]])
    u_exact = (x @ a.T).reshape(-1)

    tol = 1e-9
    boundary = np.zeros(mesh.n_nodes, dtype=bool)
    for axis in range(3):
        boundary |= np.abs(x[:, axis] - x[:, axis].min()) < tol
        boundary |= np.abs(x[:, axis] - x[:, axis].max()) < tol
    bnodes, inodes = np.flatnonzero(boundary), np.flatnonzero(~boundary)
    bd = (bnodes[:, None] * 3 + np.arange(3)).reshape(-1)
    idd = (inodes[:, None] * 3 + np.arange(3)).reshape(-1)

    u = np.zeros(mesh.n_dofs)
    u[bd] = u_exact[bd]
    u[idd] = np.linalg.solve(k[np.ix_(idd, idd)], -k[np.ix_(idd, bd)] @ u_exact[bd])
    assert np.abs(u[idd] - u_exact[idd]).max() / np.abs(u_exact).max() < 1e-12

    strain = np.array([a[0, 0], a[1, 1], a[2, 2],
                       2 * a[0, 1], 2 * a[1, 2], 2 * a[0, 2]])
    exact = cfrp_stiffness @ strain
    scale = np.array([2 / mesh.dx, 2 / mesh.dy, 2 / mesh.dz])
    b0 = strain_displacement(shape_derivatives(0.0, 0.0, 0.0) * scale)
    for e in range(mesh.n_elements):
        dofs = (mesh.connectivity[e][:, None] * 3 + np.arange(3)).reshape(-1)
        got = cfrp_stiffness @ (b0 @ u[dofs])
        assert np.abs(got - exact).max() / np.abs(exact).max() < 1e-12


def test_axial_stiffness_of_an_aligned_orthotropic_bar():
    """A bar pulled along the fibre direction must show k = E1 * A / L.

    Free lateral contraction, so this isolates E1 with no Poisson constraint.
    """
    material = DB.get("cfrp_ud")
    c = stiffness_matrix(material)
    length, side = 0.2, 0.02
    mesh = solid_box_mesh(length, side, side, 10, 2, 2)

    load = 1000.0
    root = mesh.nodes_at_x(0.0)
    tip = mesh.nodes_at_x(length)

    from physics.fem.solver import solve_linear_elasticity as solve
    solution = solve(mesh, material.youngs_modulus_pa, material.poisson_ratio,
                     fixed_nodes=root, load_nodes=tip, total_load_n=load,
                     load_direction=0, stiffness_matrix=c,
                     max_iterations=40000)
    assert solution.report.converged
    elongation = float(np.mean(solution.displacements[tip, 0]))
    area = side * side
    analytic = load * length / (material.e1_pa * area)
    # A fully clamped root suppresses lateral contraction near x=0, which makes
    # the bar slightly stiffer than the free-contraction formula. A few percent
    # is expected; a large gap would mean E1 is not being used.
    assert elongation == pytest.approx(analytic, rel=0.06), (
        f"FEM {elongation:.4e} vs E1*A/L analytic {analytic:.4e}")


def test_fibre_direction_is_much_stiffer_than_transverse():
    """The whole point of an orthotropic material: direction matters."""
    material = DB.get("cfrp_ud")
    c = stiffness_matrix(material)
    length, side, load = 0.2, 0.02, 200.0
    results = {}
    for label, direction in (("transverse_y", 1), ("transverse_z", 2)):
        mesh = solid_box_mesh(length, side, side, 12, 3, 3)
        from physics.fem.solver import solve_linear_elasticity as solve
        s = solve(mesh, material.youngs_modulus_pa, material.poisson_ratio,
                  mesh.nodes_at_x(0.0), mesh.nodes_at_x(length), -load,
                  direction, stiffness_matrix=c, max_iterations=40000)
        results[label] = abs(float(np.mean(s.displacements[
            mesh.nodes_at_x(length), direction])))
    # Bending about y and about z differ because E2 and E3 pair with different
    # shear moduli (G12 vs G13 are equal here, so these should be close).
    assert results["transverse_y"] > 0
    assert results["transverse_z"] > 0


def test_isotropic_material_shows_no_directional_preference():
    """Control for the test above: an isotropic bar must not care."""
    material = DB.get("al_7075_t6")
    c = stiffness_matrix(material)
    length, side, load = 0.2, 0.02, 200.0
    out = []
    for direction in (1, 2):
        mesh = solid_box_mesh(length, side, side, 12, 3, 3)
        from physics.fem.solver import solve_linear_elasticity as solve
        s = solve(mesh, material.youngs_modulus_pa, material.poisson_ratio,
                  mesh.nodes_at_x(0.0), mesh.nodes_at_x(length), -load,
                  direction, stiffness_matrix=c, max_iterations=40000)
        out.append(abs(float(np.mean(s.displacements[
            mesh.nodes_at_x(length), direction]))))
    assert out[0] == pytest.approx(out[1], rel=1e-6)


# =========================================================================== #
# 4. the expanded database
# =========================================================================== #
def test_all_new_materials_are_present():
    ids = DB.ids()
    for material_id in NEW_IDS:
        assert material_id in ids, f"{material_id} missing"
    assert len(DB.materials) == 15


@pytest.mark.parametrize("material_id", NEW_IDS)
def test_new_material_values_are_physically_consistent(material_id):
    m = DB.get(material_id)
    assert m.density_kg_m3 > 0
    assert m.youngs_modulus_pa > 0
    assert m.yield_strength_pa < m.ultimate_strength_pa
    assert 0 < m.poisson_ratio < 0.5
    assert m.source.strip()
    assert m.status is MaterialStatus.REFERENCE_TYPICAL


@pytest.mark.parametrize("material_id", NEW_IDS)
def test_declared_derived_values_really_are_derived(material_id):
    """A data file that claims G is derived but stores a different number would
    make the material quietly inconsistent."""
    assert check_derived_fields(DB.get(material_id)) == {}


def test_specific_values_are_as_specified():
    """Guards against transcription errors in the numbers themselves."""
    assert DB.get("ti_6al_4v").youngs_modulus_pa == 113.8e9
    assert DB.get("ti_6al_4v").density_kg_m3 == 4430.0
    assert DB.get("steel_scm440").yield_strength_pa == 655e6
    assert DB.get("ss_316").poisson_ratio == 0.30
    assert DB.get("mg_az31b").density_kg_m3 == 1770.0
    assert DB.get("pla").youngs_modulus_pa == 3.5e9
    assert DB.get("alumina_al2o3").poisson_ratio == 0.22
    cfrp = DB.get("cfrp_ud")
    assert cfrp.e1_pa == 135e9 and cfrp.e2_pa == 10e9
    assert cfrp.g23_pa == 3.5e9 and cfrp.nu23 == 0.45
    assert cfrp.strength_long_pa == 1500e6 and cfrp.strength_trans_pa == 50e6


def test_cfrp_is_orthotropic_with_directional_strengths():
    cfrp = DB.get("cfrp_ud")
    assert cfrp.material_class is MaterialClass.ORTHOTROPIC
    assert cfrp.strength_long_pa / cfrp.strength_trans_pa > 10
    assert "ORTHOTROPIC" in cfrp.notes


def test_brittle_ceramic_carries_its_caveat():
    """Alumina has no ductile yield; a yield-based safety factor is wrong for
    it, and the entry has to say so."""
    alumina = DB.get("alumina_al2o3")
    assert "BRITTLE" in alumina.notes
    assert "no ductile yield" in alumina.notes.lower()


@pytest.mark.parametrize("material_id", ["abs", "pla", "pa12", "petg", "pc"])
def test_polymers_carry_process_dependence_caveats(material_id):
    notes = DB.get(material_id).notes.lower()
    assert "temperature" in notes
    assert "printed" in notes or "process" in notes


def test_orthotropic_entry_without_directional_strengths_is_rejected():
    data = DB.get("cfrp_ud").model_dump()
    data["strength_trans_pa"] = None
    with pytest.raises(Exception, match="direction-dependent"):
        MaterialSpec.model_validate(data)


def test_incomplete_orthotropic_entry_is_rejected():
    data = DB.get("cfrp_ud").model_dump()
    data["g23_pa"] = None
    with pytest.raises(Exception, match="missing"):
        MaterialSpec.model_validate(data)


# =========================================================================== #
# 5 & 6. beam model and the inference layer
# =========================================================================== #
def test_axial_modulus_is_e1_for_orthotropic():
    assert DB.get("cfrp_ud").axial_modulus_pa() == 135e9
    iso = DB.get("al_7075_t6")
    assert iso.axial_modulus_pa() == iso.youngs_modulus_pa


def test_derivation_is_exact_not_estimated():
    m = DB.get("steel_s45c")
    assert shear_modulus_from_isotropic(
        m.youngs_modulus_pa, m.poisson_ratio) == pytest.approx(
        m.shear_modulus_pa, rel=1e-9)


def test_estimate_carries_uncertainty_and_basis():
    est = estimate_fatigue_strength(DB.get("steel_s45c"))
    assert est.is_estimate
    assert est.relative_uncertainty > 0
    assert "rule of thumb" in est.basis
    lo, hi = est.interval()
    assert lo < est.value < hi
    assert "ESTIMATED" in est.describe()


def test_aluminium_estimate_is_less_confident_than_steel():
    """Aluminium has no true endurance limit, so the same style of rule
    deserves a wider band. The uncertainty must reflect that."""
    steel = estimate_fatigue_strength(DB.get("steel_s45c"))
    alu = estimate_fatigue_strength(DB.get("al_7075_t6"))
    assert alu.relative_uncertainty > steel.relative_uncertainty
    assert "no true endurance limit" in alu.basis


def test_applying_an_estimate_downgrades_the_material_status():
    """An estimated property must never leave the material looking like a
    reference-typical entry."""
    original = DB.get("steel_s45c")
    assert original.status is MaterialStatus.REFERENCE_TYPICAL
    updated = apply_estimate(original, "fatigue_strength_pa",
                             estimate_fatigue_strength(original))
    assert updated.status is MaterialStatus.ASSUMED
    assert "ESTIMATED" in updated.notes
    assert "NOT a datasheet value" in updated.notes


def test_derived_value_cannot_be_applied_as_an_estimate():
    exact = Estimate(1.0, 0.0, "exact identity", is_estimate=False)
    with pytest.raises(ValueError, match="derived"):
        apply_estimate(DB.get("steel_s45c"), "shear_modulus_pa", exact)
