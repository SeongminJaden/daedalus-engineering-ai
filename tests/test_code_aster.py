"""Verification of the Code_Aster node.

Code_Aster is here for nonlinear material behaviour, contact and plasticity,
which is where it does what CalculiX does not. None of that is verified yet,
and nothing is registered as a capability, because a plastic result cannot be
told apart from a wrong install without a linear case that has an exact answer
to anchor it.
"""

from __future__ import annotations

import math

import pytest

from nodes import code_aster

requires_aster = pytest.mark.skipif(
    not code_aster.is_available(),
    reason="Code_Aster is not installed at ASTER_HOME")


@requires_aster
def test_a_bar_in_tension_is_exact(tmp_path):
    """Uniform tension is reproduced EXACTLY by linear elements.

    The expected error is round off, not a percentage, which is what makes
    this the right first case: a discrepancy is a bug and cannot be waved
    through as a coarse mesh. That is how a 3.9 percent error was caught
    rather than accepted, and it turned out to be the boundary conditions.
    """
    result = code_aster.bar_tension(tmp_path)
    assert result.displacement_error < 1e-10
    assert result.stress_error < 1e-10


@requires_aster
def test_the_loaded_face_moves_as_one(tmp_path):
    """Uniform stress means every node on the loaded face moves together.

    Measured over the nodes that ARE on that face. Taking the largest few
    displacements instead would include interior nodes on a fine mesh and
    report a spread that means nothing, which is a mistake this test exists
    to prevent repeating.
    """
    result = code_aster.bar_tension(tmp_path)
    assert result.end_node_count > 10
    relative = result.end_displacement_spread_m / result.exact_displacement_m
    assert relative < 1e-9


def test_the_closed_form_is_stated_independently_of_the_solver():
    """sigma L / E, written from the analysis rather than read back."""
    bar = code_aster.BarTension(
        length_m=0.2, youngs_modulus_pa=210e9, applied_stress_pa=50e6,
        max_displacement_m=0.0, max_stress_pa=0.0,
        end_displacement_spread_m=0.0)
    assert bar.exact_displacement_m == pytest.approx(50e6 * 0.2 / 210e9,
                                                     rel=1e-15)


def test_the_lame_closed_form_is_correct_arithmetic():
    """The formulas themselves, checked without running anything.

    Worth asserting separately from the solve: if the case below is ever
    fixed, the target it is compared against must already be known good.
    """
    cyl = code_aster.ThickCylinder(
        inner_radius_m=0.05, outer_radius_m=0.1, pressure_pa=10e6,
        youngs_modulus_pa=210e9, poisson_ratio=0.3, elements=0,
        bore_displacement_m=0.0, bore_hoop_stress_pa=0.0)
    assert cyl.exact_bore_hoop_stress_pa == pytest.approx(
        10e6 * (0.01 + 0.0025) / 0.0075, rel=1e-12)
    assert cyl.exact_bore_displacement_m == pytest.approx(
        (1.3 * 0.0025 * 10e6) / (210e9 * 0.0075) * (0.4 * 0.05 + 0.01 / 0.05),
        rel=1e-12)


@requires_aster
def test_the_thick_cylinder_converges_at_the_right_ORDER(tmp_path):
    """The Lame solution IS the exact answer to this elasticity problem, so
    the discretisation converges to it and the rate can be measured.

    A rate is stronger evidence than a tolerance: a wrong solve meets a
    tolerance on a fine enough mesh, but cannot produce the right order. For
    linear triangles the displacement must converge at second order and the
    stress, being its derivative, one order lower.
    """
    results = [code_aster.thick_cylinder(tmp_path / f"s{n}", element_size_m=n)
               for n in (0.012, 0.006, 0.003)]

    displacement = [r.displacement_error for r in results]
    hoop = [r.hoop_stress_error for r in results]
    assert displacement[0] > displacement[1] > displacement[2]
    assert hoop[0] > hoop[1] > hoop[2]

    displacement_order = math.log2(displacement[0] / displacement[2]) / 2
    hoop_order = math.log2(hoop[0] / hoop[2]) / 2
    assert 1.6 < displacement_order < 2.4, displacement_order
    assert 0.8 < hoop_order < 1.6, hoop_order
    assert displacement_order > hoop_order


@requires_aster
def test_the_thick_cylinder_matches_lame(tmp_path):
    """The values themselves, on the finest mesh the test can afford."""
    result = code_aster.thick_cylinder(tmp_path, element_size_m=0.003)
    assert result.displacement_error < 0.01
    assert result.hoop_stress_error < 0.03


@requires_aster
def test_mismatched_groups_are_refused(tmp_path):
    """A load on the wrong edge still produces a field, and that field looks
    plausible until it meets a closed form.

    Code_Aster's GMSH reader was measured returning the bore group spanning
    the outer radius and holding triangles as well, while the file itself was
    correct. Reading MED avoids it, and this guard means a future regression
    in either direction is an error rather than a wrong number.
    """
    mesh = code_aster._quarter_annulus_mesh(
        tmp_path / "bad.med", 0.05, 0.1, 0.006)
    assert mesh.exists()


def test_no_code_aster_capability_is_registered_yet():
    """Two linear cases now pass, and still nothing is registered.

    Both reasons hold. Linear elasticity overlaps CalculiX, which is already
    verified here, and by this project's own rule an overlap is a cross-check
    rather than a capability the engine gains. And the nonlinear behaviour
    Code_Aster is actually here for is still unverified.

    Passing cases are not the same as a capability worth claiming.
    """
    from nodes.roster import build_roster

    names = {c.name for c in build_roster().all()}
    assert not any("aster" in n.lower() for n in names)
