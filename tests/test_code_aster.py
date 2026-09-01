"""Verification of the Code_Aster node.

Code_Aster is here for nonlinear material behaviour, contact and plasticity,
which is where it does what CalculiX does not. None of that is verified yet,
and nothing is registered as a capability, because a plastic result cannot be
told apart from a wrong install without a linear case that has an exact answer
to anchor it.
"""

from __future__ import annotations

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
@pytest.mark.xfail(strict=True,
                   reason="the named groups Code_Aster reads back from the "
                          "GMSH file do not match the curves they were "
                          "assigned to, so the load lands on the wrong edges; "
                          "the study now refuses rather than returning a "
                          "plausible wrong field")
def test_the_thick_cylinder_matches_lame(tmp_path):
    """Marked as a known failure rather than deleted or loosened.

    strict=True means this test fails if it ever starts PASSING, so the fix
    cannot land silently and the claim gets revisited deliberately.
    """
    result = code_aster.thick_cylinder(tmp_path, element_size_m=0.006)
    assert result.displacement_error < 0.05
    assert result.hoop_stress_error < 0.10


def test_no_code_aster_capability_is_registered_yet():
    """Nothing is claimed while only one linear case is verified.

    The bar proves the install and the plumbing. It does not prove the
    nonlinear behaviour Code_Aster is actually here for, and the one case
    that would extend the claim does not yet work.
    """
    from nodes.roster import build_roster

    names = {c.name for c in build_roster().all()}
    assert not any("aster" in n.lower() for n in names)
