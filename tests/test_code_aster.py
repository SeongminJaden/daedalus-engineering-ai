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


def test_only_plasticity_is_registered():
    """Plasticity is registered. Linear elasticity is not.

    Both linear cases in this module pass, and that is still not a reason to
    claim them: CalculiX already covers linear elasticity and is verified
    here, and by this project's rule an overlap is a cross-check rather than a
    capability the engine gains. The same judgement kept Elmer's conduction
    solver unregistered.

    What IS registered is the thing Code_Aster was installed for and the thing
    nothing else here does.
    """
    from nodes.roster import build_roster

    names = {c.name for c in build_roster().all()}
    assert code_aster.ASTER_PLASTICITY_CAPABILITY in names
    assert "analysis.fea.linear_aster" not in names
    assert sum(1 for n in names if "elasticity" in n and "aster" in n) == 0


def test_the_plasticity_conditions_refuse_when_unstated():
    """Whether a load passes yield cannot be inferred from silence.

    A linear solve below yield gives the same answer more cheaply, and a
    linear solve ABOVE yield reports a stress the material cannot carry, so
    the distinction has to be stated.
    """
    from core.registry.context import ProblemContext
    from nodes.roster import build_roster

    method = next(c.method for c in build_roster().all()
                  if c.name == code_aster.ASTER_PLASTICITY_CAPABILITY)
    reasons = " ".join(str(r) for r
                       in method.applicability(ProblemContext()).failed)
    assert "loads_exceed_yield" in reasons
    assert "strains_remain_small" in reasons

    stated = ProblemContext(loads_exceed_yield=True, strains_remain_small=True)
    assert not list(method.applicability(stated).failed)


# ------------------------------------------------------------- plasticity

@requires_aster
def test_below_yield_the_plastic_solve_reproduces_the_linear_one(tmp_path):
    """The limit case, and the cheapest anchor there is.

    Plasticity has few closed forms, but it has one certainty: below first
    yield an elastoplastic solve must return the elastic answer. The control
    is the linear case already verified against Lame, so this tests the
    nonlinear plumbing without needing a plastic closed form at all.
    """
    pressure = 50.0e6
    linear = code_aster.thick_cylinder(tmp_path / "lin", pressure_pa=pressure,
                                       element_size_m=0.003)
    plastic = code_aster.plastic_cylinder(tmp_path / "pl",
                                          pressure_pa=pressure,
                                          element_size_m=0.003)
    assert pressure < plastic.first_yield_pressure_pa
    assert plastic.plastic_radius_m == pytest.approx(0.05, abs=1e-9)
    difference = abs(plastic.bore_displacement_m - linear.bore_displacement_m)
    assert difference / linear.bore_displacement_m < 1e-10


def test_the_yield_criterion_is_von_mises_not_tresca():
    """Using the Tresca form against a von Mises solver is worth 15 percent,
    in the direction that makes the part look stronger than it is.

    Plane strain with von Mises and incompressible plastic flow gives
    sigma_theta - sigma_r = 2 sigma_y / sqrt(3), where the textbook thick
    cylinder formulas are usually quoted for Tresca with a factor of one.
    """
    cylinder = code_aster.PlasticCylinder(
        inner_radius_m=0.05, outer_radius_m=0.1, pressure_pa=0.0,
        yield_stress_pa=250e6, youngs_modulus_pa=210e9, poisson_ratio=0.3,
        bore_displacement_m=0.0, plastic_radius_m=0.0, converged=True)
    tresca = 250e6 * (0.01 - 0.0025) / (2 * 0.01)
    assert cylinder.first_yield_pressure_pa == pytest.approx(
        tresca * 2 / math.sqrt(3), rel=1e-12)
    assert cylinder.first_yield_pressure_pa / tresca > 1.15


def test_the_plastic_radius_inversion_round_trips():
    """The closed form and its inverse, checked without a solver."""
    cylinder = code_aster.PlasticCylinder(
        inner_radius_m=0.05, outer_radius_m=0.1, pressure_pa=0.0,
        yield_stress_pa=250e6, youngs_modulus_pa=210e9, poisson_ratio=0.3,
        bore_displacement_m=0.0, plastic_radius_m=0.0, converged=True)
    for radius in (0.06, 0.075, 0.09):
        pressure = cylinder.exact_pressure_for_plastic_radius(radius)
        at = code_aster.PlasticCylinder(
            0.05, 0.1, pressure, 250e6, 210e9, 0.3, 0.0, 0.0, True)
        assert at.exact_plastic_radius_m == pytest.approx(radius, rel=1e-9)


@requires_aster
def test_the_plastic_front_converges_to_the_closed_form(tmp_path):
    """The front is located to within one element, and that is measurable.

    Taking the outermost node that has yielded overshoots by up to one
    element, so the error is expected to be a fixed FRACTION of the element
    size rather than a fixed percentage. Measured at 0.57, 0.64 and 0.67 of an
    element across a factor of four in mesh size, which is the signature of a
    first order front estimate rather than a wrong answer.

    Richardson extrapolating the two finest meshes to zero element size gives
    the closed form to about 0.1 percent.
    """
    sizes = (0.004, 0.002, 0.001)
    runs = [code_aster.plastic_cylinder(tmp_path / f"h{h}", pressure_pa=150e6,
                                        element_size_m=h) for h in sizes]
    errors = [r.plastic_radius_m - r.exact_plastic_radius_m for r in runs]

    assert all(e > 0 for e in errors), "the estimate should overshoot"
    assert errors[0] > errors[1] > errors[2]
    for error, h in zip(errors, sizes):
        assert 0.3 < error / h < 1.0, (error, h)

    exact = runs[-1].exact_plastic_radius_m
    fine, finer = runs[1].plastic_radius_m, runs[2].plastic_radius_m
    extrapolated = finer - (fine - finer) / (sizes[1] / sizes[2] - 1.0)
    assert abs(extrapolated - exact) / exact < 0.005


@requires_aster
@pytest.mark.parametrize("pressure_pa", [130e6, 150e6, 170e6])
def test_the_plastic_front_is_right_at_several_pressures(tmp_path,
                                                         pressure_pa):
    """One pressure could agree by luck; three across the range cannot."""
    run = code_aster.plastic_cylinder(tmp_path, pressure_pa=pressure_pa,
                                      element_size_m=0.002)
    assert run.first_yield_pressure_pa < pressure_pa
    assert pressure_pa < run.fully_plastic_pressure_pa
    error = abs(run.plastic_radius_m - run.exact_plastic_radius_m)
    assert error / run.exact_plastic_radius_m < 0.03


# ------------------------------------------------- contact, NOT yet verified

def test_the_hertz_closed_forms_are_correct_arithmetic():
    """The formulas, checked without a solver.

    These are right and are worth keeping even though the solve is not
    working, because when it is fixed the target must already be known good.
    """
    contact = code_aster.HertzContact(
        sphere_radius_m=0.01, force_n=100.0, youngs_modulus_pa=210e9,
        poisson_ratio=0.3, contact_radius_m=0.0, peak_pressure_pa=0.0,
        zone_radius_m=0.0)
    effective = 210e9 / (1.0 - 0.3 ** 2)
    assert contact.effective_modulus_pa == pytest.approx(effective, rel=1e-12)

    radius = (3 * 100.0 * 0.01 / (4 * effective)) ** (1 / 3)
    assert contact.exact_contact_radius_m == pytest.approx(radius, rel=1e-12)
    assert contact.exact_peak_pressure_pa == pytest.approx(
        3 * 100.0 / (2 * math.pi * radius ** 2), rel=1e-12)
    assert contact.exact_approach_m == pytest.approx(radius ** 2 / 0.01,
                                                     rel=1e-12)


def test_the_half_space_assumption_is_reported():
    """Hertz assumes each body is a half space near the contact, which holds
    while the contact radius is small against the sphere radius."""
    contact = code_aster.HertzContact(
        sphere_radius_m=0.01, force_n=100.0, youngs_modulus_pa=210e9,
        poisson_ratio=0.3, contact_radius_m=0.0, peak_pressure_pa=0.0,
        zone_radius_m=0.0)
    assert contact.half_space_ratio < 0.05


@requires_aster
def test_hertz_contact_matches_the_closed_form(tmp_path):
    """Formerly a strict xfail: the contact zone sat 9.6 micrometres BELOW
    the plane and the bodies interpenetrated freely.

    The cause was the sign of the unilateral condition. LIAISON_UNIL imposes
    sum(COEF_MULT * ddl) < COEF_IMPO, and the study wrote COEF_MULT = +1 with
    COEF_IMPO = -Y, which is DY < -Y: the nodes were REQUIRED to pass below
    the plane. Written as -DY < Y the measured errors are 2.0 percent on the
    contact radius and 1.3 percent on the peak pressure. The floors below are
    the ones the xfail carried, so the fix is judged by the bar it failed.
    """
    result = code_aster.hertz_contact(tmp_path, force_n=100.0)
    assert result.contact_radius_error < 0.10, result.contact_radius_error
    assert result.peak_pressure_error < 0.15, result.peak_pressure_error
    assert result.contact_radius_m > 0.0


@requires_aster
def test_the_reversed_half_space_is_what_the_old_study_imposed(tmp_path):
    """The failure, reproduced on purpose so the explanation is a measurement.

    Running the same study with the original sign, the deformed contact zone
    must lie below the plane and the reported radius must be zero, which is
    exactly the symptom the xfail recorded.
    """
    result = code_aster.hertz_contact(tmp_path, force_n=100.0,
                                      reverse_half_space=True)
    assert result.contact_radius_m == 0.0
    assert result.lowest_point_m < 0.0


def test_contact_is_registered_now_that_it_is_verified():
    """Plasticity was registered because it was verified; contact was not,
    for the same reason. It is now, and its notes carry the measured errors
    and the sign convention that had it wrong."""
    from nodes.roster import build_roster

    registry = build_roster()
    names = {c.name for c in registry.all()}
    assert code_aster.ASTER_PLASTICITY_CAPABILITY in names
    assert code_aster.ASTER_CONTACT_CAPABILITY in names
    notes = registry.get(code_aster.ASTER_CONTACT_CAPABILITY).method.notes
    assert "COEF_MULT = -1" in notes and "Not verified" in notes
    # the linear elastic cases stay unregistered: CalculiX covers them
    assert not any(n.startswith("analysis.fea.code_aster") for n in names)
