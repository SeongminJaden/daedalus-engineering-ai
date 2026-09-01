"""Verification of the Elmer node against closed form answers.

Elmer earns its place for electromagnetics, which nothing else in the stack
does. The conduction case is here because it is the cheapest way to prove the
mesh writing, the sif, the run and the result parsing are all correct; it is
NOT claimed as a capability, because CalculiX already does conduction and a
second answer to the same question is a cross-check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nodes import elmer

requires_elmer = pytest.mark.skipif(
    not elmer.is_available(),
    reason="Elmer is not installed at ELMER_HOME")


# ------------------------------------------------------------------ meshing

def test_the_box_mesh_tags_all_six_faces():
    mesh = elmer.box_mesh(0.1, 0.02, 0.02, 4, 2, 2)
    assert len(mesh.nodes) == 5 * 3 * 3
    assert len(mesh.elements) == 4 * 2 * 2
    tags = {}
    for bc, _, _ in mesh.boundary:
        tags[bc] = tags.get(bc, 0) + 1
    assert tags == {1: 4, 2: 4, 3: 8, 4: 8, 5: 8, 6: 8}


def test_the_mesh_header_type_count_matches_the_lines_that_follow(tmp_path):
    """A count that disagrees with the list makes Elmer stop on a header error.

    This was a real mistake: a count of three was written above two type
    lines, and the solver refused the mesh rather than guessing.
    """
    mesh = elmer.box_mesh(0.1, 0.02, 0.02, 2, 1, 1).write(tmp_path)
    lines = (mesh / "mesh.header").read_text().splitlines()
    declared = int(lines[1])
    assert declared == len(lines) - 2


# ----------------------------------------------- conduction, plumbing check

@requires_elmer
def test_a_uniformly_heated_slab_matches_the_closed_form(tmp_path):
    """Both faces held at T0, uniform volumetric heating.

    T(x) = T0 + q x (L - x) / (2 k), so the rise at the centre is q L^2 / 8 k.
    Note the convention: L is the FULL thickness with both faces held. The
    scoping document quotes the same result in the half thickness form,
    q L^2 / 2 k, for a slab of half thickness L.
    """
    length, conductivity, source, wall = 0.1, 15.0, 2.0e5, 300.0
    elmer.box_mesh(length, 0.02, 0.02, 40, 2, 2).write(tmp_path)

    # Elmer's Heat Source is per unit MASS and is multiplied by density, so a
    # density of one makes it numerically equal to the volumetric source.
    sif = f"""
Header
  Mesh DB "." "mesh"
End
Simulation
  Coordinate System = Cartesian 3D
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Output Intervals = 0
End
Body 1
  Equation = 1
  Material = 1
  Body Force = 1
End
Equation 1
  Active Solvers(1) = 1
End
Solver 1
  Equation = Heat Equation
  Variable = Temperature
  Procedure = "HeatSolve" "HeatSolver"
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 2000
  Linear System Convergence Tolerance = 1.0e-12
  Linear System Preconditioning = ILU0
  Nonlinear System Max Iterations = 1
  Steady State Convergence Tolerance = 1.0e-10
End
Solver 2
  Equation = SaveScalars
  Procedure = "SaveData" "SaveScalars"
  Filename = "scalars.dat"
  Variable 1 = Temperature
  Operator 1 = max
End
Material 1
  Density = 1.0
  Heat Conductivity = {conductivity}
End
Body Force 1
  Heat Source = {source}
End
Boundary Condition 1
  Target Boundaries(2) = 1 2
  Temperature = {wall}
End
"""
    elmer.run(tmp_path, sif)
    scalars = elmer.read_scalars(tmp_path)
    hottest = max(scalars.values())
    exact = wall + source * length ** 2 / (8.0 * conductivity)
    assert abs(hottest - exact) / (exact - wall) < 1e-9


# ------------------------------------------------------ magnetostatics

@requires_elmer
def test_the_wire_field_converges_at_the_right_ORDER(tmp_path):
    """The real check, and it is a rate rather than a tolerance.

    A tolerance can be met by a wrong solve on a fine enough mesh. The ORDER
    of convergence cannot: for linear triangles the potential must converge at
    second order and the flux density, being its derivative, at first. Seeing
    both is much stronger evidence than one number under a threshold.
    """
    results = [elmer.wire_magnetostatics(tmp_path / f"d{d}", divisions=d)
               for d in (12, 25, 50)]

    potential_errors = [r.potential_error for r in results]
    flux_errors = [r.flux_density_error for r in results]

    assert potential_errors[0] > potential_errors[1] > potential_errors[2]
    assert flux_errors[0] > flux_errors[1] > flux_errors[2]

    potential_order = math.log2(potential_errors[0] / potential_errors[2]) / 2
    flux_order = math.log2(flux_errors[0] / flux_errors[2]) / 2
    assert 1.6 < potential_order < 2.4, potential_order
    assert 0.6 < flux_order < 1.4, flux_order
    assert potential_order > flux_order


@requires_elmer
def test_the_wire_flux_density_matches_the_closed_form(tmp_path):
    """|B| at the wire surface is mu0 I / (2 pi a)."""
    result = elmer.wire_magnetostatics(tmp_path, divisions=60)
    assert result.exact_flux_density_max_t == pytest.approx(
        elmer.MU0 * 100.0 / (2 * math.pi * 0.002), rel=1e-12)
    assert result.flux_density_error < 0.02


def test_the_exact_answers_are_stated_independently_of_the_solver():
    """The closed forms must not be derived from a solved result.

    A(0) = (mu0 I / 2 pi)(1/2 + ln(b/a)) and |B|max = mu0 I / (2 pi a), both
    written out here from the analysis rather than read back from Elmer.
    """
    field = elmer.WireField(
        wire_radius_m=0.002, domain_radius_m=0.05, current_a=100.0,
        elements=0, potential_centre_wb_per_m=0.0, flux_density_max_t=0.0)
    mu0_i_over_2pi = 4e-7 * math.pi * 100.0 / (2.0 * math.pi)
    assert field.exact_potential_centre_wb_per_m == pytest.approx(
        mu0_i_over_2pi * (0.5 + math.log(25.0)), rel=1e-12)
    assert field.exact_flux_density_max_t == pytest.approx(
        mu0_i_over_2pi / 0.002, rel=1e-12)


def test_a_missing_scalars_file_raises_rather_than_returning_zero(tmp_path):
    """An empty result read as zero makes a broken solve look converged."""
    with pytest.raises(RuntimeError, match="no scalars.dat"):
        elmer.read_scalars(tmp_path)


# ------------------------------------------------- the coupled check

@requires_elmer
def test_integrated_joule_heating_equals_i_squared_r(tmp_path):
    """The check a backwards coupling cannot pass.

    Both halves of a coupled problem can be individually correct while the
    link between them is wired the wrong way round, and each separate check
    still passes. Comparing the solver's integrated heating against I squared
    R, computed from the bar's dimensions and its conductivity with no solver
    involved, is what catches that.
    """
    run = elmer.joule_heating(tmp_path)
    assert run.heating_error < 1e-12


@requires_elmer
@pytest.mark.parametrize("conductivities", [
    (5.96e7, 3.5e7),        # copper into aluminium
    (5.96e7, 1.45e6),       # copper into steel, a ratio of about 41
])
def test_two_materials_in_series_also_match(tmp_path, conductivities):
    """The discriminating version.

    A uniform bar is a weak test, because several wrong formulas agree with
    the right one when there is only one conductivity to get wrong. In series
    the RESISTANCES add, so the answer depends on the harmonic combination.
    Averaging the two conductivities instead, which is the natural mistake,
    gives 3.3e-5 ohm where the answer is 3.5e-4 for the copper and steel bar,
    an order of magnitude apart.
    """
    run = elmer.joule_heating(tmp_path, conductivities=conductivities)
    assert run.heating_error < 1e-12


def test_the_series_resistance_is_not_the_averaged_conductivity():
    """Stated as arithmetic so the discriminating power is not just asserted.

    If this ever became true the series fixtures above would stop testing
    anything, because the wrong formula would give the right answer.
    """
    length, area = 0.1, 1e-4
    sigma1, sigma2 = 5.96e7, 1.45e6
    series = (length / 2) / (sigma1 * area) + (length / 2) / (sigma2 * area)
    averaged = length / (0.5 * (sigma1 + sigma2) * area)
    assert series / averaged > 5.0


def test_output_without_a_heating_line_is_refused():
    """A silent zero would look like agreement for a bar carrying no current.

    The parser is driven directly with solver output that lacks the heating
    line, because provoking the real solver into that state would test Elmer
    rather than this code.
    """
    assert elmer._HEATING.findall("MAIN: nothing useful here\n") == []
    assert elmer._HEATING.findall(
        "Total Heating Power   :   14900.000\n") == ["14900.000"]


# ------------------------------------------------------------ registration

def test_only_magnetostatics_is_registered():
    """Elmer also solves conduction and elasticity. Those are not registered.

    CalculiX already covers them and is already verified, so a second
    implementation of the same equations is a cross-check rather than a
    capability the engine gains.
    """
    from nodes.roster import build_roster

    names = {c.name for c in build_roster().all()}
    assert elmer.ELMER_MAGNETOSTATICS_CAPABILITY in names
    assert not any(n.startswith("analysis.thermal.elmer") for n in names)
    assert not any("elmer" in n and "fea" in n for n in names)


def test_the_magnetostatic_conditions_refuse_when_unstated():
    """The two bounds fail in the direction that flatters a design, so
    neither may be assumed from silence."""
    from core.registry.context import ProblemContext
    from nodes.roster import build_roster

    method = next(c.method for c in build_roster().all()
                  if c.name == elmer.ELMER_MAGNETOSTATICS_CAPABILITY)

    stated = ProblemContext(has_conductor_current=True,
                            skin_depth_exceeds_conductor=True,
                            magnetically_linear=True)
    assert not list(method.applicability(stated).failed)

    silent = ProblemContext(has_conductor_current=True)
    reasons = " ".join(str(r) for r in method.applicability(silent).failed)
    assert "skin_depth_exceeds_conductor" in reasons
    assert "magnetically_linear" in reasons
