"""Cross-validation of the fluid work against OpenFOAM.

Every benchmark here has an exact solution, so the comparison is against
mathematics rather than against another approximation. Where a discretisation
error is expected, it is PREDICTED in closed form first and the test checks
the prediction, because a test that merely records whatever number came out
cannot fail for the right reason.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.registry import Category, ProblemContext
from nodes import openfoam as of
from nodes.descriptor import CapabilityUnavailable
from nodes.roster import build_roster
from nodes.verification import (FlowCrossValidation,
                                flow_cross_validated_status)

from physics.fluids.internal import solve_pipe_flow

requires_openfoam = pytest.mark.skipif(
    not of.is_available(), reason="OpenFOAM is not installed")

RHO = 1000.0
NU = 1e-6


@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("openfoam")


@pytest.fixture(scope="module")
def couette(work):
    case = of.ChannelCase(gap_m=0.01, kinematic_viscosity_m2_s=1e-5,
                          cells=20, wall_velocity_m_s=1.0)
    return case, of.solve(case, work / "couette")


@pytest.fixture(scope="module")
def channel(work):
    case = of.ChannelCase(gap_m=0.01, kinematic_viscosity_m2_s=1e-5,
                          cells=20, body_force_m_s2=0.01)
    return case, of.solve(case, work / "channel")


@pytest.fixture(scope="module")
def pipe(work):
    case = of.PipeCase(radius_m=0.005, kinematic_viscosity_m2_s=NU,
                       body_force_m_s2=0.01, radial_cells=80,
                       wedge_angle_deg=1.0)
    return case, of.solve(case, work / "pipe")


# ---------------------------------------------------------------- the node

def test_availability_is_read_from_the_filesystem():
    """The descriptor must not be able to claim a solver that is not there."""
    descriptor = of.openfoam_descriptor()
    assert descriptor.available is of.is_available()
    if not descriptor.available:
        assert "unavailable" in descriptor.unavailable_reason


@requires_openfoam
def test_the_version_is_reported_not_guessed():
    reported = of.version()
    assert reported is not None and "OpenFOAM" in reported


def test_a_missing_solver_raises_rather_than_returning_numbers(tmp_path,
                                                               monkeypatch):
    """A missing solver must never be mistaken for a passing comparison."""
    monkeypatch.setattr(of, "is_available", lambda: False)
    case = of.ChannelCase(gap_m=0.01, kinematic_viscosity_m2_s=1e-5)
    with pytest.raises(CapabilityUnavailable):
        of.solve(case, tmp_path / "nothing")


def test_the_capability_is_registered_alongside_the_analytical_one():
    registry = build_roster()
    assert of.OPENFOAM_CAPABILITY in registry
    context = ProblemContext(has_internal_flow=True, flow_is_laminar=True)
    names = registry.query(context, Category.ANALYSIS).names()
    assert "pipe_flow" in names
    if of.is_available():
        assert of.OPENFOAM_CAPABILITY in names


def test_the_capability_is_refused_on_turbulent_flow():
    """No turbulence model is configured, so the node must decline.

    The analytical method stays available there, which is the point: the
    cross-check disappears exactly where it cannot be trusted, rather than
    returning a confident laminar answer for a turbulent duct.
    """
    registry = build_roster()
    turbulent = ProblemContext(has_internal_flow=True, flow_is_laminar=False)
    names = registry.query(turbulent, Category.ANALYSIS).names()
    assert of.OPENFOAM_CAPABILITY not in names
    assert "pipe_flow" in names


def test_an_unstated_regime_does_not_route_to_cfd():
    """Not stating the regime must fail closed, not default to laminar."""
    registry = build_roster()
    unstated = ProblemContext(has_internal_flow=True)
    assert of.OPENFOAM_CAPABILITY not in registry.query(
        unstated, Category.ANALYSIS).names()


# ------------------------------------------------------------- plane Couette

@requires_openfoam
def test_couette_matches_the_exact_linear_profile(couette):
    case, result = couette
    worst = max(abs(u - case.analytical_velocity(y))
                for y, u in zip(result.coordinates_m, result.velocity_m_s))
    assert worst < 1e-9 * case.wall_velocity_m_s


@requires_openfoam
def test_couette_carries_no_discretisation_error_because_it_is_linear(couette):
    """A linear profile is in the discrete space exactly.

    This case therefore tests the mesh, the boundary conditions, the solver
    and the parsing, but it puts no pressure at all on the scheme's accuracy.
    Agreement here is much cheaper than it looks and must not be quoted as
    though the scheme had been tested.
    """
    case, result = couette
    second = [result.velocity_m_s[i - 1] - 2 * result.velocity_m_s[i]
              + result.velocity_m_s[i + 1]
              for i in range(1, len(result.velocity_m_s) - 1)]
    assert max(abs(value) for value in second) < 1e-11


def test_couette_is_laminar_by_construction_not_by_result():
    """The mesh is one cell deep, so transition cannot occur in it.

    Plane Couette flow is linearly stable at every Reynolds number and yet
    goes turbulent in experiment near Re = 360. This case runs at Re = 1000
    and stays perfectly laminar, which is a property of the mesh rather than
    a physical finding.
    """
    case = of.ChannelCase(gap_m=0.01, kinematic_viscosity_m2_s=1e-5,
                          wall_velocity_m_s=1.0)
    assert case.reynolds_number() > 360.0


# ---------------------------------------------------------- plane Poiseuille

@requires_openfoam
def test_channel_error_is_the_predicted_uniform_offset(channel):
    """Predicted before measuring: the profile shifts by dy^2 |u''| / 8."""
    case, result = channel
    offsets = [u - case.analytical_velocity(y)
               for y, u in zip(result.coordinates_m, result.velocity_m_s)]
    mean = sum(offsets) / len(offsets)
    assert mean == pytest.approx(case.wall_offset_prediction(), rel=1e-6)


@requires_openfoam
def test_the_channel_error_is_an_offset_and_not_a_shape_error(channel):
    """The shape is exact to round-off, so only the offset is wrong."""
    case, result = channel
    offsets = [u - case.analytical_velocity(y)
               for y, u in zip(result.coordinates_m, result.velocity_m_s)]
    assert max(offsets) - min(offsets) < 1e-11


@requires_openfoam
@pytest.mark.parametrize("cells", [20, 40])
def test_the_channel_offset_falls_as_the_square_of_the_spacing(work, cells):
    case = of.ChannelCase(gap_m=0.01, kinematic_viscosity_m2_s=1e-5,
                          cells=cells, body_force_m_s2=0.01)
    result = of.solve(case, work / f"refine{cells}")
    offsets = [u - case.analytical_velocity(y)
               for y, u in zip(result.coordinates_m, result.velocity_m_s)]
    mean = sum(offsets) / len(offsets)
    assert mean == pytest.approx(case.wall_offset_prediction(), rel=1e-6)


# -------------------------------------------------------- Hagen-Poiseuille

@requires_openfoam
def test_pipe_mean_velocity_matches_hagen_poiseuille(pipe):
    case, result = pipe
    exact = case.analytical_mean_velocity()
    assert result.mean_velocity_m_s == pytest.approx(exact, rel=1e-5)


@requires_openfoam
def test_pipe_profile_is_parabolic_to_the_expected_accuracy(pipe):
    case, result = pipe
    worst = max(abs(u - case.analytical_velocity(r)) / case.analytical_velocity(0.0)
                for r, u in zip(result.coordinates_m, result.velocity_m_s))
    assert worst < 5e-3


def test_the_pipe_stays_inside_the_laminar_validity_domain():
    """The benchmark has to sit where its exact solution is actually valid."""
    case = of.PipeCase(radius_m=0.005, kinematic_viscosity_m2_s=NU,
                       body_force_m_s2=0.01)
    assert case.reynolds_number() < 2300.0


# ------------------------------------------- the wedge angle control experiment

@requires_openfoam
def test_a_five_degree_wedge_is_wrong_by_the_predicted_amount(work):
    """The control: a coarse wedge angle costs about theta^2 / 4.

    This is the error that radial refinement does NOT remove, so it is the one
    that would have been quoted as solver disagreement if it had not been
    diagnosed.
    """
    case = of.PipeCase(radius_m=0.005, kinematic_viscosity_m2_s=NU,
                       body_force_m_s2=0.01, radial_cells=80,
                       wedge_angle_deg=5.0)
    result = of.solve(case, work / "wedge5")
    exact = case.analytical_mean_velocity()
    measured = (result.mean_velocity_m_s - exact) / exact
    assert measured < 0.0
    assert abs(measured) == pytest.approx(case.wedge_deficit_prediction(),
                                          rel=0.10)


@requires_openfoam
def test_refining_the_mesh_does_not_fix_a_coarse_wedge(work):
    """Halving the cell size leaves the geometric error where it was."""
    coarse = of.PipeCase(radius_m=0.005, kinematic_viscosity_m2_s=NU,
                         body_force_m_s2=0.01, radial_cells=40,
                         wedge_angle_deg=5.0)
    fine = of.PipeCase(radius_m=0.005, kinematic_viscosity_m2_s=NU,
                       body_force_m_s2=0.01, radial_cells=80,
                       wedge_angle_deg=5.0)
    exact = coarse.analytical_mean_velocity()
    errors = []
    for case, name in ((coarse, "wedge5c"), (fine, "wedge5f")):
        result = of.solve(case, work / name)
        errors.append(abs(result.mean_velocity_m_s - exact) / exact)
    assert min(errors) > 0.5 * coarse.wedge_deficit_prediction()


# ------------------------------------------------- the three way comparison

@requires_openfoam
def test_the_project_formula_is_not_independent_of_the_closed_form(pipe):
    """Darcy-Weisbach with f = 64/Re IS Hagen-Poiseuille, rearranged.

    They agree to machine precision, which confirms the algebra and nothing
    else. Quoting this as a cross-check would be circular.
    """
    case, result = pipe
    diameter, length = 2 * case.radius_m, case.length_m
    velocity = result.mean_velocity_m_s
    flow = velocity * math.pi * case.radius_m ** 2
    closed_form = 32 * (RHO * NU) * length * velocity / diameter ** 2
    ours = solve_pipe_flow(flow, diameter, length, RHO, RHO * NU)
    assert ours.pressure_drop_pa == pytest.approx(closed_form, rel=1e-12)
    assert ours.friction_factor == pytest.approx(64.0 / ours.reynolds, rel=1e-12)


@requires_openfoam
def test_openfoam_independently_confirms_the_project_pressure_drop(pipe):
    """The only leg of the comparison that is genuinely independent."""
    case, result = pipe
    diameter, length = 2 * case.radius_m, case.length_m
    flow = result.mean_velocity_m_s * math.pi * case.radius_m ** 2
    ours = solve_pipe_flow(flow, diameter, length, RHO, RHO * NU)
    imposed = case.pressure_drop_pa(RHO)
    assert ours.pressure_drop_pa == pytest.approx(imposed, rel=1e-4)


# ------------------------------------------------------------- the limit

def test_no_amount_of_solver_agreement_reaches_physical_validation():
    """OpenFOAM is a simulation. Two simulations agreeing is not a test.

    Both solutions assume an incompressible Newtonian fluid, a smooth rigid
    wall, no entrance effect and a laminar state imposed by the mesh. Every
    one of those assumptions could be wrong together.
    """
    method = of.openfoam_capability_method()
    assert method.evidence == "SIMULATED"
    assert "not a wind tunnel" in method.notes


@requires_openfoam
def test_a_flow_result_can_be_recorded_as_cross_validated(pipe):
    """The measurement travels with the claim, as it does for the solid case."""
    case, result = pipe
    exact_mean = case.analytical_mean_velocity()
    peak = case.analytical_velocity(0.0)
    mean_error = abs(result.mean_velocity_m_s - exact_mean) / exact_mean
    profile_error = max(
        abs(u - case.analytical_velocity(r)) / peak
        for r, u in zip(result.coordinates_m, result.velocity_m_s))
    validation = FlowCrossValidation(
        solver="OpenFOAM", solver_version=result.solver_version,
        mean_velocity_relative_error=mean_error,
        profile_relative_error=profile_error, tolerance=1e-2,
        discretisation=f"{case.radial_cells} radial cells, "
                       f"{case.wedge_angle_deg} degree wedge")
    status = flow_cross_validated_status("pipe-benchmark", validation)
    assert status.is_cross_validated
    assert status.cross_validation is validation
    assert status.is_physically_validated is False


def test_a_disagreeing_flow_result_reports_the_number_it_measured():
    """Disagreement must not quietly revert to the weaker status."""
    validation = FlowCrossValidation(
        solver="OpenFOAM", solver_version="v1912",
        mean_velocity_relative_error=0.25, profile_relative_error=0.30,
        tolerance=1e-2, discretisation="20 radial cells, 5.0 degree wedge")
    status = flow_cross_validated_status("bad", validation)
    assert not status.is_cross_validated
    assert "2.500e-01" in status.reason
    assert "5.0 degree wedge" in status.reason
    assert status.cross_validation is None
