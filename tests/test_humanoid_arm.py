"""The forearm run, and the requirements added to make it a real forearm.

The point of these tests is not that the optimiser finds a number. It is that
adding a REQUIREMENT moves the design while a PRIOR cannot, which is the whole
separation `core.design_reference` exists to enforce.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.engineering_ir import LoadType
from optimization.constraints import build_optimization_problem, evaluate_design
from optimization.constraints.evaluate import constraint_jacobian
from optimization.gradient.slsqp import default_start, optimize_slsqp
from projects.humanoid_arm import (BASELINE, PACKAGED, build_reference,
                                   forearm_problem, run)
from projects.robotic_link.problem import build_mvp_problem

PROBE = np.array([0.030, 0.045, 0.003])


@pytest.fixture(scope="module")
def baseline():
    return build_optimization_problem(forearm_problem(BASELINE))


@pytest.fixture(scope="module")
def packaged():
    return build_optimization_problem(forearm_problem(PACKAGED))


# --------------------------------------------- the additions are additions

def test_a_problem_stating_neither_requirement_is_unchanged():
    """Both new constraints default to absent, so nothing else moves."""
    mvp = build_optimization_problem(build_mvp_problem())
    assert mvp.min_clear_bore_m is None
    assert mvp.applied_torque_nm == 0.0
    assert set(evaluate_design(mvp, np.array([0.05, 0.05, 0.005])).constraints) \
        == {"stress", "cavity_b", "cavity_h", "deflection"}


def test_the_baseline_forearm_has_no_bore_or_torsion_constraint(baseline):
    assert set(evaluate_design(baseline, PROBE).constraints) \
        == {"stress", "cavity_b", "cavity_h", "deflection"}


def test_the_packaged_forearm_adds_exactly_the_requirements_stated(packaged):
    """Two from the bore, one from the torque, one from the process."""
    assert set(evaluate_design(packaged, PROBE).constraints) \
        == {"stress", "cavity_b", "cavity_h", "deflection",
            "bore_b", "bore_h", "combined_stress", "manufacturing_wall"}


# ---------------------------------------------------- the constraints are right

def test_the_bore_constraint_measures_the_clear_opening(packaged):
    b, h, t = PROBE
    bore = packaged.min_clear_bore_m
    values = evaluate_design(packaged, PROBE).constraints
    assert values["bore_b"] == pytest.approx(((b - 2 * t) - bore) / bore)
    assert values["bore_h"] == pytest.approx(((h - 2 * t) - bore) / bore)


def test_a_section_too_small_to_pass_the_actuator_is_infeasible(packaged):
    """A 16 mm wide tube cannot pass a 15 mm actuator through a 1 mm wall."""
    tight = np.array([0.016, 0.045, 0.001])
    values = evaluate_design(packaged, tight).constraints
    assert values["bore_b"] < 0.0
    assert not evaluate_design(packaged, tight).is_feasible()


def test_the_combined_stress_uses_bredt_and_von_mises(packaged):
    """Checked against the formula written out, not against itself."""
    b, h, t = PROBE
    evaluation = evaluate_design(packaged, PROBE)
    enclosed = (b - t) * (h - t)          # the MIDLINE encloses this, not b*h
    shear = packaged.applied_torque_nm / (2.0 * enclosed * t)
    bending = evaluation.max_bending_stress_pa
    expected = np.sqrt(bending ** 2 + 3.0 * shear ** 2)
    assert evaluation.constraints["combined_stress"] == pytest.approx(
        1.0 - expected / packaged.allowable_stress_pa, rel=1e-9)


def test_torsion_makes_the_combined_stress_the_binding_one(packaged):
    """Combined must be worse than bending alone, or it is not doing anything."""
    values = evaluate_design(packaged, PROBE).constraints
    assert values["combined_stress"] < values["stress"]


@pytest.mark.parametrize("name", ["bore_b", "bore_h", "combined_stress"])
def test_the_new_gradients_match_finite_differences(packaged, name):
    """The optimiser trusts these, so they are checked rather than derived.

    The step is relative and not tiny: the beam evaluation runs in single
    precision on the GPU, and a step small enough to look rigorous just
    measures round-off.
    """
    analytic = constraint_jacobian(packaged, PROBE)[name]
    numeric = np.zeros(3)
    for i in range(3):
        step = 1e-4 * PROBE[i]
        up, down = PROBE.copy(), PROBE.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (evaluate_design(packaged, up).constraints[name]
                      - evaluate_design(packaged, down).constraints[name]) \
            / (2 * step)
    scale = max(np.abs(numeric).max(), 1e-12)
    assert np.abs(analytic - numeric).max() / scale < 1e-3


# ------------------------------------------------ the beam model and torque

def test_the_beam_model_accepts_a_torque_beside_its_point_force():
    """Adding the torque must not make the bending model refuse the problem."""
    problem = forearm_problem(PACKAGED)
    assert sum(1 for load in problem.loads
               if load.type is LoadType.TORQUE) == 1
    build_optimization_problem(problem)          # would raise if refused


def test_the_beam_model_still_refuses_two_point_forces():
    """The original guard has to survive: one bending load, not two."""
    from physics.structural.beam import evaluate_beam

    problem = forearm_problem(BASELINE)
    doubled = problem.model_copy(update={"loads": list(problem.loads) * 2})
    with pytest.raises(NotImplementedError,
                       match="exactly one point force load"):
        evaluate_beam([build_optimization_problem(problem).genome(PROBE)],
                      doubled)


# ------------------------------------- a prior cannot move what a requirement can

@pytest.mark.parametrize("variant", [BASELINE, PACKAGED])
def test_the_reference_never_moves_the_physics(variant):
    result = run(variant)
    assert result["physics_unmoved"]
    default = result["from_default"]
    biased = result["from_reference"]
    assert np.abs(default.x - biased.x).max() < 1e-8
    assert default.evaluation.mass_kg == pytest.approx(
        biased.evaluation.mass_kg, rel=1e-9)


def test_the_requirement_does_move_the_design():
    """The contrast that makes the separation visible.

    A prior asking for a wider section changed nothing. A bore requirement
    asking for the same thing widened the member by 70 percent, because it
    entered as a constraint rather than as a suggestion.
    """
    slim = run(BASELINE)["from_reference"].x
    packaged = run(PACKAGED)["from_reference"].x
    assert packaged[0] > slim[0] * 1.5
    assert build_reference().prior("outer_width_m").minimum > slim[0]


def test_the_packaged_design_actually_passes_every_check():
    result = run(PACKAGED)["from_reference"]
    evaluation = result.evaluation
    assert evaluation.is_feasible()
    assert evaluation.safety_factor >= 2.0
    assert all(value >= -1e-4 for value in evaluation.constraints.values())
    # The bore is what binds, so it should sit on its limit.
    assert "bore_b" in evaluation.active_constraints()


# --------------------------------- the wall floor is a process, not a strength

def test_the_manufacturing_wall_defaults_to_absent():
    """A problem that does not state a process is unaffected by this."""
    mvp = build_optimization_problem(build_mvp_problem())
    assert mvp.min_manufacturing_wall_m is None
    assert "manufacturing_wall" not in evaluate_design(
        mvp, np.array([0.05, 0.05, 0.005])).constraints


def test_the_wall_constraint_measures_the_wall_against_the_floor(packaged):
    from projects.humanoid_arm.forearm import MANUFACTURING_WALL_M

    floor = packaged.min_manufacturing_wall_m
    assert floor == MANUFACTURING_WALL_M
    value = evaluate_design(packaged, PROBE).constraints["manufacturing_wall"]
    assert value == pytest.approx((PROBE[2] - floor) / floor)


def test_the_wall_gradient_matches_finite_differences(packaged):
    analytic = constraint_jacobian(packaged, PROBE)["manufacturing_wall"]
    step = 1e-4 * PROBE[2]
    up, down = PROBE.copy(), PROBE.copy()
    up[2] += step
    down[2] -= step
    numeric = (evaluate_design(packaged, up).constraints["manufacturing_wall"]
               - evaluate_design(packaged,
                                 down).constraints["manufacturing_wall"]) \
        / (2 * step)
    assert analytic[2] == pytest.approx(numeric, rel=1e-6)
    assert analytic[0] == 0.0 and analytic[1] == 0.0


def test_no_load_lifts_the_wall_off_its_floor():
    """The measurement that made the process constraint the right answer.

    A closed section carries torsion as shear flow q = T/(2A), so enlarging
    the section always beats thickening the wall on mass. The optimiser knows
    that and never buys wall.
    """
    import projects.humanoid_arm.forearm as forearm

    original = forearm.WRIST_TORQUE_NM
    try:
        walls = []
        for torque in (5.0, 100.0, 400.0):
            forearm.WRIST_TORQUE_NM = torque
            problem = build_optimization_problem(
                forearm.forearm_problem(PACKAGED))
            result = optimize_slsqp(problem, x0=default_start(problem))
            walls.append(result.x[2])
    finally:
        forearm.WRIST_TORQUE_NM = original
    floor = forearm.MANUFACTURING_WALL_M
    assert all(w == pytest.approx(floor, abs=1e-9) for w in walls)


# ----------------------------------- the reference decomposes into requirements

def test_the_wall_prior_is_satisfied_once_the_process_is_stated():
    """The reference asked for 2 to 4 mm. A process constraint delivers it."""
    prior = build_reference().prior("wall_thickness_m")
    wall = run(PACKAGED)["from_reference"].x[2]
    assert prior.minimum <= wall + 1e-9 <= prior.maximum


def test_the_width_is_set_by_the_bore_and_not_by_strength():
    """b = bore + 2t exactly, so the width prior is a packaging statement.

    Inverting it: the reference's 20 to 40 mm width corresponds to a 16 to
    36 mm clear bore, which is a checkable claim about what the reference was
    describing rather than a guess.
    """
    import projects.humanoid_arm.forearm as forearm

    original = forearm.CLEAR_BORE_M
    try:
        for bore in (0.015, 0.020, 0.036):
            forearm.CLEAR_BORE_M = bore
            problem = build_optimization_problem(
                forearm.forearm_problem(PACKAGED))
            result = optimize_slsqp(problem, x0=default_start(problem))
            assert result.x[0] == pytest.approx(bore + 2 * result.x[2],
                                                abs=1e-9)
    finally:
        forearm.CLEAR_BORE_M = original


def test_at_this_load_the_forearm_is_not_a_structural_problem():
    """Nothing structural binds: packaging and process decide the section.

    Worth asserting because it is the conclusion, and because a later change
    that makes strength bind again should say so loudly.
    """
    evaluation = run(PACKAGED)["from_reference"].evaluation
    active = set(evaluation.active_constraints())
    assert active == {"bore_b", "bore_h", "manufacturing_wall"}
    for name in ("stress", "combined_stress", "deflection"):
        assert evaluation.constraints[name] > 0.1
    assert evaluation.safety_factor > 50.0
