"""The integrated capstone: whole-assembly design with a conjunctive verdict.

The tests that carry this phase are the fault injections. Each one perturbs the
spec so that exactly one component's check should fail, and asserts that the
ASSEMBLY is rejected and that the right check is named as governing. Without
them, an aggregation layer that always said "passes" would look identical to
one that works.

The second group covers the gap rule: a failure mode with no applicable method
must be reported as unassessed and must not count as a pass.
"""

import pytest

from integration import (KNOWN_UNIMPLEMENTED_MODES, AssemblyStatus,
                         AssemblyVerdict, CheckResult, CheckStatus, JointSpec,
                         design_joint, review, routed_methods, satisfies)


@pytest.fixture(scope="module")
def baseline():
    return design_joint(JointSpec())


def failed_modes(result) -> set[str]:
    return {f"{f.component}/{f.failure_mode}" for f in result.verdict.failures()}


# --- the framework -----------------------------------------------------------

def test_a_verdict_must_name_the_method_behind_it():
    """An unauditable verdict is refused at construction."""
    with pytest.raises(ValueError, match="names no method"):
        CheckResult("link", "fatigue", CheckStatus.PASSED)


def test_a_gap_must_explain_itself():
    """An unexplained gap is indistinguishable from an oversight."""
    with pytest.raises(ValueError, match="says nothing about why"):
        CheckResult("link", "corrosion", CheckStatus.NOT_ASSESSED)


def test_the_shared_feasibility_tolerance_admits_an_active_constraint():
    """An optimiser drives its active constraints exactly to their limits.

    The design it returns therefore has a ratio of 1.0 to within its own
    convergence tolerance and lands on either side by floating point alone.
    Comparing strictly against 1.0 rejects the optimum at random, which is
    what happened before this tolerance was shared between the two layers.
    """
    assert satisfies(1.0)
    assert satisfies(0.9999997)
    assert not satisfies(0.99)


def test_one_failure_fails_the_whole_assembly():
    """Conjunctive, with no averaging.

    A joint whose bearing outlasts the machine and whose bolt separates on the
    first cycle is not a three-quarters-good joint.
    """
    verdict = AssemblyVerdict()
    for i in range(9):
        verdict.add(CheckResult(f"c{i}", "mode", CheckStatus.PASSED, "m", 5.0))
    assert verdict.status is AssemblyStatus.PASSED
    verdict.add(CheckResult("bolt", "separation", CheckStatus.FAILED,
                            "bolted_joint", 0.8))
    assert verdict.status is AssemblyStatus.FAILED
    assert not verdict.passes
    assert verdict.governing().failure_mode == "separation"


def test_an_unassessed_mode_is_not_a_pass():
    """The single most dangerous thing an integration layer can do.

    Treating an absent check as a satisfied one converts ignorance into
    confidence exactly where a human stops looking.
    """
    verdict = AssemblyVerdict()
    verdict.add(CheckResult("link", "fatigue", CheckStatus.PASSED,
                            "fatigue_sn", 3.0))
    verdict.add(CheckResult("link", "creep", CheckStatus.NOT_ASSESSED,
                            detail="no registered method"))
    assert verdict.status is AssemblyStatus.PASSED_WITH_GAPS
    assert not verdict.passes


def test_a_not_applicable_mode_does_not_count_as_a_gap():
    """A tension member cannot buckle, and that is not a hole in the coverage."""
    verdict = AssemblyVerdict()
    verdict.add(CheckResult("link", "fatigue", CheckStatus.PASSED,
                            "fatigue_sn", 3.0))
    verdict.add(CheckResult("link", "buckling", CheckStatus.NOT_APPLICABLE,
                            "buckling_euler",
                            detail="not in compression"))
    assert verdict.status is AssemblyStatus.PASSED
    assert verdict.passes


def test_the_governing_check_is_stable_across_ties():
    """Ties break on names, not on evaluation order."""
    def build(order):
        verdict = AssemblyVerdict()
        for component in order:
            verdict.add(CheckResult(component, "mode", CheckStatus.PASSED,
                                    "m", 2.0))
        return verdict.governing().component

    assert build(["b", "a", "c"]) == build(["c", "b", "a"]) == "a"


# --- routing and coverage ----------------------------------------------------

def test_the_registry_routes_the_applicable_methods():
    routed = routed_methods(JointSpec())
    analysis = routed["analysis"]
    assert {"fatigue_sn", "shaft_combined", "bolted_joint", "gear_tooth",
            "motor_thermal"} <= set(analysis)
    assert "bearing_l10" in routed["selection"]


def test_the_beam_model_is_excluded_at_this_slenderness():
    """The Phase 7 lesson, still enforced inside the capstone.

    The link is 0.35 m over a 0.05 m envelope, a slenderness of 7, and
    Euler-Bernoulli is measured 2.5% low there. The registry excludes it and
    the capstone uses the Timoshenko path instead.
    """
    analysis = routed_methods(JointSpec())["analysis"]
    assert "beam_eb" not in analysis
    assert "beam_timoshenko" in analysis


def test_inapplicable_methods_are_excluded_rather_than_run(baseline):
    """Buckling is reported as not applicable, not silently skipped."""
    buckling = [r for r in baseline.verdict.results
                if r.failure_mode == "buckling"]
    assert len(buckling) == 1
    assert buckling[0].status is CheckStatus.NOT_APPLICABLE
    assert "compression" in buckling[0].detail


def test_the_known_gaps_are_all_listed(baseline):
    """Every mode this project can name and cannot evaluate appears."""
    listed = {f"{r.component}/{r.failure_mode}"
              for r in baseline.verdict.unassessed()}
    for component, modes in KNOWN_UNIMPLEMENTED_MODES.items():
        for mode in modes:
            assert f"{component}/{mode}" in listed


def test_gearbox_internals_are_unassessed_rather_than_invented(baseline):
    """A verdict about a mesh nobody specified would be a fabricated result.

    The catalogue gives a ratio, a torque rating and a mass. It does not give
    tooth geometry, and an earlier version invented a representative mesh and
    reported the gearbox FAILED on the strength of it.
    """
    gearbox = [r for r in baseline.verdict.results
               if r.component == "gearbox" and "tooth" in r.failure_mode]
    assert gearbox
    assert all(r.status is CheckStatus.NOT_ASSESSED for r in gearbox)
    assert any("not" in r.detail and "geometry" in r.detail for r in gearbox)


def test_a_stated_mesh_is_actually_checked():
    """The method exists and runs the moment the geometry is supplied."""
    result = design_joint(JointSpec(gear_mesh=(0.002, 20, 60, 0.014)))
    gearbox = [r for r in result.verdict.results
               if r.component == "gearbox" and r.is_verdict]
    assert {r.failure_mode for r in gearbox} == {"tooth_bending",
                                                 "tooth_pitting"}
    assert all(r.method == "gear_tooth" for r in gearbox)


# --- the baseline design -----------------------------------------------------

def test_the_capstone_produces_a_fully_checked_assembly(baseline):
    assert baseline.verdict.status is AssemblyStatus.PASSED_WITH_GAPS
    assert not baseline.verdict.failures()
    assert len(baseline.verdict.verdicts()) >= 12
    assert {"link", "drivetrain", "shaft", "bearing", "mount"} <= set(
        baseline.verdict.components())
    assert baseline.total_mass_kg > 0.0
    assert baseline.selected_bearing is not None


def test_every_verdict_carries_its_own_optimistic_assumption(baseline):
    """The review needs the assumption belonging to whichever check governs."""
    for result in baseline.verdict.verdicts():
        assert result.optimistic_assumption, (
            f"{result.component}/{result.failure_mode} states no assumption")


def test_the_review_names_the_governing_constraint_and_its_assumption(baseline):
    report = review(baseline.verdict)
    governing = baseline.verdict.governing()
    assert report.governing is governing
    assert report.weakest_assumption == governing.optimistic_assumption
    assert report.unassessed
    assert any("SIMULATED" in line for line in report.recommendations)
    assert any("not assessed" in line for line in report.recommendations)


def test_the_run_is_deterministic():
    first, second = design_joint(JointSpec()), design_joint(JointSpec())
    assert first.verdict.summary() == second.verdict.summary()
    assert [r.status for r in first.verdict.results] == \
        [r.status for r in second.verdict.results]


# --- fault injection: one failing check must fail the assembly ---------------

@pytest.mark.parametrize("label,overrides,expected", [
    ("bearing life", dict(required_bearing_life_h=1.0e11), "bearing/l10_life"),
    ("motor thermal", dict(ambient_c=120.0), "drivetrain/winding_temperature"),
    ("bolt mount", dict(mount_bolt_size="M3", mount_lever_arm_m=0.006,
                        mount_bolt_count=1), "mount/separation"),
    ("shaft fatigue", dict(shaft_diameter_m=0.006), "shaft/fatigue"),
    ("drivetrain torque", dict(payload_kg=40.0),
     "drivetrain/torque_and_speed"),
    ("gear teeth", dict(gear_mesh=(0.0006, 14, 30, 0.004)),
     "gearbox/tooth_bending"),
])
def test_a_single_failing_check_rejects_the_assembly(label, overrides,
                                                     expected):
    """Each failure mode injected on its own, and the assembly must refuse.

    The expected check must be among the failures, and the assembly must be
    FAILED rather than passing with a gap or passing outright.
    """
    result = design_joint(JointSpec(**overrides))
    assert result.verdict.status is AssemblyStatus.FAILED, (
        f"{label}: assembly was not rejected")
    assert not result.passes
    assert expected in failed_modes(result), (
        f"{label}: expected {expected} to fail, got "
        f"{sorted(failed_modes(result))}")


def test_the_governing_check_is_the_worst_failure():
    """Whichever check is furthest under is the one reported as governing."""
    result = design_joint(JointSpec(shaft_diameter_m=0.006))
    governing = result.verdict.governing()
    assert governing.status is CheckStatus.FAILED
    assert governing.safety_factor == min(
        r.safety_factor for r in result.verdict.verdicts()
        if r.safety_factor is not None)


def test_a_failure_produces_a_fix_recommendation():
    result = design_joint(JointSpec(shaft_diameter_m=0.006))
    report = review(result.verdict)
    assert report.status is AssemblyStatus.FAILED
    assert report.failures
    assert any(line.startswith("FIX") for line in report.recommendations)


def test_a_thicker_shaft_relieves_the_shaft_failure():
    """The injection is a real physical effect, not a switch being flipped."""
    thin = design_joint(JointSpec(shaft_diameter_m=0.006))
    thick = design_joint(JointSpec(shaft_diameter_m=0.020))
    assert "shaft/fatigue" in failed_modes(thin)
    assert "shaft/fatigue" not in failed_modes(thick)


# --- the surrogate gate ------------------------------------------------------

from brain.semantic.evidence import EvidenceKind, EvidenceLevel  # noqa: E402
from integration import SurrogateVerdict  # noqa: E402


def test_a_surrogate_cannot_pass_a_check():
    """The number may be right. The layer still refuses it, because a model of
    the solver has no way of knowing when it is wrong."""
    with pytest.raises(SurrogateVerdict, match="may screen, not decide"):
        CheckResult("link", "yield", CheckStatus.PASSED, "surrogate_mlp", 3.0,
                    evidence_kind=EvidenceKind.SURROGATE)


def test_a_surrogate_cannot_fail_a_check_either():
    """Rejecting a design on a prediction is a verdict too, and a wrong one
    throws away a design the solver would have accepted."""
    with pytest.raises(SurrogateVerdict):
        CheckResult("link", "yield", CheckStatus.FAILED, "surrogate_mlp", 0.7,
                    evidence_kind=EvidenceKind.SURROGATE)


def test_a_solver_check_still_passes_by_default():
    """The default evidence kind is a simulation, so nothing registered today
    changes behaviour."""
    r = CheckResult("link", "yield", CheckStatus.PASSED, "beam_theory", 3.0)
    assert r.evidence_kind is EvidenceKind.SIMULATION
    assert r.evidence_level is EvidenceLevel.SIMULATED
    assert r.is_verdict


def test_a_screened_check_is_a_gap_not_a_verdict():
    verdict = AssemblyVerdict()
    verdict.add(CheckResult("link", "fatigue", CheckStatus.PASSED,
                            "fatigue_sn", 3.0))
    screened = CheckResult("link", "yield", CheckStatus.SCREENED,
                           "surrogate_mlp",
                           detail="surrogate predicts 2.1; run the solver",
                           evidence_kind=EvidenceKind.SURROGATE)
    verdict.add(screened)
    assert not screened.is_verdict
    assert screened.evidence_level is EvidenceLevel.SURROGATE
    assert verdict.status is AssemblyStatus.PASSED_WITH_GAPS
    assert not verdict.passes
    assert verdict.screened() == [screened]
    assert verdict.gaps() == [screened]
    assert verdict.unassessed() == []
    # a screened mode never governs, whatever it predicted
    assert verdict.governing().failure_mode == "fatigue"


def test_a_screened_check_may_not_carry_a_safety_factor():
    """A predicted factor in the solved slot would be read back as solved."""
    with pytest.raises(ValueError, match="predicted factor belongs in detail"):
        CheckResult("link", "yield", CheckStatus.SCREENED, "surrogate_mlp",
                    2.1, detail="predicted",
                    evidence_kind=EvidenceKind.SURROGATE)


def test_a_screened_check_must_name_its_model_and_say_what_it_predicted():
    with pytest.raises(ValueError, match="name the model"):
        CheckResult("link", "yield", CheckStatus.SCREENED,
                    detail="predicted 2.1", evidence_kind=EvidenceKind.SURROGATE)
    with pytest.raises(ValueError, match="says nothing about what"):
        CheckResult("link", "yield", CheckStatus.SCREENED, "surrogate_mlp",
                    evidence_kind=EvidenceKind.SURROGATE)


def test_the_review_lists_screened_modes_and_says_to_run_the_solver():
    verdict = AssemblyVerdict()
    verdict.add(CheckResult("link", "fatigue", CheckStatus.PASSED,
                            "fatigue_sn", 3.0))
    verdict.add(CheckResult("bolt", "separation", CheckStatus.SCREENED,
                            "surrogate_mlp", detail="predicted 1.4",
                            evidence_kind=EvidenceKind.SURROGATE))
    r = review(verdict)
    assert r.status is AssemblyStatus.PASSED_WITH_GAPS
    assert r.screened == ("bolt/separation",)
    assert r.unassessed == ()
    assert any("only screened by a surrogate" in line
               for line in r.recommendations)
