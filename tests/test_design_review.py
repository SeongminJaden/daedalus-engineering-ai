"""Minimum sizing, multi-design review and one-way coupled analysis.

Three tests carry the phase. `test_the_monotonicity_assumption_is_checked_not_assumed`
pins that bisection rests on a physical assumption which does not always hold.
`test_the_ranking_criteria_disagree_and_that_is_the_output` pins that choosing
between designs is a judgement rather than a computation. `test_heating_can_relieve_rather_than_worsen`
pins that a coupled result has a direction, not just a magnitude.
"""

import math

import pytest

from core.materials import get_material
from integration import (AssemblyStatus, AssemblyVerdict, CheckResult,
                         CheckStatus, DesignEntry, MultiDesignReview, RankBy,
                         is_monotonic_increasing, minimum_dimension,
                         thermal_structural)
from physics.thermal import ThermalPath


def shaft_safety_factor(diameter_m: float, moment_nm: float = 50.0,
                        yield_pa: float = 655e6) -> float:
    """A bending safety factor that grows as the cube of diameter."""
    section_modulus = math.pi * diameter_m ** 3 / 32.0
    return yield_pa * section_modulus / moment_nm


def cantilever_frequency_factor(wall_m: float, side: float = 0.040,
                                length: float = 0.5, modulus: float = 71.7e9,
                                density: float = 2810.0,
                                target_hz: float = 50.0) -> float:
    """First natural frequency over a target, for a fixed outer envelope."""
    inner = max(side - 2.0 * wall_m, 0.0)
    second_moment = (side * side ** 3 - inner * inner ** 3) / 12.0
    area = side * side - inner * inner
    omega = 3.516 / length ** 2 * math.sqrt(modulus * second_moment
                                            / (density * area))
    return omega / (2.0 * math.pi) / target_hz


# --- minimum sizing ----------------------------------------------------------

def test_the_minimum_diameter_matches_the_closed_form():
    """Bisection against algebra: d = (n M 32 / (Sy pi))^(1/3)."""
    result = minimum_dimension(shaft_safety_factor, 0.005, 0.100, target=2.0)
    expected = (2.0 * 50.0 * 32.0 / (655e6 * math.pi)) ** (1.0 / 3.0)
    assert result.dimension == pytest.approx(expected, rel=1e-4)
    assert result.converged
    assert result.monotonic


def test_a_minimum_sits_on_its_constraint_and_therefore_has_no_margin():
    """Which is what minimum means, and why it is not by itself a design."""
    result = minimum_dimension(shaft_safety_factor, 0.005, 0.100, target=2.0)
    assert result.safety_factor == pytest.approx(2.0, rel=1e-3)
    assert result.on_the_boundary


def test_the_target_is_not_one_by_default():
    """Sizing to exactly 1.0 leaves nothing for modelling error or variation,
    and offering it as the default would encourage it."""
    default_target = minimum_dimension(shaft_safety_factor, 0.005, 0.100)
    to_unity = minimum_dimension(shaft_safety_factor, 0.005, 0.100, target=1.0)
    assert default_target.target > 1.0
    assert default_target.dimension > to_unity.dimension


def test_the_monotonicity_assumption_is_checked_not_assumed():
    """Bisection rests on physics, and the physics does not always oblige.

    Growing the wall of a fixed outer envelope raises stiffness AND mass, and
    for a cantilever's natural frequency the mass wins: the factor falls from
    3.60 to 2.61 between 1 mm and 20 mm of wall. Bisection would still find a
    root there and it would not be the minimum, so the assumption is sampled
    before the search runs.
    """
    assert is_monotonic_increasing(shaft_safety_factor, 0.005, 0.100)
    assert not is_monotonic_increasing(cantilever_frequency_factor, 0.001,
                                       0.020)
    # And the flag travels with the result rather than being silently dropped.
    flagged = minimum_dimension(cantilever_frequency_factor, 0.001, 0.020,
                                target=3.0)
    assert not flagged.monotonic


def test_section_modulus_stays_monotonic_even_as_the_cavity_closes():
    """The counter-example this module first reached for, and it was wrong.

    Growing the wall flattens the section modulus as the cavity closes, but it
    never decreases, so it remains a valid case for bisection.
    """
    def modulus_factor(wall_m: float, side: float = 0.040) -> float:
        inner = max(side - 2.0 * wall_m, 0.0)
        second_moment = (side * side ** 3 - inner * inner ** 3) / 12.0
        return 655e6 * (second_moment / (side / 2.0)) / 50.0

    assert is_monotonic_increasing(modulus_factor, 0.002, 0.0195)


def test_an_unreachable_target_is_reported_rather_than_returned_quietly():
    """Returning the bracket top silently would look like a passing design."""
    result = minimum_dimension(shaft_safety_factor, 0.001, 0.003, target=100.0)
    assert not result.converged
    assert result.dimension == pytest.approx(0.003)
    assert result.safety_factor < 100.0


def test_an_already_sufficient_lower_bound_needs_no_search():
    result = minimum_dimension(shaft_safety_factor, 0.050, 0.100, target=2.0)
    assert result.converged
    assert result.iterations == 0
    assert result.dimension == pytest.approx(0.050)


def test_a_reversed_bracket_is_refused():
    with pytest.raises(ValueError, match="upper bracket"):
        minimum_dimension(shaft_safety_factor, 0.100, 0.005)


# --- multi-design review -----------------------------------------------------

def build_entry(name: str, factor: float, gaps: int, mass: float, cost: float,
                failing: bool = False) -> DesignEntry:
    verdict = AssemblyVerdict()
    verdict.add(CheckResult(
        "link", "stress",
        CheckStatus.FAILED if failing else CheckStatus.PASSED, "m", factor,
        optimistic_assumption="stated"))
    verdict.add(CheckResult("link", "fatigue", CheckStatus.PASSED, "f",
                            factor * 1.5, optimistic_assumption="stated"))
    for index in range(gaps):
        verdict.add(CheckResult("link", f"gap{index}",
                                CheckStatus.NOT_ASSESSED, detail="none"))
    return DesignEntry(name, verdict, mass, cost)


@pytest.fixture
def review():
    return MultiDesignReview([
        build_entry("steel", 3.20, 2, 0.51, 0.51),
        build_entry("aluminium", 1.35, 2, 0.31, 1.83),
        build_entry("titanium", 2.10, 5, 0.35, 14.08),
        build_entry("broken", 0.60, 1, 0.28, 0.90, failing=True),
    ])


def test_a_failing_design_is_rejected_not_ranked(review):
    """Sorting it by margin would imply it is merely worse rather than
    inadmissible."""
    assert [e.name for e in review.rejected] == ["broken"]
    assert "broken" not in [e.name for e in review.ranked()]
    assert all(e.verdict.status is not AssemblyStatus.FAILED
               for e in review.admissible)


def test_the_ranking_criteria_disagree_and_that_is_the_output(review):
    """Choosing between designs is a judgement, not a computation.

    Margin and cost pick the steel design; mass and gap count pick the
    aluminium one. Reporting a single winner would hide that the answer depends
    on what is being traded.
    """
    assert not review.criteria_agree()
    picks = review.disagreement()
    assert picks[RankBy.GOVERNING_MARGIN] == "steel"
    assert picks[RankBy.MASS] == "aluminium"
    assert picks[RankBy.COST] == "steel"
    assert len(set(picks.values())) > 1


def test_each_criterion_orders_by_what_it_says_it_does(review):
    by_mass = review.ranked(RankBy.MASS)
    assert [e.mass_kg for e in by_mass] == sorted(e.mass_kg for e in by_mass)
    by_margin = review.ranked(RankBy.GOVERNING_MARGIN)
    assert by_margin[0].governing_margin >= by_margin[-1].governing_margin
    by_gaps = review.ranked(RankBy.FEWEST_GAPS)
    assert by_gaps[0].gap_count <= by_gaps[-1].gap_count


def test_ranking_is_reproducible_across_ties():
    tied = MultiDesignReview([build_entry("b", 2.0, 1, 1.0, 1.0),
                              build_entry("a", 2.0, 1, 1.0, 1.0)])
    assert [e.name for e in tied.ranked()] == ["a", "b"]


def test_designs_are_only_comparable_on_checks_they_both_ran(review):
    """A mode one design assessed and another did not is not a shared axis."""
    shared = review.comparable_checks()
    assert shared == {"link/stress", "link/fatigue"}
    # The gap count travels with each design so the difference is visible.
    counts = {e.name: e.gap_count for e in review.admissible}
    assert counts["titanium"] > counts["steel"]


def test_the_governing_check_is_named_per_design(review):
    for entry in review.admissible:
        assert "/" in entry.governing_check


# --- coupled analysis --------------------------------------------------------

@pytest.fixture(scope="module")
def aluminium():
    return get_material("al_7075_t6")


def heat_path() -> ThermalPath:
    return ThermalPath().add("part to bracket", 0.9).add("bracket to air", 1.6)


def test_the_coupled_result_carries_the_thermal_solution(aluminium):
    result = thermal_structural(heat_path(), 40.0, aluminium, ambient_c=25.0,
                                reference_c=25.0, mechanical_stress_pa=200e6)
    assert result.thermal_resistance_k_w == pytest.approx(2.5)
    assert result.temperature_rise_k == pytest.approx(100.0)
    assert result.operating_c == pytest.approx(125.0)
    assert result.dominant_resistance == "bracket to air"


def test_heating_can_relieve_rather_than_worsen(aluminium):
    """A coupled result has a DIRECTION, not just a magnitude.

    A restrained part that is heated goes into compression, which relieves a
    tensile mechanical stress rather than adding to it. The safety factor
    therefore PEAKS where the thermal contribution cancels the mechanical one
    and falls again beyond it, which a magnitude-only model could not express.
    """
    path = heat_path()
    mild = thermal_structural(path, 10.0, aluminium, 25.0, 25.0, 200e6)
    cancelling = thermal_structural(path, 40.0, aluminium, 25.0, 25.0, 200e6)
    excessive = thermal_structural(path, 80.0, aluminium, 25.0, 25.0, 200e6)

    assert mild.thermal_stress_pa < 0.0            # compression
    assert not mild.thermal_worsens_it
    assert cancelling.safety_factor > mild.safety_factor
    assert excessive.safety_factor < cancelling.safety_factor
    assert excessive.combined_stress_pa < 0.0      # thermal has overshot


def test_the_stress_comes_from_the_difference_to_the_reference_temperature():
    """Not from the operating temperature, and using that would invent stress.

    A part assembled hot and running hot has no thermal stress at all.
    """
    material = get_material("al_7075_t6")
    path = heat_path()
    assembled_cold = thermal_structural(path, 40.0, material, 25.0, 25.0, 0.0)
    assembled_hot = thermal_structural(path, 40.0, material, 25.0, 125.0, 0.0)
    assert abs(assembled_cold.thermal_stress_pa) > 1e6
    assert assembled_hot.thermal_stress_pa == pytest.approx(0.0, abs=1.0)


def test_more_power_means_more_temperature_and_more_thermal_stress(aluminium):
    path = heat_path()
    low = thermal_structural(path, 10.0, aluminium, 25.0, 25.0, 0.0)
    high = thermal_structural(path, 40.0, aluminium, 25.0, 25.0, 0.0)
    assert high.temperature_rise_k == pytest.approx(
        4.0 * low.temperature_rise_k)
    assert abs(high.thermal_stress_pa) == pytest.approx(
        4.0 * abs(low.thermal_stress_pa))


# ------------------------------------------------- dominance, added later
# These reuse build_entry above rather than defining a second way to make a
# design, so a change to the verdict shape cannot leave half the file behind.


def test_a_design_worse_on_every_axis_is_dominated():
    """Heavier, dearer and weaker at once. No criterion can prefer it, so it
    can be discarded without arguing about which criterion to use."""
    review = MultiDesignReview([
        build_entry("good", factor=3.0, gaps=0, mass=1.0, cost=10.0),
        build_entry("bad", factor=2.0, gaps=1, mass=2.0, cost=20.0)])

    assert [e.name for e in review.non_dominated()] == ["good"]
    assert [e.name for e in review.dominated()] == ["bad"]


def test_a_trade_off_survives_dominance():
    """Lighter but weaker against heavier but stronger is a real trade.

    Dominance must NOT resolve it: that is a judgement, and a filter that
    returned one of them would be deciding by arithmetic.
    """
    review = MultiDesignReview([
        build_entry("light", factor=2.0, gaps=0, mass=1.0, cost=10.0),
        build_entry("strong", factor=4.0, gaps=0, mass=3.0, cost=10.0)])

    assert len(review.non_dominated()) == 2
    assert review.dominated() == []


def test_the_strict_winner_on_an_axis_is_never_dominated():
    """A design uniquely best on some axis cannot be dominated.

    Anything dominating it would have to be no worse everywhere, which on that
    axis is impossible. Worth asserting because it is a property of dominance
    rather than of this particular data.
    """
    review = MultiDesignReview([
        build_entry("lightest", factor=1.6, gaps=0, mass=0.5, cost=99.0),
        build_entry("strongest", factor=9.0, gaps=0, mass=8.0, cost=99.0),
        build_entry("cheapest", factor=1.7, gaps=0, mass=7.9, cost=1.0)])

    surviving = {e.name for e in review.non_dominated()}
    assert {"lightest", "strongest", "cheapest"} <= surviving


def test_a_failing_design_is_not_considered_for_dominance():
    """It is inadmissible, not merely worse, so it is neither kept as
    non-dominated nor reported as dominated."""
    review = MultiDesignReview([
        build_entry("good", factor=3.0, gaps=0, mass=1.0, cost=10.0),
        build_entry("broken", factor=0.4, gaps=0, mass=0.1, cost=1.0,
                    failing=True)])

    assert [e.name for e in review.non_dominated()] == ["good"]
    assert "broken" not in [e.name for e in review.dominated()]
    assert "broken" in [e.name for e in review.rejected]


def test_more_margin_wins_when_nothing_else_separates_designs():
    """The shared dominance filter minimises every column, so the governing
    margin has to be negated on the way in.

    Passing it raw would select exactly the wrong designs, which is the
    mistake the filter's own docstring warns about. Here mass, cost and gaps
    are identical, so only the margin separates them and MORE must win.
    """
    review = MultiDesignReview([
        build_entry("weak", factor=1.6, gaps=0, mass=1.0, cost=5.0),
        build_entry("strong", factor=8.0, gaps=0, mass=1.0, cost=5.0)])

    assert [e.name for e in review.non_dominated()] == ["strong"]


def test_an_axis_carrying_no_information_decides_nothing():
    """Mass and cost default to zero. When a caller leaves them unset they are
    constant across the entries, and a constant axis must not decide."""
    review = MultiDesignReview([
        build_entry("a", factor=3.0, gaps=0, mass=0.0, cost=0.0),
        build_entry("b", factor=2.0, gaps=0, mass=0.0, cost=0.0)])

    assert [e.name for e in review.non_dominated()] == ["a"]


def test_the_capability_refuses_a_single_candidate():
    """A review of one design is a verdict, and that question is answered
    elsewhere. Selecting this method for it would add ranking machinery to a
    problem that has nothing to rank."""
    from core.registry.context import ProblemContext
    from nodes.roster import build_roster

    method = next(c.method for c in build_roster().all()
                  if c.name == "multi_design_review")
    reasons = " ".join(str(r) for r
                       in method.applicability(ProblemContext()).failed)
    assert "has_multiple_candidates" in reasons
    assert not list(method.applicability(
        ProblemContext(has_multiple_candidates=True)).failed)


def test_the_review_reports_its_own_numbers_from_the_verdicts():
    """It must not recompute anything.

    Its whole claim to trust is that the numbers are the verified checks'
    numbers. If it derived a safety factor itself there would be two sources
    for one quantity and they could disagree.
    """
    entry = build_entry("only", factor=2.75, gaps=0, mass=1.0, cost=1.0)
    assert entry.governing_margin == pytest.approx(2.75)
    assert entry.governing_margin == pytest.approx(
        entry.verdict.governing_safety_factor)
