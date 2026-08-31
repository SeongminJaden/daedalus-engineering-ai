"""Multi-objective optimisation: dominance, NSGA-II and the material front.

Two tests carry the phase. `test_fast_sort_front_zero_matches_brute_force`
checks the sort against an independent O(N^2) implementation, which is how the
inverted domination count in the first version was caught: it produced a
plausible front that was nearly the WORST layer. `test_zdt1_front_matches_the_analytical_solution`
checks the whole optimiser against a problem whose true front is known in
closed form.
"""

import numpy as np
import pytest

from core.registry import ProblemContext, build_default_registry
from optimization.multi_objective.nsga2 import (Nsga2Result, Sense,
                                                constrained_dominates,
                                                crowding_distance, dominates,
                                                fast_non_dominated_sort, nsga2,
                                                to_minimisation)
from optimization.multi_objective.pareto import non_dominated_mask


def zdt1(design: np.ndarray):
    """f1 = x1, f2 = g (1 - sqrt(f1/g)), g = 1 + 9 mean(x[1:]).

    Its true front is f2 = 1 - sqrt(f1) for x1 in [0, 1] with every other
    variable zero, which is what makes it a check rather than a demonstration.
    """
    design = np.atleast_2d(design)
    f1 = design[:, 0]
    g = 1.0 + 9.0 * design[:, 1:].sum(axis=1) / (design.shape[1] - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.column_stack([f1, f2]), np.zeros(design.shape[0])


def brute_force_front(objectives: np.ndarray) -> set[int]:
    """Every pair compared directly. Slow, obvious, and independent."""
    n = objectives.shape[0]
    keep = set()
    for i in range(n):
        if not any(dominates(objectives[j], objectives[i])
                   for j in range(n) if j != i):
            keep.add(i)
    return keep


# --- dominance ---------------------------------------------------------------

def test_dominance_is_no_worse_everywhere_and_better_somewhere():
    assert dominates(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert dominates(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert not dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    # Equal rows do not dominate one another, which keeps duplicates alive.
    assert not dominates(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_maximised_objectives_must_be_converted_explicitly():
    """Dominance assumes minimisation; a maximised column inverts the answer.

    Left unconverted, the sort would return the LOWEST natural frequency
    designs and call them optimal. The conversion is a named step so it cannot
    be skipped silently, and an unstated sense is refused rather than assumed.
    """
    objectives = np.array([[1.0, 10.0], [2.0, 20.0]])
    minimised = to_minimisation(objectives, [Sense.MIN, Sense.MAX])
    assert minimised[0, 1] == -10.0 and minimised[1, 1] == -20.0
    # Row 1 is better on the maximised column, so neither dominates.
    assert not dominates(minimised[0], minimised[1])
    assert not dominates(minimised[1], minimised[0])
    with pytest.raises(ValueError, match="senses"):
        to_minimisation(objectives, [Sense.MIN])


def test_fast_sort_front_zero_matches_brute_force():
    """The gate. An independent all-pairs check on random and clustered data.

    The first implementation counted how many solutions each row DOMINATED
    instead of how many dominated IT, which inverts the sort and returns
    nearly the worst layer as the front. It looked entirely plausible and only
    this comparison caught it.
    """
    rng = np.random.default_rng(0)
    for shape in ((200, 2), (150, 3), (80, 5)):
        points = rng.random(shape)
        assert set(fast_non_dominated_sort(points)[0].tolist()) == \
            brute_force_front(points)
    # And with heavy duplication, where dominance ties are common.
    duplicated = np.repeat(rng.random((20, 2)), 5, axis=0)
    assert set(fast_non_dominated_sort(duplicated)[0].tolist()) == \
        brute_force_front(duplicated)


def test_fast_sort_agrees_with_the_existing_mask():
    rng = np.random.default_rng(3)
    points = rng.random((120, 3))
    assert set(fast_non_dominated_sort(points)[0].tolist()) == \
        set(np.flatnonzero(non_dominated_mask(points)).tolist())


def test_every_row_lands_in_exactly_one_front():
    rng = np.random.default_rng(4)
    points = rng.random((100, 3))
    fronts = fast_non_dominated_sort(points)
    assigned = np.concatenate(fronts)
    assert sorted(assigned.tolist()) == list(range(100))


def test_later_fronts_are_dominated_by_earlier_ones():
    """The defining property of the ranking."""
    rng = np.random.default_rng(5)
    points = rng.random((80, 2))
    fronts = fast_non_dominated_sort(points)
    for order in range(1, len(fronts)):
        for i in fronts[order]:
            assert any(dominates(points[j], points[i])
                       for j in fronts[order - 1])


def test_an_empty_population_sorts_to_nothing():
    assert fast_non_dominated_sort(np.zeros((0, 2))) == []


# --- crowding ----------------------------------------------------------------

def test_boundary_solutions_are_always_kept():
    """Otherwise the front collapses inward and loses its own extremes."""
    front = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    distance = crowding_distance(front)
    assert np.isinf(distance[0]) and np.isinf(distance[2])
    assert np.isfinite(distance[1])


def test_a_solution_between_close_neighbours_scores_as_crowded():
    """The measure is the gap to the IMMEDIATE neighbours on each axis.

    So a point needs close neighbours on BOTH sides to count as crowded. One
    at the edge of a cluster has a near neighbour on one side and a distant one
    on the other, and is correctly not crowded, which is the behaviour that
    keeps a front from collapsing into its densest region.
    """
    front = np.array([[0.0, 1.0], [0.10, 0.90], [0.11, 0.89], [0.12, 0.88],
                      [1.0, 0.0]])
    distance = crowding_distance(front)
    interior = distance[np.isfinite(distance)]
    # The middle of the cluster is the most crowded point on the front.
    assert distance[2] == interior.min()
    # And the one at the cluster edge, with open space to its right, is not.
    assert distance[3] > distance[2]


def test_a_degenerate_objective_contributes_nothing():
    """A column with no range cannot separate anything, and must not divide
    by zero trying."""
    front = np.array([[0.0, 5.0], [0.5, 5.0], [1.0, 5.0]])
    distance = crowding_distance(front)
    assert np.isfinite(distance[1])
    assert not np.isnan(distance).any()


# --- constraints -------------------------------------------------------------

def test_feasible_beats_infeasible_whatever_the_objectives():
    good = np.array([10.0, 10.0])
    bad = np.array([0.0, 0.0])
    assert constrained_dominates(good, 0.0, bad, 1.0)
    assert not constrained_dominates(bad, 1.0, good, 0.0)


def test_between_two_infeasible_designs_the_smaller_violation_wins():
    a = np.array([0.0, 0.0])
    assert constrained_dominates(a, 0.5, a, 2.0)
    assert not constrained_dominates(a, 2.0, a, 0.5)


# --- the optimiser against a known answer ------------------------------------

@pytest.fixture(scope="module")
def zdt1_result():
    n = 10
    return nsga2(zdt1, np.zeros(n), np.ones(n), [Sense.MIN, Sense.MIN],
                 population=64, generations=200, seed=0)


def test_zdt1_front_matches_the_analytical_solution(zdt1_result):
    """The true front is f2 = 1 - sqrt(f1). Measured, not asserted."""
    front = zdt1_result.front_objectives()
    assert front.shape[0] > 30
    error = np.abs(front[:, 1] - (1.0 - np.sqrt(front[:, 0])))
    assert error.mean() < 0.01
    assert error.max() < 0.05


def test_the_front_spreads_across_the_whole_trade_off(zdt1_result):
    """Diversity. A converged but bunched front answers only one question."""
    front = zdt1_result.front_objectives()
    f1 = np.sort(front[:, 0])
    assert f1.min() < 0.05 and f1.max() > 0.95
    # No single gap swallows a quarter of the range.
    assert np.diff(f1).max() < 0.25 * (f1.max() - f1.min())


def test_no_member_of_the_returned_front_dominates_another(zdt1_result):
    """The definition of a front, checked on the actual output."""
    front = zdt1_result.front_objectives()
    for i in range(front.shape[0]):
        for j in range(front.shape[0]):
            if i != j:
                assert not dominates(front[i], front[j])


def test_solutions_off_the_front_are_dominated_by_it(zdt1_result):
    """The other half of the definition."""
    off = zdt1_result.objectives[~zdt1_result.front_mask]
    front = zdt1_result.front_objectives()
    for row in off:
        assert any(dominates(candidate, row) for candidate in front)


def test_a_seed_reproduces_a_run_exactly():
    n = 6
    def run(seed):
        return nsga2(zdt1, np.zeros(n), np.ones(n), [Sense.MIN, Sense.MIN],
                     population=16, generations=20, seed=seed)
    assert np.array_equal(run(1).objectives, run(1).objectives)
    assert not np.array_equal(run(1).objectives, run(2).objectives)


def test_preference_is_applied_after_the_front_is_known(zdt1_result):
    """The point of a front: see the trade-off, then choose.

    Weighting one objective heavily must select an extreme of the front, and
    the two extremes must be different designs.
    """
    mostly_f1 = zdt1_result.choose(np.array([1.0, 0.0]))
    mostly_f2 = zdt1_result.choose(np.array([0.0, 1.0]))
    assert mostly_f1 != mostly_f2
    assert (zdt1_result.objectives[mostly_f1, 0]
            < zdt1_result.objectives[mostly_f2, 0])
    assert (zdt1_result.objectives[mostly_f2, 1]
            < zdt1_result.objectives[mostly_f1, 1])


def test_bad_configuration_is_refused():
    with pytest.raises(ValueError, match="upper bound"):
        nsga2(zdt1, np.ones(3), np.zeros(3), [Sense.MIN, Sense.MIN])
    with pytest.raises(ValueError, match="population"):
        nsga2(zdt1, np.zeros(3), np.ones(3), [Sense.MIN, Sense.MIN],
              population=3)


# --- registry ----------------------------------------------------------------

def test_nsga2_is_gated_on_having_a_trade_off_at_all():
    registry = build_default_registry()
    single = ProblemContext(geometry="prismatic_beam",
                            representations=("prismatic_beam",),
                            n_objectives=1, has_discrete_variables=False)
    assert "nsga2" not in registry.query(single).names()
    assert "trade-off" in registry.query(single).reason("nsga2")[0]


def test_nsga2_is_gated_on_continuous_variables():
    """The operators interpolate; nothing lies between two materials."""
    registry = build_default_registry()
    discrete = ProblemContext(geometry="prismatic_beam",
                              representations=("prismatic_beam",),
                              n_objectives=4, has_discrete_variables=True)
    assert "nsga2" not in registry.query(discrete).names()
    assert "between two materials" in registry.query(discrete).reason("nsga2")[0]


# --- the engineering front ---------------------------------------------------

def test_every_material_carries_a_price():
    from core.materials import get_material

    for material_id in ("al_7075_t6", "al_6061_t6", "steel_s45c",
                        "steel_scm440", "ti_6al_4v", "cfrp_ud", "abs"):
        assert get_material(material_id).price_per_kg_usd is not None


def test_the_expected_price_ordering():
    """Titanium and CFRP are the expensive ones; plain steel is the cheap one."""
    from core.materials import get_material

    price = {m: get_material(m).price_per_kg_usd
             for m in ("steel_s45c", "steel_scm440", "al_6061_t6",
                       "al_7075_t6", "ss_316", "ti_6al_4v", "cfrp_ud")}
    # Plain carbon steel is the cheapest thing here and CFRP the dearest.
    assert price["steel_s45c"] == min(price.values())
    assert price["cfrp_ud"] == max(price.values())
    assert price["steel_s45c"] < price["steel_scm440"] < price["al_6061_t6"]
    assert price["al_6061_t6"] < price["al_7075_t6"] < price["ti_6al_4v"]
    # Aerospace aluminium against stainless is NOT an obvious ordering, and at
    # these figures 7075 is the dearer of the two. Asserting the intuitive
    # ordering here failed, and the data was right.
    assert price["al_7075_t6"] > price["ss_316"]


def test_material_cost_is_mass_times_price():
    from optimization.multi_objective.objectives import material_cost_usd

    assert material_cost_usd(np.array([2.0, 0.5]), 6.0) == pytest.approx(
        np.array([12.0, 3.0]))
    with pytest.raises(ValueError, match="positive"):
        material_cost_usd(np.array([1.0]), 0.0)


@pytest.fixture(scope="module")
def material_fronts():
    from projects.robotic_link.problem import build_mvp_problem
    from optimization.multi_objective.objectives import sweep_materials

    return sweep_materials(build_mvp_problem(),
                           ["steel_s45c", "al_6061_t6", "al_7075_t6",
                            "ti_6al_4v"],
                           population=24, generations=20, seed=0)


def test_the_merged_front_is_non_dominated_across_materials(material_fronts):
    from optimization.multi_objective.objectives import merged_front

    _, objectives, labels = merged_front(material_fronts)
    assert objectives.shape[0] > 0
    assert len(labels) == objectives.shape[0]
    for i in range(objectives.shape[0]):
        for j in range(objectives.shape[0]):
            if i != j:
                assert not dominates(objectives[i], objectives[j])


def test_no_single_material_wins_every_objective(material_fronts):
    """Which is the entire reason to compute a front rather than a winner.

    Steel is the cheapest and the stiffest; aluminium is the lightest. A single
    weighted objective would have picked one and hidden the other.
    """
    from optimization.multi_objective.objectives import (OBJECTIVE_NAMES,
                                                         merged_front)

    _, objectives, labels = merged_front(material_fronts)
    winners = {name: labels[int(np.argmin(objectives[:, column]))]
               for column, name in enumerate(OBJECTIVE_NAMES)}
    assert len(set(winners.values())) > 1, (
        f"one material won every objective: {winners}")
    assert "steel" in winners["material_cost_usd"]
    assert "al_" in winners["mass_kg"]


def test_the_material_sweep_is_deterministic():
    from projects.robotic_link.problem import build_mvp_problem
    from optimization.multi_objective.objectives import sweep_materials

    def run():
        return sweep_materials(build_mvp_problem(), ["al_6061_t6"],
                               population=16, generations=8, seed=7)[0]

    assert np.array_equal(run().objectives, run().objectives)


def test_a_material_without_a_price_cannot_carry_a_cost_objective():
    from projects.robotic_link.problem import build_mvp_problem
    from optimization.multi_objective.objectives import sweep_materials

    problem = build_mvp_problem()
    with pytest.raises(ValueError, match="no price"):
        # Patch the lookup to a material stripped of its price.
        from core.materials import get_material
        import optimization.multi_objective.objectives as module

        stripped = get_material("al_6061_t6").model_copy(
            update={"price_per_kg_usd": None})
        original = module.get_material
        module.get_material = lambda mid: (stripped if mid == "al_6061_t6"
                                           else original(mid))
        try:
            sweep_materials(problem, ["al_6061_t6"], population=8,
                            generations=2)
        finally:
            module.get_material = original
