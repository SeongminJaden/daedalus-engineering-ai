"""Phase 5 verification: the engineering Brain.

The load-bearing check is the evidence ladder. A store of accumulated
simulation results is dangerous exactly when it starts presenting agreement as
fact, so the tests below pin the one rule that prevents that:
**no amount of simulation reaches EXPERIMENTALLY_VALIDATED.**
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.loop import DesignLoop, LoopConfig  # noqa: E402
from brain import Brain  # noqa: E402
from brain.retrieval import FEATURE_NAMES, design_vector  # noqa: E402
from brain.semantic import (  # noqa: E402
    Counterexample, Evidence, EvidenceKind, EvidenceLevel, Knowledge,
    PromotionPolicy, compute_confidence, derive_level, independent_runs,
)
from brain.strategy import derive_stiffness_strategy  # noqa: E402
from optimization.constraints import build_optimization_problem  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402

FAST = dict(local_max_iter=30)


@pytest.fixture
def brain():
    with Brain(":memory:") as b:
        yield b


def sim(n: int, run_prefix: str = "r") -> list[Evidence]:
    """n simulation evidence items, each from its own run (independent)."""
    return [Evidence(EvidenceKind.SIMULATION, f"{run_prefix}-e{i}",
                     run_id=f"{run_prefix}{i}")
            for i in range(n)]


# =========================================================================== #
# 2. the evidence-level state machine
# =========================================================================== #
def test_no_evidence_is_unverified():
    assert derive_level([], []) is EvidenceLevel.UNVERIFIED


def test_single_simulation_is_simulated():
    assert derive_level(sim(1), []) is EvidenceLevel.SIMULATED


def test_many_episodes_from_one_run_stay_simulated():
    """Independence is by run. Five iterations of one search are five samples
    of one observation, not five observations."""
    same_run = [Evidence(EvidenceKind.SIMULATION, f"e{i}", run_id="r0")
                for i in range(20)]
    assert independent_runs(same_run) == 1
    assert derive_level(same_run, []) is EvidenceLevel.SIMULATED


def test_independent_runs_reach_repeated():
    assert derive_level(sim(3), []) is EvidenceLevel.REPEATED


def test_enough_independent_evidence_reaches_high_confidence():
    assert derive_level(sim(8), []) is EvidenceLevel.HIGH_CONFIDENCE


def test_simulation_alone_NEVER_reaches_experimentally_validated():
    """The safety rule. Simulation cannot become experimental validation, at
    any volume, from any number of runs, with any agreement."""
    for n in (1, 10, 100, 1000):
        level = derive_level(sim(n), [])
        assert level is not EvidenceLevel.EXPERIMENTALLY_VALIDATED, (
            f"{n} simulations wrongly reached {level}"
        )
        assert level.rank <= EvidenceLevel.HIGH_CONFIDENCE.rank


def test_test_suite_and_analytical_evidence_also_cannot_validate():
    """Not just simulation - only a physical test opens that gate."""
    evidence = (
        [Evidence(EvidenceKind.TEST_SUITE, f"t{i}") for i in range(20)]
        + [Evidence(EvidenceKind.ANALYTICAL, f"a{i}") for i in range(20)]
    )
    assert derive_level(evidence, []) is EvidenceLevel.HIGH_CONFIDENCE


def test_physical_test_evidence_reaches_experimentally_validated():
    evidence = sim(2) + [Evidence(EvidenceKind.PHYSICAL_TEST, "rig-001")]
    assert derive_level(evidence, []) is EvidenceLevel.EXPERIMENTALLY_VALIDATED


def test_unresolved_counterexample_caps_below_high_confidence():
    strong = sim(20)
    assert derive_level(strong, []) is EvidenceLevel.HIGH_CONFIDENCE
    capped = derive_level(strong, [Counterexample("c1", "contradicts")])
    assert capped is EvidenceLevel.REPEATED
    assert capped.rank < EvidenceLevel.HIGH_CONFIDENCE.rank


def test_unresolved_counterexample_blocks_experimental_validation():
    evidence = sim(5) + [Evidence(EvidenceKind.PHYSICAL_TEST, "rig-001")]
    level = derive_level(evidence, [Counterexample("c1", "part failed early")])
    assert level is not EvidenceLevel.EXPERIMENTALLY_VALIDATED


def test_resolving_a_counterexample_restores_promotion():
    strong = sim(20)
    open_c = [Counterexample("c1", "x", resolved=False)]
    closed_c = [Counterexample("c1", "x", resolved=True)]
    assert derive_level(strong, open_c) is EvidenceLevel.REPEATED
    assert derive_level(strong, closed_c) is EvidenceLevel.HIGH_CONFIDENCE


def test_promotion_policy_is_parameterized():
    strict = PromotionPolicy(repeat_independent_runs=10,
                             high_confidence_evidence=50,
                             high_confidence_runs=50)
    assert derive_level(sim(5), [], strict) is EvidenceLevel.SIMULATED


# =========================================================================== #
# 3. the confidence function
# =========================================================================== #
def test_confidence_is_bounded():
    for n in (0, 1, 5, 50, 5000):
        c = compute_confidence(sim(n), [])
        assert 0.0 <= c <= 1.0


def test_confidence_increases_monotonically_with_evidence():
    values = [compute_confidence(sim(n), []) for n in range(0, 40)]
    for earlier, later in zip(values, values[1:]):
        assert later >= earlier


def test_confidence_drops_when_a_counterexample_appears():
    evidence = sim(6)
    before = compute_confidence(evidence, [])
    after = compute_confidence(evidence, [Counterexample("c", "x")])
    assert after < before


def test_confidence_respects_the_level_ceiling():
    """One run, many episodes: SIMULATED, so confidence cannot exceed 0.60."""
    same_run = [Evidence(EvidenceKind.SIMULATION, f"e{i}", run_id="r0")
                for i in range(1000)]
    assert compute_confidence(same_run, []) <= 0.60


def test_resolved_counterexamples_do_not_penalize():
    evidence = sim(6)
    assert compute_confidence(evidence, [Counterexample("c", "x", resolved=True)]) \
        == compute_confidence(evidence, [])


# =========================================================================== #
# 1. storage round trips
# =========================================================================== #
def test_knowledge_round_trip(brain):
    k = Knowledge(statement="deflection binds", domain="link",
                  source="test", evidence=sim(4),
                  counterexamples=[Counterexample("c", "odd case")],
                  assumptions=["beam theory"])
    brain.semantic.store(k)
    loaded = brain.semantic.get(k.knowledge_id)
    assert loaded.statement == k.statement
    assert len(loaded.evidence) == 4
    assert len(loaded.counterexamples) == 1
    assert loaded.evidence_level is k.evidence_level
    assert loaded.confidence == pytest.approx(k.confidence)


def test_design_and_episode_round_trip(brain):
    genome = {"outer_width_m": 0.01, "outer_height_m": 0.08,
              "wall_thickness_m": 0.001, "material_id": "al_7075_t6"}
    metrics = {"mass_kg": 0.25, "tip_deflection_m": 1e-3}
    brain.episodic.record_run("run1", "mvp")
    design_id = brain.episodic.record_design(
        genome, metrics, feasible=True, active_constraints=["deflection"],
        run_id="run1")
    loaded = brain.episodic.get_design(design_id)
    assert loaded["genome"] == genome
    assert loaded["metrics"] == metrics
    assert loaded["feasible"] is True
    assert loaded["active_constraints"] == ["deflection"]


def test_repeated_generalization_consolidates_into_one_claim(brain):
    """The statement text embeds live counts, so identity must come from a
    stable claim key - otherwise each pass files a fresh, separately-weak row
    and the evidence never accumulates."""
    from brain.semantic import generalize_binding_constraint
    for runs in (1, 2, 3):
        generalize_binding_constraint(
            _synthetic_episodes(n_runs=runs, per_run=3, active="deflection"),
            brain.semantic)
    items = brain.semantic.by_domain("cantilever_link")
    assert len(items) == 1, [k.statement for k in items]
    assert independent_runs(items[0].evidence) == 3
    assert items[0].evidence_level is EvidenceLevel.REPEATED
    assert "9/9" in items[0].statement          # restated over the full corpus


def test_upsert_merges_evidence_instead_of_duplicating(brain):
    a = Knowledge(statement="same claim", domain="d", source="s", evidence=sim(2, "a"))
    brain.semantic.upsert_by_statement(a)
    b = Knowledge(statement="same claim", domain="d", source="s", evidence=sim(2, "b"))
    merged = brain.semantic.upsert_by_statement(b)
    assert len(brain.semantic.by_domain("d")) == 1
    assert len(merged.evidence) == 4


def test_run_round_trip(brain):
    brain.episodic.record_run("r1", "mvp", termination="converged",
                              iterations=5, best_mass_kg=0.25,
                              meta={"budget": {"evaluations": 10}})
    run = brain.episodic.get_run("r1")
    assert run["termination"] == "converged"
    assert run["iterations"] == 5
    assert run["meta"]["budget"]["evaluations"] == 10


# =========================================================================== #
# 4. retrieval
# =========================================================================== #
def _populate(brain, n=25, seed=0):
    rng = np.random.default_rng(seed)
    brain.episodic.record_run("run1", "mvp")
    ids = []
    for i in range(n):
        genome = {"outer_width_m": float(rng.uniform(0.01, 0.1)),
                  "outer_height_m": float(rng.uniform(0.01, 0.1)),
                  "wall_thickness_m": float(rng.uniform(0.001, 0.01)),
                  "material_id": "al_7075_t6"}
        metrics = {"mass_kg": float(rng.uniform(0.1, 3.0)),
                   "max_bending_stress_pa": float(rng.uniform(1e6, 1e8)),
                   "tip_deflection_m": float(rng.uniform(1e-5, 5e-3)),
                   "safety_factor": float(rng.uniform(1.0, 50.0)),
                   "first_natural_frequency_hz": float(rng.uniform(50, 500))}
        ids.append(brain.episodic.record_design(genome, metrics, feasible=True,
                                                run_id="run1"))
    return ids


def test_nearest_matches_naive_brute_force(brain):
    """The vectorized search must agree with an explicit per-row loop - a
    genuinely different code path, so this is not a restatement."""
    _populate(brain, n=40)
    query = design_vector(
        {"outer_width_m": 0.03, "outer_height_m": 0.06, "wall_thickness_m": 0.002},
        {"mass_kg": 1.0, "max_bending_stress_pa": 3e7, "tip_deflection_m": 1e-3,
         "safety_factor": 12.0, "first_natural_frequency_hz": 250.0})

    got = brain.similar_designs(k=5, feasible_only=True,
                                genome={"outer_width_m": 0.03,
                                        "outer_height_m": 0.06,
                                        "wall_thickness_m": 0.002},
                                metrics={"mass_kg": 1.0,
                                         "max_bending_stress_pa": 3e7,
                                         "tip_deflection_m": 1e-3,
                                         "safety_factor": 12.0,
                                         "first_natural_frequency_hz": 250.0})

    naive = []
    for d in brain.episodic.designs(feasible_only=True):
        v = design_vector(d["genome"], d["metrics"])
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, query)))
        naive.append((d["design_id"], dist))
    naive.sort(key=lambda t: (t[1], t[0]))

    assert [d["design_id"] for d, _ in got] == [i for i, _ in naive[:5]]
    for (_, got_dist), (_, want_dist) in zip(got, naive[:5]):
        assert got_dist == pytest.approx(want_dist, rel=1e-12)


def test_identical_query_retrieves_itself_first(brain):
    ids = _populate(brain, n=20)
    target = brain.episodic.get_design(ids[7])
    got = brain.similar_designs(genome=target["genome"],
                                metrics=target["metrics"], k=3)
    assert got[0][0]["design_id"] == ids[7]
    assert got[0][1] == pytest.approx(0.0, abs=1e-12)


def test_feature_vector_is_deterministic_and_bounded():
    genome = {"outer_width_m": 0.05, "outer_height_m": 0.08,
              "wall_thickness_m": 0.005}
    metrics = {"mass_kg": 1.686, "max_bending_stress_pa": 3.96e6,
               "tip_deflection_m": 1.15e-4, "safety_factor": 126.9,
               "first_natural_frequency_hz": 324.8}
    a = design_vector(genome, metrics)
    b = design_vector(genome, metrics)
    assert np.array_equal(a, b)
    assert a.shape == (len(FEATURE_NAMES),)
    assert np.all(a >= 0.0) and np.all(a <= 1.0)


def test_retrieval_on_empty_brain_is_empty(brain):
    assert brain.similar_designs(genome={}, metrics={}) == []
    assert brain.warm_start() is None


def test_best_design_is_the_lightest_feasible(brain):
    brain.episodic.record_run("run1", "mvp")
    brain.episodic.record_design({"material_id": "m"}, {"mass_kg": 2.0},
                                 feasible=True, run_id="run1")
    light = brain.episodic.record_design({"material_id": "m"}, {"mass_kg": 0.5},
                                         feasible=True, run_id="run1")
    brain.episodic.record_design({"material_id": "m"}, {"mass_kg": 0.1},
                                 feasible=False, run_id="run1")
    assert brain.episodic.best_design()["design_id"] == light


# =========================================================================== #
# 5. knowledge graph
# =========================================================================== #
def _small_graph(brain):
    ev = [Evidence(EvidenceKind.SIMULATION, "e1", run_id="r1")]
    brain.graph.add_edge("al_7075_t6", "is_a", "material", ev)
    brain.graph.add_edge("link", "made_of", "al_7075_t6", ev)
    brain.graph.add_edge("link", "loaded_in", "bending", ev)
    brain.graph.add_edge("bending", "causes", "deflection", ev)
    brain.graph.add_edge("deflection", "limits", "mass_reduction", ev)


def test_graph_neighbors(brain):
    _small_graph(brain)
    out = brain.graph.neighbors("link", direction="out")
    assert {e["target"] for e in out} == {"al_7075_t6", "bending"}
    incoming = brain.graph.neighbors("al_7075_t6", direction="in")
    assert {e["source"] for e in incoming} == {"link"}
    typed = brain.graph.neighbors("link", relation="loaded_in")
    assert len(typed) == 1 and typed[0]["target"] == "bending"


def test_graph_path(brain):
    _small_graph(brain)
    path = brain.graph.path("link", "mass_reduction")
    assert [e["relation"] for e in path] == ["loaded_in", "causes", "limits"]
    assert brain.graph.path("mass_reduction", "link") is None
    assert brain.graph.path("link", "link") == []


def test_graph_edges_require_evidence(brain):
    with pytest.raises(ValueError, match="evidence"):
        brain.graph.add_edge("a", "rel", "b", [])


def test_graph_edge_merges_evidence(brain):
    e1 = [Evidence(EvidenceKind.SIMULATION, "e1", run_id="r1")]
    e2 = [Evidence(EvidenceKind.SIMULATION, "e2", run_id="r2")]
    brain.graph.add_edge("a", "rel", "b", e1)
    brain.graph.add_edge("a", "rel", "b", e2)
    edges = brain.graph.edges()
    assert len(edges) == 1
    assert len(edges[0]["evidence"]) == 2


def test_graph_creates_concepts_for_edge_endpoints(brain):
    _small_graph(brain)
    names = {c["name"] for c in brain.graph.concepts()}
    assert {"link", "bending", "deflection", "material"} <= names


# =========================================================================== #
# 6. semantic generalization
# =========================================================================== #
def _synthetic_episodes(n_runs: int, per_run: int, active: str):
    episodes = []
    for r in range(n_runs):
        for i in range(per_run):
            episodes.append({
                "episode_id": f"r{r}-e{i}",
                "run_id": f"run{r}",
                "feasible": True,
                "design_genome": {"outer_width_m": 0.010,
                                  "outer_height_m": 0.081,
                                  "wall_thickness_m": 0.001},
                "constraint_status": {"active": [active]},
            })
    return episodes


def test_generalization_finds_the_planted_pattern(brain):
    episodes = _synthetic_episodes(n_runs=4, per_run=3, active="deflection")
    from brain.semantic import generalize_binding_constraint
    produced = generalize_binding_constraint(episodes, brain.semantic)
    assert len(produced) == 1
    k = produced[0]
    assert "deflection" in k.statement
    assert len(k.evidence) == 12                 # one per episode
    assert independent_runs(k.evidence) == 4     # but only four independent runs


def test_generalization_is_tagged_simulated_not_fact(brain):
    """Generalizing over one run must not look like corroboration."""
    from brain.semantic import generalize_binding_constraint
    produced = generalize_binding_constraint(
        _synthetic_episodes(n_runs=1, per_run=10, active="deflection"),
        brain.semantic)
    assert produced[0].evidence_level is EvidenceLevel.SIMULATED
    assert all(e.kind is EvidenceKind.SIMULATION for e in produced[0].evidence)


def test_generalization_across_runs_can_reach_repeated(brain):
    """3 independent runs clears REPEATED but not the HIGH_CONFIDENCE
    thresholds (8 evidence items from 5 runs)."""
    from brain.semantic import generalize_binding_constraint
    produced = generalize_binding_constraint(
        _synthetic_episodes(n_runs=3, per_run=2, active="deflection"),
        brain.semantic)
    assert produced[0].evidence_level is EvidenceLevel.REPEATED


def test_generalization_across_many_runs_reaches_high_confidence(brain):
    from brain.semantic import generalize_binding_constraint
    produced = generalize_binding_constraint(
        _synthetic_episodes(n_runs=5, per_run=2, active="deflection"),
        brain.semantic)
    assert produced[0].evidence_level is EvidenceLevel.HIGH_CONFIDENCE


def test_generalization_never_reaches_experimental_validation(brain):
    from brain.semantic import generalize_binding_constraint
    produced = generalize_binding_constraint(
        _synthetic_episodes(n_runs=50, per_run=10, active="deflection"),
        brain.semantic)
    assert produced[0].evidence_level is not EvidenceLevel.EXPERIMENTALLY_VALIDATED


def test_generalization_ignores_weak_patterns(brain):
    from brain.semantic import generalize_binding_constraint
    mixed = (_synthetic_episodes(2, 2, "deflection")
             + _synthetic_episodes(2, 2, "stress"))
    for i, e in enumerate(mixed):
        e["episode_id"] = f"m{i}"
    assert generalize_binding_constraint(mixed, brain.semantic) == []


def test_generalization_records_its_assumptions(brain):
    from brain.semantic import generalize_binding_constraint
    produced = generalize_binding_constraint(
        _synthetic_episodes(3, 2, "deflection"), brain.semantic)
    joined = " ".join(produced[0].assumptions).lower()
    assert "beam theory" in joined or "euler" in joined


def test_generalization_flags_bound_activity(brain):
    from brain.semantic import generalize_bound_activity
    produced = generalize_bound_activity(
        _synthetic_episodes(3, 2, "deflection"), brain.semantic)
    statements = " ".join(k.statement for k in produced)
    assert "wall_thickness_m" in statements
    assert "sets the achievable mass" in statements


# =========================================================================== #
# strategies
# =========================================================================== #
def test_strategy_promoted_when_one_variable_dominates(brain):
    samples = [
        {"ref": f"d{i}", "run_id": f"run{i}",
         "d_deflection": {"outer_width_m": -0.0016, "outer_height_m": -0.0036,
                          "wall_thickness_m": -0.018},
         "d_mass": {"outer_width_m": 14.05, "outer_height_m": 14.05,
                    "wall_thickness_m": 309.1}}
        for i in range(5)
    ]
    strategy = derive_stiffness_strategy(samples, brain.strategies)
    assert strategy is not None
    assert "outer_height_m" in strategy.name
    assert strategy.context == {"binding_constraint": "deflection"}
    assert strategy.evidence_level is EvidenceLevel.REPEATED


def test_strategy_not_promoted_without_a_consistent_winner(brain):
    samples = [
        {"ref": "a", "run_id": "r1",
         "d_deflection": {"h": -1.0, "b": -0.1}, "d_mass": {"h": 1.0, "b": 1.0}},
        {"ref": "b", "run_id": "r2",
         "d_deflection": {"h": -0.1, "b": -1.0}, "d_mass": {"h": 1.0, "b": 1.0}},
    ]
    assert derive_stiffness_strategy(samples, brain.strategies) is None


def test_strategy_retrieval_by_context(brain):
    samples = [
        {"ref": f"d{i}", "run_id": f"run{i}",
         "d_deflection": {"outer_height_m": -1.0, "outer_width_m": -0.1},
         "d_mass": {"outer_height_m": 1.0, "outer_width_m": 1.0}}
        for i in range(4)
    ]
    derive_stiffness_strategy(samples, brain.strategies)
    assert brain.applicable_strategies({"binding_constraint": "deflection"})
    assert brain.applicable_strategies({"binding_constraint": "stress"}) == []


def test_strategy_store_round_trip(brain):
    samples = [
        {"ref": f"d{i}", "run_id": f"run{i}",
         "d_deflection": {"outer_height_m": -1.0}, "d_mass": {"outer_height_m": 1.0}}
        for i in range(3)
    ]
    s = derive_stiffness_strategy(samples, brain.strategies)
    loaded = brain.strategies.get(s.name)
    assert loaded.statement == s.statement
    assert loaded.confidence == pytest.approx(s.confidence)


def test_empty_samples_promote_nothing(brain):
    assert derive_stiffness_strategy([], brain.strategies) is None


# =========================================================================== #
# skills stub
# =========================================================================== #
def test_skill_applicability():
    from brain.skills import Skill, SkillLibrary
    lib = SkillLibrary()
    lib.add(Skill(name="deepen-section",
                  description="raise section height",
                  preconditions=["deflection_binding"]))
    assert lib.applicable({"deflection_binding": True})
    assert lib.applicable({"deflection_binding": False}) == []


# =========================================================================== #
# 7. model independence
# =========================================================================== #
def test_brain_is_usable_without_any_model_or_reasoner(tmp_path):
    """Open and query the Brain in a fresh interpreter that imports only
    `brain` - no agent, no physics, no optimizer, no GPU."""
    db_path = tmp_path / "brain.sqlite3"
    with Brain(db_path) as b:
        b.episodic.record_run("run1", "mvp")
        b.episodic.record_design(
            {"outer_width_m": 0.01, "outer_height_m": 0.08,
             "wall_thickness_m": 0.001, "material_id": "al_7075_t6"},
            {"mass_kg": 0.25}, feasible=True, run_id="run1")
        b.semantic.store(Knowledge(statement="deflection binds", domain="link",
                                   source="test", evidence=sim(3)))

    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from brain import Brain
with Brain({str(db_path)!r}) as b:
    assert len(b.episodic.designs()) == 1
    assert len(b.knowledge('link')) == 1
    assert b.warm_start() is not None
    forbidden = [m for m in sys.modules
                 if m.split('.')[0] in ('torch', 'warp', 'scipy')]
    assert not forbidden, forbidden
    print("OK")
"""
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# =========================================================================== #
# 8. loop integration
# =========================================================================== #
@pytest.fixture(scope="module")
def op():
    return build_optimization_problem(build_mvp_problem())


def test_loop_populates_the_brain(op, brain):
    result = DesignLoop(op, LoopConfig(max_iterations=2, seed=1, **FAST),
                        brain=brain).run()
    counts = brain.db.counts()
    assert counts["runs"] == 1
    assert counts["episodes"] == result.iterations
    assert counts["designs"] == result.iterations
    run = brain.episodic.get_run(result.run_id)
    assert run["termination"] == result.termination.value
    assert run["meta"]["warm_started"] is False


def test_second_run_warm_starts_and_still_converges(op, brain):
    """Closing the loop: run 2 begins from run 1's best and must not regress."""
    first = DesignLoop(op, LoopConfig(max_iterations=2, seed=1, **FAST),
                       brain=brain).run()
    assert brain.warm_start() is not None

    second_loop = DesignLoop(op, LoopConfig(max_iterations=2, seed=2, **FAST),
                             brain=brain)
    assert second_loop.warm_start_x is not None
    second = second_loop.run()

    assert second.best_evaluation.is_feasible()
    assert second.best_mass_kg <= first.best_mass_kg * 1.001   # no regression
    assert brain.episodic.get_run(second.run_id)["meta"]["warm_started"] is True
    assert brain.db.counts()["runs"] == 2


def test_generalization_over_real_loop_episodes(op, brain):
    DesignLoop(op, LoopConfig(max_iterations=2, seed=1, **FAST), brain=brain).run()
    produced = brain.generalize()
    assert produced
    for k in produced:
        assert k.evidence_level.rank <= EvidenceLevel.HIGH_CONFIDENCE.rank
        assert all(e.kind is EvidenceKind.SIMULATION for e in k.evidence)


def test_brain_summary_reports_levels(op, brain):
    DesignLoop(op, LoopConfig(max_iterations=2, seed=1, **FAST), brain=brain).run()
    brain.generalize()
    summary = brain.summary()
    assert summary["counts"]["episodes"] > 0
    assert summary["knowledge_by_level"]["experimentally_validated"] == 0
