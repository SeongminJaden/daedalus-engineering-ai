"""Phase 6 verification: the surrogate and the screen-and-verify discipline.

The load-bearing check is not the model's accuracy - it is that the surrogate
is never allowed to decide. `screen_and_verify` must return a design the real
solver evaluated, and the tests below pin that.

Honest scope: the surrogate approximates the Phase 2 beam evaluator, not 3D
FEM, so its error stacks on top of beam theory's own.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from optimization.constraints import build_optimization_problem, evaluate_batch  # noqa: E402
from physics.structural import evaluate_beam_case, load_case_from_problem  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402
from surrogate.datasets import (  # noqa: E402
    INPUT_NAMES, OUTPUT_NAMES, Dataset, SamplingRanges, generate_dataset,
    resolve_dataset_size,
)
from surrogate.inference import (  # noqa: E402
    SurrogatePredictor, brute_force_best, build_inputs, screen_and_verify,
)
from surrogate.models import SurrogateBundle, evaluate_predictions, train_surrogate  # noqa: E402

# Small enough to train inside a test run, large enough to clear R^2 > 0.99 on
# a context-grouped held-out set (which is a much harder target than a random
# row split - see Dataset.split).
TRAIN_CFG = dict(seed=0, epochs=200, hidden=(128, 128), patience=80)
DATASET_CFG = dict(n_samples=12000, seed=0)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(**DATASET_CFG)


@pytest.fixture(scope="module")
def trained(dataset):
    return train_surrogate(dataset, **TRAIN_CFG)


@pytest.fixture(scope="module")
def predictor(trained):
    return SurrogatePredictor(trained[0])


@pytest.fixture(scope="module")
def op():
    return build_optimization_problem(build_mvp_problem())


def _candidates(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.010, 0.100, n)
    h = rng.uniform(0.010, 0.100, n)
    t = rng.uniform(0.001, np.minimum(0.020, 0.45 * np.minimum(b, h)))
    return np.column_stack([b, h, t])


# =========================================================================== #
# 1. dataset consistency - the data must BE the solver's output
# =========================================================================== #
def test_dataset_rows_reproduce_under_the_solver(dataset):
    """Re-evaluate stored rows with the solver. If these disagree, the model
    would be learning something that is not the physics."""
    rng = np.random.default_rng(0)
    picks = rng.choice(len(dataset), size=40, replace=False)

    from physics.structural import BeamLoadCase
    for i in picks:
        L, P, E, rho, sy, b, h, t = dataset.inputs[i]
        case = BeamLoadCase(length_m=L, tip_load_n=P, youngs_modulus_pa=E,
                            density_kg_m3=rho, yield_strength_pa=sy)
        got = evaluate_beam_case(np.array([b]), np.array([h]), np.array([t]),
                                 case).candidate(0)
        for j, name in enumerate(dataset.output_names):
            assert got[name] == pytest.approx(dataset.outputs[i, j], rel=1e-9), (
                f"row {i} field {name} does not reproduce"
            )


def test_dataset_shapes_and_names(dataset):
    assert dataset.inputs.shape[1] == len(INPUT_NAMES)
    assert dataset.outputs.shape[1] == len(OUTPUT_NAMES)
    assert dataset.input_names == INPUT_NAMES
    assert "safety_factor" not in dataset.output_names   # derived, not learned


def test_dataset_is_finite_and_positive(dataset):
    assert np.all(np.isfinite(dataset.inputs))
    assert np.all(np.isfinite(dataset.outputs))
    assert np.all(dataset.outputs > 0)


def test_dataset_samples_the_problem_context_not_just_geometry(dataset):
    """A single-problem dataset would give a single-problem surrogate."""
    for name in ("length_m", "tip_load_n", "youngs_modulus_pa"):
        column = dataset.inputs[:, INPUT_NAMES.index(name)]
        assert column.std() > 0, f"{name} was not varied"


def test_dataset_respects_geometric_validity(dataset):
    b = dataset.inputs[:, INPUT_NAMES.index("outer_width_m")]
    h = dataset.inputs[:, INPUT_NAMES.index("outer_height_m")]
    t = dataset.inputs[:, INPUT_NAMES.index("wall_thickness_m")]
    assert np.all(t < np.minimum(b, h) / 2.0)


def test_dataset_round_trip(dataset, tmp_path):
    path = dataset.save(tmp_path / "d.npz")
    loaded = Dataset.load(path)
    assert np.array_equal(loaded.inputs, dataset.inputs)
    assert np.array_equal(loaded.outputs, dataset.outputs)
    assert loaded.output_names == dataset.output_names


def test_split_holds_out_whole_contexts(dataset):
    """A random row split would put designs from the same problem in both
    train and test, and the resulting score would flatter the model."""
    train, val, test = dataset.split(seed=0)
    train_ctx = set(np.unique(train.context_ids))
    val_ctx = set(np.unique(val.context_ids))
    test_ctx = set(np.unique(test.context_ids))
    assert not (train_ctx & test_ctx)
    assert not (train_ctx & val_ctx)
    assert not (val_ctx & test_ctx)


def test_context_sampling_covers_the_low_end(dataset):
    """Load and length are drawn log-uniformly: sampled uniformly, almost every
    context would land in the top decade and the low-load corner where the MVP
    problem actually sits would be nearly unvisited."""
    for name in ("length_m", "tip_load_n"):
        column = dataset.inputs[:, INPUT_NAMES.index(name)]
        lo, hi = column.min(), column.max()
        share_low = float(np.mean(column < lo + 0.25 * (hi - lo)))
        assert share_low > 0.4, f"{name}: only {share_low:.1%} in the bottom quarter"


def test_split_is_deterministic_and_disjoint(dataset):
    a = dataset.split(seed=3)
    b = dataset.split(seed=3)
    for x, y in zip(a, b):
        assert np.array_equal(x.inputs, y.inputs)
    total = sum(len(part) for part in a)
    assert total == len(dataset)


def test_dataset_size_comes_from_profile():
    assert resolve_dataset_size("laptop_4gb") == 20000
    assert resolve_dataset_size("cloud_a100") == 500000
    assert resolve_dataset_size("laptop_4gb", n_samples=123) == 123


def test_generation_is_deterministic_for_a_seed():
    a = generate_dataset(n_samples=400, n_contexts=8, seed=11)
    b = generate_dataset(n_samples=400, n_contexts=8, seed=11)
    assert np.array_equal(a.inputs, b.inputs)
    assert np.array_equal(a.outputs, b.outputs)


# =========================================================================== #
# 2. accuracy on held-out data
# =========================================================================== #
def test_surrogate_accuracy_on_held_out_test_set(trained):
    """The held-out set contains only contexts the model never trained on, so
    this measures generalization to a NEW problem, not interpolation within a
    seen one."""
    _, report = trained
    for name, stats in report.test_metrics.items():
        assert stats["r2"] > 0.99, f"{name}: R2={stats['r2']:.5f}\n{stats}"
        assert stats["mean_rel_err"] < 0.05, f"{name}: {stats}"


def test_mass_is_predicted_best(trained):
    """Mass is close to multilinear in the section dimensions, so it should be
    the easiest of the four - if it is not, something is wrong upstream."""
    _, report = trained
    assert report.test_metrics["mass_kg"]["r2"] > 0.999


def test_early_stopping_kept_the_best_epoch(trained):
    _, report = trained
    assert report.best_epoch <= report.epochs_run
    assert np.isfinite(report.val_loss)


def test_bundle_round_trip(trained, tmp_path):
    bundle, _ = trained
    inputs = np.array([[0.5, 196.2, 71.7e9, 2810.0, 503e6, 0.05, 0.08, 0.005]])
    before = bundle.predict_array(inputs)
    reloaded = SurrogateBundle.load(bundle.save(tmp_path / "m.pt"))
    assert np.allclose(reloaded.predict_array(inputs), before, rtol=1e-6)
    assert reloaded.test_metrics == bundle.test_metrics


def test_predictor_rejects_wrong_input_width(predictor):
    with pytest.raises(ValueError, match="expected 8 inputs"):
        predictor.predict(np.zeros((2, 5)))


# =========================================================================== #
# uncertainty is always attached
# =========================================================================== #
def test_prediction_carries_expected_error(predictor, op):
    case = load_case_from_problem(op.problem)
    pred = predictor.predict_designs([0.05], [0.08], [0.005], case)
    assert pred.verified is False            # a prediction is never "verified"
    for name in OUTPUT_NAMES:
        assert pred.expected_relative_error[name] > 0
    lo, hi = pred.interval("mass_kg", 0)
    assert lo < pred.values["mass_kg"][0] < hi


def test_safety_factor_is_derived_not_predicted(predictor, op):
    case = load_case_from_problem(op.problem)
    pred = predictor.predict_designs([0.05], [0.08], [0.005], case)
    expected = case.yield_strength_pa / pred.values["max_bending_stress_pa"][0]
    assert pred.values["safety_factor"][0] == pytest.approx(expected, rel=1e-12)


def test_predictions_track_the_solver(predictor, op):
    """Predicted values must be near the solver's, within the model's own
    stated error - this is what makes the screening ranking meaningful."""
    case = load_case_from_problem(op.problem)
    candidates = _candidates(200, seed=3)
    pred = predictor.predict_designs(candidates[:, 0], candidates[:, 1],
                                     candidates[:, 2], case)
    truth = evaluate_beam_case(candidates[:, 0], candidates[:, 1],
                               candidates[:, 2], case)
    for name in ("mass_kg", "max_bending_stress_pa", "tip_deflection_m"):
        rel = np.abs(pred.values[name] - getattr(truth, name)) / getattr(truth, name)
        assert np.median(rel) < 0.05, f"{name} median rel err {np.median(rel):.3%}"


# =========================================================================== #
# 3. THE DISCIPLINE: the surrogate never decides alone
# =========================================================================== #
def test_screen_and_verify_returns_a_solver_verified_design(predictor, op):
    candidates = _candidates(4000)
    result = screen_and_verify(predictor, op, candidates, top_k=16)

    assert result.verified is True
    assert result.n_screened == 4000
    assert result.n_verified <= 16          # only the shortlist cost solver time

    # The reported mass must be the SOLVER's number, not the surrogate's.
    mass, _, _ = evaluate_batch(op, result.best_x.reshape(1, 3))
    assert result.best_mass_kg == pytest.approx(float(mass[0]), rel=1e-12)
    assert result.best_mass_kg != result.predicted_best_mass_kg


def test_screened_winner_matches_brute_force_solver_search(predictor, op):
    """Screening must not cost quality: compare against evaluating every
    candidate with the solver."""
    candidates = _candidates(4000)
    result = screen_and_verify(predictor, op, candidates, top_k=16)
    _, brute_mass = brute_force_best(op, candidates)
    assert brute_mass is not None
    rel = (result.best_mass_kg - brute_mass) / brute_mass
    assert rel < 0.02, (
        f"screening winner {result.best_mass_kg:.6f} kg is {rel:.2%} heavier "
        f"than the brute-force optimum {brute_mass:.6f} kg"
    )


def test_screened_winner_is_feasible(predictor, op):
    result = screen_and_verify(predictor, op, _candidates(2000), top_k=12)
    assert result.is_feasible()
    assert result.best_constraints


def test_screening_reports_its_own_error_on_the_winner(predictor, op):
    result = screen_and_verify(predictor, op, _candidates(2000), top_k=12)
    assert result.surrogate_error_on_winner is not None
    assert result.surrogate_error_on_winner >= 0.0
    assert result.expected_relative_error


def test_screening_with_no_valid_candidates_is_unverified(op, predictor):
    impossible = np.array([[0.02, 0.02, 0.02], [0.01, 0.01, 0.05]])
    result = screen_and_verify(predictor, op, impossible, top_k=4)
    assert result.verified is False
    assert result.best_x is None


def test_screening_rejects_bad_arguments(predictor, op):
    with pytest.raises(ValueError):
        screen_and_verify(predictor, op, _candidates(10), top_k=0)
    with pytest.raises(ValueError, match="columns"):
        screen_and_verify(predictor, op, np.zeros((5, 2)), top_k=2)


def test_top_k_of_one_still_verifies(predictor, op):
    result = screen_and_verify(predictor, op, _candidates(500), top_k=1)
    assert result.n_verified == 1
    if result.verified:
        mass, _, _ = evaluate_batch(op, result.best_x.reshape(1, 3))
        assert result.best_mass_kg == pytest.approx(float(mass[0]), rel=1e-12)


# =========================================================================== #
# 4. speed - measured, not assumed
# =========================================================================== #
def test_speed_benchmark_is_measurable_and_reported(predictor, op, capsys):
    """Records throughput honestly. At Phase 2 fidelity the base evaluator is
    a closed-form arithmetic kernel, so the surrogate is NOT expected to win;
    this test measures rather than asserts a speedup."""
    case = load_case_from_problem(op.problem)
    candidates = _candidates(20000, seed=11)
    inputs = build_inputs(candidates[:, 0], candidates[:, 1], candidates[:, 2],
                          case)

    def bench(fn, warm=2, runs=3):
        for _ in range(warm):
            fn()
        return min(_timed(fn) for _ in range(runs))

    def _timed(fn):
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    t_sur = bench(lambda: predictor.predict(inputs))
    t_solver = bench(lambda: evaluate_beam_case(
        candidates[:, 0], candidates[:, 1], candidates[:, 2], case))

    assert t_sur > 0 and t_solver > 0
    with capsys.disabled():
        print(f"\n  surrogate: {20000 / t_sur:,.0f} cand/s | "
              f"batched solver: {20000 / t_solver:,.0f} cand/s | "
              f"ratio {t_solver / t_sur:.2f}x")


# =========================================================================== #
# 5. monotonicity - ML can violate physics, so measure how often
# =========================================================================== #
def test_monotonic_trends_are_mostly_preserved(predictor, op, capsys):
    """Thicker wall -> more inertia -> less stress and less deflection. A
    learned model has no guarantee of respecting that, so the violation rate is
    measured and bounded rather than assumed to be zero."""
    case = load_case_from_problem(op.problem)
    rng = np.random.default_rng(5)
    n = 800
    b = rng.uniform(0.02, 0.09, n)
    h = rng.uniform(0.02, 0.09, n)
    t1 = rng.uniform(0.0015, 0.006, n)
    t2 = t1 * 1.5

    p1 = predictor.predict_designs(b, h, t1, case)
    p2 = predictor.predict_designs(b, h, t2, case)

    violations = {
        "stress_should_fall": float(np.mean(
            p2.values["max_bending_stress_pa"] >= p1.values["max_bending_stress_pa"])),
        "deflection_should_fall": float(np.mean(
            p2.values["tip_deflection_m"] >= p1.values["tip_deflection_m"])),
        "mass_should_rise": float(np.mean(
            p2.values["mass_kg"] <= p1.values["mass_kg"])),
    }
    with capsys.disabled():
        print("\n  monotonicity violation rates: " + ", ".join(
            f"{k}={v:.2%}" for k, v in violations.items()))
    for name, rate in violations.items():
        assert rate < 0.05, f"{name} violated in {rate:.2%} of pairs"


# =========================================================================== #
# 6. determinism
# =========================================================================== #
def test_predictions_are_deterministic(predictor, op):
    case = load_case_from_problem(op.problem)
    c = _candidates(200, seed=1)
    a = predictor.predict_designs(c[:, 0], c[:, 1], c[:, 2], case)
    b = predictor.predict_designs(c[:, 0], c[:, 1], c[:, 2], case)
    for name in a.values:
        assert np.array_equal(a.values[name], b.values[name])


def test_screening_is_deterministic(predictor, op):
    c = _candidates(1000, seed=2)
    a = screen_and_verify(predictor, op, c, top_k=8)
    b = screen_and_verify(predictor, op, c, top_k=8)
    assert np.array_equal(a.best_x, b.best_x)
    assert a.best_mass_kg == b.best_mass_kg


def test_training_is_reproducible_for_a_seed():
    small = generate_dataset(n_samples=1200, n_contexts=12, seed=4)
    _, r1 = train_surrogate(small, seed=1, epochs=30, hidden=(32, 32))
    _, r2 = train_surrogate(small, seed=1, epochs=30, hidden=(32, 32))
    assert r1.val_loss == pytest.approx(r2.val_loss, rel=1e-9)
    assert r1.best_epoch == r2.best_epoch


def test_evaluate_predictions_is_exact_on_perfect_predictions():
    truth = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]])
    stats = evaluate_predictions(truth, truth, ("a", "b"))
    for name in ("a", "b"):
        assert stats[name]["r2"] == pytest.approx(1.0)
        assert stats[name]["max_rel_err"] == pytest.approx(0.0)
