"""Phase 2 verification: GPU beam physics against an independent reference.

The Warp kernel is checked against tests/reference_beam.py, a float64 numpy
implementation that imports neither Warp nor any project module. A bug shared
between the kernel and core.design_genome therefore cannot pass both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from core.design_genome import DesignGenome, HollowRectangleSection  # noqa: E402
from physics.solver import evaluate_population, resolve_batch_size  # noqa: E402
from physics.structural import (  # noqa: E402
    METRIC_NAMES, evaluate_beam, load_case_from_problem,
)
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402
from reference_beam import reference_metrics  # noqa: E402

# fp32 kernel vs float64 reference.
FP32_RELATIVE_TOLERANCE = 1e-4

# A spread of shapes, including a thin wall where computing I as a difference
# of two near-equal products loses the most precision in fp32.
DESIGNS = [
    (0.050, 0.080, 0.0050),
    (0.040, 0.040, 0.0040),
    (0.100, 0.020, 0.0020),
    (0.030, 0.060, 0.0015),
    (0.060, 0.100, 0.0100),
    (0.050, 0.080, 0.0005),   # thin wall: worst-case cancellation
    (0.012, 0.012, 0.0010),   # small section
]


def genome(b, h, t, material_id="al_7075_t6"):
    return DesignGenome(
        section=HollowRectangleSection(
            outer_width_m=b, outer_height_m=h, wall_thickness_m=t
        ),
        material_id=material_id,
    )


@pytest.fixture(scope="module")
def problem():
    return build_mvp_problem()


@pytest.fixture(scope="module")
def case(problem):
    """Reference parameters mirroring the DEFAULT model, which now includes the
    Timoshenko shear term. The reference is told the same, so this stays a
    genuine independent check of whatever model the kernel is running."""
    lc = load_case_from_problem(problem)
    return dict(
        length_m=lc.length_m,
        tip_load_n=lc.tip_load_n,
        youngs_modulus_pa=lc.youngs_modulus_pa,
        density_kg_m3=lc.density_kg_m3,
        yield_strength_pa=lc.yield_strength_pa,
        poisson_ratio=lc.poisson_ratio,
        shear_factor=lc.shear_factor,
    )


@pytest.fixture(scope="module")
def euler_case(problem):
    """Same, with shear switched off: the pure Euler-Bernoulli path."""
    lc = load_case_from_problem(problem, shear_deformation=False)
    return dict(
        length_m=lc.length_m,
        tip_load_n=lc.tip_load_n,
        youngs_modulus_pa=lc.youngs_modulus_pa,
        density_kg_m3=lc.density_kg_m3,
        yield_strength_pa=lc.yield_strength_pa,
        poisson_ratio=lc.poisson_ratio,
        shear_factor=0.0,
    )


# --------------------------------------------------------------------------- #
# 1. GPU vs independent analytical reference
# --------------------------------------------------------------------------- #
def test_gpu_matches_independent_reference(problem, case):
    """Every metric, every design, against outside algebra."""
    genomes = [genome(b, h, t) for b, h, t in DESIGNS]
    got = evaluate_beam(genomes, problem)

    b, h, t = (np.array([d[i] for d in DESIGNS]) for i in range(3))
    want = reference_metrics(b, h, t, **case)

    for name in METRIC_NAMES:
        gpu = getattr(got, name)
        ref = want[name]
        rel = np.abs(gpu - ref) / np.abs(ref)
        assert np.all(rel < FP32_RELATIVE_TOLERANCE), (
            f"{name}: max rel err {rel.max():.3e} at design "
            f"{DESIGNS[int(rel.argmax())]}\n  gpu={gpu}\n  ref={ref}"
        )


@pytest.mark.parametrize("b,h,t", DESIGNS)
def test_each_design_matches_reference(problem, case, b, h, t):
    got = evaluate_beam([genome(b, h, t)], problem).candidate(0)
    want = reference_metrics(b, h, t, **case)
    for name in METRIC_NAMES:
        assert got[name] == pytest.approx(
            float(want[name]), rel=FP32_RELATIVE_TOLERANCE
        ), name


def test_mvp_metrics_hand_checked_euler_bernoulli(problem):
    """Hand-computed for b=0.05 h=0.08 t=0.005, P=196.2 N, L=0.5 m, Al 7075-T6,
    with shear off so the bending term is isolated.

        A = 0.0012 m^2, I = 9.9e-7 m^4, c = 0.04 m
        m     = 0.0012*0.5*2810            = 1.686 kg
        sigma = 196.2*0.5*0.04/9.9e-7      = 3.9636e6 Pa
        delta = 196.2*0.125/(3*71.7e9*9.9e-7) = 1.1517e-4 m
        SF    = 503e6/3.9636e6             = 126.9
    """
    m = evaluate_beam([genome(0.05, 0.08, 0.005)], problem,
                      shear_deformation=False).candidate(0)
    assert m["mass_kg"] == pytest.approx(1.686, rel=1e-5)
    assert m["max_bending_stress_pa"] == pytest.approx(3.9636e6, rel=1e-4)
    assert m["tip_deflection_m"] == pytest.approx(1.1517e-4, rel=1e-4)
    assert m["safety_factor"] == pytest.approx(126.9, rel=1e-3)
    assert m["mean_transverse_shear_stress_pa"] == pytest.approx(163500.0, rel=1e-5)


def test_mvp_metrics_hand_checked_timoshenko(problem):
    """Same design with the shear term on. Hand-computed addition:

        G   = 71.7e9 / (2 * 1.33)           = 2.6955e10 Pa
        A_s = 2 * t * h = 2*0.005*0.08      = 8.0e-4 m^2
        d_s = 196.2*0.5/(2.6955e10*8.0e-4)  = 4.5503e-6 m
        delta = 1.1517e-4 + 4.5503e-6       = 1.1972e-4 m
    """
    m = evaluate_beam([genome(0.05, 0.08, 0.005)], problem).candidate(0)
    assert m["tip_deflection_m"] == pytest.approx(1.1972e-4, rel=1e-3)
    # mass and stress are unaffected by the shear term
    assert m["mass_kg"] == pytest.approx(1.686, rel=1e-5)
    assert m["max_bending_stress_pa"] == pytest.approx(3.9636e6, rel=1e-4)


def test_shear_term_only_affects_deflection(problem):
    euler = evaluate_beam([genome(0.05, 0.08, 0.005)], problem,
                          shear_deformation=False).candidate(0)
    timo = evaluate_beam([genome(0.05, 0.08, 0.005)], problem).candidate(0)
    assert timo["tip_deflection_m"] > euler["tip_deflection_m"]
    for key in ("mass_kg", "max_bending_stress_pa", "safety_factor",
                "first_natural_frequency_hz"):
        assert timo[key] == pytest.approx(euler[key], rel=1e-9), key


# --------------------------------------------------------------------------- #
# 2. batch consistency
# --------------------------------------------------------------------------- #
def test_chunked_batches_match_single_launch(problem):
    """Chunking is an orchestration detail and must not change any number."""
    genomes = [genome(b, h, t) for b, h, t in DESIGNS]
    full = evaluate_beam(genomes, problem)
    chunked = evaluate_population(genomes, problem, batch_size=2)

    for name in METRIC_NAMES:
        assert np.array_equal(getattr(full, name), getattr(chunked, name)), name


def test_batch_matches_one_at_a_time(problem):
    """Same kernel per element, so a batch equals per-candidate launches."""
    genomes = [genome(b, h, t) for b, h, t in DESIGNS]
    batched = evaluate_beam(genomes, problem)
    for i, g in enumerate(genomes):
        single = evaluate_beam([g], problem).candidate(0)
        for name in METRIC_NAMES:
            assert single[name] == getattr(batched, name)[i], f"{name} at {i}"


def test_batch_size_comes_from_profile():
    assert resolve_batch_size("laptop_4gb") == 4
    assert resolve_batch_size("cloud_a100") == 128
    assert resolve_batch_size("laptop_4gb", batch_size=32) == 32


def test_population_length_preserved(problem):
    genomes = [genome(b, h, t) for b, h, t in DESIGNS]
    assert len(evaluate_population(genomes, problem, batch_size=3)) == len(DESIGNS)


# --------------------------------------------------------------------------- #
# 3. physical monotonicity and scaling
# --------------------------------------------------------------------------- #
def test_thicker_wall_reduces_stress_and_deflection(problem):
    thin = evaluate_beam([genome(0.05, 0.08, 0.002)], problem).candidate(0)
    thick = evaluate_beam([genome(0.05, 0.08, 0.006)], problem).candidate(0)
    assert thick["max_bending_stress_pa"] < thin["max_bending_stress_pa"]
    assert thick["tip_deflection_m"] < thin["tip_deflection_m"]
    assert thick["mass_kg"] > thin["mass_kg"]          # stiffness costs mass
    assert thick["safety_factor"] > thin["safety_factor"]


def test_taller_section_cuts_deflection_faster_than_linearly(problem):
    """delta ~ 1/I and I ~ h^3, so doubling h must cut delta by well over 2x."""
    short = evaluate_beam([genome(0.04, 0.04, 0.003)], problem).candidate(0)
    tall = evaluate_beam([genome(0.04, 0.08, 0.003)], problem).candidate(0)
    assert tall["tip_deflection_m"] < short["tip_deflection_m"] / 4.0


def test_stress_and_deflection_are_linear_in_load(problem):
    doubled = problem.model_copy(deep=True)
    doubled.loads[0].magnitude_n = problem.loads[0].magnitude_n * 2.0

    base = evaluate_beam([genome(0.05, 0.08, 0.005)], problem).candidate(0)
    twice = evaluate_beam([genome(0.05, 0.08, 0.005)], doubled).candidate(0)

    assert twice["max_bending_stress_pa"] == pytest.approx(
        2.0 * base["max_bending_stress_pa"], rel=1e-5)
    assert twice["tip_deflection_m"] == pytest.approx(
        2.0 * base["tip_deflection_m"], rel=1e-5)
    # load-independent quantities must not move
    assert twice["mass_kg"] == pytest.approx(base["mass_kg"], rel=1e-6)
    assert twice["first_natural_frequency_hz"] == pytest.approx(
        base["first_natural_frequency_hz"], rel=1e-6)


def test_deflection_scales_with_length_cubed(problem):
    """delta ~ P*L^3/(3EI): doubling L must raise delta ~8x (P, section fixed).

    Shear is switched off here on purpose: the shear term is linear in L, so
    with it the total is deliberately NOT a pure cube. That is the point of the
    correction, and it is checked separately.
    """
    longer = problem.model_copy(deep=True)
    longer.geometry.length_m = problem.geometry.length_m * 2.0

    base = evaluate_beam([genome(0.05, 0.08, 0.005)], problem,
                         shear_deformation=False).candidate(0)
    long_ = evaluate_beam([genome(0.05, 0.08, 0.005)], longer,
                          shear_deformation=False).candidate(0)
    assert long_["tip_deflection_m"] == pytest.approx(
        8.0 * base["tip_deflection_m"], rel=1e-4)
    # f1 ~ 1/L^2 -> quarter. Frequency uses the bending stiffness only, so it is
    # unaffected by the shear switch.
    assert long_["first_natural_frequency_hz"] == pytest.approx(
        base["first_natural_frequency_hz"] / 4.0, rel=1e-4)


def test_shear_contribution_shrinks_as_the_beam_gets_slender(problem):
    """Shear matters for a stubby beam and fades for a slender one: it is
    linear in L while bending goes as L^3."""
    shares = []
    for length in (0.2, 0.5, 1.5):
        p = problem.model_copy(deep=True)
        p.geometry.length_m = length
        euler = evaluate_beam([genome(0.05, 0.08, 0.005)], p,
                              shear_deformation=False).candidate(0)
        timo = evaluate_beam([genome(0.05, 0.08, 0.005)], p).candidate(0)
        shares.append((timo["tip_deflection_m"] - euler["tip_deflection_m"])
                      / timo["tip_deflection_m"])
    assert shares[0] > shares[1] > shares[2]
    assert shares[0] > 0.1, f"short beam shear share only {shares[0]:.2%}"


def test_safety_factor_is_yield_over_stress(problem):
    from core.materials import get_material
    sy = get_material(problem.material_id).yield_strength_pa
    m = evaluate_beam([genome(0.05, 0.08, 0.005)], problem).candidate(0)
    assert m["safety_factor"] == pytest.approx(
        sy / m["max_bending_stress_pa"], rel=1e-5)


def test_stiffer_material_deflects_less(problem):
    """Same geometry in 6061 (E=68.9 GPa) must deflect more than 7075 (71.7)."""
    softer = problem.model_copy(deep=True)
    softer.material_id = "al_6061_t6"
    g7075 = genome(0.05, 0.08, 0.005, "al_7075_t6")
    g6061 = genome(0.05, 0.08, 0.005, "al_6061_t6")
    assert (evaluate_beam([g6061], softer).candidate(0)["tip_deflection_m"]
            > evaluate_beam([g7075], problem).candidate(0)["tip_deflection_m"])


# --------------------------------------------------------------------------- #
# 4. the model refuses problems it does not actually solve
# --------------------------------------------------------------------------- #
def test_rejects_non_tip_load(problem):
    from core.engineering_ir import LoadApplication
    bad = problem.model_copy(deep=True)
    bad.loads[0].application = LoadApplication.MID_SPAN
    with pytest.raises(NotImplementedError, match="tip"):
        load_case_from_problem(bad)


def test_rejects_axial_load_direction(problem):
    from core.engineering_ir import Vec3
    bad = problem.model_copy(deep=True)
    bad.loads[0].direction = Vec3(x=1.0, y=0.0, z=0.0)
    with pytest.raises(NotImplementedError, match="transverse"):
        load_case_from_problem(bad)


def test_rejects_pinned_boundary(problem):
    from core.engineering_ir import BoundaryConditionType
    bad = problem.model_copy(deep=True)
    bad.boundary_conditions[0].type = BoundaryConditionType.PINNED
    with pytest.raises(NotImplementedError, match="cantilever"):
        load_case_from_problem(bad)


def test_rejects_multiple_loads(problem):
    bad = problem.model_copy(deep=True)
    bad.loads.append(problem.loads[0].model_copy(deep=True))
    with pytest.raises(NotImplementedError, match="one load"):
        load_case_from_problem(bad)


def test_rejects_invalid_genome(problem):
    bad = DesignGenome(
        section=HollowRectangleSection(
            outer_width_m=0.02, outer_height_m=0.02, wall_thickness_m=0.02
        ),
        material_id="al_7075_t6",
    )
    with pytest.raises(ValueError, match="invalid"):
        evaluate_beam([bad], problem)


def test_rejects_empty_batch(problem):
    with pytest.raises(ValueError):
        evaluate_beam([], problem)
