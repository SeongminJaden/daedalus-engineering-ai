"""Phase 2 verification: Warp autodiff gradients vs central differences.

The finite differences are taken on tests/reference_beam.py - the independent
float64 implementation - not on the kernel itself. Differencing the kernel
would only prove the autodiff is self-consistent with its own forward pass; a
wrong forward pass would still pass. Differencing outside algebra checks the
value and the derivative at once.
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
from physics.solver import population_gradients  # noqa: E402
from physics.structural import beam_gradients, load_case_from_problem  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402
from reference_beam import central_difference  # noqa: E402

GRADIENT_RELATIVE_TOLERANCE = 1e-3

DESIGNS = [
    (0.050, 0.080, 0.0050),
    (0.040, 0.040, 0.0040),
    (0.030, 0.060, 0.0015),
    (0.060, 0.100, 0.0100),
]

METRICS = [
    "mass_kg",
    "max_bending_stress_pa",
    "tip_deflection_m",
    "safety_factor",
    "first_natural_frequency_hz",
]

VARIABLE_BY_FIELD = {
    "outer_width_m": "b",
    "outer_height_m": "h",
    "wall_thickness_m": "t",
}


def genome(b, h, t):
    return DesignGenome(
        section=HollowRectangleSection(
            outer_width_m=b, outer_height_m=h, wall_thickness_m=t
        ),
        material_id="al_7075_t6",
    )


@pytest.fixture(scope="module")
def problem():
    return build_mvp_problem()


@pytest.fixture(scope="module")
def case(problem):
    lc = load_case_from_problem(problem)
    return dict(
        length_m=lc.length_m,
        tip_load_n=lc.tip_load_n,
        youngs_modulus_pa=lc.youngs_modulus_pa,
        density_kg_m3=lc.density_kg_m3,
        yield_strength_pa=lc.yield_strength_pa,
    )


@pytest.mark.parametrize("metric", METRICS)
def test_autodiff_matches_central_difference(problem, case, metric):
    """All designs, all three design variables, one metric."""
    genomes = [genome(b, h, t) for b, h, t in DESIGNS]
    grads = beam_gradients(genomes, problem, metric)

    worst = 0.0
    for i, (b, h, t) in enumerate(DESIGNS):
        for field, var in VARIABLE_BY_FIELD.items():
            got = float(grads[field][i])
            want = central_difference(metric, b, h, t, var, **case)
            rel = abs(got - want) / max(abs(want), 1e-30)
            worst = max(worst, rel)
            assert rel < GRADIENT_RELATIVE_TOLERANCE, (
                f"d({metric})/d{var} at {(b, h, t)}: autodiff={got:.8g} "
                f"central_diff={want:.8g} rel_err={rel:.3e}"
            )
    assert worst < GRADIENT_RELATIVE_TOLERANCE


def test_mass_gradient_is_hand_checkable(problem):
    """dA/db = h - (h-2t) = 2t, so dm/db = 2*t*L*rho.
    t=0.005, L=0.5, rho=2810 -> 14.05 kg/m."""
    grads = beam_gradients([genome(0.05, 0.08, 0.005)], problem, "mass_kg")
    assert float(grads["outer_width_m"][0]) == pytest.approx(
        2 * 0.005 * 0.5 * 2810.0, rel=1e-4)
    # symmetric: dA/dh = b - (b-2t) = 2t as well
    assert float(grads["outer_height_m"][0]) == pytest.approx(
        2 * 0.005 * 0.5 * 2810.0, rel=1e-4)


def test_gradient_signs_are_physical(problem):
    """Adding material everywhere: mass up, stress and deflection down."""
    g = [genome(0.05, 0.08, 0.005)]
    mass = beam_gradients(g, problem, "mass_kg")
    stress = beam_gradients(g, problem, "max_bending_stress_pa")
    defl = beam_gradients(g, problem, "tip_deflection_m")
    sf = beam_gradients(g, problem, "safety_factor")

    for field in VARIABLE_BY_FIELD:
        assert mass[field][0] > 0, f"d(mass)/d{field} should be positive"
        assert stress[field][0] < 0, f"d(stress)/d{field} should be negative"
        assert defl[field][0] < 0, f"d(deflection)/d{field} should be negative"
        assert sf[field][0] > 0, f"d(SF)/d{field} should be positive"


def test_height_dominates_deflection_gradient(problem):
    """I ~ h^3 but only ~b^1, so height is the stronger lever on deflection."""
    grads = beam_gradients([genome(0.05, 0.05, 0.005)], problem, "tip_deflection_m")
    assert abs(grads["outer_height_m"][0]) > abs(grads["outer_width_m"][0])


def test_gradients_are_finite_everywhere(problem):
    genomes = [genome(b, h, t) for b, h, t in DESIGNS]
    for metric in METRICS:
        for field, arr in beam_gradients(genomes, problem, metric).items():
            assert np.all(np.isfinite(arr)), f"{metric}/{field} not finite"


def test_chunked_gradients_match_single_launch(problem):
    genomes = [genome(b, h, t) for b, h, t in DESIGNS]
    full = beam_gradients(genomes, problem, "tip_deflection_m")
    chunked = population_gradients(
        genomes, problem, "tip_deflection_m", batch_size=2)
    for field in full:
        assert np.array_equal(full[field], chunked[field]), field


def test_unknown_metric_rejected(problem):
    with pytest.raises(ValueError, match="unknown metric"):
        beam_gradients([genome(0.05, 0.08, 0.005)], problem, "vibes")
