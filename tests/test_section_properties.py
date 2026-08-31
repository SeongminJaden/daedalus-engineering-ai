"""Independent verification of the closed-form section formulas.

The formulas in HollowRectangleSection.section_properties() are the single
most load-bearing calculation in Phase 1: every stress and deflection in
Phase 2 is built on I_x and S_x. Asserting them against the same algebra that
produced them would prove nothing, so here they are checked against a
*numerically integrated* section - a genuinely different method.

    A   = integral of dA
    I_x = integral of y^2 dA      (horizontal centroidal axis)
    I_y = integral of x^2 dA      (vertical centroidal axis)

The integrator itself is anchored first against a solid rectangle, whose
I = b*h^3/12 is textbook, so a broken integrator cannot quietly bless a
broken formula.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.design_genome.section import HollowRectangleSection  # noqa: E402

# Grid pitch. Every test dimension below is an exact multiple of this, so cell
# edges land on section boundaries and no cell is partially filled.
GRID_PITCH_M = 1e-4

# Midpoint integration of y^2 has a known truncation floor of (pitch^2/12)*A;
# for these sections that is ~1e-6 relative, so 1e-4 is a real bound, not a
# rubber stamp.
RELATIVE_TOLERANCE = 1e-4

SECTIONS = [
    (0.05, 0.08, 0.005),    # tall rectangle
    (0.04, 0.04, 0.004),    # square
    (0.10, 0.02, 0.002),    # wide and flat
    (0.03, 0.06, 0.0015),   # thin wall
]


def _integrate_section(b: float, h: float, t: float | None, pitch: float):
    """Numerically integrate A, I_x, I_y over a rectangle or hollow rectangle.

    Builds an explicit occupancy mask on a uniform grid of cell centres and
    sums; deliberately shares no algebra with the closed form under test.
    Pass t=None for a solid rectangle.
    """
    nx = int(round(b / pitch))
    ny = int(round(h / pitch))
    assert abs(nx * pitch - b) < 1e-15, "grid must align with the width"
    assert abs(ny * pitch - h) < 1e-15, "grid must align with the height"

    # cell centres, measured from the centroid at (0, 0)
    x = (np.arange(nx) + 0.5) * pitch - b / 2.0
    y = (np.arange(ny) + 0.5) * pitch - h / 2.0
    xx, yy = np.meshgrid(x, y, indexing="xy")

    inside_outer = (np.abs(xx) <= b / 2.0) & (np.abs(yy) <= h / 2.0)
    if t is None:
        occupied = inside_outer
    else:
        bi, hi = b - 2.0 * t, h - 2.0 * t
        assert abs(round(t / pitch) * pitch - t) < 1e-15, "grid must align with t"
        inside_cavity = (np.abs(xx) < bi / 2.0) & (np.abs(yy) < hi / 2.0)
        occupied = inside_outer & ~inside_cavity

    cell_area = pitch * pitch
    area = float(occupied.sum()) * cell_area
    i_x = float((yy[occupied] ** 2).sum()) * cell_area
    i_y = float((xx[occupied] ** 2).sum()) * cell_area
    return area, i_x, i_y


def _rel_err(numeric: float, closed_form: float) -> float:
    return abs(numeric - closed_form) / abs(closed_form)


# --------------------------------------------------------------------------- #
# 0. anchor the integrator itself against a textbook result
# --------------------------------------------------------------------------- #
def test_integrator_reproduces_solid_rectangle():
    """A solid rectangle has A=b*h and I_x=b*h^3/12. If the integrator cannot
    reproduce that, nothing it says about hollow sections is worth anything."""
    b, h = 0.05, 0.08
    area, i_x, i_y = _integrate_section(b, h, None, GRID_PITCH_M)

    assert _rel_err(area, b * h) < 1e-12
    assert _rel_err(i_x, b * h**3 / 12.0) < RELATIVE_TOLERANCE
    assert _rel_err(i_y, h * b**3 / 12.0) < RELATIVE_TOLERANCE


# --------------------------------------------------------------------------- #
# 1. the actual check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("b,h,t", SECTIONS)
def test_area_matches_numerical_integration(b, h, t):
    props = HollowRectangleSection(
        outer_width_m=b, outer_height_m=h, wall_thickness_m=t
    ).section_properties()
    area, _, _ = _integrate_section(b, h, t, GRID_PITCH_M)
    assert _rel_err(area, props.area_m2) < RELATIVE_TOLERANCE, (
        f"A: numeric={area:.12g} closed_form={props.area_m2:.12g}"
    )


@pytest.mark.parametrize("b,h,t", SECTIONS)
def test_i_x_matches_numerical_integration(b, h, t):
    """The critical one: I_x drives bending stress and deflection."""
    props = HollowRectangleSection(
        outer_width_m=b, outer_height_m=h, wall_thickness_m=t
    ).section_properties()
    _, i_x, _ = _integrate_section(b, h, t, GRID_PITCH_M)
    err = _rel_err(i_x, props.i_x_m4)
    assert err < RELATIVE_TOLERANCE, (
        f"I_x: numeric={i_x:.12g} closed_form={props.i_x_m4:.12g} rel_err={err:.3e}"
    )


@pytest.mark.parametrize("b,h,t", SECTIONS)
def test_i_y_matches_numerical_integration(b, h, t):
    props = HollowRectangleSection(
        outer_width_m=b, outer_height_m=h, wall_thickness_m=t
    ).section_properties()
    _, _, i_y = _integrate_section(b, h, t, GRID_PITCH_M)
    err = _rel_err(i_y, props.i_y_m4)
    assert err < RELATIVE_TOLERANCE, (
        f"I_y: numeric={i_y:.12g} closed_form={props.i_y_m4:.12g} rel_err={err:.3e}"
    )


# --------------------------------------------------------------------------- #
# 2. section moduli follow from I and the extreme fibre distance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("b,h,t", SECTIONS)
def test_section_moduli(b, h, t):
    props = HollowRectangleSection(
        outer_width_m=b, outer_height_m=h, wall_thickness_m=t
    ).section_properties()
    assert props.s_x_m3 == pytest.approx(props.i_x_m4 / (h / 2.0), rel=1e-12)
    assert props.s_y_m3 == pytest.approx(props.i_y_m4 / (b / 2.0), rel=1e-12)


def test_square_section_has_equal_moments():
    """Sanity: a square tube must have I_x == I_y."""
    props = HollowRectangleSection(
        outer_width_m=0.04, outer_height_m=0.04, wall_thickness_m=0.004
    ).section_properties()
    assert props.i_x_m4 == pytest.approx(props.i_y_m4, rel=1e-12)


def test_taller_section_is_stiffer_in_bending():
    """I_x scales with height^3, so height buys bending stiffness fast."""
    short = HollowRectangleSection(
        outer_width_m=0.04, outer_height_m=0.04, wall_thickness_m=0.003
    ).section_properties()
    tall = HollowRectangleSection(
        outer_width_m=0.04, outer_height_m=0.08, wall_thickness_m=0.003
    ).section_properties()
    assert tall.i_x_m4 > 4.0 * short.i_x_m4
