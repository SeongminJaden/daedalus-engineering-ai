"""Wall thickness and draft, and which of the two measures is usable.

The wall check has exact answers on shapes whose thickness is known by
construction. The draft check does not, and the interesting content here is
which draft statistic survives contact with a voxel derived surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from geometry.surfacing import isosurface
from geometry.surfacing.manufacturability import draft, wall_thickness

SPACING = 0.001


def plate(thickness_voxels: int) -> np.ndarray:
    field = np.zeros((30, 30, 30))
    start = 15 - thickness_voxels // 2
    field[start:start + thickness_voxels, 5:25, 5:25] = 1.0
    return field


def square_rod(width_voxels: int) -> np.ndarray:
    field = np.zeros((30, 30, 30))
    start = 15 - width_voxels // 2
    field[5:25, start:start + width_voxels, start:start + width_voxels] = 1.0
    return field


def tapered(slope: float) -> np.ndarray:
    field = np.zeros((30, 30, 30))
    for k in range(6, 24):
        half = 4.0 + (k - 6) * slope
        low, high = int(round(15 - half)), int(round(15 + half))
        field[k, low:high, low:high] = 1.0
    return field


def box_field() -> np.ndarray:
    field = np.zeros((24, 24, 24))
    field[6:18, 6:18, 6:18] = 1.0
    return field


# --------------------------------------------------- wall, against exact answers

@pytest.mark.parametrize("voxels", [4, 6, 8])
def test_a_plate_reports_the_thickness_it_was_built_with(voxels):
    """Twice the distance to the surface, at the ridge, is the wall. For a
    plate of n voxels the ridge sits at n/2 and the answer is n exactly."""
    report = wall_thickness(plate(voxels), SPACING)
    assert report.minimum_wall_m == pytest.approx(voxels * SPACING, rel=1e-12)


@pytest.mark.parametrize("voxels", [4, 6, 8])
def test_a_square_rod_reports_its_width(voxels):
    report = wall_thickness(square_rod(voxels), SPACING)
    assert report.minimum_wall_m == pytest.approx(voxels * SPACING, rel=1e-12)


def test_a_thinner_plate_reports_a_thinner_wall():
    thin = wall_thickness(plate(4), SPACING).minimum_wall_m
    thick = wall_thickness(plate(8), SPACING).minimum_wall_m
    assert thin < thick


def test_the_wall_floor_is_a_pass_or_fail():
    report = wall_thickness(plate(4), SPACING, floor_m=0.006)
    assert not report.passes(0.006)
    assert report.passes(0.004)
    assert report.voxels_below_floor > 0


def test_an_empty_field_is_refused():
    with pytest.raises(ValueError, match="no material"):
        wall_thickness(np.zeros((10, 10, 10)), SPACING)


# ------------------------------------ draft, and which statistic is usable

def test_the_minimum_draft_separates_nothing_on_voxel_geometry():
    """The finding that decided the shape of this check.

    A square box, a tapered box and a sphere all report a minimum draft of
    zero, because a surface extracted from voxels always contains individual
    facets lying along an axis. A gate on the minimum would reject every one
    of them equally, including the sphere, whose zero is CORRECT: its equator
    really is parallel to any pull.
    """
    ball = np.zeros((30, 30, 30))
    zz, yy, xx = np.mgrid[0:30, 0:30, 0:30]
    ball[((xx - 15) ** 2 + (yy - 15) ** 2 + (zz - 15) ** 2) < 81] = 1.0

    for field, axis in ((box_field(), 2), (tapered(0.6), 0), (ball, 2)):
        assert draft(*isosurface(field, SPACING),
                     pull_axis=axis).minimum_draft_deg == pytest.approx(0.0,
                                                                        abs=1e-9)


def test_the_area_fraction_does_separate_them():
    """More taper means less surface dragging along the pull, monotonically."""
    fractions = [
        draft(*isosurface(field, SPACING), pull_axis=axis)
        .area_fraction_below(3.0)
        for field, axis in ((box_field(), 2), (tapered(0.30), 0),
                            (tapered(0.60), 0))]
    assert fractions[0] > fractions[1] > fractions[2]


def test_the_draft_distribution_is_quantised_on_voxel_geometry():
    """Thresholds between facet angles give identical answers.

    Marching cubes on a binary field produces facets at a handful of discrete
    angles, so asking for one degree, three degrees or ten gives the same
    fraction. Worth knowing before treating a draft requirement here as a
    continuous dial.
    """
    report = draft(*isosurface(box_field(), SPACING), pull_axis=2)
    at_one = report.area_fraction_below(1.0)
    assert report.area_fraction_below(3.0) == pytest.approx(at_one)
    assert report.area_fraction_below(10.0) == pytest.approx(at_one)


def test_the_allowance_must_be_asked_for():
    """A caller tolerating faceting has to say so, rather than inheriting a
    tolerance chosen for them."""
    report = draft(*isosurface(tapered(0.60), SPACING), pull_axis=0)
    assert not report.passes(3.0)
    assert report.passes(3.0, allowed_area_fraction=0.5)


def test_a_degenerate_surface_is_refused():
    vertices = np.zeros((3, 3))
    faces = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="no surface"):
        draft(vertices, faces)
