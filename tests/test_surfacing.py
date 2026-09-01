"""Organic surfacing from a density field.

The checks that matter here are not "does it produce a mesh". They are whether
the surface encloses the volume it should, whether smoothing quietly removes
material, and whether a symmetric input stays symmetric. Each of those is a
way the result can look perfectly good and be wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geometry.surfacing import (SurfaceReport, enclosed_volume_m3,
                                field_integral_m3, isosurface, smooth,
                                surface_from_density, thresholded_volume_m3)

SPACING = 0.001


def ball_field(n: int, radius_cells: float, ramp: float = 1.0) -> np.ndarray:
    """A ball whose 0.5 level sits exactly at `radius_cells`.

    `ramp` controls how grey the boundary is: 1.0 gives a wide grey band, a
    small value approaches a binary field.
    """
    centre = (n - 1) / 2.0
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    signed = radius_cells - np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2
                                    + (zz - centre) ** 2)
    return np.clip(signed / (radius_cells * ramp) + 0.5, 0.0, 1.0)


def exact_ball_volume(radius_cells: float) -> float:
    return 4.0 / 3.0 * math.pi * (radius_cells * SPACING) ** 3


def _area_weighted_centroid(vertices: np.ndarray,
                            faces: np.ndarray) -> np.ndarray:
    """Centre of the SURFACE, which is what symmetry is a claim about."""
    triangles = vertices[faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    centres = triangles.mean(axis=1)
    return (centres * areas[:, None]).sum(axis=0) / areas.sum()


# ------------------------------------------------------ the enclosed volume

def test_the_surface_encloses_the_analytic_volume():
    field = ball_field(48, 15.0)
    report = surface_from_density(field, SPACING, smoothing_passes=0)
    exact = exact_ball_volume(15.0)
    assert abs(report.enclosed_volume_m3 - exact) / exact < 0.01


def test_the_volume_error_falls_as_the_grid_is_refined():
    """A fixed tolerance can be met by luck at one resolution.

    Marching cubes converges, so the error must fall when the same shape is
    represented on a finer grid. That is a statement about the method rather
    than about one mesh.
    """
    errors = []
    for cells, radius in ((24, 7.5), (48, 15.0), (96, 30.0)):
        report = surface_from_density(ball_field(cells, radius), SPACING,
                                      smoothing_passes=0)
        exact = exact_ball_volume(radius)
        errors.append(abs(report.enclosed_volume_m3 - exact) / exact)
    assert errors[0] > errors[1] > errors[2]


def test_the_three_volumes_are_not_the_same_quantity():
    """The trap this module exists to avoid.

    On a grey field the density integral is NOT the volume the surface
    encloses, and calling either "the volume" silently answers a different
    question than the caller asked. Asserted so the distinction cannot be
    quietly collapsed later.
    """
    field = ball_field(48, 15.0, ramp=1.0)
    report = surface_from_density(field, SPACING, smoothing_passes=0)

    assert report.field_integral_m3 > report.enclosed_volume_m3 * 1.1
    assert report.enclosed_vs_thresholded < 0.02


def test_a_nearly_binary_field_brings_the_integral_back_into_line():
    """The converse, which shows the difference above is the grey band and
    not a defect in either measure."""
    field = ball_field(48, 15.0, ramp=0.08)
    report = surface_from_density(field, SPACING, smoothing_passes=0)
    ratio = report.field_integral_m3 / report.enclosed_volume_m3
    assert 0.95 < ratio < 1.05


# ---------------------------------------------------------------- smoothing

def test_smoothing_barely_changes_the_volume():
    field = ball_field(48, 15.0)
    report = surface_from_density(field, SPACING, smoothing_passes=10)
    assert abs(report.volume_change_from_smoothing) < 0.01


def thin_plate() -> np.ndarray:
    """The worst case for smoothing: most surface per unit volume."""
    field = np.zeros((40, 40, 40))
    field[18:22, 5:35, 5:35] = 1.0
    return field


@pytest.mark.parametrize("passes", [5, 25, 100])
def test_the_default_holds_volume_better_than_taubins_correction(passes):
    """This pins a result that contradicted the reason the code was written.

    The textbook expectation is that Laplacian smoothing shrinks and Taubin's
    negative step corrects for it. Measured here it is the other way round:
    the correction overshoots and INFLATES the shape, by more than plain
    Laplacian shrinks it, at every pass count and on every shape tried. The
    default was changed to match the measurement, and this test exists so a
    future change back has to argue with numbers.
    """
    vertices, faces = isosurface(thin_plate(), SPACING)
    before = enclosed_volume_m3(vertices, faces)

    default = enclosed_volume_m3(smooth(vertices, faces, passes=passes), faces)
    taubin = enclosed_volume_m3(
        smooth(vertices, faces, passes=passes, mu=-0.53), faces)

    assert abs(default - before) < abs(taubin - before)
    assert taubin > before, "the correction inflates rather than preserving"


def test_the_inflation_grows_with_passes():
    """It is a systematic bias, not noise at one setting."""
    vertices, faces = isosurface(thin_plate(), SPACING)
    before = enclosed_volume_m3(vertices, faces)
    changes = [
        (enclosed_volume_m3(smooth(vertices, faces, passes=n, mu=-0.53),
                            faces) - before) / before
        for n in (5, 25, 100)]
    assert changes[0] < changes[1] < changes[2]


def test_smoothing_zero_passes_changes_nothing():
    field = ball_field(32, 10.0)
    vertices, faces = isosurface(field, SPACING)
    assert np.array_equal(smooth(vertices, faces, passes=0), vertices)


# ---------------------------------------------------------------- symmetry

def test_a_symmetric_field_gives_a_symmetric_surface():
    """A mirror symmetric input must not produce a lopsided part.

    Asymmetry here would mean the extraction or the smoothing has a directional
    bias, which is invisible on any single-shape check and obvious on a part.
    """
    field = ball_field(48, 15.0)
    report = surface_from_density(field, SPACING, smoothing_passes=10)
    centre = (48 - 1) / 2.0 * SPACING
    radius = 15.0 * SPACING

    # Two measures, because each catches something the other does not.
    for axis in range(3):
        offset = report.vertices[:, axis] - centre
        # Extent: the surface must reach equally far both ways. Marching cubes
        # triangulates the two halves of a cell differently, so exact bitwise
        # symmetry is not on offer, but this is tight.
        assert abs(offset.min() + offset.max()) / radius < 1e-5

    # Centroid, weighted by triangle AREA rather than counting vertices.
    # The mean vertex position is not the centre of a shape: vertex density
    # varies with how each cell was triangulated, so a perfectly symmetric
    # surface can still have an off-centre vertex mean.
    centroid = _area_weighted_centroid(report.vertices, report.faces)
    assert np.all(np.abs(centroid - centre) / radius < 1e-6)


def test_an_asymmetric_field_is_not_forced_symmetric():
    """The check above must be able to fail, or it tests nothing."""
    field = ball_field(48, 15.0)
    field[:, :, 30:] = 0.0        # cuts the LAST array axis
    report = surface_from_density(field, SPACING, smoothing_passes=0)
    centre = (48 - 1) / 2.0 * SPACING
    # marching_cubes returns vertices in array index order, so the cut axis is
    # column 2. Checking column 0 here would find the arm still symmetric and
    # pass for the wrong reason.
    offset = report.vertices[:, 2] - centre
    assert abs(offset.min() + offset.max()) > 1e-4


# ---------------------------------------------------------------- refusals

def test_an_empty_field_is_refused():
    """Returning an empty mesh would read as a part with no material rather
    than as a question that cannot be answered."""
    with pytest.raises(ValueError, match="no surface to extract"):
        isosurface(np.zeros((10, 10, 10)), SPACING)


def test_a_two_dimensional_field_is_refused():
    with pytest.raises(ValueError, match="must be 3D"):
        isosurface(np.ones((10, 10)), SPACING)


def test_laplacian_parameters_with_the_wrong_signs_are_refused():
    field = ball_field(24, 7.5)
    vertices, faces = isosurface(field, SPACING)
    with pytest.raises(ValueError, match="positive lambda and a negative mu"):
        smooth(vertices, faces, lam=-0.5, mu=-0.53)
    with pytest.raises(ValueError, match="positive lambda and a negative mu"):
        smooth(vertices, faces, lam=0.5, mu=0.53)


def test_negative_passes_are_refused():
    field = ball_field(24, 7.5)
    vertices, faces = isosurface(field, SPACING)
    with pytest.raises(ValueError, match="must not be negative"):
        smooth(vertices, faces, passes=-1)


def test_material_touching_the_grid_edge_still_closes():
    """Without padding, material at the boundary leaves an open surface, and
    an open surface has no enclosed volume worth reporting."""
    field = np.ones((16, 16, 16))
    report = surface_from_density(field, SPACING, smoothing_passes=0)
    assert report.enclosed_volume_m3 > 0.0
    # The level sits half a cell outside the outermost material cell on each
    # side, so a 16 cell block of material bounds 16 cells of space, not 15.
    expected = (16 * SPACING) ** 3
    assert abs(report.enclosed_volume_m3 - expected) / expected < 0.02
