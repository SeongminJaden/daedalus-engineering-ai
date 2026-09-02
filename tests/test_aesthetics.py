"""Shape metrics, and the guard that keeps them off the evidence ladder.

Two things are being checked. That the metrics are correct, against shapes
whose values are known exactly. And that no value any of them can take is able
to rescue a design that fails a physical check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geometry.aesthetics import (PREFERENCE_IS_NOT_EVIDENCE,
                                 SPHERE_COMPACTNESS, compactness,
                                 measure_shape)
from geometry.aesthetics.metrics import (CUBE_COMPACTNESS,
                                         dihedral_roughness_rad,
                                         mirror_asymmetry_m)
from geometry.surfacing import isosurface, smooth
from integration import DesignEntry, MultiDesignReview, RankBy
from tests.test_design_review import build_entry

EDGE = 0.05


def cube_mesh(a: float = EDGE):
    vertices = np.array([[0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
                         [0, 0, a], [a, 0, a], [a, a, a], [0, a, a]],
                        dtype=float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return vertices, faces


def tetrahedron_mesh(s: float = 0.03):
    vertices = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                        dtype=float) * s
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


def ball_field(n: int = 56, radius: float = 18.0) -> np.ndarray:
    centre = (n - 1) / 2.0
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    signed = radius - np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2
                              + (zz - centre) ** 2)
    return np.clip(signed / radius + 0.5, 0.0, 1.0)


# ------------------------------------------------------- exact anchors

def test_a_cube_has_the_compactness_of_a_cube():
    """36 pi V^2 / A^3 with V = a^3 and A = 6 a^2 gives pi/6, exactly."""
    vertices, faces = cube_mesh()
    assert compactness(vertices, faces) == pytest.approx(CUBE_COMPACTNESS,
                                                         rel=1e-12)


def test_a_regular_tetrahedron_matches_its_closed_form():
    """A second exact anchor, and a shape with sharper corners than a cube."""
    s = 0.03
    vertices, faces = tetrahedron_mesh(s)
    volume = (8.0 / 3.0) * s ** 3
    area = 8.0 * math.sqrt(3.0) * s ** 2
    expected = 36.0 * math.pi * volume ** 2 / area ** 3
    assert compactness(vertices, faces) == pytest.approx(expected, rel=1e-12)


def test_a_sphere_approaches_the_upper_bound():
    """No closed surface can exceed one, and only a sphere reaches it."""
    vertices, faces = isosurface(ball_field(), 0.001)
    value = compactness(vertices, faces)
    assert value < SPHERE_COMPACTNESS + 1e-9
    assert value > 0.99


def test_the_ordering_of_shapes_is_the_expected_one():
    """Sphere, then cube, then tetrahedron. If this inverted, the measure
    would be reporting something other than compactness."""
    sphere = compactness(*isosurface(ball_field(), 0.001))
    cube = compactness(*cube_mesh())
    tetrahedron = compactness(*tetrahedron_mesh())
    assert sphere > cube > tetrahedron


# ------------------------------------------------------------ roughness

def test_a_faceted_surface_is_rougher_than_a_smooth_one():
    sphere_vertices, sphere_faces = isosurface(ball_field(), 0.001)
    cube_vertices, cube_faces = cube_mesh()
    assert (dihedral_roughness_rad(cube_vertices, cube_faces)
            > dihedral_roughness_rad(sphere_vertices, sphere_faces))


def test_smoothing_reduces_roughness():
    """The metric has to respond to the operation it is meant to describe."""
    field = np.zeros((40, 40, 40))
    field[8:32, 8:32, 8:32] = 1.0
    vertices, faces = isosurface(field, 0.001)
    before = dihedral_roughness_rad(vertices, faces)
    after = dihedral_roughness_rad(smooth(vertices, faces, passes=12), faces)
    assert after < before


def test_roughness_is_a_spread_not_a_mean():
    """A well resolved sphere has a non zero MEAN dihedral angle by
    construction, and that is curvature rather than roughness. Reporting the
    mean would call a smooth ball rough."""
    vertices, faces = isosurface(ball_field(), 0.001)
    assert dihedral_roughness_rad(vertices, faces) < 0.05


# ------------------------------------------------------------- symmetry

def test_a_symmetric_shape_reports_almost_no_asymmetry():
    vertices, faces = isosurface(ball_field(), 0.001)
    assert mirror_asymmetry_m(vertices, faces, axis=0) < 1e-6


def test_a_cut_shape_reports_asymmetry():
    field = ball_field()
    field[:, :, 34:] = 0.0
    vertices, faces = isosurface(field, 0.001)
    assert mirror_asymmetry_m(vertices, faces, axis=2) > 1e-4


# ------------------------------------ the guard: preference is not evidence

def test_the_metrics_carry_the_label_that_they_are_not_evidence():
    metrics = measure_shape(*cube_mesh())
    assert metrics.note == PREFERENCE_IS_NOT_EVIDENCE
    assert "cannot overturn a feasibility verdict" in metrics.note


def test_the_most_beautiful_failing_design_never_outranks_a_passing_one():
    """The guard that matters, and it is structural rather than remembered.

    A failing design is inadmissible, so it is never ranked at all. No value
    of the form score can lift it, because the ranking never sees it.
    """
    review = MultiDesignReview([
        build_entry("ugly_but_sound", factor=1.4, gaps=0, mass=9.0, cost=99.0),
        build_entry("beautiful_but_broken", factor=0.2, gaps=0, mass=0.5,
                    cost=1.0, failing=True)])
    review.entries[1] = DesignEntry(
        name="beautiful_but_broken", verdict=review.entries[1].verdict,
        mass_kg=0.5, cost_usd=1.0, form_score=1e9)
    review.entries[0] = DesignEntry(
        name="ugly_but_sound", verdict=review.entries[0].verdict,
        mass_kg=9.0, cost_usd=99.0, form_score=-1e9)

    ranked = [e.name for e in review.ranked(RankBy.FORM)]
    assert ranked == ["ugly_but_sound"]
    assert review.best(RankBy.FORM).name == "ugly_but_sound"
    assert "beautiful_but_broken" in [e.name for e in review.rejected]


def test_form_ranks_admissible_designs_against_each_other():
    """It does do its job, among candidates that are all allowed."""
    plain = build_entry("plain", factor=2.0, gaps=0, mass=1.0, cost=1.0)
    shapely = build_entry("shapely", factor=2.0, gaps=0, mass=1.0, cost=1.0)
    review = MultiDesignReview([
        DesignEntry("plain", plain.verdict, 1.0, 1.0, form_score=0.1),
        DesignEntry("shapely", shapely.verdict, 1.0, 1.0, form_score=0.9)])
    assert [e.name for e in review.ranked(RankBy.FORM)] == ["shapely", "plain"]


def test_form_does_not_enter_the_physical_criteria():
    """Changing only the form score must not move a margin or mass ranking."""
    a = build_entry("a", factor=3.0, gaps=0, mass=1.0, cost=1.0)
    b = build_entry("b", factor=2.0, gaps=0, mass=2.0, cost=2.0)
    before = MultiDesignReview([a, b])
    after = MultiDesignReview([
        DesignEntry("a", a.verdict, 1.0, 1.0, form_score=-5.0),
        DesignEntry("b", b.verdict, 2.0, 2.0, form_score=+5.0)])

    for criterion in (RankBy.GOVERNING_MARGIN, RankBy.MASS, RankBy.COST):
        assert ([e.name for e in before.ranked(criterion)]
                == [e.name for e in after.ranked(criterion)])


def test_the_evidence_ladder_has_no_rung_for_preference():
    """The structural half of the guard.

    If a preference level existed on the ladder, a shape could earn confidence
    by being round. There is deliberately no such rung, and this fails if one
    is ever added without the conversation that should accompany it.
    """
    from brain.semantic.evidence import EvidenceLevel

    names = {level.name for level in EvidenceLevel}
    assert names == {"UNVERIFIED", "SIMULATED", "REPEATED", "HIGH_CONFIDENCE",
                     "EXPERIMENTALLY_VALIDATED"}


def test_no_aesthetic_capability_is_registered():
    """Shape metrics are not a capability the engine offers as analysis.

    They answer no physical question, so registering them beside stress and
    fatigue would put a preference where a verdict belongs.
    """
    from nodes.roster import build_roster

    names = {c.name for c in build_roster().all()}
    assert not any("aesthetic" in n or "beauty" in n for n in names)
