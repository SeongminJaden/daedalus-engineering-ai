"""The eight part classes added for the dataset, checked like the first five.

Every family ships with a closed form volume and the features its parameters
put there, and the engine refuses a part that disagrees with either. So the
test that matters is simply that the engine accepts random draws from each
family, which it cannot do unless the closed form and the recogniser agree
with the B-rep. Two of the eight failed that on the first try: the housing's
cavity was misplaced, and the link's rounded ends are fillets to the
recogniser, which the family now says.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.part_dataset import FAMILIES, make_part, sample_parameters
from core.part_dataset.labeller import labelling_available
from core.part_dataset.classify import UNKNOWN, rule_classify
from core.part_dataset.descriptors import describe_step
from core.part_dataset.families import ORIGINAL_FAMILIES
from geometry.cad_export.kernel import kernel_available
from nodes import step_analyzer as sa

requires_cad = pytest.mark.skipif(not (kernel_available() and sa.is_available()),
                                  reason="build123d and OCP are required")

NEW_FAMILIES = ("bracket", "flange", "housing", "keyed_shaft", "gear_blank",
                "link", "mount", "ribbed_plate")


def test_thirteen_families_five_of_them_original():
    assert len(FAMILIES) == 13
    assert set(ORIGINAL_FAMILIES) | set(NEW_FAMILIES) == set(FAMILIES)
    for name in NEW_FAMILIES:
        fam = FAMILIES[name]
        assert fam.description and fam.bounds
        rng = np.random.default_rng(0)
        for _ in range(50):
            assert fam.admissible(sample_parameters(fam, rng))


@requires_cad
@pytest.mark.parametrize("name", NEW_FAMILIES)
def test_the_engine_accepts_the_family_against_its_own_ground_truth(name, tmp_path):
    """Volume to 1e-6 of the closed form and exactly the features declared,
    or the engine would have refused the part."""
    fam = FAMILIES[name]
    rng = np.random.default_rng(7)
    for _ in range(3):
        params = sample_parameters(fam, rng)
        record, _ = make_part(fam, params, tmp_path, labelled=False)
        assert record.geometry.volume_m3 == pytest.approx(fam.volume_m3(params),
                                                          rel=1e-6)
        expected = fam.expected_features(params)
        holes = [f for f in record.features if f["kind"] == "hole"]
        assert len(holes) == expected.hole_count


@requires_cad
@pytest.mark.parametrize("name", NEW_FAMILIES)
def test_the_classifier_says_unknown_for_a_family_it_was_not_built_on(name,
                                                                        tmp_path):
    """The rules cover five families and say so. Until they are extended and
    measured on these eight, UNKNOWN is the correct answer, and a rule that
    guessed one of the five would be a wrong answer."""
    fam = FAMILIES[name]
    params = sample_parameters(fam, np.random.default_rng(3))
    record, _ = make_part(fam, params, tmp_path, labelled=False)
    result = rule_classify(describe_step(tmp_path / f"{record.part_id}.step")[0])
    assert result.family == UNKNOWN, (name, result.family)


def test_the_keyed_shaft_segment_is_the_removed_area():
    """The flat removes a circular segment; the closed form is the segment
    formula, checked here against a numeric integral."""
    from core.part_dataset.families import _segment_area

    R, d = 0.01, 0.003
    ys = np.linspace(R - d, R, 200001)
    numeric = np.trapezoid(2.0 * np.sqrt(np.clip(R * R - ys * ys, 0.0, None)), ys)
    assert _segment_area(R, d) == pytest.approx(numeric, rel=1e-6)


#: The mount that seed 0 draws first in the round trip test, at full precision.
#: Rounded to five decimals the same part meshes but CalculiX returns nothing at
#: all three sizes, which says how thin the margin is on this family.
MOUNT_SEED0 = {"length_m": 0.18852952396752043, "width_m": 0.05395172450630633,
               "thickness_m": 0.01262295052195499,
               "boss_radius_m": 0.02216809052718501,
               "boss_height_m": 0.03758560605934043,
               "hole_radius_m": 0.003688682090203389}


@requires_cad
@pytest.mark.skipif(not labelling_available(), reason="gmsh and CalculiX required")
def test_a_boundary_mesh_failure_is_retried_finer_like_a_solver_rejection(tmp_path):
    """The mount drawn first at seed 0 does not surface-mesh at the coarse
    size (12.57 mm, overlapping facets on the plate face around the boss)
    and does at the next retry size. Before this the labeller let the gmsh
    exception through and the round trip test lost the whole dataset.
    Measured over twenty mount draws at seed 1: none refused, five retried."""
    from core.materials import get_material
    from core.part_dataset.labeller import LoadCase, cantilever_labels, mesh_sizes_for
    from nodes import gmsh_node as gm

    record, _ = make_part(FAMILIES["mount"], MOUNT_SEED0, tmp_path, labelled=False)
    step = tmp_path / f"{record.part_id}.step"
    coarse, _fine = mesh_sizes_for(record.geometry.bounding_box_m)
    with pytest.raises(Exception, match="overlapping facets"):
        gm.tetrahedral_mesh_from_step(str(step), coarse, order=2)

    report = cantilever_labels(step, record.geometry.volume_m3,
                               record.geometry.bounding_box_m,
                               get_material("al_7075_t6"), LoadCase(direction=1))
    text = json.dumps(report.labels)
    assert f"{coarse * 0.7 * 1e3:.2f} mm" in text, text
    assert "was used" in text


@requires_cad
@pytest.mark.skipif(not labelling_available(), reason="gmsh and CalculiX required")
def test_a_part_the_first_ladder_refused_is_labelled_by_the_deeper_one():
    """The 70 refusals of the first industrial run were all curved parts whose
    quadratic mid side nodes CalculiX rejected. A nine part sample across all
    seven affected families solved within four steps of the 0.7 ladder, the
    deepest at 0.7 cubed, so the ladder is four steps deep now instead of two.

    Re-running the two worst cells with it: flange 100 labelled and 0 refused
    against 93 and 7, at 1104 s against 785; plate with holes the same, at
    about 1.5 times the time. The cost is paid only by the parts that need it.
    """
    from core.materials import get_material
    from core.part_dataset.labeller import (MAX_RETRIES, RETRY_FACTOR, LoadCase,
                                            cantilever_labels, mesh_sizes_for)

    assert MAX_RETRIES == 4
    fam = FAMILIES["flange"]
    rng = np.random.default_rng(0)
    params = None
    for _ in range(40):
        candidate = sample_parameters(fam, rng)
        if candidate["bolt_count"] >= 6:
            params = candidate
            break
    assert params is not None
    import tempfile
    directory = Path(tempfile.mkdtemp())
    record, _ = make_part(fam, params, directory, labelled=False)
    step = directory / f"{record.part_id}.step"
    report = cantilever_labels(step, record.geometry.volume_m3,
                               record.geometry.bounding_box_m,
                               get_material("al_7075_t6"), LoadCase(direction=1))
    assert report.labels["tip_deflection_m"]["value"] != 0.0
    coarse, fine = mesh_sizes_for(record.geometry.bounding_box_m)
    deepest = fine * RETRY_FACTOR ** MAX_RETRIES
    assert deepest < fine * 0.25


def test_a_sensitivity_between_two_nearly_equal_meshes_is_not_reported():
    """Deep retries can drive the coarse control mesh down onto the fine one.
    Two nearly identical meshes agree by construction, and reporting that as a
    mesh sensitivity would turn a failure to refine into a quality signal."""
    from core.part_dataset.labeller import SENSITIVITY_SIZE_RATIO, _sensitivity

    assert _sensitivity(1.0, 1.1) == pytest.approx(0.1)
    assert _sensitivity(1.0, 1.1, (0.010, 0.005)) == pytest.approx(0.1)
    assert _sensitivity(1.0, 1.1, (0.0055, 0.005)) is None
    assert SENSITIVITY_SIZE_RATIO == 1.25
