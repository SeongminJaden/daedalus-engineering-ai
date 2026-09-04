"""The two parts that hold the arm to the world.

These were listed as gaps for several revisions of this design: the base
mount that holds the first drive's housing to the floor, and the tool plate
that holds the payload on the last drive's output. Between them they carry
everything the arm does, and neither existed. These tests hold what they are
now designed against, and which half of each part is sourced and which half
is a choice.
"""

import numpy as np
import pytest

from projects.manipulator.mounts import (FLOOR_BOLT_SQUARE_M,
                                         TOOL_BOLT_SQUARE_M,
                                         base_mount_loads, generate_mount,
                                         tool_plate_loads, _plate_domain,
                                         _world_holes)
from projects.manipulator.spec import SPEC


def test_the_overturning_moment_is_what_sizes_the_base_mount():
    """Not the weight. A 3 kg payload at 600 mm makes 17.6 N m, and the arm's
    own mass at half that reach adds its share; the whole thing weighs 76 N.
    A mount designed for the weight alone would be designed for the smaller
    of the two loads."""
    loads = base_mount_loads(arm_mass_kg=4.8, payload_kg=3.0,
                             base_yaw_peak_nm=0.475, spec=SPEC)
    assert loads["vertical_n"] == pytest.approx(76.5, abs=0.5)
    assert loads["overturning_nm"] == pytest.approx(31.8, abs=0.5)
    payload_alone = 3.0 * 9.80665 * SPEC.reach_m
    assert payload_alone == pytest.approx(17.65, abs=0.05)
    assert loads["overturning_nm"] > payload_alone


def test_the_tool_plate_carries_the_payload_at_its_own_extent():
    """The payload is not a point. It has a stated size, and hanging it off
    half that size is a moment the plate has to take."""
    loads = tool_plate_loads(3.0, 0.0088, SPEC)
    assert loads["vertical_n"] == pytest.approx(29.42, abs=0.05)
    assert loads["overturning_nm"] == pytest.approx(
        29.42 * 0.5 * SPEC.payload_extent_m, rel=1e-3)
    assert loads["yaw_reaction_nm"] == 0.0088


def test_both_bolted_faces_are_held_solid():
    """A mount has an interface at each end and both have to survive."""
    mesh, solid = _plate_domain(0.060, 0.140, 0.140, 0.009)
    centroids = mesh.element_centroids()
    assert solid[centroids[:, 0] < 0.009].all()
    assert solid[centroids[:, 0] > 0.051].all()
    assert not solid[(centroids[:, 0] > 0.02) & (centroids[:, 0] < 0.04)].any()


def test_the_world_side_pattern_is_a_square_of_four_and_is_a_choice():
    """Nothing in this specification says what the arm stands on or what tool
    it holds, so these two patterns are CHOSEN. They are the first thing a
    real installation replaces, and the design says so rather than presenting
    them as sourced."""
    floor = _world_holes("floor", 0.060, 0.009)
    assert len(floor) == 4
    assert {round(abs(h["y_m"]), 6) for h in floor} == {
        round(0.5 * FLOOR_BOLT_SQUARE_M, 6)}
    assert {h["thread"] for h in floor} == {"M8"}
    tool = _world_holes("tool", 0.040, 0.009)
    assert {round(abs(h["z_m"]), 6) for h in tool} == {
        round(0.5 * TOOL_BOLT_SQUARE_M, 6)}
    assert {h["thread"] for h in tool} == {"M5"}


def test_a_mount_for_a_drive_with_no_drawing_is_refused():
    """The frameless motor publishes no pattern, so nothing can be designed
    to bolt to it. That is a refusal, not a plate with invented holes."""
    design = generate_mount("nowhere", "kollmorgen_tbm_6013_a", "housing",
                            "floor", base_mount_loads(4.8, 3.0, 0.5),
                            0.060, 0.140, 0.140,
                            __import__("pathlib").Path("/tmp/unused"),
                            iterations=1)
    assert not design.generated
    assert "no drawing was read" in design.reason


@pytest.mark.slow
def test_the_base_mount_generates_and_keeps_both_its_faces():
    """The volume fraction matters here in a way it does not for a link. Two
    9 mm plates in a 60 mm part are already 30 percent of the domain, so at a
    volume fraction of 0.3 there is nothing left to join them with and the
    extraction drops one whole face. Measured: it fails at 0.3 and generates
    at 0.45."""
    import tempfile
    from pathlib import Path

    loads = base_mount_loads(4.8, 3.0, 0.475, SPEC)
    with tempfile.TemporaryDirectory() as directory:
        thin = generate_mount("base_mount", "cubemars_ak80_64_kv80",
                              "housing", "floor", loads, 0.060, 0.140, 0.140,
                              Path(directory), iterations=12,
                              volume_fraction=0.30)
        assert not thin.generated
        assert "dropped" in thin.reason and "held solid" in thin.reason

        made = generate_mount("base_mount", "cubemars_ak80_64_kv80",
                              "housing", "floor", loads, 0.060, 0.140, 0.140,
                              Path(directory), iterations=12,
                              volume_fraction=0.45)
        assert made.generated, made.reason
        assert made.watertight
        assert made.mass_kg > 0.0
        assert any("CHOSEN" in note for note in made.notes)
        assert made.unresolved
