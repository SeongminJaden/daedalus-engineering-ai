"""The faces the arm's links actually bolt to.

Before this layer the arm had two ends per link and nothing on them. A "9 mm
flange" is not a fastening: it has no bolt circle, no hole, and no statement
of which side of the actuator it meets. These tests hold the drawing values
that fixed that, and they hold the refusals, which matter more. Three things
the drawings do not print are recorded as absent here, and a design that fills
any of them in with a plausible number breaks a test.
"""

import pytest

from projects.manipulator.interfaces import (AK80_9_HOUSING, AK80_9_OUTPUT,
                                             AK80_64_HOUSING, AK80_64_OUTPUT,
                                             ISO_273_MEDIUM_M, bolt_holes,
                                             clock_uncertainty_check,
                                             dowel_holes, face_for,
                                             link_interfaces,
                                             unresolved_features)
from projects.manipulator.links import mounting_holes
from projects.manipulator.spec import SPEC

DRIVES = {"j1_base_yaw": "cubemars_ak80_64_kv80",
          "j2_shoulder": "cubemars_ak80_64_kv80",
          "j3_elbow": "cubemars_ak80_64_kv80",
          "j4_wrist_roll": "cubemars_ak80_9_v3",
          "j5_wrist_pitch": "cubemars_ak80_9_v3",
          "j6_tool_roll": "kollmorgen_tbm_6013_a"}


# --- what the drawings print --------------------------------------------------

def test_the_two_faces_of_one_actuator_are_not_the_same_pattern():
    """The reason both ends of a link had to be modelled separately.

    On the AK80-64 the housing is 8-M3 on an 85 mm circle and the output is
    8-M3 on an 89 mm circle. Treating a link's two ends as one pattern puts
    every bolt on one end 2 mm off its hole.
    """
    assert AK80_64_HOUSING.largest_bolt_circle_m() == 0.085
    assert AK80_64_OUTPUT.largest_bolt_circle_m() == 0.089
    assert AK80_64_HOUSING.largest_bolt_circle_m() != AK80_64_OUTPUT.largest_bolt_circle_m()


def test_the_ak80_64_output_carries_two_bolt_circles():
    """8-M3 at 89 and 6-M4 at 28, so a link face needs 14 holes, not 8."""
    threads = {(p.thread, p.count, p.bolt_circle_m)
               for p in AK80_64_OUTPUT.patterns}
    assert threads == {("M3", 8, 0.089), ("M4", 6, 0.028)}
    assert len(bolt_holes(AK80_64_OUTPUT)) == 14


def test_the_printed_thread_depths_are_the_ones_on_the_drawing():
    depths = {p.thread: p.thread_depth_m for p in AK80_64_OUTPUT.patterns}
    assert depths == {"M3": 0.010, "M4": 0.008}
    assert AK80_64_HOUSING.patterns[0].thread_depth_m == 0.007


def test_the_link_side_is_a_clearance_hole_never_a_thread():
    """The assumption that was wrong. The threads are in the ACTUATOR, so the
    link takes an ISO 273 medium clearance hole and its thickness is a grip
    length, not an engagement length."""
    assert ISO_273_MEDIUM_M == {"M3": 0.0034, "M4": 0.0045}
    for face in (AK80_64_HOUSING, AK80_64_OUTPUT, AK80_9_HOUSING, AK80_9_OUTPUT):
        for pattern in face.patterns:
            assert pattern.clearance_hole_m > float(pattern.thread[1:]) / 1000.0


def test_a_standard_bolt_fits_the_printed_depth_at_this_flange_thickness():
    """9 mm of flange plus a catalogue bolt length has to engage at least 1.5
    diameters and still leave the hole's bottom clear. M3 x 14 into the 7 mm
    housing thread engages 5 mm with 2 mm spare, and M3 x 16 into the 10 mm
    output thread engages 7 mm with 3 mm spare. A bolt that bottoms out is
    torqued against the hole, not against the joint."""
    flange = SPEC.flange_thickness_m
    for face, length in ((AK80_64_HOUSING, 0.014), (AK80_64_OUTPUT, 0.016)):
        depth = face.patterns[0].thread_depth_m
        engagement = length - flange
        assert 1.5 * 0.003 <= engagement < depth - 0.0005, (
            f"{face.face}: an M3 x {length * 1000:.0f} engages "
            f"{engagement * 1000:.1f} mm into a {depth * 1000:.0f} mm hole")


# --- what they do not print ---------------------------------------------------

def test_the_ak80_9_prints_no_thread_depth_and_none_is_invented():
    for pattern in AK80_9_OUTPUT.patterns + AK80_9_HOUSING.patterns:
        assert pattern.thread_depth_m is None
    assert any("no thread depth is printed" in m
               for m in unresolved_features(AK80_9_OUTPUT))


def test_the_dowel_position_is_not_printed_so_no_dowel_hole_is_cut():
    """The AK80-9 prints a dowel diameter and depth and NOT an angle, and its
    3D model does not show one either. A hole at a guessed angle is worse than
    no hole: it looks located and is not. The AK80-64's angles WERE measured,
    so that one is cut."""
    assert AK80_9_OUTPUT.dowel_diameter_m == 0.003
    assert AK80_9_OUTPUT.dowel_angles_deg == ()
    assert any("ANGULAR POSITION" in m for m in unresolved_features(AK80_9_OUTPUT))
    holes, unresolved = mounting_holes(SPEC, 3, DRIVES, 0.0785)
    assert not [h for h in holes if h["kind"] == "dowel"]
    assert any("ANGULAR POSITION" in m for m in unresolved)


def test_only_the_ak80_64_output_prints_a_central_bore():
    assert AK80_64_OUTPUT.central_bore_m == 0.021
    assert AK80_9_OUTPUT.central_bore_m is None
    assert AK80_64_HOUSING.central_bore_m is None


def test_a_drive_with_no_drawing_leaves_the_link_unfastenable():
    """The frameless motor at the wrist publishes no outline, so the tool
    flange has no pattern at either end. That is reported, not filled in."""
    assert face_for("kollmorgen_tbm_6013_a", "output") is None
    holes, unresolved = mounting_holes(SPEC, 5, DRIVES, 0.0785)
    assert holes == []
    assert any("no drawing was read" in m for m in unresolved)
    assert any("tool plate" in m for m in unresolved)


# --- which end meets what -----------------------------------------------------

def test_a_link_meets_its_own_joint_at_the_output_and_the_next_at_the_housing():
    rows = link_interfaces([l.name for l in SPEC.links()],
                           [j.name for j in SPEC.joints()], DRIVES)
    assert rows[0]["proximal_mates_to"] == "output flange"
    assert rows[0]["distal_mates_to"] == "housing"
    assert rows[0]["proximal_face"] is AK80_64_OUTPUT
    assert rows[0]["distal_face"] is AK80_64_HOUSING
    assert rows[-1]["distal_mates_to"] == "tool plate"
    assert rows[-1]["distal_face"] is None


def test_the_forearm_spans_two_different_actuators():
    """It is driven by an AK80-64 and carries an AK80-9, so its two ends do
    not even come from the same drawing."""
    rows = link_interfaces([l.name for l in SPEC.links()],
                           [j.name for j in SPEC.joints()], DRIVES)
    forearm = next(r for r in rows if r["link"] == "forearm")
    assert forearm["proximal_face"].source.id == "cubemars_ak80_64_2d"
    assert forearm["distal_face"].source.id == "cubemars_ak80_9_v3_2d"


def test_every_hole_lands_inside_the_design_domain():
    """A bolt circle wider than the link is a pattern that cannot be drilled."""
    holes, _ = mounting_holes(SPEC, 1, DRIVES, 0.2)
    half = 0.5 * 0.098
    for hole in holes:
        reach = max(abs(hole["y_m"]), abs(hole["z_m"])) + 0.5 * hole["diameter_m"]
        assert reach <= half, f"{hole['kind']} reaches {reach * 1000:.1f} mm"


# --- the clock ----------------------------------------------------------------

def test_the_housing_pattern_is_clocked_half_a_pitch_from_the_output():
    """The measurement that decides whether an assembled arm bolts together.

    On the AK80-64 the housing holes sit 22.5 degrees from the output holes,
    which is exactly half the 45 degree pitch. A link whose two ends carry the
    same clock has every bolt on one end landing between two holes.
    """
    output = AK80_64_OUTPUT.patterns[0]
    housing = AK80_64_HOUSING.patterns[0]
    assert output.clock_deg == 0.0
    assert housing.clock_deg == 22.5
    assert housing.clock_deg == pytest.approx(0.5 * 360.0 / housing.count)


def test_the_dowels_share_the_outer_bolts_angles_at_a_smaller_radius():
    """What makes them locating features rather than extra holes."""
    holes = dowel_holes(AK80_64_OUTPUT)
    assert [h["angle_deg"] for h in holes] == [0.0, 180.0]
    assert AK80_64_OUTPUT.dowel_bolt_circle_m == 0.028
    outer = {round(h["y_m"], 9) for h in bolt_holes(AK80_64_OUTPUT)
             if h["bolt_circle_m"] == 0.089}
    assert round(0.5 * 0.089, 9) in outer      # an M3 hole also sits at 0 deg


def test_a_dowel_with_no_measured_angle_is_not_cut():
    assert dowel_holes(AK80_9_OUTPUT) == []
    holes, _ = mounting_holes(SPEC, 3, DRIVES, 0.0785)
    assert not [h for h in holes if h["kind"] == "dowel"]


def test_the_ak80_9_clock_uncertainty_exceeds_what_the_bolt_can_take_up():
    """A tolerance the fastener cannot absorb is not a loose tolerance, it is
    an assembly that does not go together.

    The AK80-9's 3D model is a converted one and its angles scatter by about
    1.5 degrees. On the 85 mm circle that is 1.11 mm of hole movement. An M3
    bolt in an ISO 273 medium 3.4 mm hole can move 0.20 mm. So the clock has
    to be measured on the real part before this joint is drilled.
    """
    rows = clock_uncertainty_check(AK80_9_OUTPUT)
    assert rows and all(r["verdict"] == "NOT ABSORBED" for r in rows)
    m3 = next(r for r in rows if r["thread"] == "M3")
    assert m3["clock_offset_mm"] == pytest.approx(1.11, abs=0.02)
    assert m3["clearance_allowance_mm"] == pytest.approx(0.20, abs=0.01)
    assert m3["clock_offset_mm"] > 5.0 * m3["clearance_allowance_mm"]


def test_the_ak80_64_clock_has_no_uncertainty_to_absorb():
    """Its model lands on integers, so nothing is estimated and the check has
    nothing to report."""
    assert clock_uncertainty_check(AK80_64_OUTPUT) == []
    assert clock_uncertainty_check(AK80_64_HOUSING) == []


def test_the_drawing_and_the_model_disagree_and_the_entry_says_so():
    """The 2D sheet marks 15 degrees on the housing face and the model
    measures 22.5. An entry that quietly picked one would hide a conflict
    someone has to resolve against the real part."""
    assert any("do not agree" in note for note in AK80_64_HOUSING.notes)


# --- where the drive sits ------------------------------------------------------

def test_a_joint_axis_is_read_in_the_link_own_frame():
    """The base column runs up, so the base yaw axis runs ALONG it.

    The specification states axes in the arm's frame, where y is up. A link's
    file has x along itself. Reading the arm's components as the link's made
    the base yaw drive a cross axis cylinder when it is a coaxial one, and
    drove a 98 mm pocket sideways through a part the motor runs along.
    """
    from projects.manipulator.links import local_axis

    base = local_axis(SPEC, 0, SPEC.joints()[0].axis)
    assert abs(base[0]) == pytest.approx(1.0)
    shoulder = local_axis(SPEC, 0, SPEC.joints()[1].axis)
    assert abs(shoulder[2]) == pytest.approx(1.0)
    # Every other link runs along x, so its axes pass through unchanged.
    for index in range(1, len(SPEC.links())):
        arm = SPEC.joints()[index].axis
        assert local_axis(SPEC, index, arm) == pytest.approx(arm)


def test_the_face_insets_are_the_printed_ones():
    """Where a motor sits along its own axis, from the drawings' sections."""
    assert AK80_64_OUTPUT.face_inset_m == 0.008
    assert AK80_64_HOUSING.face_inset_m == 0.0112
    assert AK80_9_OUTPUT.face_inset_m == 0.003
    assert AK80_9_HOUSING.face_inset_m == 0.011


def test_a_link_gets_a_pocket_for_the_drive_it_carries_as_well_as_its_own():
    """Two drives touch a link and only one used to be cut for.

    The base column is driven by the base yaw and CARRIES the shoulder, and
    the shoulder's 98 mm body has to fit somewhere.
    """
    from projects.manipulator.links import link_domain

    sections = {name: type("S", (), {"outer_height_m": 0.098,
                                     "outer_width_m": 0.098})()
                for name in [l.name for l in SPEC.links()]}
    built, reason = link_domain(SPEC, 0, DRIVES, sections=sections)
    assert built is not None, reason
    mesh, solid, void, span, height, width, note = built
    assert "its own drive" in note and "the drive it carries" in note
    centroids = mesh.element_centroids()
    near = void & (centroids[:, 0] < 0.5 * span)
    far = void & (centroids[:, 0] > 0.5 * span)
    assert near.any() and far.any(), "one end has no pocket"
