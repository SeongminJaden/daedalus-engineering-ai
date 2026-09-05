"""The faces the arm's links actually bolt to.

Before this layer the arm had two ends per link and nothing on them. A "9 mm
flange" is not a fastening: it has no bolt circle, no hole, and no statement
of which side of the actuator it meets. These tests hold the drawing values
that fixed that, and they hold the refusals, which matter more. Three things
the drawings do not print are recorded as absent here, and a design that fills
any of them in with a plausible number breaks a test.
"""

import numpy as np
import pytest

from projects.manipulator.interfaces import (AK80_9_HOUSING, AK80_9_OUTPUT,
                                             AK80_64_HOUSING, AK80_64_OUTPUT,
                                             ISO_273_MEDIUM_M, bolt_holes,
                                             clock_uncertainty_check,
                                             dowel_holes, face_for,
                                             link_interfaces,
                                             unresolved_features)
from projects.manipulator.links import mounting_holes
from projects.manipulator.stages import MINIMUM_DISC_LIGAMENT_M
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


def test_the_housing_clock_relationship_is_per_part_not_a_rule():
    """The AK80-64's housing ring is half a pitch from its output ring and the
    AK60-6's is aligned with it. A rule inferred from one part would put every
    bolt on the other in the wrong place, which is why the relationship is
    stored per part rather than derived."""
    from projects.manipulator.interfaces import AK60_6_HOUSING, AK60_6_OUTPUT

    assert AK80_64_HOUSING.patterns[0].clock_deg == 22.5
    assert AK80_64_OUTPUT.patterns[0].clock_deg == 0.0
    assert AK60_6_HOUSING.patterns[0].clock_deg == 0.0
    assert AK60_6_OUTPUT.patterns[0].clock_deg == 0.0


def test_the_same_uncertainty_is_absorbed_on_one_circle_and_not_the_other():
    """What a clock tolerance costs depends on the radius it acts at.

    The AK60-6's model scatters by about a degree. On its 68 mm outer ring
    that is 0.59 mm of hole movement against the 0.20 mm an M3 clearance hole
    allows, and on its 20 mm inner ring it is 0.17 mm, which the same hole
    takes up. One pattern on one part, two different answers.
    """
    from projects.manipulator.interfaces import AK60_6_OUTPUT

    rows = clock_uncertainty_check(AK60_6_OUTPUT)
    outer = next(r for r in rows if r["clock_offset_mm"] > 0.3)
    inner = next(r for r in rows if r["clock_offset_mm"] < 0.3)
    assert outer["verdict"] == "NOT ABSORBED"
    assert inner["verdict"] == "absorbed"
    assert outer["clock_offset_mm"] == pytest.approx(0.59, abs=0.02)
    assert inner["clock_offset_mm"] == pytest.approx(0.17, abs=0.02)


def test_a_drive_with_no_drawing_cannot_be_selected():
    """The rule that fixed the tool roll. A frameless motor is the lightest
    thing in the catalogue and cannot be placed in an assembly or bolted to,
    so it is not a candidate however well it meets the torque."""
    from projects.manipulator.arm import build_arm
    from projects.manipulator.loop import run_loop
    from projects.manipulator.stages import (drivetrain_stage, dynamics_stage,
                                             reflected_inertia_stage)

    loop = run_loop()
    arm = build_arm(loop.data["final_sections"], SPEC)
    dynamics = dynamics_stage(arm, SPEC, samples=40)
    first = drivetrain_stage(dynamics, SPEC, {})
    inertias = {row["joint"]: row.get("load_inertia_kg_m2")
                for row in reflected_inertia_stage(arm, first, SPEC).rows}
    drivetrain = drivetrain_stage(dynamics, SPEC, inertias)

    for row in drivetrain.rows:
        if row.get("selected"):
            assert face_for(str(row["selected"]), "output") is not None, (
                f"{row['joint']} selected {row['selected']}, which publishes "
                f"no mounting pattern")

    # The rule has to actually reject something, or it is decoration. It is
    # checked by finding the candidates it struck out rather than by naming
    # one, because which part it catches depends on the arm: at the tool roll
    # the frameless motor is now excluded on inertia ratio before the drawing
    # rule is reached, and the rule still stands behind that.
    struck = [c for rows in drivetrain.data["candidates"].values()
              for c in rows if "no drawing published" in str(c.get("why"))]
    assert struck, "no candidate was rejected for having no drawing"
    assert all(not c["feasible"] for c in struck)

    # And the joint this was found at now takes a part that has a drawing.
    selected = next(r["selected"] for r in drivetrain.rows
                    if r["joint"] == "j6_tool_roll")
    assert selected == "cubemars_ak60_6_v3_kv80"
    assert face_for(selected, "output") is not None


def test_a_link_is_clipped_to_its_domain_so_it_cannot_reach_its_neighbour():
    """The overshoot that made six parts in a row interfere.

    Marching cubes puts the surface where the field crosses the iso level and
    smoothing moves it again, so an extracted body stands about 1.7 mm proud
    of its domain at every face. For one part that is a rounding; for six in
    a row whose domains are exactly the joint spacings it means every link
    reaches past both its joint planes into its neighbours, which is what an
    assembly measured. Clipping also leaves the two mounting faces flat,
    which is what they bolt against.
    """
    import trimesh

    from projects.manipulator.links import clip_to_domain

    proud = trimesh.creation.icosphere(radius=60.0)
    proud.apply_translation([39.25, 49.0, 49.0])
    assert proud.bounds[0][0] < 0.0 and proud.bounds[1][0] > 78.5

    cut, note = clip_to_domain(proud, 0.0785, 0.098, 0.098, scale=1000.0)
    assert cut.is_watertight
    assert cut.bounds[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert cut.bounds[1] == pytest.approx([78.5, 98.0, 98.0], abs=1e-6)
    assert "overshoot" in note


def test_no_two_links_ever_hold_the_same_material():
    """The invariant, after the question it asks had to change.

    It began as "adjacent domain boxes share nothing", which the shoulder
    broke by 235,298 cubic millimetres and which putting each link on the far
    side of the face it bolts to fixed. Then the cranked links needed the
    union of two boxes and it broke again, by 117,649, and holding every
    neighbour's box empty fixed that. Then a flange around a crossing axis
    turned out to be a disc centred on the drive, with half of it behind that
    joint's own plane and inside the neighbour's box, and the invariant
    stopped being right at all: demanding no shared box would forbid the
    disc. The boxes now overlap by 1.8 million cubic millimetres at the
    shoulder on purpose.

    What must never happen is that two links claim the same MATERIAL, and
    that is what is asked now. It matters because two rules can each carve
    out an exception, a link holding its own bolt ring against a neighbour's
    box being one, and two exceptions can fire in the same place without
    either rule noticing.
    """
    from projects.manipulator.links import domain_overlaps
    from projects.manipulator.loop import run_loop

    loop = run_loop()
    rows = domain_overlaps(SPEC, dict(loop.data["history"][-1].selected),
                           loop.data["final_sections"], samples=10)
    assert rows
    assert any(row["shared_mm3"] > 0.0 for row in rows), (
        "no boxes overlap at all, which means the crossing flanges lost "
        "their discs")
    for row in rows:
        assert row["both_solid_samples"] == 0, row

    # And how much room is left for it to go wrong. A contested point is one
    # element straddling the seam between two boxes: its centroid sits
    # outside the neighbour's box so the rules leave it free, while part of
    # the element is inside. It is bounded by half a cell, which is the same
    # half cell the drive envelopes are subtracted to close, and it is worth
    # a number rather than an argument because it moves with the grid.
    worst = max(row["contested_bound_mm3"] for row in rows)
    assert worst < 4000.0, (
        f"the seam between two links leaves {worst:.0f} cubic millimetres "
        f"where both could put material; it was 1882 when this was written")


def test_two_links_are_asked_for_a_shape_a_box_cannot_be():
    """What the placement rules disagree about, stated rather than resolved.

    A link driven by a COAXIAL joint has to be centred on that axis: it bolts
    to a face perpendicular to it and turns about it. A link that carries a
    CROSSING joint has to sit clear of that drive's housing face, which is to
    one side. The base column does both, and so does the wrist roll body. A
    box cannot satisfy both, and a real arm answers it with a bracket at the
    shoulder and a cranked body at the wrist.
    """
    from projects.manipulator.links import link_placements
    from projects.manipulator.loop import run_loop

    loop = run_loop()
    rows = link_placements(SPEC, dict(loop.data["history"][-1].selected),
                           loop.data["final_sections"])
    conflicted = {row["link"] for row in rows if row.get("conflict")}
    assert conflicted == {"base_column", "wrist_roll_body"}
    for row in rows:
        if row["link"] in conflicted:
            assert "cannot do both" in row["conflict"]


def test_the_upper_arm_fits_the_machine_only_by_standing_it_up():
    """A part that does not fit cannot be made, whatever else is true of it.

    The material in this design is AlSi10Mg with its strength and fatigue
    numbers read off the EOS M 290 sheet, so the M 290's own construction
    volume is the one that applies: 250 by 250 by 325 mm, with the height
    including the build platform and stated to be application dependent.

    THIS TEST USED TO CLAIM THE WIDEST PART WAS 238.7 mm AND IT WAS STALE.
    That number was the upper arm's tightest axis, and the 11.3 mm it leaves
    against 250 is still exactly right. What changed underneath it is the
    other axis. When the crossing axis rule was written, so that a bolt
    circle centred on a drive is inside the domain rather than half outside
    it, the upper arm's long axis took half an AK80-64 outer diameter at each
    end: 193.0 mm of link between the joints, plus 98.0 mm of motor, is
    291.0. That is past the 250 mm bed.

    The part still fits, but only lying along the 325 mm build height, and
    the sheet says that height includes the build platform and is
    application dependent. So the upper arm's manufacturability is now
    conditional where it used to be plain, and neither of the old assertions
    could see it: `fits` accepts a conditional fit, and the margin check
    compared 250 against a number that had grown PAST 250, which makes the
    difference negative and the check pass for the wrong reason. Both are
    replaced with assertions that name the axis they are about.
    """
    import numpy as np

    from projects.manipulator.links import (EOS_M290_BUILD_VOLUME_M,
                                            fits_the_build_volume, world_boxes)
    from projects.manipulator.loop import run_loop

    loop = run_loop()
    boxes = world_boxes(SPEC, dict(loop.data["history"][-1].selected),
                        loop.data["final_sections"])
    standing = {}
    extents = {}
    for box in boxes:
        if not box.get("placed"):
            continue
        extent = np.sort(np.asarray(box["high"]) - np.asarray(box["low"]))
        fits, why = fits_the_build_volume(extent)
        assert fits, f"{box['link']}: {why}"
        extents[box["link"]] = extent
        if "standing it up" in why:
            standing[box["link"]] = extent

    assert set(standing) == {"upper_arm"}, (
        "exactly one part is supposed to need the build height; if this set "
        "changed, the domains or the machine did")

    bed = sorted(EOS_M290_BUILD_VOLUME_M)[1]
    upper = extents["upper_arm"]
    assert upper[-1] == pytest.approx(0.2910, abs=0.002)
    assert upper[-1] > bed
    assert upper[1] == pytest.approx(0.2387, abs=0.002)
    assert bed - upper[1] == pytest.approx(0.0113, abs=0.002), (
        "the tightest axis has moved; this is the margin the docstring "
        "quotes and it has to be read off the second longest axis, not the "
        "longest, because the longest no longer fits the bed at all")


def test_the_mounting_planes_do_not_drift_along_the_chain():
    """A placement rule has to be checked on the arm, not on a pair.

    Facing consecutive drives on a shared axis in opposite directions makes
    every PAIR of joints work and makes the ARM climb: each pitch joint's
    output face lands 140.7 mm beyond the last one and nothing brings it
    back, so three of them carry the wrist a third of a metre out of the
    plane. That rule was written and taken out again, and this is the check
    that kills it: follow the planes along the whole chain and the last one
    has to be where the first one is.
    """
    from projects.manipulator.links import mounting_plane_chain
    from projects.manipulator.loop import run_loop

    loop = run_loop()
    rows = mounting_plane_chain(SPEC, dict(loop.data["history"][-1].selected))
    total = rows[-1]
    assert total["joint"] == "TOTAL DRIFT"
    assert total["drift_from_first_mm"] == pytest.approx(0.0, abs=1e-9)
    for row in rows[:-1]:
        assert row["drift_from_first_mm"] == pytest.approx(0.0, abs=1e-9)


def test_a_crossing_drive_pocket_lands_on_the_drive_not_on_the_box_corner():
    """The pocket was 140.7 mm from the motor and every check passed.

    Every drive's output face is the arm's z = 0 plane, and a link's own
    frame starts wherever its box starts, which for the cranked links is
    140.7 mm below that. The offset from one to the other was left at zero,
    so a crossing joint's pocket was cut at the bottom of the link instead of
    at the drive. Nothing here could see it: the pocket existed, it was the
    right size, it was in the domain, and it was in the wrong place. An
    assembly measured the material left behind, 154,744 cubic millimetres of
    link inside a motor, on every joint whose axis crosses the arm and on no
    other.
    """
    import numpy as np

    from projects.manipulator.links import link_domain, world_boxes
    from projects.manipulator.loop import run_loop

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    sections = loop.data["final_sections"]
    boxes = {b["link"]: b for b in world_boxes(SPEC, drives, sections)}

    built, reason = link_domain(SPEC, 1, drives, sections=sections)
    assert built is not None, reason
    mesh, _solid, void, _span, _height, _width, _note = built
    centroids = mesh.element_centroids()[void]
    world_z = centroids[:, 2] + boxes["upper_arm"]["low"][2]

    # The shoulder and the elbow both lie between -53.9 and +8.0 mm of the
    # arm's z = 0 plane. The void also holds the neighbour's box, which
    # reaches much further, so the test is that a real share of it sits on
    # the drives rather than that all of it does. With the offset at zero
    # this was empty.
    on_the_drive = ((world_z >= -0.0539) & (world_z <= 0.008)).sum()
    assert on_the_drive > 0.2 * len(world_z), (
        f"only {on_the_drive} of {len(world_z)} void elements are where the "
        f"drives are; the pocket is not on the drive")
    assert np.abs(world_z).min() < 0.010, "nothing was cut near the drive"


def test_a_drive_is_subtracted_from_the_body_not_merely_avoided():
    """Holding elements empty cannot keep material out of a motor.

    A void is enforced on element CENTRES, so the iso surface runs between a
    void element and its solid neighbour and material stands up to half a
    cell into the pocket. Half a cell on these grids is 4.1 mm against a
    radial clearance of 1.0, so the surface reaches into the drive by
    construction. Measured across six links with every pocket in the right
    place: 104,538 cubic millimetres of link inside a motor. The drives are
    subtracted now, and a cylinder taken out of a body cannot be inside it.
    """
    import numpy as np
    import trimesh

    from projects.manipulator.links import cut_holes

    block = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
    block.apply_translation([50.0, 50.0, 50.0])
    envelope = [{"end": "drive", "kind": "envelope", "face": "j", "thread": "",
                 "diameter_m": 0.040,
                 "start_m": [0.05, 0.05, -0.02],
                 "end_m": [0.05, 0.05, 0.12],
                 "y_m": 0.0, "z_m": 0.0, "x0_m": 0.0, "x1_m": 0.0}]
    cut, report = cut_holes(block, envelope, scale=1000.0)
    assert cut.is_watertight
    # A 40 mm bore through a 100 mm cube takes pi r squared h out of it.
    removed = float(abs(block.volume) - abs(cut.volume))
    assert removed == pytest.approx(np.pi * 20.0 ** 2 * 100.0, rel=0.02)
    assert "envelope" in report[0]


def test_only_the_cranked_links_need_a_withdrawal_corridor():
    """A drive has to be able to come out, and a C traps one.

    A link that has material on both sides of a drive within that drive's own
    radius traps it, however well the two sides avoid the motor itself. The
    three cranked links do exactly that: the base column's domain reaches
    from 140.7 mm below the shoulder plane to 49 above it and closes around a
    motor living between -53.9 and +8. An independent sweep measured all
    three as trapped in both directions along the axis.

    The three straight links need nothing, because each lies entirely on one
    side of its own drive already. That is the check on the rule: if a
    corridor appeared where a link does not wrap, the corridor would be
    carving material for no reason.
    """
    from projects.manipulator.links import link_domain
    from projects.manipulator.loop import run_loop

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    sections = loop.data["final_sections"]
    carved = {}
    for index, link in enumerate(SPEC.links()):
        built, reason = link_domain(SPEC, index, drives, sections=sections)
        assert built is not None, reason
        carved[link.name] = "corridor" in built[-1]
    assert {name for name, has in carved.items() if has} == {
        "base_column", "upper_arm", "wrist_roll_body"}


def test_two_flange_discs_fit_between_a_drives_mounting_faces():
    """A joint whose axis crosses the arm has a flange disc on each side of
    it, and they have to fit in the gap the drive leaves.

    The shoulder is comfortable: its faces are 42.7 mm apart and two 9 mm
    flanges leave 24.7. The wrist is not: the AK80-9's faces are 24.5 apart
    and the same two flanges leave 6.5. The flange thickness comes from a
    bolt length and can move, and at 12 mm the two discs at the wrist would
    meet. This is the number that says so.
    """
    from projects.manipulator.interfaces import face_separation_m
    from projects.manipulator.loop import run_loop

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    gaps = {}
    for joint in SPEC.joints():
        separation = face_separation_m(str(drives.get(joint.name, "")))
        if separation is None:
            continue
        gaps[joint.name] = separation - 2.0 * SPEC.flange_thickness_m
    assert gaps, "no joint could be checked"
    for name, gap in gaps.items():
        assert gap > 0.0, (
            f"{name}: two {SPEC.flange_thickness_m * 1000:.0f} mm flanges do "
            f"not fit in the {(gap + 2 * SPEC.flange_thickness_m) * 1000:.1f} "
            f"mm between that drive's mounting faces")
    tightest = min(gaps, key=gaps.get)
    assert gaps[tightest] == pytest.approx(0.0065, abs=0.0005)
    assert "wrist" in tightest or "roll" in tightest, tightest


def test_the_volume_fraction_means_the_same_thing_on_every_link():
    """It did not, and the spread was sevenfold.

    A volume fraction read against the whole domain means whatever each
    part's interfaces leave over. On this arm 0.3 of the domain came out as
    0.128 of the decidable elements on the tool flange and 0.898 on the wrist
    pitch body, so the same number asked one part for a skeleton and another
    for a nearly solid block. Read against the free region it asks the same
    question everywhere, and that is what the links use.
    """
    import numpy as np

    from optimization.topology import SimpProblem
    from physics.fem.mesh import solid_box_mesh

    mesh = solid_box_mesh(0.1, 0.1, 0.1, 6, 6, 6)
    solid = np.zeros(mesh.n_elements, dtype=bool)
    solid[:40] = True
    void = np.zeros(mesh.n_elements, dtype=bool)
    void[40:120] = True
    made = {}
    for how in ("domain", "free"):
        made[how] = SimpProblem(
            mesh=mesh, youngs_modulus_pa=7e10, poisson_ratio=0.33,
            fixed_nodes=mesh.nodes_at_x(0.0), load_nodes=mesh.nodes_at_x(0.1),
            total_load_n=-1.0, load_direction=1, volume_fraction=0.3,
            volume_fraction_of=how, passive_solid=solid,
            passive_void=void).free_volume_fraction()
    assert made["free"] == pytest.approx(0.3)
    assert made["domain"] != pytest.approx(0.3)
    assert made["domain"] == pytest.approx(0.2583, abs=0.001)


def test_the_corridor_and_the_bolt_ring_do_not_contest_the_same_elements():
    """They look like they must and they do not, and the reason is the plane
    between them.

    Both live around the same drive axis at similar radii, so the suspicion
    that one silently deletes the other is a fair one. What separates them is
    which side of the mounting face they are on: a link's ring is on the
    link's own side of that face and the corridor is on the drive's side. The
    drive leaves the way it came, away from the link, so it never has to pass
    through the ring.

    Measured across all six links: zero elements claimed by both.
    """
    import numpy as np

    from projects.manipulator.interfaces import drive_profile, face_for
    from projects.manipulator.links import (ACTUATOR_RADIAL_CLEARANCE_M,
                                            _drive_face, actuator_for,
                                            face_separation_m, local_axis,
                                            world_boxes)
    from projects.manipulator.loop import run_loop
    from physics.fem.mesh import solid_box_mesh

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    boxes = {b["link"]: b for b in world_boxes(SPEC, drives,
                                               loop.data["final_sections"])}
    joints = SPEC.joints()
    flange = SPEC.flange_thickness_m
    seen_a_corridor = False
    for index, link in enumerate(SPEC.links()):
        box = boxes[link.name]
        nz = max(12, int(round(box["width"] / (0.098 / 12))))
        mesh = solid_box_mesh(box["span"], box["height"], box["width"],
                              28, 12, nz)
        centroids = mesh.element_centroids()
        following = joints[index + 1] if index + 1 < len(joints) else None
        corridor = np.zeros(mesh.n_elements, dtype=bool)
        ring = np.zeros(mesh.n_elements, dtype=bool)
        ends = [(joints[index], box["reach_low"], True)]
        if following is not None:
            ends.append((following, box["reach_low"] + box["joint_span"], False))
        for other, at_x, driven in ends:
            axis = local_axis(SPEC, index, other.axis)
            if abs(float(axis[0])) > 0.5:
                continue
            drive = actuator_for(other.name, drives)
            profile = drive_profile(str(drives.get(other.name, "")))
            if drive is None or profile is None:
                continue
            if face_for(str(drives.get(other.name, "")), "output") is None:
                continue
            origin = _drive_face(SPEC, index, other, at_x, box["height"],
                                 box["width"], box)
            offset = centroids - origin
            along = offset @ axis
            radial = np.linalg.norm(offset - np.outer(along, axis), axis=1)
            outer = 0.5 * drive.outer_diameter_m
            separation = face_separation_m(str(drives.get(other.name, "")))
            if driven:
                corridor |= (radial <= outer + ACTUATOR_RADIAL_CLEARANCE_M) & (along <= 0.0)
                ring |= ((along >= 0.0) & (along <= flange)
                         & (radial >= profile[-1][2]) & (radial <= outer))
            else:
                drop = separation or 0.0
                corridor |= (radial <= outer + ACTUATOR_RADIAL_CLEARANCE_M) & (along >= -drop)
                ring |= ((along >= -drop - flange) & (along <= -drop)
                         & (radial >= profile[0][2]) & (radial <= outer))
        if corridor.any():
            seen_a_corridor = True
        assert not (corridor & ring).any(), (
            f"{link.name}: {int((corridor & ring).sum())} elements are claimed "
            f"by both the withdrawal corridor and a bolt ring")
    assert seen_a_corridor, "no link had a corridor, so nothing was tested"


def test_the_narrowest_bolt_ring_clears_its_floor_and_only_just():
    """A floor, because the last defect of this kind announced itself and the
    next one would not have.

    The AK60-6's housing flange came out one millimetre wide IN THE NEGATIVE,
    and a negative number is impossible to miss. Half a millimetre would have
    built, passed every check, and arrived as a bolt seat too narrow to
    tighten against. The sign of a number is not a substitute for a number.

    The floor is 8.0 mm: an M3 socket head is 5.5 across and the ring has to
    leave 1.25 mm of material each side. Every ring on this arm clears it,
    and the AK80-64's output faces clear it by half a millimetre, so this is
    a live limit rather than a formality. It is the same shape of number as
    the 6.5 mm left between two flange discs at the wrist.
    """
    from projects.manipulator.links import (MINIMUM_RING_WIDTH_M,
                                            interface_solids, world_boxes)
    from projects.manipulator.loop import run_loop

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    boxes = {b["link"]: b for b in world_boxes(SPEC, drives,
                                               loop.data["final_sections"])}
    widths = {}
    for index, link in enumerate(SPEC.links()):
        box = boxes[link.name]
        for ring in interface_solids(SPEC, index, drives, box["span"],
                                     box["height"], box["width"], box):
            assert not ring.get("refused"), ring["note"]
            widths[f"{link.name}/{ring['face']}"] = 0.5 * (
                ring["outer_diameter_m"] - ring["inner_diameter_m"])

    assert widths
    for name, width in widths.items():
        assert width >= MINIMUM_RING_WIDTH_M, f"{name}: {width * 1000:.1f} mm"
    tightest = min(widths, key=widths.get)
    assert widths[tightest] == pytest.approx(0.0085, abs=0.0003)
    assert widths[tightest] - MINIMUM_RING_WIDTH_M < 0.001, (
        "the narrowest ring has more margin than expected, so either a drive "
        "changed or the floor did; re-read both")


def test_every_loaded_joint_needs_the_same_stiffness_and_it_is_reachable():
    """The number, and a correction to how it was first reported.

    Every deflection this design computes is link elasticity only, because
    six actuators sit between the links and not one publishes a torsional
    stiffness. Turning the question round asks what the joints would have to
    be, and needs no source to answer.

    The first answer was wrong by a factor of four, and wrong in the
    direction that matters: it made a reachable design look impossible. It
    split the joints' allowance EQUALLY, which hands a sixth of it to the
    base yaw and the two roll axes, none of which carries any gravity moment
    at full reach, and the same sixth to the shoulder, which carries 67
    percent of the whole demand. That reported 205,000 N m/rad.

    Weighted by torque times lever the answer is 50,689 for every loaded
    joint, and it is the same for all of them by construction: a joint's
    contribution is torque times lever over stiffness, so making the
    contributions proportional to torque times lever makes the stiffness
    constant. That is 1.15 times the stiffest gear unit this catalogue
    prints, which puts the limit at the edge of what is available rather
    than beyond it.
    """
    from projects.manipulator.arm import build_arm
    from projects.manipulator.loop import run_loop
    from projects.manipulator.stages import (PRINTED_STIFFNESS_RANGE_NM_RAD,
                                             drivetrain_stage, dynamics_stage,
                                             joint_stiffness_stage,
                                             reflected_inertia_stage)

    loop = run_loop()
    arm = build_arm(loop.data["final_sections"], SPEC)
    dynamics = dynamics_stage(arm, SPEC, samples=40)
    first = drivetrain_stage(dynamics, SPEC, {})
    inertias = {row["joint"]: row.get("load_inertia_kg_m2")
                for row in reflected_inertia_stage(arm, first, SPEC).rows}
    stage = joint_stiffness_stage(dynamics, drivetrain_stage(
        dynamics, SPEC, inertias), arm, SPEC)

    needed = {row["joint"]: row["required_stiffness_nm_rad"]
              for row in stage.rows if row["required_stiffness_nm_rad"]}
    assert set(needed) == {"j2_shoulder", "j3_elbow", "j5_wrist_pitch"}
    assert len(set(round(value) for value in needed.values())) == 1, needed
    assert next(iter(needed.values())) == pytest.approx(50_689, rel=0.02)
    assert next(iter(needed.values())) < 1.3 * PRINTED_STIFFNESS_RANGE_NM_RAD[1]
    assert any("LINK ELASTICITY ONLY" in item for item in stage.could_not)


def test_an_output_face_has_two_mounting_planes():
    """One face, two planes, and the offset is a number already held.

    An output face carries an outer bolt circle in the mounting face and an
    inner one on the END OF THE BOSS, which stands proud of it. Measured on
    all three drives the offset is 8.0 mm, 3.0 and 1.5, and each equals that
    drive's published output face inset, because the boss height IS the
    inset. So the second plane needed no new measurement, only the
    recognition that it exists.

    Treating one face as one plane put the inner holes and their seat inside
    the motor. The wrist pitch body's inner ring then reported metal at 0 of
    64 sample points, which was correct about the material and wrong about
    what it meant, and the tool flange's did the same.
    """
    from projects.manipulator.interfaces import (AK60_6_OUTPUT, AK80_9_OUTPUT,
                                                 AK80_64_HOUSING,
                                                 AK80_64_OUTPUT)

    for face, expected in ((AK80_64_OUTPUT, 0.008), (AK80_9_OUTPUT, 0.003),
                           (AK60_6_OUTPUT, 0.0015)):
        offsets = {pattern.plane_offset_m for pattern in face.patterns}
        assert offsets == {0.0, expected}, (face.actuator, offsets)
        assert face.face_inset_m == expected, (
            "the inner plane's offset should be the face inset, because the "
            "boss height is the inset")
        inner = next(p for p in face.patterns if p.plane_offset_m)
        outer = next(p for p in face.patterns if not p.plane_offset_m)
        assert inner.bolt_circle_m < outer.bolt_circle_m

    # A housing face has one plane. Nothing stands proud of it.
    assert {p.plane_offset_m for p in AK80_64_HOUSING.patterns} == {0.0}
    assert AK80_64_OUTPUT.dowel_plane_offset_m == 0.008, (
        "the dowels are on the boss end with the inner circle")


def test_the_inner_bolt_circle_gets_a_seat_on_the_boss_end():
    """And the seat is a separate solid, because it is in a separate plane."""
    from projects.manipulator.links import interface_solids, world_boxes
    from projects.manipulator.loop import run_loop

    loop = run_loop()
    drives = dict(loop.data["history"][-1].selected)
    boxes = {b["link"]: b for b in world_boxes(SPEC, drives,
                                               loop.data["final_sections"])}
    for index, link in enumerate(SPEC.links()):
        box = boxes[link.name]
        solids = interface_solids(SPEC, index, drives, box["span"],
                                  box["height"], box["width"], box)
        kinds = [solid["kind"] for solid in solids]
        assert kinds.count("seat") == 1, (link.name, kinds)
        seat = next(s for s in solids if s["kind"] == "seat")
        assert seat["inner_diameter_m"] == 0.0, "a seat is a disc, not a ring"


def test_the_gravity_moment_at_the_shoulder_lies_on_the_joint_axis():
    """Why the bearing is not the answer to the deflection budget.

    An earlier reading of this had the crossed roller bearing carrying the
    tool sag, and it was wrong. The shoulder turns about z. At full reach the
    tool hangs off along x and gravity pulls along minus y, so the moment it
    makes is the cross product of those, which points along minus z. That is
    the joint axis itself, to the last digit.

    A joint bearing resists moments about the two axes ACROSS the joint. The
    one direction it cannot resist is the one the joint turns in, and that is
    exactly where this moment sits. So the sag comes out of the drive train's
    torsional compliance, not the bearing's tilting rigidity, and the two
    belong to different budgets.
    """
    axis = np.array([0.0, 0.0, 1.0])
    lever = np.array([SPEC.reach_m, 0.0, 0.0])
    moment = np.cross(lever, np.array([0.0, -1.0, 0.0]))
    along = float(np.dot(moment / np.linalg.norm(moment), axis))
    assert abs(along) == pytest.approx(1.0, abs=1e-12)


def test_no_ring_pin_gets_a_lever_longer_than_the_pitch_radius():
    """The geometric fact that cost the first torsion estimate a factor of 3.3.

    A cycloidal contact normal passes through the instantaneous pitch point.
    In the disc's own frame that point sits at the eccentricity times the
    lobe count from the disc centre, so no ring pin's moment arm about that
    centre can exceed it, whatever radius the pin circle is drawn at. With
    2.5 mm and ten lobes the bound is 25 mm and the pin circle is 45.

    The first estimate used the pin circle radius as the lever, which is
    where a sum of squares of 5,569 mm^2 came from instead of 1,700. This
    test computes the arms from the envelope and checks them against the
    bound rather than against the number they happened to produce, so it
    would still hold if the geometry changed.
    """
    from projects.manipulator.cycloidal import (CycloidalGeometry,
                                                ring_pin_moment_arms)

    geometry = CycloidalGeometry()
    assert geometry.pitch_radius_m == pytest.approx(0.030)
    assert geometry.pitch_radius_m < geometry.ring_pin_circle_radius_m

    worst = 0.0
    for angle in np.linspace(0.0, 2.0 * np.pi, 97):
        arms = ring_pin_moment_arms(geometry, float(angle))
        worst = max(worst, float(np.abs(arms).max()))
    assert worst <= geometry.pitch_radius_m + 1e-9
    assert worst == pytest.approx(geometry.pitch_radius_m, rel=0.002), (
        "the envelope should reach the pitch point bound almost exactly; if "
        "it does not, the normals are being taken wrongly")


def test_the_output_pin_circle_cannot_be_opened_to_sixty():
    """A 44 percent gain that the disc's root radius does not allow.

    Widening the output pin circle buys stiffness as the radius squared, and
    Ø60 was proposed on the grounds that the disc's outline is at 42.5 mm. It
    is not. 42.5 is the TIP radius. The binding one is the ROOT, at the pin
    circle less the pin radius less the eccentricity, which is 37.5, and an
    output pin hole is the pin plus the eccentricity across its own radius.

    At the eccentricity this was proposed against, 2.5 mm, the drawn Ø50
    circle left a 5 mm web and Ø60 left 0.00 mm exactly. The eccentricity is
    3.0 mm now, which moves the root in by half a millimetre, so the circle
    is Ø48 for the same 5 mm web and Ø60 is a millimetre worse than it was.
    The tip and the root differ by twice the eccentricity and that is the
    whole of the error.
    """
    import dataclasses

    from projects.manipulator.cycloidal import CycloidalGeometry

    geometry = CycloidalGeometry()
    assert geometry.disc_tip_radius_m - geometry.disc_root_radius_m == (
        pytest.approx(2.0 * geometry.eccentricity_m))
    assert geometry.output_web_m == pytest.approx(0.005, abs=1e-9)

    wider = dataclasses.replace(geometry, output_pin_circle_radius_m=0.030)
    assert wider.output_web_m == pytest.approx(-0.001, abs=1e-9)
    assert wider.output_web_m < MINIMUM_DISC_LIGAMENT_M

    as_proposed = dataclasses.replace(geometry, eccentricity_m=0.0025,
                                      output_pin_circle_radius_m=0.030)
    assert as_proposed.output_web_m == pytest.approx(0.0, abs=1e-9)


def test_the_eccentric_bearings_lever_is_the_pitch_radius_squared():
    """Derived rather than assumed, because it is the one term with no value.

    The disc centre sits at the eccentricity along the input angle and the
    disc turns at minus that angle over the ratio. A tangential shift d at
    the centre is indistinguishable from the input angle being larger by
    d / e, so the disc's rotation errs by d / (e * N). The force follows from
    power: the input torque is the output torque over the ratio and acts at
    the orbit radius, so it is T / (N * e). Put together, the torsional
    stiffness is the radial stiffness times (e * N) squared, and each disc
    brings its own bearing in parallel.

    The lever is therefore the pitch radius, 30 mm, not the eccentricity,
    3.0 mm. Those two readings differ by a hundred in the answer, which is
    why this is a test and not a comment.
    """
    from projects.manipulator.cycloidal import (
        CycloidalGeometry, eccentric_bearing_stiffness_nm_rad,
        required_bearing_stiffness_n_m)

    geometry = CycloidalGeometry()
    radial = 1.0e8
    stiffness = eccentric_bearing_stiffness_nm_rad(radial, geometry)
    assert stiffness == pytest.approx(
        radial * geometry.pitch_radius_m ** 2 * geometry.disc_count)
    assert stiffness == pytest.approx(180_000.0, rel=1e-9)
    assert required_bearing_stiffness_n_m(stiffness, geometry) == (
        pytest.approx(radial, rel=1e-9))


def test_the_reducers_torsion_clears_the_requirement_but_only_as_an_estimate():

    """The drive train's own stiffness, against the 50,689 the arm asks for.

    The chain is the output flange, six output pins, the disc, eleven ring
    pins and the housing, with the discs' in plane shear alongside. Four
    terms have numbers and one, the eccentric bearing, has none.

    252,682 N m/rad, a factor of 5.0. THIS REPLACES 682,012 AND A FACTOR OF
    13.5 reported the same day, and the whole of the difference is in lever
    arms rather than loads: the ring pins were given their pin circle radius
    instead of the pitch radius bound, and the output pins were given a hand
    picked count of engaged pins at full radius instead of a load share
    solved from their arms. Both errors made the drive look stiffer. The
    corrected model first read 241,801, and raising K1 from 0.611 to 0.733
    took it to 252,682 while cutting the bearing requirement by a third.

    It is a pass and it is not verified. Palmgren's approach is still a
    roller bearing relation applied to a cycloidal flank and to a pin in a
    hole, and those two terms carry 86 percent of the compliance. The test
    pins the numbers so that the next correction shows up as a change rather
    than as a new opinion.
    """
    from projects.manipulator.stages import joint_torsion_stage

    stage = joint_torsion_stage()
    terms = {row["term"]: row["stiffness_nm_rad"] for row in stage.rows}
    assert set(terms) == {"ring pin contact", "output pin contact",
                          "discs, in plane shear", "housing, in torsion"}
    assert terms["output pin contact"] < terms["ring pin contact"]
    assert terms["ring pin contact"] < terms["housing, in torsion"]
    assert terms["housing, in torsion"] < terms["discs, in plane shear"]
    assert stage.data["k1_factor"] == pytest.approx(0.7333, rel=1e-3)

    shares = {row["term"]: row["share_of_compliance"] for row in stage.rows}
    assert shares["output pin contact"] + shares["ring pin contact"] == (
        pytest.approx(0.85, abs=0.02))

    known = stage.data["known_terms_nm_rad"]
    assert known < min(terms.values())
    assert known == pytest.approx(252_682, rel=0.01)
    assert stage.data["margin_before_the_bearing"] == pytest.approx(4.98,
                                                                    rel=0.02)
    assert stage.data[
        "eccentric_bearing_radial_stiffness_needed_n_m"] == pytest.approx(
            3.52e7, rel=0.02)
    assert any("not a verified result" in item.lower()
               for item in stage.could_not)


def test_no_computed_floor_comes_near_the_discs_eight_millimetres():
    """The thickness stays CHOSEN, and now four computations say so.

    In plane shear is linear in thickness, so the stiffness floor is 0.028
    mm. The contact terms go as thickness to the 0.8, putting theirs at 0.292
    mm. Pin contact stress and output pin hole bearing were already clear by
    factors of six and forty. Nothing computed reaches within an order of
    magnitude of 8 mm, so the thickness answers to handling and flatness of a
    wire cut part, which this repository cannot compute and does not claim to.
    """
    from projects.manipulator.cycloidal import (CycloidalGeometry,
                                                disc_shear_stiffness_nm_rad)
    from projects.manipulator.stages import (CYCLOIDAL_DISC_THICKNESS_BASIS,
                                             CYCLOIDAL_DISC_THICKNESS_M)

    required = 50_689.0
    geometry = CycloidalGeometry()
    assert geometry.disc_thickness_m == CYCLOIDAL_DISC_THICKNESS_M
    floor = CYCLOIDAL_DISC_THICKNESS_M * required / disc_shear_stiffness_nm_rad(
        geometry)
    assert floor == pytest.approx(3.2e-5, rel=0.05)
    assert floor < CYCLOIDAL_DISC_THICKNESS_M / 100.0
    assert CYCLOIDAL_DISC_THICKNESS_BASIS.startswith("CHOSEN")
    assert "handling" in CYCLOIDAL_DISC_THICKNESS_BASIS
    assert "0.879 mm" in CYCLOIDAL_DISC_THICKNESS_BASIS, (
        "the contact floor moved from 0.292 to 0.940 when the lever arms "
        "were corrected, and to 0.879 when K1 rose; the basis string has to "
        "carry the current number")


def test_disc_shear_stiffness_is_linear_in_thickness():
    """The property the floor above is read off, asserted rather than assumed."""
    import dataclasses

    from projects.manipulator.cycloidal import (CycloidalGeometry,
                                                disc_shear_stiffness_nm_rad)

    base = CycloidalGeometry()
    one = disc_shear_stiffness_nm_rad(dataclasses.replace(base,
                                                          disc_thickness_m=0.004))
    two = disc_shear_stiffness_nm_rad(dataclasses.replace(base,
                                                          disc_thickness_m=0.008))
    assert two == pytest.approx(2.0 * one, rel=1e-12)


def test_the_eccentricity_is_not_a_free_variable():
    """K1, and the band it has to stay inside.

    Raising the eccentricity was the strongest lever available on this
    reducer, because the pitch radius is the ring pins' moment arm bound and
    the eccentric bearing's lever squared. But e is not free: it is K1 times
    the pin circle radius over the pin count, and K1 carries the design band,
    usually 0.5 to 0.75. 2.5 mm was 0.611 and 3.0 is 0.733, which takes the
    pitch radius from 25 to 30 mm and stays inside. 3.5 mm would have been
    0.856 and outside it.

    The band is a convention this project has no source for, so what is
    actually asserted is the thing the band stands in for: how much curvature
    is left at the lobe tips before offsetting inward by the pin radius
    undercuts the profile. That is computed from the envelope.
    """
    import dataclasses

    from projects.manipulator.cycloidal import (CycloidalGeometry,
                                                undercut_margin_m)

    geometry = CycloidalGeometry()
    assert geometry.k1_factor == pytest.approx(
        geometry.eccentricity_m * geometry.ring_pin_count
        / geometry.ring_pin_circle_radius_m)
    assert 0.5 <= geometry.k1_factor <= 0.75
    assert geometry.pitch_radius_m == pytest.approx(0.030)

    margins = []
    for factor in (0.60, 0.75, 0.95, 1.05):
        wider = dataclasses.replace(
            geometry, eccentricity_m=factor * geometry.ring_pin_circle_radius_m
            / geometry.ring_pin_count)
        margins.append(undercut_margin_m(wider))
    assert margins[0] > margins[1] > margins[2] > 0.0
    assert margins[3] < 0.0, "the profile has to undercut somewhere past K1 = 1"
    assert undercut_margin_m(geometry) == pytest.approx(0.0071, abs=0.0005)


def test_raising_the_eccentricity_moves_the_bearings_two_loads_apart():
    """The objection to raising e, answered with the arithmetic rather than
    with a judgement.

    Raising the eccentricity lowers the eccentric bearing's TANGENTIAL load,
    which is T / (N e) straight from power, and raises its RADIAL load,
    because the pressure angle grows with K1. The two have opposite signs and
    only the numbers say which wins. They very nearly cancel: across the band
    the resultant moves by a few percent while the requirement on the
    bearing's stiffness falls by a third.

    And the radial share does not touch the stiffness at all. Only the
    tangential deflection turns the output, and an isotropic radial stiffness
    has no cross term between the two directions.
    """
    import dataclasses

    import numpy as np

    from projects.manipulator.cycloidal import (CycloidalGeometry,
                                                eccentric_bearing_load_n)

    torque = 22.76
    tighter = dataclasses.replace(CycloidalGeometry(), eccentricity_m=0.0025,
                                  output_pin_circle_radius_m=0.025)
    looser = CycloidalGeometry()

    def worst(geometry):
        rows = [eccentric_bearing_load_n(geometry, torque, float(angle))
                for angle in np.linspace(0.0, 2.0 * np.pi / 11.0, 16)]
        return max(rows, key=lambda row: row["magnitude_n"])

    low, high = worst(tighter), worst(looser)
    assert high["tangential_n"] < low["tangential_n"]
    assert high["radial_n"] > low["radial_n"]
    assert high["magnitude_n"] < 1.05 * low["magnitude_n"]

    for row, geometry in ((low, tighter), (high, looser)):
        assert row["tangential_n"] == pytest.approx(row["power_estimate_n"],
                                                    rel=0.02), (
            "the tangential component has to agree with T / (N e), which is "
            "the independent check on the whole force model")


def test_the_orbiting_discs_leave_a_couple_too_small_to_design_for():
    """Checked because it was raised, and it is not a factor AT THIS SPEED.

    Two discs at 180 degrees cancel each other's centrifugal force and leave
    a rocking couple, m e omega squared times their spacing. The arm's own
    duty is 90 degrees in 2 seconds, so a joint turns at about 1.2 rad/s at
    its trapezoidal peak and the reducer input at ten times that. The couple
    comes out under a thousandth of a newton metre against a 23.35 N m peak
    joint torque.

    The speed is the whole of the reason. A reducer running its input at a
    few thousand rpm would see this term four orders larger, so the finding
    is written with its operating point attached and not as a general one.
    """
    import numpy as np

    from projects.manipulator.cycloidal import (CycloidalGeometry,
                                                orbit_couple_nm)

    geometry = CycloidalGeometry()
    mass = 7850.0 * np.pi * (geometry.disc_tip_radius_m ** 2 - 0.015 ** 2) * (
        geometry.disc_thickness_m)
    assert mass == pytest.approx(0.31, abs=0.02)

    joint_speed = 1.5 * (SPEC.move_angle_rad / SPEC.move_time_s)
    couple = orbit_couple_nm(geometry, geometry.ratio * joint_speed, mass,
                             0.009)
    assert couple == pytest.approx(0.0012, abs=0.0002)
    assert couple < 1.0e-4 * 23.35

    fast = orbit_couple_nm(geometry, 300.0, mass, 0.009)
    assert fast > 100.0 * couple, (
        "the conclusion is about this operating point and has to fail at a "
        "high speed input, or it is not saying anything")


def test_the_base_yaws_gravity_moment_is_perpendicular_to_its_axis():
    """The mirror image of the shoulder, and the reason there are two budgets.

    At the shoulder the gravity moment lies exactly ALONG the joint axis, so
    the bearing cannot resist it and the drive train does. At the base yaw
    the axis is vertical and the gravity moment is exactly PERPENDICULAR to
    it, so the drive train never sees it and the bearing carries all of it.

    Same arm, same gravity, opposite answers, turning on nothing more than
    which way the axis points. A stage that computes "the joint stiffness"
    without saying which of the two it means is wrong at half the joints, and
    that is what happened here once already.
    """
    axis = np.array([0.0, 1.0, 0.0])
    lever = np.array([SPEC.reach_m, SPEC.base_height_m, 0.0])
    moment = np.cross(lever, np.array([0.0, -1.0, 0.0]))
    along = float(np.dot(moment / np.linalg.norm(moment), axis))
    assert abs(along) == pytest.approx(0.0, abs=1e-12)


def test_the_base_yaws_structure_alone_misses_a_forty_micron_allowance():
    """The out of plane requirement, and what is already in the way of it.

    The whole arm stands on the base yaw and the tool is 618.5 mm off along
    the hypotenuse of the reach and the base height, so the bearing's tilt
    multiplies straight into the tip. At 43.3 N m of overturning a 0.04 mm
    allowance asks 669,221 N m/rad and a 0.08 mm one asks 334,610.

    The housing shell in bending gives 2.62 million and the ring of eight M3
    on the 85 mm circle gives 581,468 with no preload, which is 475,877 in
    series. That is already under the 0.04 mm figure BEFORE any bearing is
    added, so a 0.04 mm allowance is not clearly affordable and a 0.08 mm one
    probably is. That is narrower than the question this started as and it is
    as far as it can be taken without a preload and a bearing curve.
    """
    from projects.manipulator.stages import out_of_plane_stage

    stage = out_of_plane_stage()
    assert stage.data["overturning_nm"] == pytest.approx(43.28, rel=0.01)
    assert stage.data["lever_m"] == pytest.approx(0.6185, rel=0.001)

    required = {row["tip_allowance_m"]: row["stiffness_nm_rad"]
                for row in stage.rows}
    assert required[4.0e-5] == pytest.approx(669_221, rel=0.01)
    assert required[8.0e-5] == pytest.approx(334_610, rel=0.01)

    structure = stage.data["structure_lower_bound_nm_rad"]
    assert structure == pytest.approx(475_877, rel=0.01)
    assert structure < required[4.0e-5]
    assert structure > required[8.0e-5]
    assert any("LOWER BOUND" in item for item in stage.could_not)
    assert any("presser flange" in item for item in stage.could_not)


def test_the_pitch_joints_out_of_plane_load_is_twenty_times_smaller():
    """Measured rather than dismissed, because negligible is not a number.

    Gravity makes NO out of plane moment at the shoulder, elbow or wrist
    pitch, since it lies along those axes. What is left is the payload's own
    z offset, which exists only because the payload is a stated 100 mm cube,
    and the sideways force the base yaw makes while it accelerates. Together
    they are under 2 N m against the base yaw's 43.3.
    """
    gravity = 9.80665
    offset = 0.5 * SPEC.payload_extent_m
    payload = float(np.linalg.norm(np.cross(
        np.array([0.0, 0.0, offset]),
        np.array([0.0, -SPEC.payload_kg * gravity, 0.0]))))
    assert payload == pytest.approx(1.471, rel=0.01)

    from projects.manipulator.stages import out_of_plane_stage

    stage = out_of_plane_stage()
    assert payload + 0.4737 < stage.data["overturning_nm"] / 20.0


def test_every_stage_that_reports_a_stiffness_says_which_one():
    """The check that would have caught the mistake this session made twice.

    A bending stiffness was computed, called the joint stiffness, and used to
    conclude that the bearing was the whole of it. The number was right and
    the question was the wrong one. The fix is not a better number, it is a
    docstring that says what the value is a value OF before it says anything
    else, so the next reader cannot take it for the other quantity.
    """
    import ast
    import pathlib

    source = pathlib.Path("projects/manipulator/stages.py").read_text()
    wanted = {"compliance_stage", "joint_stiffness_stage",
              "joint_torsion_stage", "out_of_plane_stage",
              "joint_module_stiffness_stage"}
    seen = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef) or node.name not in wanted:
            continue
        seen.add(node.name)
        doc = ast.get_docstring(node) or ""
        opening = doc[:900]
        assert any(word in opening for word in
                   ("TORSIONAL", "OUT OF PLANE", "along the joint axis",
                    "ACROSS", "SCOPE")), (
            f"{node.name} does not say which stiffness it means")
    assert seen == wanted
