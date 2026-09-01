"""Rule based feature recognition, on parts whose answers are known.

Each test here corresponds to a rule that a simpler version got wrong. The
simpler versions are recorded as the cases that would fail under them, so the
reasoning cannot be lost and quietly reverted.
"""

from __future__ import annotations

import re
import collections
from pathlib import Path

import pytest

from nodes import feature_recognizer as fr

requires_occ = pytest.mark.skipif(not fr.is_available(),
                                  reason="OCP is not installed")

try:
    import build123d as _bd
    HAS_BUILD123D = True
except ImportError:
    HAS_BUILD123D = False

requires_cad = pytest.mark.skipif(not HAS_BUILD123D,
                                  reason="build123d is not installed")

FIXTURE_C = Path("tests/fixtures/cad/fixtureC.step")


def plate(hole_radius_mm: float, fillet_radius_mm: float):
    """A plate with four through holes and four filleted vertical corners."""
    import build123d as bd

    with bd.BuildPart() as part:
        bd.Box(100, 60, 10)
        bd.fillet(part.edges().filter_by(bd.Axis.Z), radius=fillet_radius_mm)
        with bd.Locations((30, 18, 0), (-30, 18, 0), (30, -18, 0),
                          (-30, -18, 0)):
            bd.Hole(radius=hole_radius_mm)
    return part.part.wrapped


# ------------------------------------------- a hole is concave, not small

@requires_occ
@requires_cad
@pytest.mark.parametrize("hole_r,fillet_r", [(4.0, 3.0), (2.5, 6.0)])
def test_holes_are_found_with_their_diameters(hole_r, fillet_r):
    report = fr.recognise(plate(hole_r, fillet_r), 1e-3)
    assert report.hole_count == 4
    assert report.hole_diameters_m() == pytest.approx(
        [2 * hole_r * 1e-3] * 4, rel=1e-9)


@requires_occ
@requires_cad
def test_equal_radii_do_not_merge_holes_into_fillets():
    """The trap. A radius based rule sees eight cylinders of one size.

    Only concavity separates them, and this is the case that proves the rule
    is doing the work rather than the numbers happening to differ.
    """
    report = fr.recognise(plate(4.0, 4.0), 1e-3)
    assert report.hole_count == 4
    assert report.fillet_count == 4
    assert report.hole_diameters_m() == pytest.approx([8e-3] * 4, rel=1e-9)
    assert report.fillet_radii_m() == pytest.approx([4e-3] * 4, rel=1e-9)


# ------------------------ a fillet blends, and blending means two neighbours

@requires_occ
@requires_cad
def test_corner_fillets_are_found_though_they_meet_the_ends_squarely():
    """The case that refuted 'tangent to every neighbour'.

    A corner fillet blends the two side walls and runs into the top and bottom
    faces at a right angle. Requiring every neighbour to be tangent found none
    of them.
    """
    report = fr.recognise(plate(4.0, 3.0), 1e-3)
    assert report.fillet_count == 4
    assert report.fillet_radii_m() == pytest.approx([3e-3] * 4, rel=1e-9)
    assert all(f.surface_kind == "cylinder" for f in report.fillets)


@requires_occ
@requires_cad
def test_a_fillet_is_a_cylinder_a_sphere_or_a_torus():
    """Filleting every edge of a box makes spheres, and no torus at all."""
    import build123d as bd

    with bd.BuildPart() as box:
        bd.Box(60, 40, 20)
        bd.fillet(box.edges(), radius=4.0)
    report = fr.recognise(box.part.wrapped, 1e-3)
    kinds = collections.Counter(f.surface_kind for f in report.fillets)
    assert kinds == {"cylinder": 12, "sphere": 8}
    assert report.fillet_radii_m() == pytest.approx([4e-3] * 20, rel=1e-9)


@requires_occ
@pytest.mark.skipif(not FIXTURE_C.exists(), reason="fixture C is not present")
def test_a_body_wall_is_not_reported_as_a_fillet():
    """The trap that came with the Fusion fixture, and the better one.

    The part is a cylinder with a filleted rim, so its wall is a CONVEX
    cylinder. Any convexity rule reports the part as its own fillet. The wall
    is tangent to the fillet and square to the base, so it blends only one
    face and is not a blend.
    """
    from nodes.step_analyzer import read_step

    contents = read_step(FIXTURE_C)
    report = fr.recognise(contents.shapes[0], contents.unit_to_metres)
    assert report.hole_count == 0
    assert report.fillet_count == 1
    assert report.fillets[0].surface_kind == "torus"
    assert report.fillets[0].radius_m == pytest.approx(3e-3, rel=1e-9)
    # The wall is left unclassified rather than forced into a category.
    assert report.unclassified_faces == 1


@requires_occ
@pytest.mark.skipif(not FIXTURE_C.exists(), reason="fixture C is not present")
def test_the_torus_radius_reported_is_the_minor_one():
    """The major radius is the path the fillet runs along, not its size."""
    from nodes.step_analyzer import read_step

    contents = read_step(FIXTURE_C)
    report = fr.recognise(contents.shapes[0], contents.unit_to_metres)
    assert report.fillet_radii_m()[0] == pytest.approx(3e-3, rel=1e-9)
    # 17 mm is the major radius, and reporting it would be wrong by a factor
    # of nearly six.
    assert report.fillet_radii_m()[0] != pytest.approx(17e-3, rel=1e-3)


# ------------------------------------------------------------- the limits

def test_the_capability_states_what_it_will_not_decide():
    method = fr.feature_recognizer_capability_method()
    assert "never what it is FOR" in method.notes
    assert "through versus blind is not decided" in method.notes
    assert method.evidence == "SIMULATED"


def test_a_missing_kernel_raises_rather_than_returning_nothing(monkeypatch):
    from nodes.descriptor import CapabilityUnavailable

    monkeypatch.setattr(fr, "_occ", lambda: None)
    with pytest.raises(CapabilityUnavailable):
        fr.recognise(object(), 1.0)


# ------------------------- fixture A, authored in Fusion with known answers

FIXTURE_A = Path("tests/fixtures/cad/fixtureA.step")
requires_a = pytest.mark.skipif(not FIXTURE_A.exists(),
                                reason="fixture A is not present")



def _faces_of(shape):
    """Every face of a shape, once each."""
    from OCP.TopAbs import TopAbs_ShapeEnum
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        faces.append(TopoDS.Face_s(explorer.Current()))
        explorer.Next()
    return faces

def read_fixture(path):
    from nodes.step_analyzer import read_step

    contents = read_step(path)
    return contents.shapes[0], contents.unit_to_metres


@requires_occ
@requires_a
def test_fixture_a_holes_match_the_supplied_answers():
    """A plate authored in another kernel, with the answers sent alongside."""
    report = fr.recognise(*read_fixture(FIXTURE_A))
    assert report.hole_count == 4
    assert report.hole_diameters_m() == pytest.approx([8e-3] * 4, rel=1e-9)


@requires_occ
@requires_a
def test_fixture_a_fillets_match_the_supplied_answers():
    report = fr.recognise(*read_fixture(FIXTURE_A))
    assert report.fillet_count == 4
    assert report.fillet_radii_m() == pytest.approx([3e-3] * 4, rel=1e-9)
    assert all(f.surface_kind == "cylinder" for f in report.fillets)


@requires_occ
@requires_a
def test_fixture_a_hole_positions_match():
    """Positions, not just counts: four holes in the wrong places would pass
    a count check and be entirely wrong."""
    report = fr.recognise(*read_fixture(FIXTURE_A))
    centres = sorted((round(h.point_on_axis_m[0] * 1000, 6),
                      round(h.point_on_axis_m[1] * 1000, 6))
                     for h in report.holes)
    assert centres == [(15.0, 15.0), (15.0, 45.0), (85.0, 15.0), (85.0, 45.0)]


@requires_occ
@requires_a
def test_the_hole_axis_sign_is_canonical():
    """A hole is a line and does not point anywhere.

    OpenCASCADE returned (0,0,-1) where Fusion reported (0,0,1). Both describe
    the same line, so the sign is canonicalised rather than left to look like
    a disagreement.
    """
    report = fr.recognise(*read_fixture(FIXTURE_A))
    for hole in report.holes:
        assert hole.axis[2] == pytest.approx(1.0)


@requires_occ
@requires_a
def test_nothing_in_fixture_a_is_left_unclassified():
    """Fourteen faces: six planar, four holes, four fillets."""
    report = fr.recognise(*read_fixture(FIXTURE_A))
    assert report.unclassified_faces == 0


# ------------------- fixture B, the same trap authored in a different kernel

FIXTURE_B = Path("tests/fixtures/cad/fixtureB.step")
requires_b = pytest.mark.skipif(not FIXTURE_B.exists(),
                                reason="fixture B is not present")


@requires_occ
@requires_b
def test_fixture_b_every_cylinder_shares_one_radius():
    """The premise of the trap, asserted so it cannot quietly stop being true.

    All eight cylindrical faces are R4. Any rule that sorts holes from fillets
    by radius has nothing to sort on here.
    """
    text = FIXTURE_B.read_text(errors="ignore")
    assert text.count("=CYLINDRICAL_SURFACE") == 8
    assert text.count("CYLINDRICAL_SURFACE('',#") == 8
    assert len(re.findall(r"=CYLINDRICAL_SURFACE\('',#\d+,4\.\);", text)) == 8


@requires_occ
@requires_b
def test_fixture_b_splits_equal_radii_by_concavity():
    """Four holes and four fillets, all R4, separated only by concavity."""
    report = fr.recognise(*read_fixture(FIXTURE_B))
    assert report.hole_count == 4
    assert report.fillet_count == 4
    assert report.hole_diameters_m() == pytest.approx([8e-3] * 4, rel=1e-9)
    assert report.fillet_radii_m() == pytest.approx([4e-3] * 4, rel=1e-9)
    assert report.unclassified_faces == 0


@requires_occ
@requires_b
def test_fixture_b_hole_positions_match():
    """Counts alone would pass even if holes and fillets were swapped: both
    groups are four faces of radius 4. The positions are what prove which is
    which, since the holes are inboard and the fillets are at the corners."""
    report = fr.recognise(*read_fixture(FIXTURE_B))
    centres = sorted((round(h.point_on_axis_m[0] * 1000, 6),
                      round(h.point_on_axis_m[1] * 1000, 6))
                     for h in report.holes)
    assert centres == [(15.0, 15.0), (15.0, 45.0), (85.0, 15.0), (85.0, 45.0)]


# --------------------------------- fixture D, a chamfer, which is neither

FIXTURE_D = Path("tests/fixtures/cad/fixtureD.step")
requires_d = pytest.mark.skipif(not FIXTURE_D.exists(),
                                reason="fixture D is not present")


@requires_occ
@requires_d
def test_a_chamfer_is_not_reported_as_a_feature():
    """The rules must decline to name what they cannot name.

    A chamfer is a cone. It is not a hole, and a 45 degree cone is not tangent
    to either neighbour, so it is not a fillet either. Claiming it as a fillet
    would mean the tangency test is not actually testing tangency. Recognising
    chamfers properly is later work; the requirement here is only that nothing
    is invented.
    """
    report = fr.recognise(*read_fixture(FIXTURE_D))
    assert report.hole_count == 0
    assert report.fillet_count == 0
    assert report.unclassified_faces == 2


@requires_occ
@requires_d
def test_fixture_d_really_does_contain_a_cone():
    """The premise, asserted so the fixture cannot quietly stop being a trap."""
    text = FIXTURE_D.read_text(errors="ignore")
    assert text.count("=CONICAL_SURFACE") == 1
    assert text.count("=CYLINDRICAL_SURFACE") == 1


# ------------- fixture E, the one that refuted "a hole is a concave cylinder"

FIXTURE_E = Path("tests/fixtures/cad/fixtureE.step")
requires_e = pytest.mark.skipif(not FIXTURE_E.exists(),
                                reason="fixture E is not present")


@requires_occ
@requires_e
def test_a_concave_fillet_is_not_a_hole():
    """An L bracket's reentrant corner blend is concave, and is not a hole.

    This is the case that refuted the original rule. Concavity cannot separate
    a bore from a reentrant blend because both are concave; the blend is a
    ninety degree sector while a bore wraps a full turn.
    """
    report = fr.recognise(*read_fixture(FIXTURE_E))
    assert report.hole_count == 0
    assert report.fillet_count == 1
    assert report.fillets[0].radius_m == pytest.approx(5e-3, rel=1e-9)
    assert report.fillets[0].surface_kind == "cylinder"
    assert report.unclassified_faces == 0


@requires_occ
@requires_e
def test_the_fillet_in_fixture_e_really_is_concave():
    """The premise. If this face ever stopped being concave the fixture would
    still pass while testing nothing."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType

    shape, _ = read_fixture(FIXTURE_E)
    cylinders = [f for f in _faces_of(shape)
                 if BRepAdaptor_Surface(f).GetType()
                 == GeomAbs_SurfaceType.GeomAbs_Cylinder]
    assert len(cylinders) == 1
    face = cylinders[0]
    assert fr._is_concave_cylinder(face, BRepAdaptor_Surface(face))


@requires_occ
@requires_e
def test_a_hole_must_wrap_a_full_turn():
    """The new condition, stated directly rather than only through outcomes."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    shape, _ = read_fixture(FIXTURE_A)
    spans = [BRepAdaptor_Surface(f) for f in _faces_of(shape)]
    full = [a for a in spans if fr._wraps_a_full_turn(a)]
    assert len(full) == 4, "the four bores wrap a full turn"

    shape_e, _ = read_fixture(FIXTURE_E)
    assert not any(fr._wraps_a_full_turn(BRepAdaptor_Surface(f))
                   for f in _faces_of(shape_e)), \
        "the reentrant blend is a sector, not a full turn"

