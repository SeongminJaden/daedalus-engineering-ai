"""Turning an analysis shape into a manufacturing one: fillets, fastener
features and tolerances.

The measurements that justify the module are in docs/manufacturing_shape.md.
What is pinned here is that the tables are the standard ones, that the hole
and the screw come from the same source, that a tolerance outside the table is
refused rather than extrapolated, and that the fillet study reports a
comparison it can defend rather than a stress concentration factor it cannot.
"""

from __future__ import annotations

import json

import pytest

from geometry.cad_export.manufacturing_features import (ISO_273_CLEARANCE,
                                                        ISO_2768_LINEAR,
                                                        DrawingNotes,
                                                        ToleranceOutOfRange,
                                                        fastener_feature,
                                                        format_fillet_table,
                                                        general_tolerance_mm)
from geometry.cad_export.standard_parts import ISO_4762


# --- the general tolerance table ---------------------------------------------

def test_the_general_tolerance_is_the_printed_band():
    assert general_tolerance_mm(50.0, "m") == 0.3
    assert general_tolerance_mm(50.0, "f") == 0.15
    assert general_tolerance_mm(50.0, "c") == 0.8
    assert general_tolerance_mm(2.0, "m") == 0.1
    assert general_tolerance_mm(1000.0, "m") == 0.8
    # The bands are half open on the low side, which is how the table reads.
    assert general_tolerance_mm(6.0, "m") == 0.1
    assert general_tolerance_mm(6.001, "m") == 0.2


def test_a_dimension_outside_the_table_is_refused():
    with pytest.raises(ToleranceOutOfRange, match="outside"):
        general_tolerance_mm(0.2, "m")
    with pytest.raises(ToleranceOutOfRange, match="outside"):
        general_tolerance_mm(5000.0, "m")


def test_the_very_coarse_class_has_no_small_band_and_says_so():
    """The table has no entry under 3 mm for class v. A large number would be
    an invention and a small one a lie."""
    with pytest.raises(ToleranceOutOfRange, match="starts at 3 mm"):
        general_tolerance_mm(2.0, "v")
    assert general_tolerance_mm(4.0, "v") == 0.5


def test_the_bands_are_contiguous_and_never_tighten_with_size():
    """The first two bands share the fine and medium values in the printed
    table, so the rule is not decreasing rather than strictly increasing."""
    for (low, high, values), (next_low, _next_high, next_values) in zip(
            ISO_2768_LINEAR, ISO_2768_LINEAR[1:]):
        assert high == next_low
        for grade in ("f", "m", "c"):
            assert next_values[grade] >= values[grade]
    first, last = ISO_2768_LINEAR[0][2], ISO_2768_LINEAR[-1][2]
    for grade in ("f", "m", "c"):
        assert last[grade] > first[grade]


# --- fastener features -------------------------------------------------------

def test_the_hole_and_the_screw_come_from_the_same_table():
    """A counterbore sized from a second head table would eventually disagree
    with the screw the catalogue builds."""
    for size in sorted(set(ISO_273_CLEARANCE) & set(ISO_4762)):
        feature = fastener_feature(size)
        _pitch, head_diameter, head_height, _socket = ISO_4762[size]
        assert feature.counterbore_diameter_mm == pytest.approx(head_diameter + 0.4)
        assert feature.counterbore_depth_mm == pytest.approx(head_height + 0.4)
        assert feature.clearance_diameter_mm > float(size[1:])
        assert "ISO 273" in feature.source and "standard_parts" in feature.source


def test_the_three_clearance_fits_are_ordered():
    close = fastener_feature("M6", "close").clearance_diameter_mm
    normal = fastener_feature("M6", "normal").clearance_diameter_mm
    free = fastener_feature("M6", "free").clearance_diameter_mm
    assert close < normal < free
    assert (close, normal, free) == ISO_273_CLEARANCE["M6"]


def test_an_unknown_size_or_fit_is_refused():
    with pytest.raises(KeyError, match="no clearance data"):
        fastener_feature("M12")
    with pytest.raises(ValueError, match="close, normal or free"):
        fastener_feature("M6", "snug")


# --- drawing notes -----------------------------------------------------------

def test_the_notes_carry_the_step_limitation_in_the_record(tmp_path):
    """AP203 has no tolerance entity, so the notes are written beside the STEP
    and the file itself says why."""
    notes = DrawingNotes(part_id="demo", general_tolerance_class="m",
                         dimensions_mm={"length": 200.0, "bore": 20.0})
    notes.with_fit("bore", 20.0, 7, "g", 6)
    path = notes.write(tmp_path / "demo.notes.json")
    data = json.loads(path.read_text())
    assert "AP203" in data["step_limitation"]
    assert data["general_tolerance_source"].startswith("ISO 2768-1")
    assert notes.tolerance_for("length") == 0.5
    fit = data["fits"][0]
    assert fit["designation"] == "H7/g6"
    assert fit["type"] == "clearance"
    assert 0.0 < fit["min_clearance_mm"] < fit["max_clearance_mm"]
    assert "physics/elements/fits.py" in fit["source"]


# --- the fillet study --------------------------------------------------------

def test_the_fillet_table_reports_against_the_first_row():
    from geometry.cad_export.manufacturing_features import FilletMeasurement

    rows = [FilletMeasurement(0.0, 71.45e6, 8.685e-4, 0.2698, 18659, 0.003),
            FilletMeasurement(0.004, 67.35e6, 8.427e-4, 0.2701, 19113, 0.003)]
    table = format_fillet_table(rows)
    assert "| 0.00 | 71.45 | 1.00 |" in table
    assert "| 4.00 | 67.35 | 0.94 |" in table


@pytest.mark.slow
def test_a_larger_fillet_lowers_the_peak_and_the_sharp_case_does_not_converge():
    """The measurement the module exists for, at a size a test can afford.

    At a fixed mesh the peak falls with radius. The sharp corner is not a
    reference: measured at 4, 3 and 2 mm elements it reads 65.33, 71.45 and
    81.91 MPa, rising with refinement because it is a singularity, while the
    4 mm fillet reads 63.39, 67.35 and 71.72, still rising but far less.
    """
    from core.materials import get_material
    from core.part_dataset.labeller import labelling_available
    from geometry.cad_export.kernel import kernel_available, require_kernel
    from geometry.cad_export.manufacturing_features import fillet_study

    if not (kernel_available() and labelling_available()):
        pytest.skip("build123d, gmsh and CalculiX are required")

    kernel = require_kernel()
    b = kernel.module
    root_length, root_height, beam_height, width, total = 40.0, 40.0, 20.0, 20.0, 200.0

    def build(radius_m):
        root = b.Box(root_length, root_height, width,
                     align=(b.Align.MIN, b.Align.CENTER, b.Align.CENTER))
        beam = b.Pos(root_length, 0, 0) * b.Box(
            total - root_length, beam_height, width,
            align=(b.Align.MIN, b.Align.CENTER, b.Align.CENTER))
        part = root + beam
        if radius_m > 0:
            shoulder = [e for e in part.edges().filter_by(b.Axis.Z)
                        if abs(e.center().X - root_length) < 1e-6
                        and abs(abs(e.center().Y) - beam_height / 2) < 1e-6]
            part = b.fillet(shoulder, radius=radius_m * 1000.0)
        return part

    material = get_material("al_7075_t6")
    rows = fillet_study(build, [0.002, 0.008], material, mesh_size_m=0.005,
                        total_load_n=-500.0)
    assert rows[1].peak_von_mises_pa < rows[0].peak_von_mises_pa
    assert rows[1].mass_kg > rows[0].mass_kg      # a fillet adds material here
    assert all(row.elements > 1000 for row in rows)
