"""Catalogue solids: real where the dimensions exist, labelled where they do not.

The whole point of this module is that the two cases stay distinguishable, so
most of these tests are about the labelling rather than the geometry.
"""

from __future__ import annotations

import math

import pytest

from drivetrain.bearings.catalog import all_bearings
from drivetrain.gearboxes.catalog import gearboxes
from drivetrain.motors.catalog import motors
from geometry.cad_export.kernel import kernel_available

requires_cad = pytest.mark.skipif(not kernel_available(),
                                  reason="no CAD kernel installed")

if kernel_available():
    from geometry.cad_export.catalog_parts import (
        ASSUMED_PLACEHOLDER_DENSITY_KG_M3, PLACEHOLDER, bearing_solid,
        check_interference, gearbox_placeholder, interference_m3,
        motor_placeholder, place)


def ring_volume_m3(spec) -> float:
    return (math.pi / 4.0
            * (spec.outer_diameter_m ** 2 - spec.bore_m ** 2) * spec.width_m)


# ------------------------------------------------- bearings: real geometry

@requires_cad
@pytest.mark.parametrize("spec", all_bearings(), ids=lambda s: s.designation)
def test_a_bearing_solid_matches_its_iso_dimensions(spec):
    """A 6206 is 30 by 62 by 16 from any manufacturer. That is a standard."""
    part = bearing_solid(spec)
    assert part.geometry_is_real
    assert part.volume_m3 == pytest.approx(ring_volume_m3(spec), rel=1e-9)
    x, y, z = part.bounding_box_m
    assert x == pytest.approx(spec.outer_diameter_m, rel=1e-9)
    assert y == pytest.approx(spec.outer_diameter_m, rel=1e-9)
    assert z == pytest.approx(spec.width_m, rel=1e-9)


@requires_cad
def test_a_bearing_is_not_labelled_a_placeholder():
    part = bearing_solid(all_bearings()[0])
    assert not part.is_placeholder
    assert PLACEHOLDER not in part.name
    assert "ISO" in part.source


# --------------------------------- motors and gearboxes: labelled inventions

@requires_cad
@pytest.mark.parametrize("spec", motors(), ids=lambda s: s.id)
def test_a_motor_placeholder_carries_only_its_mass(spec):
    """The shape is invented, so mass is the one thing it may claim."""
    part = motor_placeholder(spec)
    assert part.is_placeholder
    assert PLACEHOLDER in part.name
    assert part.mass_kg == spec.mass_kg
    assert part.volume_m3 * ASSUMED_PLACEHOLDER_DENSITY_KG_M3 == pytest.approx(
        spec.mass_kg, rel=1e-9)


@requires_cad
@pytest.mark.parametrize("spec", gearboxes(), ids=lambda s: s.id)
def test_a_gearbox_placeholder_carries_only_its_mass(spec):
    part = gearbox_placeholder(spec)
    assert part.is_placeholder
    assert PLACEHOLDER in part.name
    assert part.volume_m3 * ASSUMED_PLACEHOLDER_DENSITY_KG_M3 == pytest.approx(
        spec.mass_kg, rel=1e-9)


@requires_cad
def test_the_placeholder_marker_survives_into_anything_a_human_reads():
    """Grepping for it must find every invented shape."""
    part = motor_placeholder(motors()[0])
    assert PLACEHOLDER in part.name
    assert PLACEHOLDER in part.describe()
    assert "invented" in part.source


@requires_cad
def test_the_assumed_density_is_stated_in_the_source_not_hidden():
    part = motor_placeholder(motors()[0], density_kg_m3=2700.0)
    assert "2700" in part.source
    assert part.volume_m3 * 2700.0 == pytest.approx(motors()[0].mass_kg,
                                                    rel=1e-9)


@requires_cad
def test_a_massless_part_cannot_become_a_placeholder():
    """Volume comes from mass, so a zero mass has no shape to derive."""
    spec = motors()[0].model_copy(update={"mass_kg": 0.0})
    with pytest.raises(ValueError, match="positive catalogue mass"):
        motor_placeholder(spec)


# ------------------------------------------------ placement and interference

@requires_cad
def test_two_identical_bearings_at_one_place_overlap_completely():
    """Checked against the closed form, not against the checker itself."""
    spec = all_bearings()[0]
    part = bearing_solid(spec)
    overlap = interference_m3(place(part, (0, 0, 0)), place(part, (0, 0, 0)))
    assert overlap == pytest.approx(ring_volume_m3(spec), rel=1e-9)


@requires_cad
def test_half_an_offset_along_the_axis_overlaps_half():
    spec = all_bearings()[0]
    part = bearing_solid(spec)
    overlap = interference_m3(place(part, (0, 0, 0)),
                              place(part, (0, 0, spec.width_m / 2)))
    assert overlap == pytest.approx(ring_volume_m3(spec) / 2.0, rel=1e-6)


@requires_cad
def test_separated_parts_report_no_interference():
    spec = all_bearings()[0]
    part = bearing_solid(spec)
    report = check_interference([place(part, (0, 0, 0)),
                                 place(part, (0, 0, spec.width_m * 1.5))])
    assert report.is_clear
    assert report.summary() == "no interference"


@requires_cad
def test_a_small_bearing_nests_inside_a_larger_bore_without_touching():
    """Not a false negative: a 26 mm outer ring clears a 30 mm bore."""
    small = min(all_bearings(), key=lambda s: s.outer_diameter_m)
    large = max(all_bearings(), key=lambda s: s.bore_m)
    assert small.outer_diameter_m < large.bore_m
    report = check_interference([place(bearing_solid(large), (0, 0, 0)),
                                 place(bearing_solid(small), (0, 0, 0))])
    assert report.is_clear


@requires_cad
def test_an_interference_involving_a_placeholder_says_so():
    """A clash against an invented shape is a claim about an assumption."""
    bearing = place(bearing_solid(all_bearings()[0]), (0, 0, 0))
    motor = place(motor_placeholder(motors()[0]), (0, 0, 0))
    report = check_interference([bearing, motor])
    assert not report.is_clear
    assert report.involves_placeholder
    assert "placeholder" in report.summary()


@requires_cad
def test_placement_does_not_consume_the_original():
    """Two placements of one catalogue solid must not alias."""
    part = bearing_solid(all_bearings()[0])
    first = place(part, (0, 0, 0))
    second = place(part, (0.5, 0, 0))
    assert first.bounding_box_m == pytest.approx(second.bounding_box_m)
    assert interference_m3(first, second) == 0.0
    assert part.volume_m3 == pytest.approx(first.volume_m3, rel=1e-12)
