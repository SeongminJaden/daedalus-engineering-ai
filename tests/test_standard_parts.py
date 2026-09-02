"""Standard parts as solids: dimensions from named tables, volumes against closed forms."""

from __future__ import annotations

import math

import pytest

from geometry.cad_export import standard_parts as sp
from geometry.cad_export.kernel import kernel_available

requires_cad = pytest.mark.skipif(not kernel_available(), reason="build123d is required")


def test_every_table_names_a_source_read_on_the_retrieval_date():
    for source in sp.SOURCES.values():
        assert source.retrieved == "2026-09-03" and source.url.startswith("https://")
    assert set(sp.ISO_4762) == set(sp.ISO_4032) == {"M3", "M4", "M5", "M6", "M8", "M10"}


def test_gt2_geometry_follows_from_pitch_and_the_pitch_line_differential():
    """PowerDrive: 12 grooves P.D. 0.301 in, O.D. 0.281 in; 60 grooves 1.504 and
    1.484 in. The formula reproduces both to the table's rounding."""
    assert sp.gt2_pitch_diameter_m(12) * 1e3 / 25.4 == pytest.approx(0.301, abs=0.0006)
    assert sp.gt2_outside_diameter_m(12) * 1e3 / 25.4 == pytest.approx(0.281, abs=0.0006)
    assert sp.gt2_pitch_diameter_m(60) * 1e3 / 25.4 == pytest.approx(1.504, abs=0.0006)
    assert sp.gt2_outside_diameter_m(60) * 1e3 / 25.4 == pytest.approx(1.484, abs=0.0006)
    assert sp.gt2_belt_length_m(200) == pytest.approx(0.400)


@requires_cad
@pytest.mark.parametrize("size", ["M3", "M5", "M8"])
def test_screw_and_nut_volumes_match_their_closed_forms(size):
    screw = sp.socket_head_screw(size, 0.020)
    assert screw.volume_m3 == pytest.approx(sp.socket_head_screw_volume_m3(size, 0.020), rel=1e-6)
    assert screw.geometry_is_real
    pitch, dk, k, s = sp.ISO_4762[size]
    assert screw.bounding_box_m[0] == pytest.approx(dk * 1e-3, rel=1e-6)
    nut = sp.hex_nut(size)
    assert nut.volume_m3 == pytest.approx(sp.hex_nut_volume_m3(size), rel=1e-6)
    s_flats, m = sp.ISO_4032[size]
    assert nut.bounding_box_m[2] == pytest.approx(m * 1e-3, rel=1e-6)


@requires_cad
def test_pulley_is_an_envelope_and_says_so():
    pulley = sp.gt2_pulley(20, 0.007, 0.010, 0.006, 0.005)
    assert pulley.volume_m3 == pytest.approx(
        sp.gt2_pulley_volume_m3(20, 0.007, 0.010, 0.006, 0.005), rel=1e-6)
    assert not pulley.geometry_is_real
    assert "NOT modelled" in pulley.source
    assert pulley.bounding_box_m[0] == pytest.approx(sp.gt2_outside_diameter_m(20), rel=1e-6)
    with pytest.raises(ValueError, match="inside"):
        sp.gt2_pulley(20, 0.007, 0.020, 0.006, 0.005)


@requires_cad
def test_insert_volume_and_the_material_the_database_does_not_have():
    insert = sp.heat_set_insert("M3")
    assert insert.volume_m3 == pytest.approx(sp.heat_set_insert_volume_m3("M3"), rel=1e-6)
    material_id, mass, note = sp.material_for(insert)
    assert material_id is None and mass is None and "brass" in note


@requires_cad
def test_material_links_give_a_mass_where_the_database_has_the_material():
    material_id, mass, note = sp.material_for(sp.socket_head_screw("M6", 0.03))
    assert material_id == "steel_scm440"
    assert mass == pytest.approx(sp.socket_head_screw_volume_m3("M6", 0.03) * 7850.0, rel=1e-6)
    assert "nearest" in note
