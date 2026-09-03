"""Three ways the material table is not the material.

A printed part is not the bulk material, a laminate does not fail by a single
yield number, and a ceramic does not yield at all. The first and third are new
here; the second already existed and is checked from the material side so the
CFRP entry cannot be used with the wrong criterion by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.materials import get_material
from core.materials.printed import (PRINTED_MATERIALS, BuildOrientation,
                                    MissingPrintedData, bulk_is_an_upper_bound,
                                    printed_material, printed_strength_pa)
from physics.failure.brittle import (BrittleDataMissing, BrittleLimit,
                                     WeibullStrength, effective_volume_ratio,
                                     max_principal_stress, principal_stresses,
                                     size_scaled_strength_pa)


# --- printed anisotropy ------------------------------------------------------

def test_every_printed_entry_carries_its_document_and_machine():
    for entry in PRINTED_MATERIALS.values():
        assert entry.document_url.startswith("https://")
        assert entry.read_on == "2026-09-03"
        assert entry.machine and entry.standard
        assert entry.directions
        for direction in entry.directions.values():
            assert direction.printed_as, (entry.id, direction.orientation)


def test_the_same_material_on_two_machines_is_two_entries():
    """The measurement that decides the shape of this module: ABS-M30 on an
    F900 and on an F770 are the same material with different anisotropy, so a
    single factor for FDM ABS would be wrong on at least one of them."""
    f900 = printed_material("abs_m30_fdm_f900_t16")
    f770 = printed_material("abs_m30_fdm_f770")
    assert f900.base_material_id == f770.base_material_id == "abs"
    assert f900.machine != f770.machine

    yield_ratio_900 = (f900.directions[BuildOrientation.UPRIGHT_ZX].yield_strength_pa
                       / f900.directions[BuildOrientation.ON_EDGE_XZ].yield_strength_pa)
    yield_ratio_770 = (f770.directions[BuildOrientation.UPRIGHT_ZX].yield_strength_pa
                       / f770.directions[BuildOrientation.ON_EDGE_XZ].yield_strength_pa)
    assert yield_ratio_900 == pytest.approx(0.89, abs=0.01)
    assert yield_ratio_770 == pytest.approx(0.71, abs=0.01)
    assert abs(yield_ratio_900 - yield_ratio_770) > 0.15


def test_the_upright_direction_is_the_weak_one_everywhere_measured():
    for entry in PRINTED_MATERIALS.values():
        weakest = entry.weakest()
        assert weakest.orientation in (BuildOrientation.UPRIGHT_ZX,
                                       BuildOrientation.AXIS_Z)
        assert entry.anisotropy_ratio() < 1.0


def test_ductility_falls_further_than_strength():
    """The number that matters for impact: PA 2200 loses 12 percent of its
    strength across the layers and four fifths of its strain at break."""
    entry = printed_material("pa2200_sls_eos")
    x = entry.directions[BuildOrientation.AXIS_X]
    z = entry.directions[BuildOrientation.AXIS_Z]
    assert z.tensile_strength_pa / x.tensile_strength_pa == pytest.approx(0.875)
    assert z.elongation_at_break_percent / x.elongation_at_break_percent == \
        pytest.approx(4.0 / 18.0)


def test_an_orientation_the_sheet_does_not_print_is_refused():
    with pytest.raises(MissingPrintedData, match="no printed tensile strength"):
        printed_strength_pa("pa2200_sls_eos", BuildOrientation.ON_EDGE_XZ)
    with pytest.raises(MissingPrintedData, match="no printed data"):
        printed_material("pla_fdm_anything")


def test_the_bulk_values_are_declared_an_upper_bound():
    """The database note says a printed part is not isotropic. This is that
    sentence somewhere a check can use it."""
    message = bulk_is_an_upper_bound("abs")
    assert "UPPER BOUND" in message
    assert "not a safety factor" in message
    material = get_material("abs")
    assert "upper bound" in material.notes.lower()


# --- brittle materials -------------------------------------------------------

def test_the_criterion_is_the_largest_principal_stress_not_von_mises():
    """A ceramic under hydrostatic compression is fine and the same von Mises
    value in tension breaks it."""
    tension = np.array([[100e6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    compression = np.array([[-200e6, -200e6, -200e6, 0.0, 0.0, 0.0]])
    assert max_principal_stress(tension)[0] == pytest.approx(100e6)
    assert max_principal_stress(compression)[0] == pytest.approx(-200e6)
    ordered = principal_stresses(np.array([[10e6, -5e6, 3e6, 0.0, 0.0, 0.0]]))[0]
    assert list(ordered) == sorted(ordered, reverse=True)


def test_a_bigger_part_is_weaker_by_the_weibull_rule():
    reference = WeibullStrength(strength_pa=350e6, volume_m3=1e-6, modulus=10.0,
                                source="hypothetical, used only in this test")
    hundred_times = size_scaled_strength_pa(reference, 1e-4)
    assert hundred_times < reference.strength_pa
    assert hundred_times == pytest.approx(350e6 * (1e-6 / 1e-4) ** 0.1)
    # A lower modulus means more scatter and a steeper size effect.
    scattered = WeibullStrength(350e6, 1e-6, 5.0, "hypothetical")
    assert size_scaled_strength_pa(scattered, 1e-4) < hundred_times


def test_a_weibull_strength_needs_a_source_and_a_positive_modulus():
    with pytest.raises(ValueError, match="source"):
        WeibullStrength(350e6, 1e-6, 10.0, "   ")
    with pytest.raises(ValueError, match="modulus must be positive"):
        WeibullStrength(350e6, 1e-6, 0.0, "somewhere")


def test_the_alumina_in_this_database_cannot_be_sized_and_the_check_says_so():
    """The honest state of the database: alumina has a flexural strength from
    a secondary source and no characteristic strength, test volume or modulus,
    so a brittle check refuses rather than treating flexural strength as an
    allowable."""
    alumina = get_material("alumina_al2o3")
    assert "BRITTLE" in alumina.notes
    assert "Weibull" in alumina.notes
    limit = BrittleLimit("alumina_al2o3", None, part_volume_m3=1e-4)
    with pytest.raises(BrittleDataMissing, match="no Weibull strength"):
        limit.allowable_pa()
    with pytest.raises(BrittleDataMissing):
        limit.check(np.array([[10e6, 0.0, 0.0, 0.0, 0.0, 0.0]]))


def test_an_effective_volume_this_module_cannot_integrate_is_refused():
    assert effective_volume_ratio("uniform_tension") == 1.0
    with pytest.raises(BrittleDataMissing, match="will not\\s+guess|will not guess"):
        effective_volume_ratio("four_point_bending")


def test_a_check_with_a_sourced_strength_reports_the_margin_and_the_criterion():
    reference = WeibullStrength(300e6, 1e-6, 12.0, "hypothetical, for the test")
    limit = BrittleLimit("test_ceramic", reference, part_volume_m3=1e-5)
    result = limit.check(np.array([[50e6, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert result["passes"] is True
    assert result["max_principal_stress_pa"] == pytest.approx(50e6)
    assert "not the criterion for a brittle material" in result["note"]
    assert 0.0 < result["margin"] < 1.0


# --- the laminate criterion, from the material side --------------------------

def test_the_cfrp_entry_is_orthotropic_and_has_no_single_yield():
    """A laminate is checked ply by ply with Tsai-Wu or maximum stress, which
    physics.composite already implements. What matters here is that the
    material entry cannot be mistaken for something a single yield number
    describes."""
    from core.materials.db import MaterialClass

    cfrp = get_material("cfrp_ud")
    assert cfrp.material_class is not MaterialClass.ISOTROPIC
    # The entry does carry a fatigue number, and the note says what it is
    # for; what it does not carry is a single strength that describes the
    # lamina in every direction.
    assert cfrp.strength_long_pa > 20 * cfrp.strength_trans_pa
    assert "single yield number is meaningless" in cfrp.notes

    from physics.composite import max_stress_ratio, tsai_wu_index
    from physics.composite.failure import LaminaStrength

    strength = LaminaStrength(longitudinal_tension_pa=cfrp.strength_long_pa,
                              longitudinal_compression_pa=cfrp.strength_long_compressive_pa,
                              transverse_tension_pa=cfrp.strength_trans_pa,
                              transverse_compression_pa=cfrp.strength_trans_compressive_pa,
                              shear_pa=cfrp.strength_shear_pa)
    stress = np.array([500e6, 20e6, 30e6])
    index = tsai_wu_index(stress, strength)
    ratio, mode = max_stress_ratio(stress, strength)
    assert index > 0.0
    assert ratio > 1.0                    # this stress is below first ply failure
    assert mode is not None
