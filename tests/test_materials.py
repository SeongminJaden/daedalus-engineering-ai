"""Material database: schema, values, provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.materials import MaterialSpec, MaterialStatus, load_materials  # noqa: E402

DB = load_materials()


def test_expected_materials_present():
    assert "al_7075_t6" in DB.ids()
    assert "al_6061_t6" in DB.ids()


@pytest.mark.parametrize("material_id", [m.id for m in DB.materials])
def test_all_properties_positive(material_id):
    m = DB.get(material_id)
    for field in ("density_kg_m3", "youngs_modulus_pa", "yield_strength_pa",
                  "ultimate_strength_pa", "poisson_ratio", "shear_modulus_pa",
                  "fatigue_strength_pa"):
        assert getattr(m, field) > 0, f"{material_id}.{field} must be > 0"


@pytest.mark.parametrize("material_id", [m.id for m in DB.materials])
def test_yield_below_ultimate(material_id):
    m = DB.get(material_id)
    assert m.yield_strength_pa < m.ultimate_strength_pa


@pytest.mark.parametrize("material_id", [m.id for m in DB.materials])
def test_provenance_recorded(material_id):
    """Every value must carry where it came from and how far to trust it."""
    m = DB.get(material_id)
    assert m.source.strip()
    assert isinstance(m.status, MaterialStatus)


def test_al_7075_t6_values():
    """Exact values as specified - guards against silent edits."""
    m = DB.get("al_7075_t6")
    assert m.density_kg_m3 == 2810.0
    assert m.youngs_modulus_pa == 71.7e9
    assert m.yield_strength_pa == 503.0e6
    assert m.ultimate_strength_pa == 572.0e6
    assert m.poisson_ratio == 0.33
    assert m.shear_modulus_pa == 26.9e9
    assert m.fatigue_strength_pa == 159.0e6
    assert m.status is MaterialStatus.REFERENCE_TYPICAL


def test_al_6061_t6_values():
    m = DB.get("al_6061_t6")
    assert m.density_kg_m3 == 2700.0
    assert m.youngs_modulus_pa == 68.9e9
    assert m.yield_strength_pa == 276.0e6
    assert m.ultimate_strength_pa == 310.0e6
    assert m.poisson_ratio == 0.33
    assert m.shear_modulus_pa == 26.0e9
    assert m.fatigue_strength_pa == 96.5e6
    assert m.status is MaterialStatus.REFERENCE_TYPICAL


def test_unknown_material_raises():
    with pytest.raises(KeyError):
        DB.get("unobtainium")


def test_yield_above_ultimate_rejected():
    with pytest.raises(ValidationError):
        MaterialSpec(
            id="bad", name="Bad", density_kg_m3=1000.0, youngs_modulus_pa=1e9,
            yield_strength_pa=500e6, ultimate_strength_pa=400e6,
            poisson_ratio=0.3, shear_modulus_pa=1e9, fatigue_strength_pa=1e6,
            source="test", status=MaterialStatus.ASSUMED,
        )


def test_negative_property_rejected():
    with pytest.raises(ValidationError):
        MaterialSpec(
            id="bad", name="Bad", density_kg_m3=-1.0, youngs_modulus_pa=1e9,
            yield_strength_pa=100e6, ultimate_strength_pa=200e6,
            poisson_ratio=0.3, shear_modulus_pa=1e9, fatigue_strength_pa=1e6,
            source="test", status=MaterialStatus.ASSUMED,
        )


def test_allowable_stress():
    m = DB.get("al_7075_t6")
    assert m.allowable_stress_pa(2.0) == pytest.approx(503.0e6 / 2.0)
