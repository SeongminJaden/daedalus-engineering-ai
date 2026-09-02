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
                  "ultimate_strength_pa", "poisson_ratio", "shear_modulus_pa"):
        assert getattr(m, field) > 0, f"{material_id}.{field} must be > 0"
    # fatigue is the one value allowed to be absent, and absent means no
    # source gave one, never zero
    assert m.fatigue_strength_pa is None or m.fatigue_strength_pa > 0


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


# --- provenance per value ------------------------------------------------------

from core.materials import MissingMaterialValue, SourceGrade  # noqa: E402


@pytest.mark.parametrize("material_id", [m.id for m in DB.materials])
def test_every_stored_value_names_its_document(material_id):
    """No number without a document. The one-line `source` is a summary; the
    per-value entry is what lets a reader open the table it came from."""
    m = DB.get(material_id)
    assert m.unsourced_fields() == [], m.unsourced_fields()
    assert m.sources, "no documents listed"
    ids = {d.id for d in m.sources}
    for name, vs in m.value_sources.items():
        assert vs.source in ids
        if vs.grade is SourceGrade.PRIMARY:
            assert vs.location or vs.as_printed, (
                f"{material_id}.{name}: a primary citation has to say where in "
                f"the document")


def test_a_supplier_datasheet_status_rests_on_primary_load_bearing_values():
    """The status claims a named producer's sheet; the yield or the modulus
    had better come from one."""
    for m in DB.materials:
        if m.status is MaterialStatus.SUPPLIER_DATASHEET:
            grades = {m.value_sources[f].grade for f in
                      ("youngs_modulus_pa", "yield_strength_pa",
                       "ultimate_strength_pa", "density_kg_m3")
                      if f in m.value_sources}
            assert SourceGrade.PRIMARY in grades, m.id


def test_the_requested_materials_are_present():
    """The set the coordinating session asked for, by id."""
    for wanted in ("al_6061_t6", "al_7075_t6", "ss_304", "ss_316", "steel_s45c",
                   "steel_scm440", "steel_4140_annealed", "ti_6al_4v",
                   "inconel_718", "mg_az31b", "peek", "pa12", "abs", "pla",
                   "cfrp_ud"):
        assert wanted in DB.ids(), wanted


def test_a_missing_fatigue_value_is_a_refusal_not_a_number():
    m = DB.get("inconel_718")
    assert m.fatigue_strength_pa is None
    with pytest.raises(MissingMaterialValue, match="no sourced fatigue"):
        m.require_fatigue_strength_pa()
    assert DB.get("al_7075_t6").require_fatigue_strength_pa() == 159.0e6


def test_the_fatigue_methods_refuse_a_material_without_an_endurance_value():
    """The consumers ask through the guard, so a None cannot reach the
    arithmetic as a zero or a TypeError."""
    import inspect
    import physics.fatigue.sn as sn
    import physics.fatigue.miner as miner
    import physics.shaft.design as shaft
    for module in (sn, miner, shaft):
        source = inspect.getsource(module)
        assert "material.fatigue_strength_pa" not in source, module.__name__
        assert "require_fatigue_strength_pa()" in source, module.__name__


def test_inconel_modulus_falls_with_temperature_and_refuses_to_extrapolate():
    """Special Metals Table 4: 29.0 to 14.3 x 10^3 ksi from 70 to 2000 F."""
    m = DB.get("inconel_718")
    curve = m.temperature_dependence["youngs_modulus_pa"]
    values = [v for _, v in curve]
    assert values == sorted(values, reverse=True)
    assert m.modulus_at_temperature_pa(294.3) == pytest.approx(199.9e9)
    assert 150e9 < m.modulus_at_temperature_pa(1000.0) < 160e9
    with pytest.raises(MissingMaterialValue, match="no extrapolation"):
        m.modulus_at_temperature_pa(100.0)
    with pytest.raises(MissingMaterialValue, match="no sourced modulus"):
        DB.get("al_7075_t6").modulus_at_temperature_pa(300.0)


def test_the_quasi_isotropic_estimate_is_derived_and_strengthless():
    """3/8 E1 + 5/8 E2 for the Hexcel 8552/AS4 lamina: 59.1 GPa. A stiffness
    only; there is no isotropic CFRP entry because its strength would have
    to be invented."""
    m = DB.get("cfrp_ud")
    assert m.quasi_isotropic_modulus_estimate_pa() == pytest.approx(
        3 / 8 * 141e9 + 5 / 8 * 10e9)
    assert "cfrp_qi" not in " ".join(DB.ids())
    with pytest.raises(MissingMaterialValue):
        DB.get("al_7075_t6").quasi_isotropic_modulus_estimate_pa()


def test_pinned_values_did_not_move_when_provenance_was_added():
    """The MVP result depends on these, so the provenance pass had to leave
    them exactly where they were and record the primary numbers beside."""
    m = DB.get("al_7075_t6")
    assert (m.density_kg_m3, m.youngs_modulus_pa) == (2810.0, 71.7e9)
    assert "71.0 GPa" in m.notes and "2.80" in m.notes


def test_a_value_source_must_name_a_listed_document():
    from core.materials import SourceDocument, ValueSource
    base = dict(id="x", name="x", density_kg_m3=1.0, youngs_modulus_pa=1.0,
                yield_strength_pa=1.0, ultimate_strength_pa=2.0, poisson_ratio=0.3,
                shear_modulus_pa=1.0, source="s", status="reference_typical")
    with pytest.raises(ValidationError, match="not in sources"):
        MaterialSpec(**base, value_sources={"density_kg_m3": ValueSource(
            source="nowhere", grade=SourceGrade.PRIMARY)})
    with pytest.raises(ValidationError, match="unknown field"):
        MaterialSpec(**base, sources=[SourceDocument(
            id="d", title="t", publisher="p", grade=SourceGrade.PRIMARY)],
            value_sources={"not_a_field": ValueSource(source="d",
                                                      grade=SourceGrade.PRIMARY)})
    with pytest.raises(ValidationError, match="has no source"):
        MaterialSpec(**base, temperature_dependence={
            "youngs_modulus_pa": [[300.0, 1.0], [400.0, 0.9]]})
