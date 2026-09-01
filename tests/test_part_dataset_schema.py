"""The Part Dataset schema, and the things it must refuse to store.

Most of these tests are about refusal. A schema that accepts a record it
cannot account for is how a public repository ends up shipping somebody else's
CAD, and that failure is silent until it is expensive.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from core.part_dataset import (SCHEMA_VERSION, GeometrySummary, Licence,
                               PartRecord, Provenance, ProvenanceKind,
                               TopologySummary, validate_record)


def permissive() -> Licence:
    return Licence(identifier="CC-BY-4.0", url="https://creativecommons.org",
                   redistributable=True)


def geometry() -> GeometrySummary:
    return GeometrySummary(volume_m3=1.112e-4, surface_area_m2=0.112,
                           bounding_box_m=(0.55, 0.04, 0.03),
                           centre_of_mass_m=(-0.118, 0.0, 0.0))


def topology() -> TopologySummary:
    return TopologySummary(solids=2, shells=2, faces=20, edges=96,
                           vertices=192)


def record(**overrides) -> PartRecord:
    fields = dict(
        part_id="arm-link",
        provenance=Provenance(kind=ProvenanceKind.SYNTHETIC_PARAMETRIC,
                              source="daedalus parametric generator",
                              licence=permissive(),
                              generator="hollow_rect"),
        geometry=geometry(), topology=topology())
    fields.update(overrides)
    return PartRecord(**fields)


# ------------------------------------------------------- provenance is required

def test_a_part_cannot_exist_without_provenance():
    """Not optional metadata. A record without it cannot be constructed."""
    with pytest.raises(ValidationError):
        PartRecord(part_id="x", geometry=geometry(), topology=topology())


def test_a_provenance_cannot_exist_without_a_licence():
    with pytest.raises(ValidationError):
        Provenance(kind=ProvenanceKind.PUBLIC_DATASET, source="somewhere")


@pytest.mark.parametrize("vague", ["unknown", "Unspecified", "none", "n/a",
                                   "TBD"])
def test_an_unnamed_licence_cannot_be_redistributable(vague):
    """It was on the internet is not a licence."""
    with pytest.raises(ValidationError, match="not a licence"):
        Licence(identifier=vague, redistributable=True)


def test_an_unnamed_licence_is_fine_when_it_admits_it_is_not_redistributable():
    """Recording a part you may not publish is allowed. Publishing is not."""
    licence = Licence(identifier="unknown", redistributable=False)
    assert not licence.redistributable


def test_proprietary_local_cannot_be_marked_redistributable():
    """The exact combination by which private CAD reaches a public repo."""
    with pytest.raises(ValidationError, match="proprietary local"):
        Provenance(kind=ProvenanceKind.PROPRIETARY_LOCAL, source="customer",
                   licence=Licence(identifier="proprietary",
                                   redistributable=True))


def test_publishability_needs_both_a_licence_and_a_permitted_origin():
    assert record().is_publishable

    private = record(provenance=Provenance(
        kind=ProvenanceKind.PROPRIETARY_LOCAL, source="customer",
        licence=Licence(identifier="proprietary", redistributable=False)))
    assert not private.is_publishable

    unlicensed = record(provenance=Provenance(
        kind=ProvenanceKind.PUBLIC_DATASET, source="a website",
        licence=Licence(identifier="unknown", redistributable=False)))
    assert not unlicensed.is_publishable


# ------------------------------------------------ the geometry has to cohere

def test_a_volume_larger_than_its_own_bounding_box_is_refused():
    """The cheapest possible check that two numbers came from one solid."""
    with pytest.raises(ValidationError, match="exceeds its own bounding box"):
        GeometrySummary(volume_m3=1.0, surface_area_m2=1.0,
                        bounding_box_m=(0.1, 0.1, 0.1),
                        centre_of_mass_m=(0.0, 0.0, 0.0))


def test_a_degenerate_bounding_box_is_refused():
    with pytest.raises(ValidationError):
        GeometrySummary(volume_m3=1e-9, surface_area_m2=1e-6,
                        bounding_box_m=(0.1, 0.0, 0.1),
                        centre_of_mass_m=(0.0, 0.0, 0.0))


def test_a_solid_needs_at_least_one_face():
    with pytest.raises(ValidationError):
        TopologySummary(solids=1, shells=1, faces=0, edges=0, vertices=0)


# ------------------------------------------------------------- versioning

def test_a_record_from_an_unknown_schema_is_refused_not_guessed():
    """A silently misread dataset is worse than an absent one."""
    payload = json.loads(record().model_dump_json())
    payload["schema_version"] = "9.9.9"
    with pytest.raises(ValidationError, match="refusing rather than guessing"):
        validate_record(payload)


def test_a_record_round_trips_through_json():
    """The stored form has to survive being stored."""
    original = record()
    restored = validate_record(json.loads(original.model_dump_json()))
    assert restored == original
    assert restored.schema_version == SCHEMA_VERSION


def test_unexpected_fields_are_refused_rather_than_dropped():
    """Silently discarding a field is how two producers drift apart."""
    payload = json.loads(record().model_dump_json())
    payload["mystery"] = 1
    with pytest.raises(ValidationError):
        validate_record(payload)


# -------------------------------------------- one schema, two producers

def test_both_producers_yield_the_same_record_shape():
    """A synthetic part and a scraped one differ in provenance, not in shape.

    That is the point of one schema: a model trained on synthetic parts can be
    evaluated on real ones, and the two remain tellable apart afterwards.
    """
    synthetic = record()
    scraped = record(provenance=Provenance(
        kind=ProvenanceKind.PUBLIC_DATASET,
        source="a permissively licensed public set",
        licence=permissive(), retrieved="2026-09-01"))
    assert set(synthetic.model_dump()) == set(scraped.model_dump())
    assert synthetic.provenance.kind is not scraped.provenance.kind


def test_labels_carry_their_own_evidence_rather_than_the_record():
    """A part may have an exact mass and a surrogate stress at once.

    Averaging those into one number for the record would destroy exactly the
    distinction that decides whether a result may be acted on.
    """
    part = record(labels={
        "mass_kg": {"value": 0.31, "evidence": "simulated"},
        "max_stress_pa": {"value": 1.2e8, "evidence": "surrogate"},
    })
    assert part.labels["mass_kg"]["evidence"] != \
        part.labels["max_stress_pa"]["evidence"]
