"""The Part Dataset schema: one record shape, whatever produced the part."""

from .schema import (SCHEMA_VERSION, GeometrySummary, Licence, PartRecord,
                     Provenance, ProvenanceKind, TopologySummary,
                     validate_record)

__all__ = ["SCHEMA_VERSION", "GeometrySummary", "Licence", "PartRecord",
           "Provenance", "ProvenanceKind", "TopologySummary",
           "validate_record"]
