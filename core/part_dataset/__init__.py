"""The Part Dataset: one record shape, whatever produced the part, and the
synthetic engine that produces parts in bulk without scraped CAD."""

from .schema import (LABEL_CEILING, SCHEMA_VERSION, GeometrySummary, Licence,
                     PartRecord, Provenance, ProvenanceKind, TopologySummary,
                     label, validate_record)
from .families import (FAMILIES, Family, family, sample_parameters,
                       part_id_for)
from .labeller import (LabelReport, LoadCase, cantilever_labels,
                       labelling_available)
from .store import read_jsonl, write_jsonl
from .engine import (SYNTHETIC_PROVENANCE, GenerationReport, generate_dataset,
                     make_part)

__all__ = ["FAMILIES", "Family", "GenerationReport", "GeometrySummary",
           "LABEL_CEILING", "LabelReport", "Licence", "LoadCase",
           "PartRecord", "Provenance", "ProvenanceKind", "SCHEMA_VERSION",
           "SYNTHETIC_PROVENANCE", "TopologySummary", "cantilever_labels",
           "family", "generate_dataset", "label", "labelling_available",
           "make_part", "part_id_for", "read_jsonl", "sample_parameters",
           "validate_record", "write_jsonl"]
