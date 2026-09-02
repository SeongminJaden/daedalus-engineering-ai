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
from .descriptors import DESCRIPTOR_NAMES, ShapeDescriptor, describe_step
from .classify import (UNKNOWN, Classification, NearestNeighbourClassifier,
                       classification_label, evaluate, rule_classify)
from .pointcloud import (canonical_frame, d2_signature, normalise,
                         point_cloud_of, sample_surface, tessellate)
from .embedding import (EMBEDDING_DIM, POINTS_PER_PART, EmbeddingBundle,
                        embedding_label, nearest_neighbour_precision,
                        train_embedding)
from .shape_surrogate import (ShapePrediction, ShapeScreeningResult,
                              ShapeSurrogate, ShapeTrainingSet, beam_proxy_m,
                              screen_and_verify_parts, train_shape_surrogate,
                              training_set_from)
from .intent import (AblationResult, Direction, IntentClaim, Outcome, ablate,
                     ablated_parameters, intent_claims, record_in_brain)

__all__ = ["AblationResult", "DESCRIPTOR_NAMES", "Classification",
           "Direction", "EMBEDDING_DIM", "IntentClaim", "Outcome", "ablate",
           "ablated_parameters", "intent_claims", "record_in_brain",
           "ShapePrediction", "ShapeScreeningResult", "ShapeSurrogate",
           "ShapeTrainingSet", "beam_proxy_m", "screen_and_verify_parts",
           "train_shape_surrogate", "training_set_from",
           "EmbeddingBundle", "FAMILIES", "Family", "POINTS_PER_PART",
           "canonical_frame", "d2_signature", "embedding_label",
           "nearest_neighbour_precision", "normalise", "point_cloud_of",
           "sample_surface", "tessellate", "train_embedding",
           "GenerationReport", "GeometrySummary", "NearestNeighbourClassifier",
           "ShapeDescriptor", "UNKNOWN", "classification_label",
           "describe_step", "evaluate", "rule_classify",
           "LABEL_CEILING", "LabelReport", "Licence", "LoadCase",
           "PartRecord", "Provenance", "ProvenanceKind", "SCHEMA_VERSION",
           "SYNTHETIC_PROVENANCE", "TopologySummary", "cantilever_labels",
           "family", "generate_dataset", "label", "labelling_available",
           "make_part", "part_id_for", "read_jsonl", "sample_parameters",
           "validate_record", "write_jsonl"]
