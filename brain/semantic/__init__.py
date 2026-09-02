"""brain.semantic - evidence-graded knowledge and generalization."""

from .evidence import (
    DEFAULT_POLICY,
    LEVEL_CONFIDENCE_CEILING,
    LEVEL_ORDER,
    VERDICT_FLOOR,
    Counterexample,
    Evidence,
    EvidenceKind,
    EvidenceLevel,
    PromotionPolicy,
    compute_confidence,
    derive_level,
    grounded,
    independent_runs,
    may_decide,
    unresolved,
)
from .generalize import (
    DOMINANCE_THRESHOLD,
    generalize_all,
    generalize_binding_constraint,
    generalize_bound_activity,
)
from .knowledge import Knowledge, SemanticMemory
from .lessons import record_fidelity_lesson

__all__ = [
    "DEFAULT_POLICY", "DOMINANCE_THRESHOLD", "LEVEL_CONFIDENCE_CEILING",
    "LEVEL_ORDER", "VERDICT_FLOOR", "Counterexample", "Evidence", "EvidenceKind",
    "EvidenceLevel", "Knowledge", "PromotionPolicy", "SemanticMemory",
    "compute_confidence", "derive_level", "generalize_all",
    "generalize_binding_constraint", "generalize_bound_activity", "grounded",
    "independent_runs", "may_decide", "record_fidelity_lesson", "unresolved",
]
