"""brain.retrieval.features - numeric feature vectors for design similarity.

WHAT THIS IS NOT: a semantic or text embedding. There is no language model
here and no notion of meaning. A vector is the design's own engineering
quantities - problem parameters, design variables, resulting metrics - put on a
common scale so "similar" means *numerically similar in engineering terms*.

Calling this semantic search would be an overclaim, so the API says
`retrieve_similar`, not `semantic_search`.

EXTENSION POINT: text/semantic embeddings would need an embedding model, which
is a dependency this phase deliberately does not take (stdlib sqlite3 + numpy
only). Adding one later means implementing another `FeatureSpace` whose
`vector()` returns model embeddings; nothing else in retrieval changes.
Likewise, brute-force search is exact and fine for thousands of designs; an ANN
index (faiss and friends) is the swap for millions, behind the same API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Reference scales, in SI. Fixed rather than data-derived so that a vector
# means the same thing across runs and a stored index never silently shifts
# when new designs arrive. Log scaling is used where a quantity ranges over
# orders of magnitude (stress, deflection, frequency).
FEATURE_NAMES = (
    "length_m",
    "tip_load_n",
    "outer_width_m",
    "outer_height_m",
    "wall_thickness_m",
    "mass_kg",
    "log_max_bending_stress_pa",
    "log_tip_deflection_m",
    "log_safety_factor",
    "log_first_natural_frequency_hz",
)

FEATURE_SCALES = {
    "length_m": (0.0, 2.0),
    "tip_load_n": (0.0, 2000.0),
    "outer_width_m": (0.0, 0.2),
    "outer_height_m": (0.0, 0.2),
    "wall_thickness_m": (0.0, 0.05),
    "mass_kg": (0.0, 10.0),
    "log_max_bending_stress_pa": (0.0, 10.0),      # log10 Pa
    "log_tip_deflection_m": (-9.0, 0.0),           # log10 m
    "log_safety_factor": (-2.0, 4.0),              # log10
    "log_first_natural_frequency_hz": (-1.0, 5.0),  # log10 Hz
}


def _log10(value: float, floor: float = 1e-30) -> float:
    return math.log10(max(float(value), floor))


def _normalize(name: str, value: float) -> float:
    lo, hi = FEATURE_SCALES[name]
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def design_vector(genome: dict, metrics: dict,
                  problem_params: dict | None = None) -> np.ndarray:
    """Feature vector for one evaluated design. Missing values become 0.

    Deterministic and independent of what else is in the store: the same design
    always maps to the same vector.
    """
    p = problem_params or {}
    raw = {
        "length_m": float(p.get("length_m", 0.0)),
        "tip_load_n": float(p.get("tip_load_n", 0.0)),
        "outer_width_m": float(genome.get("outer_width_m", 0.0)),
        "outer_height_m": float(genome.get("outer_height_m", 0.0)),
        "wall_thickness_m": float(genome.get("wall_thickness_m", 0.0)),
        "mass_kg": float(metrics.get("mass_kg", 0.0)),
        "log_max_bending_stress_pa": _log10(
            metrics.get("max_bending_stress_pa", 1e-30)),
        "log_tip_deflection_m": _log10(metrics.get("tip_deflection_m", 1e-30)),
        "log_safety_factor": _log10(metrics.get("safety_factor", 1e-30)),
        "log_first_natural_frequency_hz": _log10(
            metrics.get("first_natural_frequency_hz", 1e-30)),
    }
    return np.array([_normalize(n, raw[n]) for n in FEATURE_NAMES], dtype=float)


def problem_vector(problem_params: dict, target_metrics: dict | None = None,
                   genome: dict | None = None) -> np.ndarray:
    """Query vector when only the problem (and maybe a target) is known."""
    return design_vector(genome or {}, target_metrics or {}, problem_params)


@dataclass
class FeatureSpace:
    """Named so an embedding-backed implementation can replace it later."""

    names: tuple[str, ...] = FEATURE_NAMES

    @property
    def dimension(self) -> int:
        return len(self.names)

    def vector(self, genome: dict, metrics: dict,
               problem_params: dict | None = None) -> np.ndarray:
        return design_vector(genome, metrics, problem_params)
