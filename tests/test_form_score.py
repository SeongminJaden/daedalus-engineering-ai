"""Form targets consumed: a score that ranks admissible designs and is nothing else."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.design_reference import DesignReference, FormTarget, Provenance
from geometry.aesthetics.form import (SMOOTH_SCALE_RAD, VOCABULARY, form_score,
                                      target_score)
from geometry.aesthetics.metrics import (CUBE_COMPACTNESS,
                                         PREFERENCE_IS_NOT_EVIDENCE,
                                         ShapeMetrics)


def metrics(compactness=CUBE_COMPACTNESS, roughness=0.0, asymmetry=0.0):
    return ShapeMetrics(surface_area_m2=6.0, volume_m3=1.0,
                        compactness=compactness, dihedral_roughness_rad=roughness,
                        mirror_asymmetry_m=asymmetry)


def target(name, word, weight=1.0, confidence=0.5,
           provenance=Provenance.CITED):
    return FormTarget(name=name, source="a design brief", confidence=confidence,
                      provenance=provenance, target=word, weight=weight)


def test_each_word_maps_to_a_stated_function():
    m = metrics(compactness=0.4, roughness=SMOOTH_SCALE_RAD, asymmetry=0.1)
    assert target_score("compact", m, 1.0) == 0.4
    assert target_score("slender", m, 1.0) == pytest.approx(0.6)
    assert target_score("smooth", m, 1.0) == pytest.approx(math.exp(-1.0))
    assert target_score("symmetric", m, 1.0) == pytest.approx(math.exp(-0.1))
    assert target_score("baroque", m, 1.0) is None
    assert set(VOCABULARY) == {"compact", "smooth", "symmetric", "slender"}


def test_a_sphere_is_the_most_compact_and_a_cube_is_pi_over_six():
    assert target_score("compact", metrics(compactness=1.0), 1.0) == 1.0
    assert target_score("compact", metrics(), 1.0) == pytest.approx(math.pi / 6)


def test_weights_are_weight_times_confidence_so_a_guess_pulls_less():
    cited = DesignReference("cited", items=(
        target("round", "compact", weight=1.0, confidence=0.6),))
    guessed = DesignReference("guessed", items=(
        target("round", "compact", weight=1.0, confidence=0.2,
               provenance=Provenance.ASSUMED),))
    m = metrics(compactness=1.0)
    assert form_score(cited, m, 1.0).score == pytest.approx(0.6)
    assert form_score(guessed, m, 1.0).score == pytest.approx(0.2)


def test_unknown_targets_are_listed_not_dropped():
    reference = DesignReference("mixed", items=(
        target("round", "compact"), target("ornate", "baroque")))
    result = form_score(reference, metrics(compactness=1.0), 1.0)
    assert result.unmapped == ("baroque",)
    assert set(result.contributions) == {"round"}
    assert result.note == PREFERENCE_IS_NOT_EVIDENCE


def test_a_reference_without_form_targets_scores_zero():
    result = form_score(DesignReference("plain"), metrics(), 1.0)
    assert result.score == 0.0 and result.contributions == {}


def test_symmetry_needs_a_length_to_be_relative_to():
    with pytest.raises(ValueError, match="characteristic length"):
        target_score("symmetric", metrics(), 0.0)


def test_the_score_ranks_admissible_designs_and_never_rescues_a_failing_one():
    """Through the review: the only consumer, and the guard is structural."""
    from integration import AssemblyVerdict, CheckResult, CheckStatus
    from integration.multi_review import DesignEntry, MultiDesignReview, RankBy

    reference = DesignReference("brief", items=(target("round", "compact"),))
    def entry(name, compactness, passing):
        verdict = AssemblyVerdict()
        verdict.add(CheckResult("link", "yield",
                                CheckStatus.PASSED if passing else CheckStatus.FAILED,
                                "beam", 2.0 if passing else 0.5))
        score = form_score(reference, metrics(compactness=compactness), 1.0).score
        return DesignEntry(name, verdict, 1.0, 1.0, form_score=score)
    review = MultiDesignReview([entry("boxy_sound", CUBE_COMPACTNESS, True),
                                entry("round_sound", 0.9, True),
                                entry("round_broken", 1.0, False)])
    assert [e.name for e in review.ranked(RankBy.FORM)] == ["round_sound", "boxy_sound"]
    assert "round_broken" in [e.name for e in review.rejected]
