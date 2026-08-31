"""brain.semantic.lessons: knowledge learned from a model being wrong.

The most valuable thing an autonomous loop can record is not a design, it is a
discovered limitation of its own cheap model. Phase 7 found one: a design that
passed Euler-Bernoulli beam theory failed 3D FEM, because the beam model omits
shear deformation and the link was not slender. Phase 7.5 corrected the model
and the re-optimized design passed.

That whole episode is one durable lesson, and this is where it is stored so a
later run does not have to rediscover it.

Evidence stays `SIMULATION`. A cross-check between two simulations, however
much more faithful one of them is, is still not a physical test, so this can
never reach EXPERIMENTALLY_VALIDATED.
"""

from __future__ import annotations

from .evidence import Evidence, EvidenceKind
from .knowledge import Knowledge, SemanticMemory


def record_fidelity_lesson(
    memory: SemanticMemory | None,
    cheap_model: str,
    corrected_model: str,
    reference_model: str,
    mean_error_before: float,
    mean_error_after: float,
    slenderness_range: tuple[float, float],
    evidence_refs: list[str],
    run_id: str | None = None,
    domain: str = "cantilever_link",
) -> Knowledge:
    """Record that a cheap model was found wrong, and how it was corrected."""
    lo, hi = slenderness_range
    lesson = Knowledge(
        claim_key=f"model_limitation:{domain}:{cheap_model}:shear",
        statement=(
            f"{cheap_model} under-predicts tip deflection for {domain} designs "
            f"that are not slender, because it omits shear deformation. Measured "
            f"against {reference_model} over L/h from {lo:g} to {hi:g}, mean "
            f"error fell from {mean_error_before:.2%} to {mean_error_after:.2%} "
            f"once the {corrected_model} shear term was added. A design "
            f"optimized to sit exactly on a deflection limit under "
            f"{cheap_model} is therefore infeasible in {reference_model}."
        ),
        domain=domain,
        source="multi-fidelity comparison during the Phase 7 verification gate",
        evidence=[
            Evidence(kind=EvidenceKind.SIMULATION, ref=ref, run_id=run_id,
                     note=f"{reference_model} vs {cheap_model} deflection")
            for ref in evidence_refs
        ],
        assumptions=[
            f"{reference_model} is treated as the reference here, but it is "
            "still a simulation: linear elastic, small strain, idealised "
            "clamped support.",
            "The effective shear area of the box is [ASSUMED] to be the two "
            "vertical webs; the factor is calibrated against the reference "
            "model, not derived.",
        ],
    )
    if memory is not None:
        lesson = memory.upsert_by_claim(lesson)
    return lesson
