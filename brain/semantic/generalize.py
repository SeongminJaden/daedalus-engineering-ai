"""brain.semantic.generalize - promote repeated observations into knowledge.

Scans episodes for patterns that hold often enough to be worth stating, and
records them as evidence-graded knowledge rather than as facts.

Honesty constraint baked in: every statement produced here is derived from
simulation at beam-theory fidelity, so its evidence is tagged
`EvidenceKind.SIMULATION` and it can never reach EXPERIMENTALLY_VALIDATED.
Episodes from a single run count as ONE independent observation, so one run -
however many iterations it contains - yields at most SIMULATED.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .evidence import Evidence, EvidenceKind
from .knowledge import Knowledge, SemanticMemory

# A pattern must hold in at least this share of episodes to be stated at all.
DOMINANCE_THRESHOLD = 0.6


def _active_constraints(episode: dict) -> list[str]:
    return list(episode.get("constraint_status", {}).get("active", []))


def generalize_binding_constraint(
    episodes: list[dict],
    memory: SemanticMemory | None = None,
    domain: str = "cantilever_link",
    threshold: float = DOMINANCE_THRESHOLD,
) -> list[Knowledge]:
    """"Which constraint actually limits this family of designs?"

    Counts the active constraint across feasible episodes. If one dominates,
    states it with the supporting episodes as evidence - one evidence item per
    episode, tagged with its run so independence is counted correctly.
    """
    feasible = [e for e in episodes if e.get("feasible")]
    if not feasible:
        return []

    supporters: dict[str, list[dict]] = defaultdict(list)
    for e in feasible:
        for name in _active_constraints(e):
            supporters[name].append(e)

    total = len(feasible)
    produced: list[Knowledge] = []
    for name, eps in sorted(supporters.items()):
        share = len(eps) / total
        if share < threshold:
            continue
        k = Knowledge(
            claim_key=f"binding_constraint:{domain}:{name}",
            statement=(
                f"For {domain} designs, the binding constraint is '{name}' "
                f"(active in {len(eps)}/{total} feasible episodes)."
            ),
            domain=domain,
            source="semantic generalization over loop episodes",
            evidence=[
                Evidence(
                    kind=EvidenceKind.SIMULATION,
                    ref=e["episode_id"],
                    run_id=e.get("run_id"),
                    note=f"active constraints: {_active_constraints(e)}",
                )
                for e in eps
            ],
            assumptions=[
                "Euler-Bernoulli beam theory: no root stress concentration, "
                "no shear deformation, no buckling check.",
                "Design bounds as configured, including the ASSUMED 1 mm "
                "minimum wall thickness.",
            ],
        )
        if memory is not None:
            k = memory.upsert_by_claim(k)
        produced.append(k)
    return produced


def generalize_bound_activity(
    episodes: list[dict],
    memory: SemanticMemory | None = None,
    domain: str = "cantilever_link",
    tolerance: float = 1e-6,
    threshold: float = DOMINANCE_THRESHOLD,
) -> list[Knowledge]:
    """"Do optimized designs keep landing on a variable's bound?"

    Worth knowing precisely because it means the *bounds*, not the physics,
    are setting the answer - and one of those bounds is an assumption.
    """
    feasible = [e for e in episodes if e.get("feasible") and e.get("design_genome")]
    if not feasible:
        return []

    counts: Counter = Counter()
    supporters: dict[str, list[dict]] = defaultdict(list)
    for e in feasible:
        genome = e["design_genome"]
        for var, bound_value in (("wall_thickness_m", 0.001),
                                 ("outer_width_m", 0.010)):
            value = genome.get(var)
            if value is not None and abs(value - bound_value) <= max(
                    tolerance, bound_value * 1e-3):
                counts[var] += 1
                supporters[var].append(e)

    total = len(feasible)
    produced: list[Knowledge] = []
    for var, n in sorted(counts.items()):
        if n / total < threshold:
            continue
        k = Knowledge(
            claim_key=f"bound_active:{domain}:{var}",
            statement=(
                f"For {domain}, optimized designs sit on the lower bound of "
                f"'{var}' ({n}/{total} feasible episodes), so that bound - not "
                f"the physics - sets the achievable mass."
            ),
            domain=domain,
            source="semantic generalization over loop episodes",
            evidence=[
                Evidence(kind=EvidenceKind.SIMULATION, ref=e["episode_id"],
                         run_id=e.get("run_id"))
                for e in supporters[var]
            ],
            assumptions=[
                "The bound itself is a modelling choice; for wall thickness it "
                "is the ASSUMED CNC-aluminium manufacturability limit.",
            ],
        )
        if memory is not None:
            k = memory.upsert_by_claim(k)
        produced.append(k)
    return produced


def generalize_all(episodes: list[dict], memory: SemanticMemory | None = None,
                   domain: str = "cantilever_link") -> list[Knowledge]:
    return (generalize_binding_constraint(episodes, memory, domain)
            + generalize_bound_activity(episodes, memory, domain))
