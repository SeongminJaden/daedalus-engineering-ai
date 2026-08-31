"""Which registered method wins on which class of problem.

This generalises over runs: given episodes tagged with the strategy that
produced them and the outcome they reached, it asks whether one strategy
consistently wins on a given problem class, and promotes that to a stored
strategy if it does.

**Evidence stays honest by construction.** Every sample here is a simulation
result, so the evidence carried is `SIMULATION` and the level comes out of the
ordinary ladder: SIMULATED from a single run, REPEATED once enough independent
runs agree. It can never reach EXPERIMENTALLY_VALIDATED, because that gate
opens only for physical-test evidence and nothing here can produce any. A
statement that one method beats another is a statement about the models, not
about the world.
"""

from __future__ import annotations

from collections import defaultdict

from brain.semantic.evidence import Evidence, EvidenceKind
from brain.strategy.strategies import PROMOTION_THRESHOLD, Strategy, StrategyStore


def _winners_per_run(samples: list[dict]) -> dict[str, list[tuple[str, dict]]]:
    """For each (problem class, run), which strategy reached the best score.

    Grouping by run before counting matters. Without it a strategy that simply
    ran more episodes would accumulate more wins and look better for no reason
    other than having been tried more often.

    Lower `score` is better; it is whatever the caller is minimising. Samples
    that are not feasible are dropped: a design that violates its constraints
    has no score worth comparing.
    """
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        if not sample.get("feasible", False):
            continue
        score = sample.get("score")
        strategy = sample.get("strategy_used")
        if score is None or not strategy:
            continue
        key = (str(sample.get("problem_class", "unknown")),
               str(sample.get("run_id", "unknown")))
        grouped[key].append(sample)

    winners: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for (problem_class, _run), rows in sorted(grouped.items()):
        best = min(rows, key=lambda r: (float(r["score"]), str(r["strategy_used"])))
        winners[problem_class].append((str(best["strategy_used"]), best))
    return winners


def derive_method_strategies(
    samples: list[dict],
    store: StrategyStore | None = None,
    threshold: float = PROMOTION_THRESHOLD,
) -> list[Strategy]:
    """Promote "method M wins on problem class C" where the data supports it.

    Each sample is one episode:

        {"ref": <episode id>, "run_id": <run>, "problem_class": <class>,
         "strategy_used": <registry method name>, "score": <lower is better>,
         "feasible": <bool>}

    A class promotes only when one strategy wins at least `threshold` of that
    class's runs. Otherwise nothing is stored, because there is no consistent
    recommendation to make and inventing one would be the whole failure mode
    this ladder exists to prevent.

    Returns the promoted strategies ordered by problem class, so repeated calls
    on the same data give the same answer in the same order.
    """
    promoted: list[Strategy] = []
    for problem_class, wins in sorted(_winners_per_run(samples).items()):
        counts: dict[str, list[dict]] = defaultdict(list)
        for name, sample in wins:
            counts[name].append(sample)
        total = len(wins)
        # Sort by count, then name, so a tie resolves the same way every time.
        best_name, supporters = max(sorted(counts.items()),
                                    key=lambda kv: len(kv[1]))
        share = len(supporters) / total
        if share < threshold:
            continue

        strategy = Strategy(
            name=f"method-for-class:{problem_class}",
            statement=(
                f"On problems of class '{problem_class}', the registered "
                f"method '{best_name}' reached the best feasible result in "
                f"{len(supporters)} of {total} independent runs. This "
                f"generalises over simulated runs only and says nothing about "
                f"physical behaviour."),
            context={"problem_class": problem_class},
            evidence=[
                Evidence(
                    kind=EvidenceKind.SIMULATION,
                    ref=str(s.get("ref", "unknown")),
                    run_id=s.get("run_id"),
                    note=f"best feasible score {float(s['score']):.6g} "
                         f"using {best_name}",
                )
                for s in supporters
            ],
        )
        if store is not None:
            store.store(strategy)
        promoted.append(strategy)
    return promoted
