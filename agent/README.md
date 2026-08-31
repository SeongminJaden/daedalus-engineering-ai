# agent — the autonomous design loop

Implements the state machine:

```
OBSERVE → REASON → PLAN → DESIGN → SIMULATE → EVALUATE → LEARN → UPDATE_BRAIN → (REASON …)
```

One pass is one iteration and produces exactly one episode.

| package | role |
|---|---|
| `loop/` | the state machine, termination rules, run bookkeeping |
| `reasoner/` | **pluggable** decision policy: `decide(state, history) -> Action` |
| `planner/` | binds an action to concrete optimizer settings |
| `evaluator/` | judges a result into a conclusion + confidence |
| `experiment_manager/` | episode JSONL log and compute budget |

## What the "reasoner" actually is — no overclaiming

The reasoner shipped in Phase 4 is a **deterministic, rule-based heuristic**.
It is **not a language model**, and it does not reason in any sense beyond
following an explore/exploit schedule written into it. Describing this as "AI
reasoning" would be an overclaim, so the code and the docs both say heuristic.

What it *is* is the seam. `Reasoner` is an ABC with a single method, and the
loop only ever sees `decide(state, history) -> Action`. An LLM-backed policy
drops in there without the engine changing — that is the documented extension
point. In the wider system the language model is the **outer orchestrator**
(a session driving this engine), not this class.

## Why a loop rather than one optimizer call

SLSQP converges to the nearest KKT point. With two of three design variables
pinned to their bounds, the real question is whether a *different basin* does
better. The loop is therefore a **multi-start orchestrator**: exploit refines
the incumbent from a jittered start, explore restarts local search from a fresh
random point, and a stall forces exploration. That is the value it adds over
Phase 3 alone — robustness against local optima and against the bounds.

## Termination conditions

All parameterized, none hard-coded:

| reason | rule |
|---|---|
| `TARGET_REACHED` | a feasible design reached `target_mass_kg` |
| `CONVERGED` | `convergence_patience` consecutive iterations improved by < `convergence_epsilon` |
| `COMPUTE_BUDGET_EXCEEDED` | evaluations or seconds spent, from the GPU profile's `budget` block |
| `CONSTRAINTS_UNSATISFIABLE` | no feasible design after `unsatisfiable_after` independent starts |
| `USER_STOP` | `stop_flag()` returned true, or the reasoner returned `STOP` |
| `MAX_ITERATIONS` | iteration cap reached with nothing else firing |

Unsatisfiability needs several failed *independent* starts: one failure is a
bad start, several from different basins is evidence about the problem.

## Episodes — the Phase 5 Brain's seed

Every iteration is appended to `runs/<run>/episodes.jsonl`, pydantic-validated,
one JSON object per line (so a killed run still leaves a readable log). The
schema records `hypothesis` and `parent_design_id` alongside the numbers, so a
run reads back as a search tree of intentions rather than a list of results —
provenance that would otherwise have to be back-filled later.

Structured memory, retrieval and strategy generalization are Phase 5. This is
only the capture format.
