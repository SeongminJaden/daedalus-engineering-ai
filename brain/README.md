# brain: evidence-graded engineering experience

Accumulates what the loop has tried, generalizes repeated observations into
statements, and grades every statement by how far it has actually earned trust.

| package | role |
|---|---|
| `episodic/` | runs, designs (the Design Repository), episodes |
| `semantic/` | knowledge items, the evidence ladder, generalization |
| `strategy/` | reusable moves promoted from measured data |
| `knowledge_graph/` | typed concept graph, evidence on every edge |
| `retrieval/` | numeric-feature similarity search + warm start |
| `skills/` | reusable procedures (stub) |

Dependencies: **stdlib `sqlite3` + numpy**. No ML stack, no service, no model.

## Read this before trusting anything in here

**This is not a store of validated facts.** It is a store of *evidence-graded
experience*. Three claims are worth stating plainly:

1. **Everything from a run came from simulation** at Euler-Bernoulli beam
   fidelity (`physics/README.md`): no root stress concentration, no shear
   deformation, no buckling. Every generalized statement inherits those
   assumptions, and records them in its `assumptions` field.

2. **Retrieval is numeric feature similarity, not semantic search.** A vector
   is the design's own engineering quantities on a common scale. There is no
   embedding model and no notion of meaning. That is why the API is
   `retrieve_similar`, not `semantic_search`.

3. **`EXPERIMENTALLY_VALIDATED` is unreachable from simulation.** Only
   `EvidenceKind.PHYSICAL_TEST` opens that gate: not a thousand agreeing
   simulations, not a passing test suite, not a closed-form derivation. This is
   the single rule that keeps the Brain from talking itself into false
   confidence, and it is pinned by an explicit test.

## The evidence ladder

```
UNVERIFIED → SURROGATE → SIMULATED → REPEATED → HIGH_CONFIDENCE → EXPERIMENTALLY_VALIDATED
```

| transition | rule |
|---|---|
| → `SURROGATE` | only `EvidenceKind.SURROGATE` items: a learned model said so and no solver has. Ceiling 0.40 |
| → `SIMULATED` | any supporting evidence from a solver, a test or a derivation |
| → `REPEATED` | consistent across ≥ `repeat_independent_runs` **independent runs** |
| → `HIGH_CONFIDENCE` | ≥ `high_confidence_evidence` items from ≥ `high_confidence_runs` runs, **and zero unresolved counterexamples** |
| → `EXPERIMENTALLY_VALIDATED` | physical-test evidence, and nothing unresolved contradicting it |

**Independence is counted per run, not per episode.** Twenty iterations inside
one optimizer run are twenty samples of a single search, not twenty
observations. Counting them as independent is exactly how a memory talks itself
into confidence it has not earned, so one run, however long, yields at most
`SIMULATED`.

An **unresolved counterexample caps the level at `REPEATED`**: a statement with
a standing contradiction is not high-confidence, whatever else supports it.
Resolving it restores promotion.

**A surrogate is not a simulation.** Surrogate evidence is set aside before any
counting: a thousand predictions from a thousand runs promote nothing, add
nothing to a solver-backed statement's confidence, and cannot block physical
validation either. `may_decide(level)` is false for `SURROGATE` and
`UNVERIFIED`, and the verdict layer in `integration/checks.py` refuses to
build a PASSED or FAILED on surrogate evidence. The honest status for a
surrogate result there is `SCREENED`, which is a gap. All of this is pinned by
tests rather than by convention.

## Confidence

Explicit, bounded, monotone: never invented:

```
support = n / (n + k)            in [0,1), increasing in evidence count
penalty = 1 / (1 + unresolved)   in (0,1], decreasing in counterexamples
confidence = min(support * penalty, ceiling(level))
```

Adding evidence never lowers it; an unresolved counterexample always does; the
level ceiling caps it (SIMULATED 0.60, REPEATED 0.80, HIGH_CONFIDENCE 0.95).

## Model / brain separation

The Brain is a plain SQLite file. It opens and answers queries with no
reasoner, no GPU and no ML libraries loaded: verified by a test that queries it
from a subprocess importing only `brain`.

## Extension points

- **Semantic/text retrieval** needs an embedding model. Implement another
  `FeatureSpace` whose `vector()` returns embeddings; nothing else changes.
- **ANN indexing** (faiss and similar) swaps in behind the same search API when
  exact brute force stops being adequate.
- **Skills** are shape-only. They stay empty until enough episodes exist to
  earn them: inventing procedures would put unsupported content in a store
  whose whole point is that everything carries evidence.

## Storage

The runtime database is a **run artifact** and is gitignored (`*.sqlite3`). The
schema lives in `brain/db.py`, which is source.
