# Resume state

Read this first after a restart. It is rewritten at the end of every unit of
work, so whatever it says is the latest known state. Everything in the
repository is SIMULATED or below; nothing has been physically tested.

## Where things stand

- Branch `master`, remote `origin/main`. Commit after every unit of work, push
  after every commit.
- Capabilities registered: 57 (`nodes.roster.build_roster`, `len`). Nodes: 13.
- Tests collected: 1674, all passing, no xfail left (the Code_Aster contact one was fixed on 2026-09-02).
- Last full suite run: see the bottom of this file.

## Generative design track (the order is fixed)

| step | what | state |
|---|---|---|
| gate | SURROGATE evidence level below SIMULATED, verdict guard in code and tests | DONE |
| P5 | synthetic data engine: `core/part_dataset/{families,labeller,store,engine}.py`, five families, ground truth checked per record, CalculiX labels with mesh sensitivity | DONE |
| P3 | descriptors and classification: `core/part_dataset/{descriptors,classify}.py`, `nodes/shape_classifier.py`, capability `analysis.cad.classify` | DONE |
| P6 | embeddings: `core/part_dataset/{pointcloud,embedding}.py`, D2 baseline and PointNet (SURROGATE) | DONE |
| P7 | shape surrogate: `core/part_dataset/shape_surrogate.py`, beam proxy feature, screen_and_verify_parts | DONE |
| P8 | design intent by ablation: `core/part_dataset/intent.py` | DONE |
| P9 | generative CAD loop: `agent/execution/cad.py`, method `generative_cad` | DONE, first pass |

Parallel task: GitHub README refresh (54 capabilities, 7 external solvers,
the evidence ladder, roadmap with P3 to P9 marked in progress or planned,
architecture diagram replaced). Not started.

## Second work list (from the coordinating session, 2026-09-02)

| item | what | state |
|---|---|---|
| 1 | sourced materials table with per-value citations, temperature ranges, optional fatigue | DONE: 19 materials, `sources` and `value_sources` on MaterialSpec, MatWeb values kept but graded secondary, 718 modulus curve |
| 2 | dataset spec (`docs/dataset_spec.md`), 5 load cases, 13 families, material scaling, resumable batch | DONE; generation at scale not run, user's call (about 45 min at 8 workers, mount family retries a quarter of its parts) |
| 3 | form score (B) and per-process DFM rules (C), grade rule_based_dfm_guideline | DONE |
| 4 | Code_Aster contact xfail | DONE: sign of LIAISON_UNIL; contact registered |
| 5 | catalogue part shapes with material links | pending |
| 6 | Gazebo and Isaac Sim verification pipeline; machine has Gazebo Fortress 6.18 and ros_gz Humble, no Isaac Sim | pending |
| 7 | measurement and evaluation guideline document | pending |

## P9 generative CAD loop, as built

- `agent/execution/cad.py`: `run(op, candidates, top_k, seed, families,
  ranker, step_dir)`; `proxy_ranker` default, `surrogate_ranker(surrogate,
  step_dir)` optional; three families with length imposed and the section
  inside the problem envelope; winner is the lightest solver-verified part
  within the deflection limit, else the closest marked infeasible.
- `DesignOutcome.cad_record` is the third representation; a CAD outcome must
  carry a labelled record. `LoopConfig.cad_options` (candidates 8, top_k 2).
  `_genome_of` writes family, part_id, step_path, parameters, evidence.
- Registry: `generative_cad` (56th capability), condition `supports("cad_family")`.
- Tests: `tests/test_cad_loop.py` (6, slow, about 10 s);
  `test_loop_execution` now expects four executables.
- Next after this track: more families and load cases, then hardware, then
  measurement. Neither hardware nor measurement is software work here.

## P8 design intent, as built

- `IntentClaim(ReferenceItem)`: family, parameter, role, quantity, direction
  (UP, DOWN, NONE), factor, expected_ratio, tolerance; provenance-capped
  confidence inherited.
- `ablate` builds and labels base and ablated parts; `_judge` gives SUPPORTED,
  REFUTED or INCONCLUSIVE against 2x the larger mesh sensitivity; directional
  claims without a ratio must exceed `tolerance` to count.
- `record_in_brain`: SUPPORTED adds SIMULATION evidence, REFUTED a
  counterexample, INCONCLUSIVE nothing; consolidated by claim_key.
- Measured: wall x2 gives deflection x0.625; length x2 gives 7.98 against 8;
  holes 2 to 4 gives +1.8 percent (refutes structural, supports clearance);
  height x1.001 is inconclusive.
- Tests: `tests/test_design_intent.py` (8, slow, about 30 s).

## P7 shape surrogate, as built

- Features: 22 descriptors + log volume, longest side, E, load, direction
  indicators + log beam proxy (bounding box cantilever scaled by fill).
- `train_shape_surrogate` (MLP 32, wd 1e-3, 2000 epochs, seconds on GPU),
  `ShapeSurrogate` save/load, `ShapePrediction` (SURROGATE, as_evidence,
  screened_check), `screen_and_verify_parts` (solver-verified winner only).
- Measured: on 40 parts raw R2 0.94, log R2 0.97, Spearman 0.99 with the
  proxy, below zero without it; 20-part draws Spearman never below 0.79 and
  log R2 never below 0.41, raw R2 down to 0.29 (the wrong metric to read).
- Mesher: no high-order optimiser (it terminated the process once); the
  labeller retries at 0.7 size up to twice and records the size used.
- Labelled corpus for experiments is in the session scratchpad only; nothing
  labelled is committed. Cost 2.5 to 7 s per part.
- Tests: `tests/test_shape_surrogate.py` (7, marked slow, about 2 min).

## P6 embeddings, as built

- `pointcloud.py`: `tessellate` (OCP, 0.1 mm deflection), `sample_surface`
  (area weighted), `normalise`, `canonical_frame` (PCA, third-moment signs),
  `d2_signature` (64 bins), `point_cloud_of`.
- `embedding.py`: `PointNetEncoder` (32-d unit vector), `train_embedding`
  (family head, sign-flip augmentation, 150 epochs, 4 s on GPU),
  `EmbeddingBundle` save/load, `nearest_neighbour_precision`,
  `embedding_label` (SURROGATE; D2 as ANALYTICAL).
- Measured: retrieval descriptors 1.00, PointNet 0.88, D2 0.64; rotation
  cosine 1.00 after alignment; Fusion plates land nearest box in the learned
  space while the rules say plate.
- Tests: `tests/test_cad_embeddings.py` (10).

## P3 descriptors and classification, as built

- 22 scale-free descriptors (`DESCRIPTOR_NAMES`); Euler characteristic is
  V - E + F - inner loops (wires minus faces), which the first version got
  wrong and measurement caught.
- `rule_classify` gives a family or UNKNOWN with reasons, graded SIMULATED;
  1.00 on generated parts, Fusion plates A and B classify as plates, C to G
  UNKNOWN.
- `NearestNeighbourClassifier` (numpy, k=5, standardised, open-set rejection
  at 1.0 times the 99th percentile leave-one-out distance) graded SURROGATE;
  1.00 held-out after logging compactness, rejects the cone fixture D.
- Registered as `analysis.cad.classify` on node `shape.classifier` (55th
  capability, 13th node).
- Tests: `tests/test_shape_classifier.py` (11).

## P5 synthetic engine, as built

- `core/part_dataset/families.py`: box, hollow_rect, l_bracket,
  plate_with_holes, stepped_shaft. Each has bounds, an admissibility rule, a
  build123d builder, a closed-form volume and expected features. Sampler is
  rejection sampling on a seeded generator; part ids are sha1 of rounded
  parameters.
- `core/part_dataset/labeller.py`: one cantilever case (x-min clamped, x-max
  loaded, -100 N), two mesh sizes (longest side / 15 and / 22), CalculiX
  C3D10. Labels: mass_kg (analytical), tip_deflection_m, max_displacement_m,
  max_von_mises_pa (each with mesh_sensitivity), load_case. All SIMULATED.
- `core/part_dataset/schema.py`: `label()` grades by kind; `LABEL_CEILING` is
  SIMULATED; a record with a computed label above it is refused.
- `core/part_dataset/store.py`: JSONL, validated both ways, refuses
  unpublishable records on public files before writing anything.
- `core/part_dataset/engine.py`: `make_part`, `generate_dataset`; refuses a
  part whose analyzer volume or recognised features disagree with its
  parameters; `GenerationReport` lists refusals.
- Decisions taken by default: CalculiX only for labels, one load case,
  al_7075_t6, dataset scale left to what P3 needs.
- Cost measured: 0.2 to 5.7 s per part; five families in 13 s.
- Tests: `tests/test_synthetic_engine.py` (27).

## The SURROGATE gate, as built

- `brain/semantic/evidence.py`: `EvidenceLevel.SURROGATE` between UNVERIFIED
  and SIMULATED, ceiling 0.40. `EvidenceKind.SURROGATE`. `derive_level` sets
  surrogate evidence aside before counting. `VERDICT_FLOOR`, `may_decide`,
  `grounded`.
- `integration/checks.py`: `CheckResult.evidence_kind`, `SurrogateVerdict`
  raised on PASSED or FAILED with surrogate evidence, new `CheckStatus.SCREENED`
  which is a gap and may not carry a safety factor. `AssemblyVerdict.screened`
  and `gaps`.
- `integration/review.py`: `Review.screened` and a recommendation to run the
  solver.
- `surrogate/inference`: `Prediction.evidence_level` is always SURROGATE,
  `ScreeningResult.evidence_level` is SIMULATED only when verified,
  `as_evidence` on both, `screened_check`.
- Tests: `tests/test_brain.py` (8), `tests/test_integration_capstone.py` (7),
  `tests/test_surrogate.py` (6).

## Rules that do not bend

- Commit author SeongminJaden. No tool or assistant attribution in commits or
  code. No em dashes anywhere.
- Evidence honesty: zero experimental validation, everything SIMULATED or
  below, confidence ceiling 0.60. What cannot be done is said so in code and
  documentation.
- Overlap between methods is cross-validation, not a new capability. Limits
  are pinned by tests.
- Measurement beats expectation, including the user's own instructions.
- Robot and hardware execution is done by the user, not by the software.

## Last full suite run

2026-09-02, after P5: 1573 passed, 1 xfailed, 603 s with `-n 8 --dist
loadfile`, plus `test_committed_tree_can_import_every_package`, which fails
before a commit that adds modules and passes after it. After the SURROGATE
gate: 1547 passed, 1 xfailed, 619 s.
