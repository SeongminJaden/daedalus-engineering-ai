# Resume state

Read this first after a restart. It is rewritten at the end of every unit of
work, so whatever it says is the latest known state. Everything in the
repository is SIMULATED or below; nothing has been physically tested.

## Where things stand

- Branch `master`, remote `origin/main`. Commit after every unit of work, push
  after every commit.
- Capabilities registered: 54 (`nodes.roster.build_roster`, `len`). Nodes: 12.
- Tests collected: 1575, 1574 passing and 1 xfail (1548 after the SURROGATE gate, 27 more for P5).
- Last full suite run: see the bottom of this file.

## Generative design track (the order is fixed)

| step | what | state |
|---|---|---|
| gate | SURROGATE evidence level below SIMULATED, verdict guard in code and tests | DONE |
| P5 | synthetic data engine: `core/part_dataset/{families,labeller,store,engine}.py`, five families, ground truth checked per record, CalculiX labels with mesh sensitivity | DONE |
| P3 | shape descriptors and classification | next |
| P6 | CAD embeddings | not started |
| P7 | surrogate prediction, search acceleration only, behind the gate | not started |
| P8 | design intent, measured by ablation against real solvers | not started |
| P9 | generative design and the autonomous CAD loop | not started |

Parallel task: GitHub README refresh (54 capabilities, 7 external solvers,
the evidence ladder, roadmap with P3 to P9 marked in progress or planned,
architecture diagram replaced). Not started.

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
