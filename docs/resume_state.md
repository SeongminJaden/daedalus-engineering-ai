# Resume state

Read this first after a restart. It is rewritten at the end of every unit of
work, so whatever it says is the latest known state. Everything in the
repository is SIMULATED or below; nothing has been physically tested.

## Where things stand

- Branch `master`, remote `origin/main`. Commit after every unit of work, push
  after every commit.
- Capabilities registered: 54 (`nodes.roster.build_roster`, `len`). Nodes: 12.
- Tests collected: 1527 before the SURROGATE gate, 1548 after (21 new).
- Last full suite run: see the bottom of this file.

## Generative design track (the order is fixed)

| step | what | state |
|---|---|---|
| gate | SURROGATE evidence level below SIMULATED, verdict guard in code and tests | DONE |
| P5 | synthetic data engine: build123d parametric shapes, Analyzer, labels from external solvers (CalculiX and the rest), labels recorded SIMULATED | next |
| P3 | shape classification | not started |
| P6 | CAD embeddings | not started |
| P7 | surrogate prediction, search acceleration only, behind the gate | not started |
| P8 | design intent, measured by ablation against real solvers | not started |
| P9 | generative design and the autonomous CAD loop | not started |

Parallel task: GitHub README refresh (54 capabilities, 7 external solvers,
the evidence ladder, roadmap with P3 to P9 marked in progress or planned,
architecture diagram replaced). Not started.

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

2026-09-02, after the SURROGATE gate: 1547 passed, 1 xfailed, 619 s with
`-n 8 --dist loadfile`. The one failure on the first run was the ladder pin in
`tests/test_aesthetics.py`, widened to admit SURROGATE and nothing else.
