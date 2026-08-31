# surrogate — learned fast approximation of the physics evaluator

Generates training data from the solver, fits a small MLP, and — critically —
**never lets the model decide anything on its own**.

| package | role |
|---|---|
| `datasets/` | sample designs *and* problem contexts, evaluate on GPU, store pairs |
| `models/` | MLP, scalers, training with early stopping, held-out scoring |
| `inference/` | predictions with expected error, and `screen_and_verify` |

## Read this first: what is being approximated, and what it is worth today

**The surrogate approximates the Phase 2 Euler–Bernoulli evaluator, not 3D
FEM.** There is no higher-fidelity model in the system yet. So its error stacks
*on top of* beam theory's own error: a surrogate prediction is an approximation
of an approximation.

**There is no speedup today, and claiming one would be false.** Measured on
20,000 candidates:

| path | throughput |
|---|---|
| surrogate (one forward pass) | ~8.5M candidates/s |
| Phase 2 solver (one batched launch) | ~22M candidates/s |
| **ratio** | **~0.38× — the surrogate is ~2.6× SLOWER** |

That is expected. The beam kernel is closed-form arithmetic; an MLP forward
pass is more work than the physics it replaces. The value of this
infrastructure is **deferred to Phase 7**, when the base evaluator becomes an
expensive 3D FEM solve and the ratio inverts.

A caution about how *not* to measure this: comparing the surrogate against the
profile-chunked solver path (batch size 4, so 5,000 launches) shows a ~5,000×
"speedup". That number is launch overhead, not physics, and it is meaningless.
The fair comparison is against a single batched launch.

## The discipline: screen, then verify

```
20,000 candidates ──surrogate──> rank ──top 16──> Phase 2 solver ──> winner
                    (approximate)                   (authoritative)
```

`screen_and_verify` returns a design the **solver** evaluated. The surrogate
only chooses which 16 of 20,000 are worth the solver's time; the reported mass,
stress and deflection are always the solver's numbers, and `verified=True`
means step 2 actually ran.

Screening *can* be wrong — it may mis-rank and discard a good candidate. That
is a recall risk and the price of the shortlist. What it cannot do is put an
unverified design in front of a user as a result.

Predictions are never returned bare: every `Prediction` carries the p95
held-out relative error for each metric. Safety factor is **derived** from
predicted stress (yield / stress), never predicted separately, so it can never
disagree with the stress the model produced.

## Accuracy, measured honestly

The held-out set contains **only problem contexts the model never trained on**
(`Dataset.split` groups by context). This matters more than it sounds:

> Rows are generated in groups sharing a context — one kernel launch per
> context, many designs each. A random *row* split therefore puts designs from
> the *same* problem in both train and test, and the resulting score measures
> interpolation between designs of a problem the model already saw. It flatters
> the model and says nothing about a new problem. Under a random row split this
> model scored R² ≈ 0.999 with ~2% p95 error; under the honest context-grouped
> split the same configuration scored materially worse. The grouped split is
> the default.

Production configuration (20,000 rows / 400 contexts, 128×3, 300 epochs):

| metric | R² | mean rel. err | p95 rel. err |
|---|---|---|---|
| mass | 0.99986 | 0.51% | 1.44% |
| max bending stress | 0.99869 | 1.34% | 3.93% |
| tip deflection | 0.99809 | 1.59% | 4.19% |
| 1st natural frequency | 0.99974 | 0.69% | 1.82% |

Two sampling choices that materially affect this:

- **Context coverage**, not row count, limits generalization. ~50 designs per
  context, not a few hundred contexts with thousands of designs each.
- **Length and tip load are sampled log-uniformly.** Each spans ~2 orders of
  magnitude; drawn uniformly, nearly every context lands in the top decade and
  the low-load corner — where the MVP problem actually sits — goes unvisited.
  Measured effect at the MVP context: stress error 8.9% → 3.1%, deflection
  11.1% → 2.9%.

**Monotonicity is not guaranteed.** A learned model has no built-in respect for
physics, so trend violations are measured rather than assumed away: with the
configuration above, a thicker wall fails to reduce predicted stress in ~0.1%
of sampled pairs. Small, but not zero — another reason the solver, not the
surrogate, decides.
