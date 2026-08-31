# optimization: constrained design optimization

Phase 3 minimizes link mass subject to stress, deflection and geometric
constraints, reusing the Phase 2 differentiable GPU evaluator.

## Two methods, on purpose

| | `gradient/` (SLSQP) | `evolutionary/` (differential evolution) |
|---|---|---|
| kind | local, gradient-based | global, gradient-free |
| derivatives | exact, from Warp autodiff | none |
| GPU use | one design per launch | **whole population per launch** (`vectorized=True`) |
| constraints | passed to SLSQP with an exact Jacobian | quadratic penalty on normalized violations |
| cost | ~200 evaluations | ~24 000 evaluations |

They share only `constraints/`: the problem definition, the allowable stress
and the feasibility test. Algorithms, search behaviour and constraint handling
are all different, so their agreeing on the same optimum is evidence rather
than a restatement. `multi_objective/` is a Pareto stub; Phase 3 is
single-objective.

## Things worth knowing

- **Allowable stress is the tighter of two limits**: the explicit ceiling and
  `yield / min_safety_factor`. Honouring only the looser one would leave the
  other silently violated.
- **Constraints are normalized** to `1 - value/limit`. Raw stress is O(1e8) and
  deflection O(1e-3); an un-normalized penalty would enforce only one of them.
- **Feasibility needs a tolerance.** A mass-minimal design sits *exactly* on
  its binding constraint, so every solver lands within numerical noise of the
  boundary: SLSQP typically just inside, a penalty method just outside.
  `FEASIBILITY_TOL = 1e-4` is 0.01% of each limit. It is a numerical tolerance,
  **not** an engineering allowance; the real margin lives in the safety factor.
- **SLSQP's `ftol` is matched to fp32.** The physics kernel is single precision,
  so mass carries ~1e-7 relative noise. A tighter `ftol` makes the line search
  fail on that noise and report failure at a point that is actually converged.
- **`polish=False` for DE.** The penalty jumps across the geometric-validity
  boundary, and scipy's L-BFGS-B polish cannot line-search across a
  discontinuity. Staying gradient-free also keeps DE genuinely independent of
  the SLSQP result it is checked against.

## The MVP result, and what actually drives it

The optimum is **deflection-limited, not strength-limited**: tip deflection
sits on its 1 mm cap while the stress constraint keeps >70% margin (safety
factor ~14 against a required 2). Stiffness is what this link is paying for.

Two of the three design variables end up **on their bounds**: `b` at its 10 mm
minimum and `t` at its 1 mm minimum. That means the answer is set as much by
the bounds as by the physics, and `t_min = 1 mm` is an **[ASSUMED]**
manufacturability limit (CNC aluminium), not a derived one. Change the process
and the optimized mass changes with it. See `core/design_genome/bounds.py`.


## Topology optimization (`topology/`, Phase 13)

Optimizes a **density field** rather than section parameters, so the shape is
free. SIMP on the Phase 7 structured mesh, solved with the same matrix-free GPU
FEM.

```
E_e(x_e) = E_min + x_e^p (E0 - E_min),   p = 3
minimise c(x) = U^T K(x) U
subject to sum_e x_e v_e <= V_frac * V_domain
```

`E_min` is not decorative: a truly void element makes K singular and the solve
fails rather than returning a soft region. The penalty `p = 3` makes grey a bad
deal, since half density buys only an eighth of the stiffness.

### What is verified

- **The sensitivity, against finite differences.** Everything the optimizer does
  follows from `dc/dx_e`, and a wrong derivative produces a plausible shape that
  optimises nothing. Worst measured relative error: **2.6e-05**.
- **Compliance two ways**: `F.U` from the solver against
  `sum(scale * u_e^T Ke0 u_e)`, agreeing to **1.9e-14**.
- **The solid bridge**: with `x = 1` everywhere the scaled path reproduces the
  Phase 7 solid FEM to **1.4e-14**, so the density path has not drifted from the
  verified one.
- **The volume constraint** is met exactly at every iteration.
- **The filter's effect is measured, not asserted**: a checkerboard metric
  (mean density difference between face neighbours) drops from 0.365 unfiltered
  to 0.144 filtered.
- **The load path is rediscovered**: material migrates to the top and bottom of
  the section and hollows the middle, which is what bending demands, and
  concentrates toward the root where the moment is largest.

### A convergence trap worth knowing

The textbook criterion is "maximum density change below a tolerance". On its own
it is misleading: heavier damping makes every step small, the criterion is met
early, and the run reports convergence at a clearly worse design. Measured here,
damping 0.3 stopped after 14 iterations at compliance 1.78e-2 while a gentler
step reached 1.01e-2. **The design had not converged, the step size had.**

So convergence requires both a small density change *and* the objective to have
stopped improving.

### What this output is not

- **SIMP leaves grey.** No material is 40% present, so the field must be
  thresholded, and the thresholded shape is not the field that was optimized.
  The grey fraction is reported for exactly that reason.
- **STL, not a clean STEP.** Consistent with the Phase 9 boundary: recovering
  analytic faces from a density field is surface reconstruction. The export is a
  voxel surface, blocky by construction, with a volume that is exact from the
  voxel count.
- **Often not watertight**, and that is not a bug. Voxels meeting only along an
  edge leave non-manifold edges. Material attached to the structure *only* that
  way is dropped, since it carries no load, but a diagonal contact inside an
  otherwise connected body still breaks manifoldness.
- **Compliance minimisation, not stress.** There is no stress constraint here
  and the result says nothing about peak stress. Stress-constrained topology is
  substantially harder and is later work.

A topology result is a **design concept**, not a verified part. It still has to
pass the 3D FEM gate and acquire manufacturing features. Still SIMULATED.
