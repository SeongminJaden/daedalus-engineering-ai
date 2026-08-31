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
