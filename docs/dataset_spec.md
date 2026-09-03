# Industrial scale dataset: specification and cost

Written before generation, so the scale is a decision and not an accident.
Everything in the dataset is SIMULATED. Every label keeps its mesh sensitivity.

## What exists today

Five families (box, hollow_rect, l_bracket, plate_with_holes, stepped_shaft),
one load case (cantilever bending, the x minimum face clamped, a force on the
x maximum face), one material in the labelled tests (al_7075_t6), labels from
Gmsh quadratic tetrahedra and CalculiX at two mesh sizes.

Cost per part, measured on 2026-09-02 on this machine (RTX 3050 laptop, 16
cores; CalculiX runs one thread per solve; each part is two meshes and two
solves):

| family | seconds per part |
|---|---|
| box | 1.9 |
| stepped_shaft | 0.2 to 2 |
| plate_with_holes | 1.2 to 10 (curvature refinement on the holes) |
| hollow_rect | 5.7 |
| l_bracket | 5.6 |
| mean over 55 parts, two seeds | 3.2 |

## What the specification adds

**Families, 5 to 13.** New: bracket (two plates at a right angle with holes),
flange (disc with a bolt circle), housing (open box with wall and ribs),
keyed shaft (shaft with a key seat and a cross hole), gear blank (disc with
hub and bore), link (two-eye link), mount (base plate with a boss), ribbed
plate. Each family must ship with a closed-form volume and the features its
parameters put there, because the engine refuses a part whose analyzer volume
or recognised features disagree with its parameters. That is the cost that
dominates, and it is engineering time, not solver time.

**Load cases, 1 to 5.** Cantilever bending (exists), torsion about x, axial
tension, combined bending and torsion, thermal gradient. Each new case needs
its own boundary conditions in the labeller, a closed-form check on a box, and
a cross-check against the in-house solver where one exists (the Warp hex FEM
covers bending and axial on boxes; torsion and thermal have closed forms for
shafts and bars). No case ships without that check.

**Materials, 1 to 19.** The database now holds 19 sourced materials. For a
linear elastic, isotropic, homogeneous part the solver's answer scales
exactly: displacement is proportional to 1/E, mass to density, and stress
does not depend on E at all. What stress DOES depend on in a three
dimensional solve is Poisson's ratio. So labelling every material separately
would spend nineteen solves to learn what one solve and a multiplication give,
except across Poisson's ratio.

Measured before the plan was fixed, on a hollow rectangle cantilever (200 by
40 by 30 mm, 4 mm wall, quadratic tetrahedra, 100 N):

| change | tip deflection | max displacement | peak von Mises |
|---|---|---|---|
| E doubled, same Poisson | x 0.500000 | x 0.500000 | x 1.000000 |
| Poisson 0.33 to 0.22 | +0.23 percent | +0.21 percent | +0.66 percent |
| Poisson 0.33 to 0.29 | +0.13 percent | +0.13 percent | +0.23 percent |
| Poisson 0.33 to 0.40 | -0.44 percent | -0.44 percent | -0.30 percent |

The E scaling is exact to the last printed digit, as linear elasticity says
it must be. The Poisson effect across the whole span of the database (0.22 for
alumina to 0.40 for PEEK) is under one percent on every label for this load
case, which is smaller than the mesh sensitivity of most parts. So the plan is
ONE solve per geometry and load case with a reference material, and exact
scaling to the other eighteen: mass by density, displacements by E_ref / E,
stresses unchanged, and for torsion the twist by G_ref / G where
G = E / 2(1 + nu). Scaled labels are graded derived from a SIMULATED solve,
carry the reference material id, and carry the measured Poisson residual for
their load case as a bound in the note. The residual is measured per load case
before that case's scaled labels are stored, because torsion and thermal cases
need not behave like bending. cfrp_ud is orthotropic, is excluded from the
scaling, and needs its own solves.

**Samples.** One hundred parameter draws per family per load case, since the
Poisson grouping is no longer needed and the solves can go to geometry instead.

## The bill

    13 families x 5 load cases x 100 samples = 6,500 solves

    at 3.2 s per part (two meshes, two solves)      5.8 hours on one worker
    at 8 workers                                     about 45 minutes

    after scaling to 18 isotropic materials         about 117,000 records

The user's threshold was one day. This is well under it, so no decision is
needed on scale. The engineering time for eight new families and four new
load cases is the real cost and is not in the table. The first version of this
document planned 10,400 solves across four Poisson groups; the measurement
above removed the groups.

## Resumable generation

A batch writes one JSONL per (family, load case, group) cell, appends records
as they finish, and keeps a `done` file of part ids. A restart reads the done
set and skips those ids, so an interrupted run loses at most one part. Refused
parts (ground truth mismatch, solver returned nothing) are written to a
`refused.jsonl` with the reason and are counted in the report; a run that
drops parts silently is biased toward the easy ones and does not know it.

## The driver

`scripts/generate_industrial_dataset.py --workers 8 --samples 100 --root
data/generated/industrial_v1` runs every cell through a process pool, each
cell resuming from its own files, then writes the scaled copies under
`scaled/` and a `manifest.json` with counts, timing, md5 sums per file and the
sha256 of the specification (families, cases, samples, seed, reference
material, commit). `/data/generated/` is ignored by git; the manifest's
numbers are copied into this document when a run finishes.

The first launch found a defect the tests had not: a cell's seed was derived
with `hash(name)`, which Python salts per process, so every spawned worker
and every resume drew different parts and the specification could not
reproduce the set it described. The seed now uses crc32 and a test compares
the value across two processes. The run was restarted from nothing after the
fix.

## Measured on the mount family before generation

The full test suite caught the first mount drawn at seed 0: Gmsh refused the
coarse surface mesh (overlapping facets on the plate face around the boss) and
the labeller only retried solver rejections, so the exception escaped. The
labeller now retries a mesher failure at 0.7 of the size like a solver
rejection; on that part 12.57 mm fails and 8.80 mm meshes. Over twenty mount
draws at seed 1, none were refused and five needed a retry, so a quarter of
this family's parts cost one extra mesh. The same part rounded to five
decimals meshes at every size and CalculiX rejects all three meshes with a
nonpositive Jacobian in eight quadratic tetrahedra. That is the known limit
of second-order nodes on curved faces and such a part is refused and listed
in `refused.jsonl`, not hidden. The bill above does not include refusals;
measure the rate on the first cell before trusting the total.

## Run 1, generated 2026-09-03

Started 08:12 KST, `scripts/generate_industrial_dataset.py --workers 8
--samples 100 --seed 0`, output `data/generated/industrial_v1` (not in git).
The code state was the tree of commit a4d7030 (the driver and the crc32 seed);
`spec.json` records c27931a because the driver commit landed after the launch.

| quantity | measured |
|---|---|
| specification sha256 | d7385cf106b23b6f7e759cfc906980f92da83ebfa972a4a1c35c864e0bedcc30 |
| cells | 65 (13 families by 5 load cases) |
| solves attempted | 6,500 |
| labelled | 6,430 |
| refused | 70, all "solver returned nothing" (nonpositive Jacobian in quadratic tetrahedra after three sizes) |
| retried finer at least once | 451 of 6,430 |
| scaled copies | 106,722 (17 target materials per record) |
| scaling skipped | 2,588 (thermal records for PLA and PA12, no expansion coefficient) |
| total records | 113,152 |
| wall time for the solves | 4,439 s (74 min) on 8 workers, 16 cores, OMP_NUM_THREADS 2 |
| worker seconds per attempted part | 5.33 s (3.2 s measured serially; contention factor 1.67) |
| JSONL size | 439.9 MB in 149 files; 619 MB with the STEP files |
| largest Poisson residual bound on any scaled label | 0.07 (thermal stress) |

Refusals and retries by family (labelled of 500 attempted per family):

| family | refused | retried |
|---|---|---|
| flange | 37 | 161 |
| plate_with_holes | 22 | 64 |
| mount | 4 | 55 |
| bracket | 3 | 43 |
| stepped_shaft | 2 | 38 |
| gear_blank | 1 | 49 |
| link | 1 | 41 |
| box, hollow_rect, housing, keyed_shaft, l_bracket, ribbed_plate | 0 | 0 |

Every refused and retried part has a hole or a round boss: the quadratic
mid-side nodes on curved faces are what CalculiX rejects. The flange, with a
bolt circle of holes, loses 7 percent of its draws. The estimate of 45 minutes
assumed the serial per-part cost would hold at 8 workers; it did not, and the
bill above now carries the measured contention.

md5 of the first five solved cells, for checking a copy against this run
(the full list is in `manifest.json`):

    box__axial/records.jsonl             bd69155ead3d4cdbed139ceefde0f824
    box__bending/records.jsonl           1877f0d99470f3fb4710596e5a728e40
    box__combined/records.jsonl          b03bf59c272a804d2470896d2cbf44ee
    box__thermal_gradient/records.jsonl  b1795a03ca8509e368635d67609c00c7
    box__torsion/records.jsonl           088fe7ceb0cdc5bbcbb240b64b136344

## Surrogate on run 1

`scripts/train_industrial_surrogate.py --root data/generated/industrial_v1`
trains on 90,574 rows and reports on 22,578 held out: the last fifth of every
cell's draw order, so every family and every load case is in the test set, and
a scaled copy is always on the same side as the part it came from. 6,430
solved parts, 18 materials, 43 features, two hidden layers of 64, 3,000
epochs of full batches of 4,096, 64 seconds on the GPU.

| target | Spearman | log R2 | median error | p95 error |
|---|---|---|---|---|
| primary response | 0.997 | 0.994 | 0.07 | 0.38 |
| max displacement | 0.985 | 0.916 | 0.10 | 0.86 |
| peak von Mises | 0.997 | 0.993 | 0.08 | 0.40 |

By load case, on the primary response:

| load case | Spearman | log R2 | median error | p95 error |
|---|---|---|---|---|
| bending | 1.00 | 1.00 | 0.06 | 0.29 |
| axial | 0.99 | 0.98 | 0.08 | 0.60 |
| torsion | 1.00 | 1.00 | 0.08 | 0.39 |
| combined | 1.00 | 1.00 | 0.07 | 0.32 |
| thermal gradient | 1.00 | 0.99 | 0.05 | 0.32 |

By family the Spearman is 1.00 everywhere except link (0.95, p95 error 1.17)
and gear blank (0.99, p95 0.72); the best is keyed shaft (p95 0.10). The
per-material spread is nothing: every isotropic material lands within 0.001
Spearman of the others, which is what exact scaling should produce and is a
check on the scaling rather than a result about materials.

**The proxy alone is the baseline that keeps this honest.** Inside one family
and one load case the parts differ mostly in size, and a beam formula already
orders those, so a Spearman near one means little until the closed form is
next to it. The proxy alone, on the same held-out rows: Spearman 0.878, log R2
0.652, median error 0.36, p95 error 0.99. By load case it ranges from 0.87
(axial) to 0.99 (thermal), and by family from 0.47 (bracket) to 0.99. So the
model earns roughly a tenth of a rank correlation and a fivefold cut in median
error over the formula it is given.

**And the proxies are no longer load bearing.** With 40 training parts,
removing them cost everything (Phase 7: log R2 below zero without the proxy,
0.94 with it). With 90,574 rows the same removal costs almost nothing:
Spearman 0.997, log R2 0.994, median error 0.073 against 0.068. The model now
learns the cubic from the data, which is what more data was supposed to buy.
The proxies stay because they cost one closed form each and they are what
makes a small run usable.

**The peak stress is still the label's problem, not the model's.** Its
mesh_sensitivity between the two meshes is stored with every record and on
holed parts reaches half, so a surrogate that reproduced it perfectly would
reproduce a mesh dependent number. That is why the primary response, not the
peak, is the quantity asked to rank.

Every prediction grades SURROGATE, below SIMULATED. `screened_check` is the
only CheckResult a prediction may become, and the integration layer refuses a
verdict built on it.

## Embeddings on run 1

`scripts/measure_industrial_embeddings.py` repeats the Phase 6 comparison on
thirteen families, 120 parts each, split by the same draw order: 1,257 train,
303 held out, PointNet 150 epochs, 31 seconds on the GPU.

| method | precision at 1, five families | thirteen families |
|---|---|---|
| 22 descriptors | 1.00 | 0.99 |
| PointNet embedding | 0.88 | 0.95 |
| D2 histogram | 0.64 | 0.64 |

More data narrowed the gap the learned space had to close (0.88 to 0.95) and
left the ordering where it was: the descriptors still win, the histogram is
still last, and the embedding is still SURROGATE. The eight families added to
the set did not confuse the descriptors, which is a statement about these
families being separable by counts and moments, not about descriptors in
general.

## What this does not claim

Nothing in it is physically validated. The families are analysis shapes, not
catalogue parts. The load cases are the five listed and no other. The Poisson
scaling is a stated approximation whose error will be measured and written
down before any scaled label is stored.
