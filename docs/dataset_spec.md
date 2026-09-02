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

## What this does not claim

Nothing in it is physically validated. The families are analysis shapes, not
catalogue parts. The load cases are the five listed and no other. The Poisson
scaling is a stated approximation whose error will be measured and written
down before any scaled label is stored.
