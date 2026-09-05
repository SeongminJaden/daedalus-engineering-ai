# Resume state

Read this first after a restart. It is rewritten at the end of every unit of
work, so whatever it says is the latest known state. Everything in the
repository is SIMULATED or below; nothing has been physically tested.

## Where things stand

- Branch `master`, remote `origin/main`. Commit after every unit of work, push
  after every commit.
- Capabilities registered: 57 (`nodes.roster.build_roster`, `len`). Nodes: 13.
- Tests collected: 1905, all passing, no xfail. The full suite now takes about
  50 minutes rather than 13: the topology, free form, dynamics and mesh retry
  tests are solver bound, and the deeper retry ladder made the labelling ones
  slower on purpose.
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
| 2 | dataset spec (`docs/dataset_spec.md`), 5 load cases, 13 families, material scaling, resumable batch | DONE; run 1 generated 2026-09-03, 113,152 records, numbers in docs/dataset_spec.md |
| 3 | form score (B) and per-process DFM rules (C), grade rule_based_dfm_guideline | DONE |
| 4 | Code_Aster contact xfail | DONE: sign of LIAISON_UNIL; contact registered |
| 5 | standard parts: ISO 4762, ISO 4032, GT2, heat-set inserts, with material links | DONE |
| 6 | Gazebo spring-hold statics cross-check (0.03 percent) and envelope interference; Isaac Sim not installed, machine below its requirements | DONE |
| 7 | `docs/measurement_guideline.md` | DONE |

## Item 19, the six axis manipulator (2026-09-04)

The arm is designed as parts that bolt together rather than as sections.
Every link is a generated shape with drawing-backed fastening, the two parts
that hold the arm to the world exist, and every drive on it publishes an
outline. A Fusion session on another machine stands the parts up and measures
them, and most of the defects below were found that way rather than by any
check in this repository.

| stage | state |
|---|---|
| specification and arm | `projects/manipulator/spec.py`, `arm.py`; reach 600 mm exactly, gravity along -y, payload a stated 100 mm cube so the tool roll has a requirement at all |
| kinematics | wrist spacing 80.5 mm, j3 origin_x 193.04, j4 165.46. The spacing is DERIVED from whichever drive pair is tightest and moves when the catalogue does; it has been 30, 40, 78.5 and 80.5 |
| material | `alsi10mg_slm`, not 6061. Direction resolved tables for two machines in `core/materials/printed.py`; the metal breaks the polymers' rule, being stronger across the layers on ultimate and weaker on yield |
| drive selection | a candidate with no published drawing is refused however light it is. That replaced a frameless motor at the tool roll with an AK60-6 and cost 0.34 kg |
| bus voltage | there is no 24 V or 36 V design at all: every drive with a published drawing is a 48 V part. A narrower catalogue, not a better answer |
| mounting interfaces | `projects/manipulator/interfaces.py`, from CubeMars 2D drawings and their own 3D models. Bolt circles, thread depths, face insets, clock angles, dowels, and what each drawing does NOT print |
| placement | every drive's output face lies in the arm's z = 0 plane and each link sits on the far side of whichever faces it bolts to. Three links are unions of a centred and an offset box, which is a crank, and each holds every other link's box empty |
| links | `projects/manipulator/links.py`, `scripts/generate_links.py`. Six bodies in millimetres, watertight, clipped to their domains, with clearance holes, bores and dowels cut after extraction |
| mounts | `projects/manipulator/mounts.py`, `scripts/generate_mounts.py`. Base mount 1.372 kg, tool plate 0.576 kg |
| spigot and friction | `spigot_stage`: no friction coefficient is assumed, the coefficient the joint NEEDS is reported. Worst joint 0.027; dowel bearing 66.3 N m against a 22.6 N m shoulder peak |
| assembly access | `scripts/check_assembly_access.py`: every drive comes out along its axis and none of the 120 bolt holes is blocked |
| build volume | EOS M 290, 250 x 250 x 325 mm as printed. The upper arm is 238.7 on its TIGHTEST axis, so 11.3 mm spare there, and 291.0 on its longest, which is past the 250 mm bed. It fits only lying along the build height, which the sheet says includes the platform and is application dependent |
| mass | 8.312 kg of links at a volume fraction of 0.3, which is a fraction OF A DOMAIN and the domains grew. Meaningless until the per link search finishes |
| not done | the volume fraction search, covers, wiring, bearings beyond a seat tolerance, the spigot fit (no boss tolerance is printed), the AK80-9 dowel angles (not published and not measurable on its model) |

EVERY DEFLECTION IN THIS DESIGN IS LINK ELASTICITY ONLY. Six actuators sit
between the links and all six are treated as rigid, because no integrated
actuator in this catalogue prints a torsional stiffness. The missing term
ADDS to the modelled one, so the real tool deflection is larger than any
number here.

Asked in reverse, the way the friction grip was: with the joints taking half
the budget, every loaded joint needs 50,689 N m/rad, and it is the same
number for all of them because a joint's contribution is torque times lever
over stiffness. That is 1.15 times the stiffest gear unit this catalogue
prints, 44,000, so the limit sits at the edge of what is available rather
than beyond it. With every joint at that 44,000 the joints alone would use
0.461 mm of the 1 mm limit and leave 0.539 for the links and any margin.

CORRECTION, 2026-09-05. It was written here that the bearing is the whole
of the joint stiffness. That is wrong, and the load path says so in one
line. The shoulder turns about z; at full reach the tool hangs out along x
and gravity pulls along minus y, so the moment is the cross product of
those and points along minus z. Its component on the joint axis is exactly
one. A joint bearing resists moments about the two axes ACROSS the joint,
and the single direction it does not resist is the one the joint turns in,
which is precisely where this moment sits. The tool sags because the DRIVE
TRAIN twists.

So 50,689 N m/rad is a torsional requirement on the reducer. The crossed
roller's tilting rigidity answers a different question, the out of plane
budget, and both are needed. A catalogue figure of about 1.7e6 N m/rad read
off THK 382-5E page 16 for an RB10020 at 0.4 kN m is an UPPER BOUND only:
this arm's operating moment is under five percent of that chart's range,
where the curve is steepest and unreadable.

`joint_torsion_stage` computes the reducer's own number, from
`projects/manipulator/cycloidal.py`. The chain is the output flange, six
output pins, the disc, eleven ring pins and the housing, with the discs' in
plane shear alongside. At the 22.76 N m static shoulder torque:

| term | N m/rad | share of the compliance |
|---|---:|---:|
| output pin contact | 412,943 | 61% |
| ring pin contact | 1,052,131 | 24% |
| housing, in torsion | 1,970,341 | 13% |
| discs, in plane shear | 12,834,595 | 2% |
| **four known terms in series** | **252,682** | **5.0 times the requirement** |

WHAT IS CONFIRMED HERE IS A DIRECTION AND NOTHING MORE. The reducer is not
obviously the thing that fails the 1 mm limit. That limit stays UNVERIFIED,
because every deflection this design computes is still link elasticity alone.
Do not read 4.8 as a verified margin. Palmgren's line contact relation is a
ROLLER BEARING formula applied to a cycloidal flank and to a pin in a hole,
and those two terms carry 86 percent of the compliance.

The fifth term, the eccentric bearing, has no value at all. Its lever is
derived: a tangential shift d of the disc centre is indistinguishable from
the input angle being larger by d / e, so the disc's rotation errs by
d / (e N), and the torsional stiffness is the bearing's radial stiffness
times (e N) squared. e N is the pitch radius, 30 mm, NOT the eccentricity,
3.0 mm; those two readings differ by a hundred. Asked in reverse, the
bearing needs at least 3.52e7 N/m radial for the joint to reach 50,689 at
all. That is the requirement to carry into bearing selection.

THE ECCENTRICITY IS NOT A FREE VARIABLE and this is where the design moved.
It is K1 times the pin circle radius over the pin count, and K1 carries the
usual design band, 0.5 to 0.75. It was 2.5 mm and K1 0.611; it is 3.0 mm and
K1 0.733, which takes the pitch radius from 25 to 30 mm. That is the ring
pins' moment arm bound and the eccentric bearing's lever squared, so it
raised the chain from 241,801 to 252,682 and cut the bearing requirement
from 5.13e7 to 3.52e7 N/m, for no change in the module's outer diameter at
all. The output pin circle went from Ø50 to Ø48 to hold its 5 mm web, since
a larger eccentricity moves the disc's root inward.

Opening the pin circle to 48 mm at the same K1 would give 275,497 and
2.90e7, but the pins' outer edge then stands at 53.0 mm against a 53.4 mm
motor radius, and the ring body has to go outside them, so it costs about 6
mm on the joint's outside diameter. Not taken.

3.5 mm was considered and refused: K1 would be 0.856, outside the band.
Rather than trust the band, `undercut_margin_m` computes what it stands in
for, the curvature left at the lobe tips before an inward offset by the pin
radius eats the profile. It is 7.09 mm at K1 0.733, 0.55 mm at 0.95, and
negative past 1.0. So the band is conservative and not a cliff edge, and the
refusal rests on pressure angle and contact stress rather than on undercut.

The objection that raising e helps the bearing's stiffness requirement while
hurting its load is real in mechanism and small in size. The tangential
component is T / (N e) and falls, 455 to 375 N; the radial component grows
with the pressure angle, 465 to 493 N; the resultant moves from 648 to 619.
And the radial share does not enter the stiffness at all, because only the
tangential deflection turns the output and an isotropic radial stiffness has
no cross term. The tangential component computed from the full disc
equilibrium agrees with T / (N e) to three figures, which is the independent
check on the whole force model.

The orbiting discs' unbalanced couple is not a design factor AT THIS SPEED.
The arm's duty is 90 degrees in 2 seconds, so a joint peaks near 1.2 rad/s
and the reducer input at ten times that. Two 0.31 kg discs at 180 degrees
cancel their centrifugal resultant and leave m e omega squared times their
spacing, which is 0.0012 N m against a 23.35 N m peak joint torque. At a few
thousand rpm input it would be four orders larger, so the finding carries its
operating point.

CORRECTION WITHIN THE DAY. This first read 682,012 N m/rad and a factor of
13.5. It was optimistic by 2.8 and the whole of the error was in lever arms,
not loads:

- Every cycloidal contact normal passes through the instantaneous pitch
  point, which in the disc's frame is at e N from the disc centre. So NO
  RING PIN CAN HAVE A LONGER MOMENT ARM THAN THAT, whatever radius its
  circle is drawn at: 25 mm here against a 45 mm circle. The computed
  maximum is 24.98. The first estimate used 45 and got a sum of squares of
  5,569 mm2 where the envelope gives 1,700.
- The output pins' normals are all parallel, along the line of the disc's
  offset, so a hole's moment arm is its radius times the sine of its angle
  from that line. The share is solved from those arms now. The first
  estimate put in a hand picked count of engaged pins at full radius.

A proposal to open the output pin circle from Ø50 to Ø60, worth 44 percent
on the term that dominates, is REFUSED ON GEOMETRY. It was argued from the
disc's outline at 42.5 mm, which is the TIP radius. The binding one is the
ROOT, at pin circle less pin radius less eccentricity, 37.5 mm. An output
hole is the pin plus the eccentricity across its radius, so Ø50 leaves a 5.00
mm web and Ø60 leaves 0.00 mm exactly: the hole breaks out of the disc. The
tip and the root differ by twice the eccentricity and that is the whole of
it. A 3 mm web and ligament floor is CHOSEN and is what refuses it.

Scanned against that floor, nothing available buys much. Eight output pins
give 276,173 instead of 241,801. A 3.5 mm eccentricity with the output
circle moved to suit gives 252,420 and lowers the bearing requirement to
2.59e7 N/m, which is the more useful of the two. Ten output pins, a 4.5 mm
eccentricity, and both together all fail the ligament floor.

`joint_module_stiffness_stage` computed 2.6 million N m/rad for the same
housing and concluded the bearing was the whole of the joint stiffness. That
stage uses E and a BENDING second moment, so its scope is the OUT OF PLANE
budget, and within that scope the finding stands. In torsion the same shell
is 1.97 million, using G and J, and it is 12 percent of the drive train's
compliance rather than a rounding error. Same part, different question.

The same calculation closes the disc thickness question that was left open.
In plane shear is linear in thickness, so 0.028 mm would carry the stiffness
alone. The two contacts in series go as thickness to the 0.8 and put their
floor at 0.940 mm, which replaces a 0.292 written earlier the same day from
the same too long lever arms. Pin contact stress was already six times under
its allowable and output pin hole bearing needs 0.20 mm. Four computed
floors, the largest still an order of magnitude under 8 mm. The thickness is
CHOSEN, for what a wire cut disc can be handled, stacked and kept flat at,
which this repository cannot compute and no longer pretends to.

On the out of plane side, THK's A18-1 page 18 carries NO FORMULA, only the
diagram, and it prints two conditions that both matter. The diagram's
condition is RADIAL CLEARANCE ZERO, so it is neither a preloaded nor a
clearanced figure. And THK writes in the text that rigidity is affected by
the deformation of the housing, the presser flange and the bolts, and that
their strength must be taken into account. So the 1.7e6 N m/rad for an
RB10020 is the BEARING ALONE, on top of being an upper bound read at 0.4
kN m where this arm works under five percent along the chart. Any out of
plane budget built on it has to carry the structure as well.

CORRECTION, same day. This was first reported as 205,000 N m/rad and 4.7
times the stiffest available, and that was wrong by a factor of four in the
direction that matters: it made a reachable design look impossible. The
error was splitting the joints' allowance EQUALLY, which gives a sixth of it
to the base yaw and the two roll axes, none of which carries a gravity
moment at full reach, and the same sixth to the shoulder, which carries 67
percent of the demand. The split is by torque times lever now.

WITHDRAWN RESULT. Every statement in this repository before 2026-09-05 that
a link "meets its deflection limit" was measuring the wrong quantity, and
the numbers behind those statements should not be quoted. Each link was
judged on its own loaded face moving under its own load, and those six
numbers were added. A link in a chain also ROTATES everything outboard of
it, and that term was absent: measured on a worked pair the sum reads 0.15
mm where the tool sees 0.51, and the factor is largest for the base column,
which is 150 mm long with the whole 600 mm arm above it. The judgements
were not conservative-but-crude; they understated the tool's deflection,
and by a different amount per link.

It is corrected by reading each face's rigid body ROTATION out of the
CalculiX solution that was already being computed, and crossing it with the
vector from that face to the tool. Nothing new is solved. What changes is
that the rotation is no longer discarded, and that contributions are added
as vectors rather than as scalars, since two links can move the tool in
different directions.

Defects found by standing the parts up, none of which a check here could see:

- The extracted body sat half an element low on all three axes. Volume is
  invariant under translation, so every volume check passed.
- The extracted triangles pointed inward. Every volume here is read through
  `abs()`, so a mesh boolean was the first thing to notice.
- The actuator pocket was not on the joint axis, the base column's axis was
  read in the arm's frame instead of its own, and only one of the two drives
  that touch a link was cut for. Worth 1.173 kg of material.
- Keeping the largest connected component silently discarded a slab held
  solid for an interface, shortening the tool flange by 23 mm.
- Two links claimed 235 cubic centimetres of the same space at the shoulder.
- A placement rule that works on every PAIR of joints made the arm climb
  140.7 mm at each pitch joint and never come back.
- The reach was measured as the sum of the link lengths. The base column is
  150 mm long and adds no reach at all, because it stands up rather than
  out, so the tool came out 750 mm along a 600 mm arm. The reach check
  itself passed throughout, because it uses the joint origins; the wrong
  number lived only inside the deflection weighting and appeared the moment
  that was written. A right length and a wrong reach.
- The build volume test pinned 238.7 mm as the widest part, and 238.7 was
  right about the MARGIN and wrong about the AXIS. When the crossing axis
  rule put a drive's bolt circle inside the domain, the upper arm's long
  axis took half an AK80-64 outer diameter at each end and became 291.0 mm,
  past the 250 mm bed. 238.7 is now its second longest axis, and the 11.3 mm
  it leaves is still exact. Neither assertion could see the change: `fits`
  accepts a part that fits only standing up, and the margin check compared
  250 against a number that had grown past 250, so the difference went
  negative and the check passed for the wrong reason. The upper arm's
  manufacturability is conditional now, not plain.

Operational notes: killing a `ProcessPoolExecutor` parent does NOT kill its
spawned workers, and two abandoned runs once held 11.7 of 16 cores while a
third crawled. The FEM inside the topology loop runs on the GPU through Warp
above 10,000 degrees of freedom; CalculiX is only in the verification path.
A STEP file's checksum cannot test reproducibility because Open CASCADE
writes a timestamp into its header.

## Third work list (2026-09-03), items 8 to 10

| item | what | state |
|---|---|---|
| 8 | surrogate and embeddings retrained on the generated run | DONE, numbers in docs/dataset_spec.md |
| 11a | topology extraction re-verified in CalculiX, passive regions added, `docs/topology_design.md` | DONE |
| 11b | marching cubes, Taubin smoothing, STL, re-solve with linear tets, DFM on the surface | DONE |
| 11c | manufacturing projections (symmetry, support, pull) with the price of each measured | DONE |
| 11e | stress constraint measured against CalculiX; the peak does not converge and the check says so | DONE |
| 11d | weighted multi load compliance and the cross evaluation table | DONE |
| 11f and 10 | end to end demo, both paths, `docs/demo_end_to_end.md`, `scripts/demo_end_to_end.py` | DONE |
| 11g | README, KR and DESIGN brought to the measurements | DONE |
| 12 | trajectories, torque profiles, friction that refuses defaults, gear ratio, three engine cross check | DONE |
| 13 | sourced motor and gearbox entries (maxon, CubeMars, Harmonic Drive, Nabtesco) with per value provenance | DONE, first pass |
| 17 | curved mesh refusals: deeper retry ladder, zero refusals on the two worst cells | DONE |
| 14 | manufacturing shape: fillet study, fastener features, ISO 2768 and ISO 286 notes, `docs/manufacturing_shape.md` | DONE |
| 15 | free form topology strategy, registered and dispatchable, result labelled by CalculiX | DONE |
| 16 | policy seam: rule policy, language model policy behind a caller supplied callable, injection tests | DONE, no API key needed or stored |
| 18 | material limits: printed anisotropy from two machines, brittle criterion that refuses without Weibull data, laminate criterion checked from the material side | DONE |
| 11'-1 | bisection on the volume fraction, judged on the extracted part | DONE, stops at connectivity (0.225, 3.366 kg) not at the limit |
| 11'-2 | chain rule through the additive support filter | DONE, cost 5.07x to 1.15x, unsupported elements 0.025 to 0 |
| 9 | generative CAD loop widened to 11 searchable families (2 discs refused by name), 5 load cases, materials by scaling with the winner re-solved, DFM as a preference | DONE |
| 10 | end to end demo: requirement, generation, solve, DFM, assembly, Gazebo statics | after 9 |

- Item 8 code: `core/part_dataset/industrial_surrogate.py`,
  `scripts/train_industrial_surrogate.py`,
  `scripts/measure_industrial_embeddings.py`,
  `tests/test_industrial_surrogate.py`. Artefacts (not in git):
  `data/generated/surrogate_v1`, `data/generated/embedding_v1`.
- The measured result that changed a belief: with 90,574 rows the closed form
  proxies are no longer needed (0.997 Spearman with or without them), the
  opposite of the 40-part finding. The proxy-alone baseline (0.878) is
  reported next to every model number.

## Second work list, as built (2026-09-03)

- Load cases: `core/part_dataset/labeller.py` LoadKind (bending, axial,
  torsion, combined, thermal_gradient); `nodes/calculix.py` takes
  `nodal_forces` and `ThermalLoad`. Closed-form checks in
  `tests/test_load_cases.py`.
- Scaling: `core/part_dataset/scaling.py`, tags on every label,
  `POISSON_RESIDUAL_BOUND` per case from measurement.
- Batch: `core/part_dataset/batch.py` (cells, done files, refused.jsonl,
  `expand_materials`, `plan`).
- Families: 13 in `families.py`; `ORIGINAL_FAMILIES` names the five the
  classifier and CAD executor were measured on.
- Form score: `geometry/aesthetics/form.py`. DFM: `geometry/manufacturability/`
  (measures.py, processes.py), grade `rule_based_dfm_guideline`.
- Tessellation now flips reversed faces (`core/part_dataset/pointcloud.py`).
- Standard parts: `geometry/cad_export/standard_parts.py`.
- Gazebo: `integration/simulation/gazebo.py` (spring hold, envelope
  interference). Runs need `ign` and real time; about 15 s per test file.
- Guideline: `docs/measurement_guideline.md`. Dataset spec: `docs/dataset_spec.md`.
- Generation run 1 finished 2026-09-03 09:26 KST: 6,430 labelled of 6,500, 70 refused
  (all nonpositive Jacobian), 113,152 records with scaling, 440 MB JSONL in
  `data/generated/industrial_v1` (ignored by git; manifest numbers and md5 in
  docs/dataset_spec.md). 74 min on 8 workers, 1.67 times the serial per-part cost.
- Not done: classifier rules for the eight new
  families; Isaac Sim (hardware below requirements); anything physical.

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
