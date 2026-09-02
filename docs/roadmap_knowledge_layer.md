# CAD-native engineering design knowledge layer

A roadmap, and a record of what the existing code already provides. This is an
addition to Daedalus, not a parallel stack: the physics, the registry, the
Brain and the evidence ladder are the ones already here.

Everything below is SIMULATED unless it says otherwise. Nothing in this
project has been physically tested, and nothing in this layer changes that.

## The shape of it

One Part Dataset schema with two producers. A real STEP file and a
parametric design rendered to CAD both go through the SAME analyzer and come
out as the same schema. The parametric Design Genome does not get replaced by
a CAD representation; the two coexist, and the provenance field says which
produced a given record.

No new physics. A STEP solid is meshed by Gmsh into tetrahedra and routed to
the CalculiX general shape capability that already exists. A part that a
structured hex grid can cover still goes to Warp, which is faster and is
cross-validated. The knowledge layer decides WHICH existing solver runs; it
does not add a solver.

Synthetic data first. The parametric side can already generate and label parts
in bulk, and the optimisation path is now fast enough to do it at scale, so a
training corpus can exist before any dependence on scraped CAD.

## Three rules that are not negotiable

**Licensing.** No scraped CAD is bundled into this Apache licensed
repository. Synthetic parts and permissively licensed public sets only, each
checked individually rather than by reputation. Proprietary CAD stays local
and untracked. The schema carries source and licence as REQUIRED fields, so a
record whose provenance is unknown cannot be stored at all.

**Rules before learning.** Feature recognition starts rule based, each rule
with a stated validity domain. The rules are less tidy than they sound, and
the tidy version was wrong. Measured on parts built for the purpose:

    A hole is a CONCAVE cylindrical face. Radius does not identify it, because
    a fillet is a cylinder too; what separates them is which side the material
    is on. A part whose fillet radius happens to equal its hole radius would
    merge the two under a radius rule.

    A fillet is a cylinder, a SPHERE or a torus depending on the edge it runs
    along. Filleting every edge of a box gives twelve cylinders and eight
    spheres and no toruses at all. A torus appears where a fillet follows a
    circular rim, and there the fillet radius is the MINOR radius.

A learned recogniser comes after there are labels to learn from, and the rule
based one stays as the control to compare against. A learned model that agrees
with nothing is not evidence.

**A surrogate is not a verification.** The evidence ladder has a SURROGATE
level BELOW SIMULATED (implemented, `brain/semantic/evidence.py`). A surrogate
may screen, rank and suggest. It may never produce a final verdict:
`may_decide` is false for it, `integration/checks.py` refuses a PASSED or
FAILED on surrogate evidence at construction, and the Phase 6 predictor and
`screen_and_verify` grade their own output. Tests enforce each of those rather
than a convention. Human preference is a separate axis and does not enter the
physical ladder at all.

## What the existing code already gives us

Checked against the code rather than assumed.

| Need | State today |
|---|---|
| OpenCASCADE in the venv | ALREADY PRESENT. OCP 7.9.3.1 arrives with build123d, and its STEP reader works. Nothing to install. |
| STEP reading | Verified on this project's own export: 2 solids, 20 faces, 96 edges, and a volume of 111200 mm3 that matches the closed form for the two links exactly. |
| General shape to CalculiX | Done. Gmsh tetrahedra to CalculiX, C3D10, agreeing with the hex solution to 0.5 percent on the one shape both meshers cover. |
| Registry for new nodes | Straightforward. Five external nodes have been added this way already. |
| Brain SURROGATE level | DONE. Inserted between UNVERIFIED and SIMULATED with ceiling 0.40. Ranks are only ever compared to each other, so nothing else moved; the existing 140 Brain and integration tests passed unchanged. |
| Material provenance | Already better than a new vocabulary would be, so the existing MaterialStatus is reused: reference_typical, supplier_datasheet, measured, assumed. Nothing parallel is added beside it, because two vocabularies for one idea eventually disagree. |
| Design Genome extension | It already reserves `topology` and `structure` dict fields for later phases, so a CAD field fits the existing shape. It sets `extra="forbid"`, so the field has to be declared, which is the right way round. |
| Design reference priors | Already present, with mandatory provenance and a confidence ceiling. Design Intent extends it rather than replacing it. |

### Two CAD backends, for two different reasons

**build123d is the primary parametric backend.** It is already installed, it
is OpenCASCADE underneath, it runs natively on Linux and it costs nothing. The
parameters to CAD to STEP half is built on it, which is what keeps the
synthetic data engine and the generative phases from depending on a tool this
machine cannot run.

**The Fusion node stays honestly unavailable on Linux.** It is a deliberate
stub reporting `unavailable: requires Fusion paid entitlement`, and Fusion
runs on Windows and macOS. Marking it promoted without a working round trip
would be exactly the kind of false claim the rest of this project is arranged
to prevent. The Fusion round trip runs on the Windows host instead.

Having both is worth more than having either. build123d and Fusion are
INDEPENDENT kernels, so a STEP file from one is a second opinion on the
analyzer reading a STEP file from the other, in the same way CalculiX is a
second opinion on Warp. Once the analyzer exists, parts authored in Fusion
become an independent check on it rather than merely more input.

## Phases

Phase 0 and the solver queue interleave: CAD work is CPU and OpenCASCADE
bound, physics is GPU bound, so they do not contend.

- **Phase 0** Foundations. OpenCASCADE confirmed, licence policy written, Part
  Dataset schema v0, the existing code cross-check above, and a STEP round
  trip spike using exports this project already produces.
- **Phase 0b** A parametric CAD backend on build123d, so that parameters can
  become a STEP file without leaving this machine.
- **Phase 1** STEP analyzer: geometry, topology, and the face adjacency graph,
  each quantity checked against a closed form where one exists.
- **Phase 2** Rule based feature recognition, with the validity domain of each
  rule stated before it is written.
- **Phase 3** Shape descriptors and classification. DONE
  (`core/part_dataset/descriptors.py`, `classify.py`, capability
  `analysis.cad.classify`): 22 scale-free descriptors, topology rules that
  classify the five families and say UNKNOWN otherwise, and a nearest
  neighbour model graded SURROGATE that rejects parts beyond its training
  set. The Euler count had to learn about inner face loops first.
- **Phase 4** Physical labelling through the EXISTING solvers, including
  boundary condition tagging.
- **Phase 5** Synthetic data engine. DONE (`core/part_dataset/engine.py`):
  five families with closed-form volumes, every record checked against the
  parameters that made it, labelled through Gmsh and CalculiX with
  `mesh_sensitivity` on every solver label, all labels graded SIMULATED by
  construction. Dataset scale is decided by what Phase 3 needs.
- **Phase 6** CAD embeddings. DONE (`core/part_dataset/pointcloud.py`,
  `embedding.py`): area-weighted surface points in a canonical frame, the D2
  histogram as the no-learning baseline, a PointNet embedding graded
  SURROGATE. Measured nearest-neighbour retrieval: descriptors 1.00,
  embedding 0.88, D2 0.64. The embedding has not beaten the topology on
  five prismatic families and the docs say so.
- **Phase 7** Surrogate prediction, behind the SURROGATE evidence guard. DONE
  (`core/part_dataset/shape_surrogate.py`): descriptors plus a beam-theory
  proxy predict the labeller's CalculiX deflection, held-out R2 0.94 on 40
  parts; `screen_and_verify_parts` ranks with it and returns only a
  solver-verified winner. Without the proxy feature the same model had R2
  below zero, which is recorded rather than hidden.
- **Phase 8** Design intent, measured by ablation against real solvers rather
  than asserted. DONE (`core/part_dataset/intent.py`): an IntentClaim is a
  DesignReference item with mandatory provenance; `ablate` changes the one
  parameter the claim is about, labels both parts through CalculiX, and says
  SUPPORTED, REFUTED or INCONCLUSIVE against the mesh noise; the Brain
  records evidence or counterexamples and derives the level by the ladder.
- **Phase 9** Generative design and an autonomous CAD loop, extending the
  existing agent loop to emit CAD. DONE as far as it goes
  (`agent/execution/cad.py`, registry method `generative_cad`): the loop
  builds candidates from three part families, ranks them with the proxy or
  the shape surrogate, has CalculiX label the shortlist, and records the
  winner's STEP path in the episode. Not free-form; the families are the
  design space.

## What this roadmap does not claim

It does not claim any of this is validated by physical test, because none of
it is. It does not claim a learned model can replace a solver. And it does not
claim a feature recogniser understands design intent: recognising a
cylindrical face as a hole is geometry, and why the hole is there is not
recoverable from the geometry alone.

## The hole rule, and what refuted each version of it

Three versions, two of them refuted by a fixture rather than by review.

1. **A hole is a small cylinder.** Refuted by the equal radius fixture, where
   four bores and four corner fillets are all radius 4. A radius test has
   nothing to sort on.
2. **A hole is a concave cylinder.** Refuted by the L bracket, whose reentrant
   corner blend is concave and is a fillet. Both are concave, so concavity
   cannot separate them.
3. **A hole is a concave cylinder that wraps a full turn.** Current. The blend
   is a ninety degree sector; a bore is the whole turn.

Two candidates were measured for the third version, a full turn and an absence
of tangent neighbours, and BOTH separated the first five fixtures correctly.
Those fixtures could not choose between them. The full turn was chosen on an
argument they did not test: a bore blended at both its mouth and its bottom
has two tangent neighbours, which the tangency candidate would call a fillet,
while it remains a full turn.

A sixth fixture was then authored to be exactly that bore, and it confirmed
the argument. The bore is concave, wraps 360 degrees, has two tangent
neighbours, and is reported as one hole with two toroidal fillets. The
rejected candidate would have found no hole at all. The choice is no longer
resting on reasoning alone.

The known limit is stated rather than hidden: a hole broken open by an
intersecting feature no longer wraps a full turn and will not be reported. That
gap is preferred to misreporting ordinary blended holes.
