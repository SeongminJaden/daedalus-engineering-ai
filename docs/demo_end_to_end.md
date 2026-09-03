# One requirement, two design paths, the same checks

`scripts/demo_end_to_end.py` takes a single requirement and runs it through
both routes this repository has, then applies the same downstream checks to
each. Everything below is SIMULATED or a rule set. Nothing has been made or
measured.

## The requirement

A robot forearm link: 0.5 m long inside a 0.1 by 0.1 m envelope, 196.2 N at
the tip, tip deflection under 1 mm, aluminium, milled.

## Path A, family search

The generative CAD executor built 12 candidates, ranked them with the beam
proxy, and had CalculiX label the best 3. The winner is a hollow rectangle in
6061 aluminium, 0.590 kg, tip deflection 0.661 mm against the 1 mm limit,
chosen across four candidate materials by exact scaling and then solved again
in its own. Four and a half seconds. Manufacturability for milling: 2 rules
measurable, both pass, 2 more not measurable on this shape.

## Path B, topology

SIMP in the same envelope with passive load and support patches and the three
field projection, 1536 elements, 100 iterations, volume fraction 0.35. 84
seconds. Grey fraction 0.108, field compliance 7.298e-3 J.

| threshold | part mass kg | part compliance J | part over field |
|---|---|---|---|
| 0.3 | 5.305 | 6.340e-3 | 0.87 |
| 0.5 | 5.287 | 6.390e-3 | 0.87 |
| 0.7 | 4.921 | 6.820e-3 | 0.93 |

Smoothed to a watertight single body of 2400 triangles whose volume is 11.7
percent below the field's. Manufacturability for milling: 1 of 2 measurable
rules fails, six-axis inaccessibility, which is what an organic shape does to
a mill.

## The two paths are not comparable, and that is the result

The family part weighs 0.590 kg and meets a deflection limit. The topology
part weighs 5.3 kg and meets nothing, because the topology run was given a
volume fraction rather than the deflection limit, and 35 percent of that
envelope is a great deal of aluminium. Reading the two masses side by side as
if they answered the same question would be the mistake this demo exists to
make visible: a compliance minimisation at fixed volume answers "the stiffest
shape for this much material", and the requirement asked "the lightest shape
under this deflection". Turning the second into the first needs a bisection on
the volume fraction, which is another run per step and is not in this demo.

## Downstream, identically for both

Fasteners come from the catalogue with their standards: an M6 ISO 4762 socket
head screw 30 mm long, head 10 mm across and 6 mm high, 5 mm hex key, volume
1.2545e-6 m3, and its ISO 4032 nut, both linked to the nearest steel in the
database with the note that says it is the nearest and not the specified one.

The part goes into a two link arm. This project's statics is compared with
Gazebo Fortress 6.18 by preloading each joint spring with the statics torque:
the joints settle within 5.2e-4 rad and the spring torque agrees with the
statics at the settled pose to 0.03 percent on both joints. Envelope
interference at that pose reports zero pairs checked, because the only pair is
adjacent and adjacent links share a joint by construction; a longer chain is
what makes that check say something.

## What this demo does not show

No part here was manufactured. No number was measured on hardware. The
manufacturability results are a rule set with a grade of its own that is not
on the evidence ladder, the topology stress values are voxel corner numbers
that do not converge, and the agreement with Gazebo is two simulations
agreeing with each other.
