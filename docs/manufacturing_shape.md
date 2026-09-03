# From an analysis shape to a manufacturing one

The part families produce clean prismatic solids: sharp interior corners, no
fastener features, no tolerances. That is the right input for a solver and the
wrong output for a shop. This document is what was added and what each
addition measures.

## Fillets, and why the sharp case is not a reference

`fillet_study` solves the same part at several radii at the SAME mesh size,
because a changing mesh would confound the comparison. On a stepped cantilever
(40 mm root block 40 mm tall, then a 20 mm tall beam to 200 mm, 20 mm wide,
500 N at the tip, aluminium 7075, 3 mm elements):

| fillet radius mm | peak von Mises MPa | max displacement m | mass kg |
|---|---|---|---|
| sharp | 71.45 | 8.685e-4 | 0.2698 |
| 1 | 89.50 | 8.645e-4 | 0.2698 |
| 2 | 82.00 | 8.576e-4 | 0.2699 |
| 4 | 67.35 | 8.427e-4 | 0.2701 |
| 8 | 61.24 | 8.103e-4 | 0.2713 |

Between filleted radii the trend is what a designer expects: bigger radius,
lower peak, slightly stiffer part, slightly more mass. The sharp case reads
lower than the 1 mm fillet, which is not a physical result. It is the
singularity: the peak at a sharp re-entrant corner has no converged value, and
what a solver reports there is a property of the mesh.

Measured on the same part at three mesh sizes:

| elements | sharp corner peak MPa | 4 mm fillet peak MPa |
|---|---|---|
| 4 mm mesh | 65.33 | 63.39 |
| 3 mm mesh | 71.45 | 67.35 |
| 2 mm mesh | 81.91 | 71.72 |

The sharp corner rises 25 percent over that refinement and shows no sign of
stopping. The filleted case rises 13 percent, which is not convergence either:
two millimetre elements across a four millimetre radius is two elements, and a
fillet needs more than that. So the study reports a comparison between radii at
a stated mesh, and refuses to call any of it a stress concentration factor.

## Fastener features

`fastener_feature` gives a clearance hole from the ISO 273 table and a
counterbore sized from the head diameter and height in the SAME ISO 4762 table
the catalogue screws are built from, plus a stated clearance. Taking the head
size from a second table is how a hole and a screw come to disagree six months
later. Three clearance classes are available and ordered: close, normal, free.

## Tolerances, and what STEP cannot carry

`DrawingNotes` holds the ISO 2768 general tolerance class with the band table,
and ISO 286 fits for named features computed by the fits module rather than
restated. It is written beside the STEP file as JSON, because this project
writes AP203, which has no tolerance entity at all. AP242 does; this project
does not write it. The note in every record says so, so a reader cannot
mistake a toleranced drawing for a toleranced file.

## What the DFM rules see

Nothing. The same part with a sharp corner and with an 8 mm fillet passes the
same two measurable CNC milling rules. The rule set reads wall thickness,
overhang, tool access and draft; a fillet at an interior edge changes none of
them. That is a limit of the rule set and not a statement that the fillet does
not matter, which is exactly the difference the stress table above shows.
