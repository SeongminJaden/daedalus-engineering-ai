# Topology optimisation as a source of parts

A SIMP run reports the compliance of a density FIELD. The part is that field
thresholded. Those are two different structures, and everything in this
document is the measurement of the difference, made by re-solving the
extracted part in CalculiX, which is an independent solver from this project's
matrix-free FEM. Agreement between the two is a cross validation, not a new
capability, and the grade stays SIMULATED.

The problem throughout is a cantilever: 0.5 by 0.1 by 0.05 m, aluminium 7075,
1 kN down on the free face, clamped face at the other end, volume fraction
0.35. Reproduce with `scripts/topology_design_study.py`.

## What the plain run hands over

Nothing that can be solved. On this problem the elements the point load is
applied through settle at density 0.39, below every useful threshold, so
thresholding severs the load path: 19 face-connected pieces, the largest
holding the clamped face and none of them reaching the tip. The extractor
refuses with the counts rather than handing CalculiX a singular problem.

That measurement is why `SimpProblem` now carries passive regions. The
elements touching the load and the support are held solid for the whole run,
which is the standard passive region of the literature, and the volume
constraint accounts for them so the requested fraction is still what comes
out. Without that correction the projected run delivered 0.50 against a
requested 0.40; the volume projection now measures the volume including the
passive elements.

## What the extracted part does, measured

| run | elements | grey fraction | threshold | part mass kg | part over field compliance |
|---|---|---|---|---|---|
| SIMP, no passive patches | 768 | 0.44 | 0.3, 0.5, 0.7 | disconnected | |
| SIMP, passive patches | 768 | 0.42 | 0.3, 0.5, 0.7 | disconnected | |
| three field, passive patches | 768 | 0.16 | 0.3, 0.5, 0.7 | disconnected | |
| three field, 1536 elements, radius 2.5 | 1536 | 0.09 | 0.3 | 2.598 | 0.90 |
| | | | 0.5 | 2.561 | 0.91 |
| | | | 0.7 | 2.488 | 0.92 |
| three field, 3840 elements, radius 2.0 | 3840 | 0.02 | 0.3 | 2.492 | 0.97 |
| | | | 0.5 | 2.490 | 0.97 |
| | | | 0.7 | 2.468 | 0.97 |

The pattern is the grey fraction, not the threshold. While a fifth of the
elements are intermediate, the part either does not exist or is nothing like
the field. Once the projection and the resolution push grey to two percent,
the extracted part is within three percent of the field at every threshold and
the threshold barely matters, which is the only regime in which "the optimiser
says this part has that compliance" is a true sentence.

A smaller run shows the other failure mode. At 480 elements with a coarser
projection the extracted parts are connected but the compliance ratio runs
from 0.59 at threshold 0.3 to 22 at threshold 0.5: rounding grey material up
to solid makes a heavier, stiffer part, and cutting it away makes a much
softer one. Neither is the field.

## From voxels to a surface a designer could use

`optimization/topology/smooth.py` runs marching cubes on the density field and
then Taubin smoothing, which alternates a shrinking pass with an expanding one.
The result is watertight and a single body, which the blocky export is not, and
it is inexact, which the blocky export is not either. Every surface reports its
volume against the volume of the field, because that is the quantity the
optimiser was constraining.

On the 1536 element cantilever, volume against the field:

| iso level | smoothing passes | volume error vs field | vs the voxel body |
|---|---|---|---|
| 0.3 | 0 | +24.6 percent | +18.0 |
| 0.3 | 20 | +20.3 percent | +13.9 |
| 0.5 | 0 | -7.2 percent | -10.9 |
| 0.5 | 20 | -7.6 percent | -11.3 |
| 0.7 | 0 | -36.0 percent | -36.7 |
| 0.7 | 20 | -33.4 percent | -33.1 |

The iso level dominates and the smoothing barely moves the volume, which is
the point of Taubin: twenty passes of plain Laplacian smoothing shrink the
same body by 29.7 percent, and the two-pass scheme by 1.6.

Marching cubes has its own bias, measured on a fully solid field whose answer
is arithmetic. Densities live at cell centres, so the surface of a solid block
passes through the centres of the boundary cells and the body comes out half a
cell small on every face: 9.9 percent low at 8 by 4 by 2 cells, then 2.6, 0.67
and 0.17 as the grid doubles. First order in the cell size, and reported rather
than assumed away.

### Re-solving the smoothed body

Gmsh reclassifies the STL facets, sews a shell and fills it, so the smoothed
body can be meshed and solved like any other part. On the cantilever, linear
tetrahedra at 8 mm give a compliance of 3.520e-1 J against the field's
3.582e-1, a ratio of 0.98. Two limits came with it and both are measured, not
avoided: second order tetrahedra on this surface invert (68 elements with a
nonpositive Jacobian) and CalculiX writes nothing, and a finer surface mesh is
refused by Gmsh with overlapping facets. Linear tetrahedra are stiff in
bending, so the 0.98 is a comparison between an approximate mesh and an
approximate field, not a validation of either.

### The rules already in the repository, applied to it

The smoothed surface is a triangle mesh, which is what the manufacturability
rules read, so the DFM assessment of a topology result costs nothing extra.
On this part, per process, measurable rules passed: CNC milling 1 of 2, CNC
turning 1 of 1, sheet metal 0 of 1, FDM 1 of 2, SLS 1 of 2, SLM 1 of 2, die
casting 2 of 3, injection moulding 0 of 1. The failures are the expected ones
for an organic shape: six-axis inaccessibility, overhang fraction above the
45 and 50 degree rules, and no draft anywhere. Smoothing raises the minimum
wall from 5.3 mm to 13.7 mm and leaves the overhang fraction where it was.

## Manufacturing constraints inside the loop, and what each costs

`optimization/topology/manufacturing.py` supplies projections that `SimpProblem`
applies to the density the solver sees, so the compliance a run reports belongs
to a field that already obeys the constraint. Same cantilever, 1536 elements,
three-field projection, 100 iterations, volume fraction 0.35 throughout.

| constraint | compliance J | against none | overhang area 45 deg | undercut fraction | min wall mm |
|---|---|---|---|---|---|
| none | 3.582e-1 | 1.00 | 0.23 | 0.41 | 13.7 |
| symmetry about the width | 3.582e-1 | 1.00 | 0.23 | 0.41 | 13.7 |
| symmetry about the height | 4.144e-1 | 1.16 | 0.28 | 0.50 | 15.7 |
| symmetry about the length | 1.316e0 | 3.67 | 0.20 | 0.45 | 13.1 |
| additive support filter | 1.817e0 | 5.07 | 0.25 | 0.42 | 11.3 |
| casting pull direction | 1.588e0 | 4.43 | 0.15 | 0.00 | 8.8 |

Three things in that table are worth saying plainly.

Symmetry about the width is free because the unconstrained optimum is already
symmetric about that plane, to 1e-16. A constraint that costs nothing is not
evidence that constraints are cheap; the same projection about the length,
where the cantilever is genuinely asymmetric, costs a factor of 3.7.

The casting constraint does what it claims: the undercut fraction goes from
0.41 to exactly zero, and the price is a factor of 4.4 in compliance. That is
the honest trade for a part that can leave a mould along one axis.

The additive support filter does not. It costs a factor of 5.1 and leaves the
measured overhang area where it was, 0.25 against 0.23. The filter constrains
the field (nothing floats above nothing) but the gradient the optimiser follows
is the unfiltered one, since the chain rule through the layer maximum is not
applied, so the run pays for a constraint it cannot steer with. It is left in
the module with this paragraph attached rather than removed, because the
measurement is the useful part; anyone reaching for it should know it currently
buys nothing.

### Filter radius and the thinnest member

| filter radius, cells | compliance J | min wall mm | overhang 45 | undercut |
|---|---|---|---|---|
| 1.5 | 3.726e-1 | 0.5 | 0.31 | 0.49 |
| 2.5 | 3.582e-1 | 13.7 | 0.23 | 0.41 |
| 3.5 | 4.066e-1 | 15.6 | 0.21 | 0.24 |

The cell is 15.6 mm. At radius 1.5 the extracted body has half-millimetre
slivers, which is the filter failing to impose any member size at all; at 2.5
and above the thinnest wall is about one cell and stops growing with the
radius. So on this grid the minimum member size is set by the discretisation
rather than by the filter, and a real minimum-thickness requirement needs a
mesh several cells finer than the member, not a larger radius.

## What this does not fix

The part is still a voxel body. It is blocky, its surface is non-manifold
wherever two voxels meet only at an edge, and it is not a clean STEP. The
peak von Mises reported above is a voxel-mesh value at re-entrant corners and
is not converged; it is there to compare runs with each other, not to judge a
design.
