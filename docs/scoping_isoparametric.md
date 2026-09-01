# General isoparametric hexahedra: where each equation stops being true

Written before implementation. The current element is not wrong; it is exact
within an assumption that the code states plainly, and this document is about
what breaks when that assumption is dropped.

## What the current element assumes

`element_stiffness_from_c` builds one 24x24 Ke and every element reuses it.
That is legitimate only because the mesh is an axis aligned structured grid:

    scale  = (2/dx, 2/dy, 2/dz)      constant
    det J  = dx dy dz / 8            constant

The Jacobian is diagonal, constant inside the element, and identical across
elements. Dropping the grid means dropping all three properties at once.

## What changes, equation by equation

### The Jacobian stops being constant

For a general trilinear hex with node positions x_i,

    J(xi,eta,zeta) = sum_i x_i (dN_i/d xi, dN_i/d eta, dN_i/d zeta)

**Valid when:** det J > 0 at every integration point. That is not a formality.
A hex with a node pushed past its neighbours folds, det J changes sign, and the
solve returns a number that looks like an answer.

**The rule this implies:** compute det J at every Gauss point and REFUSE on a
non positive value, naming the element. A negative Jacobian must never be
silently squared away or clamped.

### Two point Gauss stops being exact

This is the part most easily got wrong, and the current docstring's word
"exact" is what makes it worth stating.

For a parallelepiped the integrand of B^T C B det J is a polynomial that 2x2x2
Gauss integrates exactly. For a general hex, J depends on the natural
coordinates, B contains J^-1, and the integrand becomes a RATIONAL function.
No fixed Gauss rule integrates it exactly.

**So:** on distorted elements 2x2x2 is an approximation, not a quadrature that
happens to be cheap. Raising to 3x3x3 reduces the quadrature error but does not
remove it, and costs 27/8 as much work. The honest statement is that the
current "exact" claim is a property of the rectangular assumption, not of the
element.

### Incompatible modes stop being derived for the element in hand

The Wilson modes in `incompatible_mode_b` take dx, dy, dz explicitly. They are
constructed for a rectangle. Applied unchanged to a distorted hex they break
the patch test: the element no longer reproduces a state of constant strain,
which is the minimum any element must do to converge at all.

**Valid when:** the incompatible mode gradients are evaluated with the
Jacobian at the element CENTRE rather than at each Gauss point, the standard
Taylor correction. Without that correction the element is not merely less
accurate; it converges to the wrong answer, which is worse than being coarse.

### Accuracy degrades with distortion, and the limit is measurable

Trilinear hexes lose accuracy as elements are stretched or skewed. Rather than
assert a threshold, the limit is to be MEASURED on a case with a known answer
and the measured number recorded, in the same way the element size study was
done for the rectangular element.

## The cost that the grid was buying

One shared Ke is why the solve is matrix free and cheap. Per element Ke is
576 doubles, 4.6 kB. At the 150000 DOF ceiling of the small profile that is
roughly 50000 elements and 230 MB of Ke alone, on a machine profiled at 4 GB.

Two options, and the choice was to be made on a measurement, not a
preference:

1. **Store per element Ke.** One build, fast apply, memory as above.
2. **Recompute Ke inside the kernel.** No storage, but the eight Gauss point
   loop runs on every matrix vector product, and there are hundreds of those
   in a CG solve.

### Measured, on this machine

RTX 3050 Laptop, 4 GiB, 20000 elements, float64:

    shared Ke        2.186 ms per matvec      4.5 kB
    per element Ke   4.150 ms per matvec     87.9 MB

So storing costs 1.9x the time of the shared path, and 220 MB at the 150000
DOF profile ceiling. On a 4 GiB card that is about five percent of memory,
which is affordable. Option 1 is therefore implemented and option 2 is NOT,
because there is no measured problem for it to solve. Recompute becomes worth
building when a mesh is large enough that 4.6 kB per element stops fitting,
and that point has not been reached.

The 1.9x is the honest price of generality, and it is why the structured path
stays and stays the default: on a grid it is both faster and smaller, and it
is exact where the general element is not.

## What will NOT be claimed

- The structured path is not being replaced. It stays, it is faster, and it
  stays the default when the mesh is a grid.
- No claim that the general element is as accurate as the structured one on
  the same problem. It is not, and the gap is to be measured.
- No second order elements in this step.

## How it gets verified

Each of these has an answer known independently of the code.

1. **Reduces to the existing element.** Given a hex that IS an axis aligned
   box, the general Ke must equal the structured Ke to machine precision. This
   is the strongest available check because it compares against an already
   verified implementation, and any indexing or ordering error shows up at
   once.
2. **Patch test.** Impose a linear displacement field on the boundary of a
   deliberately distorted mesh. Every interior node must land on that field
   and the recovered strain must be constant, to round off. This is the test
   the uncorrected Wilson modes fail.
3. **Rigid body motion.** Translate and rotate the element: the internal force
   must be zero. Ke must have exactly six zero eigenvalues, no more and no
   fewer. Fewer means a spurious constraint; more means a hourglass mode.
4. **Refusal on a folded element.** Build a hex with an inverted corner and
   require the code to raise, naming the element, rather than return a result.
5. **Convergence under refinement** on a case with a closed form answer, with
   the mesh distortion held fixed so that refinement is the only thing varying.
   The earlier study taught that varying two things at once produces a
   non monotonic curve that means nothing.
