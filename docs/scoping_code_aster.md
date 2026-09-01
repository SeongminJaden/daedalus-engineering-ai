# Code_Aster: what it is for, and what the first cases must establish

Written before the node exists. Code_Aster overlaps CalculiX heavily, so the
justification has to be specific rather than "another solver".

## Why it is here

CalculiX already does linear elasticity and is verified. Code_Aster earns its
place on nonlinear material behaviour, contact, and plasticity, which is where
CalculiX is weaker and where an independent implementation matters most.

That is NOT what gets verified first. Plasticity has almost no closed form
answers, so a wrong plastic result cannot be distinguished from a wrong
install, a wrong material input, or a misunderstanding of the model. The first
cases are linear and have exact answers, purely to establish that the install
and the plumbing are right. Only then does a nonlinear result mean anything.

## The order, and why

1. **A bar in tension.** Uniform stress is a state linear elements reproduce
   EXACTLY, so the expected error is round off, not a percentage. That is what
   makes it a good first case: any discrepancy is a bug, and cannot be
   excused as a coarse mesh.
2. **A thick cylinder under internal pressure, the Lame solution.** Exact for
   plane strain elasticity, so the discretisation genuinely converges to it
   and the ORDER of convergence can be measured: displacement at second order
   and stress, being its derivative, at first.
3. **A cantilever, against CalculiX.** NOT against Euler-Bernoulli. A 3D solve
   converges to the true three dimensional answer, and the difference from
   beam theory is a real shear deformation of order (h/L)^2 that no amount of
   refinement removes. Demanding a convergence order against a formula the
   solution is not converging to would be fitting numbers, not verifying.
   Beam theory is used only as an order of magnitude sanity check; the
   quantitative comparison is against CalculiX.

## A measured trap, recorded because it cost real time

The bar in tension first came out 3.9 percent low in displacement and 13.9
percent high in stress. The cause was the boundary conditions, not the solver.

Holding whole lateral planes, DY = 0 on the y = 0 face and DZ = 0 on the
z = 0 face, perturbs the solution, while holding the axial face and removing
the remaining rigid body motions with three point constraints gives the exact
answer to 2e-13.

What was established by measurement:

* Refining from 433 to 25317 tetrahedra moved the error 3.94, 2.88, 2.40
  percent. It shrinks but does not go to zero, so this is a different boundary
  value problem rather than a discretisation error.
* The total reaction is exactly minus 20000 N, equal to stress times area, so
  the load is applied correctly.
* The mesh groups are exactly the intended planes: the y = 0 group spans
  y from 0 to 0, the z = 0 group likewise, and both hold 1602 triangles, so
  the face classification is not at fault.
* Constraining the two planes separately gives errors of opposite sign,
  plus 0.42 and plus 2.9 percent, and holding both gives minus 2.4 percent.
  Displacements ABOVE the exact value cannot come from added stiffness, so
  the field is being made non uniform rather than merely stiffened.

What was NOT established: why holding those planes perturbs anything, given
that the exact uniaxial solution already has zero lateral displacement on
both of them and so should satisfy the constraints with zero reaction.

This is left as an open question rather than given an invented explanation.
The node uses the configuration that measurement shows is exact, not the one
that reasoning says ought to be fine.
