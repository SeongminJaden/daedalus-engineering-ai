# Elmer: what it is for, and where each equation stops being true

Written before implementation, not after. The point of stating a validity
domain first is that it can then refute the implementation, rather than being
back-fitted to whatever the code happened to do.

## Why Elmer at all

The curation rule for solvers is that a new one must add physics the stack
does not have. Measured against the registry as it stands:

| capability | solver | physics |
|---|---|---|
| analysis.fea.calculix | CalculiX | structural, and conduction |
| analysis.cfd.openfoam | OpenFOAM | incompressible flow |
| analysis.multibody.pinocchio | Pinocchio | rigid body dynamics |
| simulation.multibody_contact | MuJoCo | contact |

Elmer adds one thing nothing else here has: **electromagnetics**, and with it
the coupled solve where a field computed by one equation is a source term in
the next. That is the whole justification. It is worth being blunt that
Elmer's structural and conduction solvers overlap CalculiX's, and that overlap
is not a reason to add it. Overlapping capability gets an independent
cross-check at best, not a new capability entry.

## The equations, and the domain of each

### Magnetostatics

Solved for the vector potential A, with B = curl A:

    curl( (1/mu) curl A ) = J

**Valid when:** currents are steady or slow enough that induced currents do
not react back on the field. The test is the skin depth

    delta = sqrt( 2 / (omega mu sigma) )

against the conductor's smallest dimension. When delta is much larger than the
conductor, the current fills it and magnetostatics holds.

**Stops being true when:** the frequency rises until delta shrinks below the
conductor size. Current crowds into the surface, the resistance rises, and a
magnetostatic answer is wrong in a direction that flatters the design. At 50 Hz
in copper delta is about 9 mm, so a 20 mm bar is already marginal. This is not
an exotic corner; it is ordinary mains-frequency hardware.

**Also stops being true when:** the material saturates. Linear mu is an
assumption about the operating point, not a property of the steel. Past the
knee of the B-H curve the permeability collapses and a linear solve
overpredicts flux, again in the flattering direction.

### Joule heating as a source term

    q = |J|^2 / sigma

**Valid when:** sigma is taken at the temperature the conductor actually
reaches, not at 20 C.

**Stops being true when:** the temperature rise is large enough to change
sigma appreciably. For copper the coefficient is about 0.00393 per K, so a
100 K rise raises resistivity by roughly 39 percent, and the heating with it.
Ignoring that underpredicts the temperature, and the error compounds: hotter
means more resistive means hotter. A one-way EM-then-thermal solve is only
honest when this feedback is small; otherwise the coupling must run both ways.

### Heat conduction with convection and radiation

    rho c dT/dt = div( k grad T ) + q

with boundary flux

    -k dT/dn = h (T - T_inf) + eps sigma_SB (T^4 - T_inf^4)

**Valid when:** h is known. It rarely is.

**Stops being true because of h, not because of the equation.** The conduction
solve is the trustworthy part; the convection coefficient is a correlation
with a factor-of-two spread in the honest literature for natural convection.
A temperature computed from an assumed h inherits that spread. Radiation is
better founded, since eps for a known surface is measurable, but the fourth
power means an emissivity guessed at 0.9 instead of 0.3 changes the radiated
flux threefold.

The consequence for this repo: an Elmer thermal result is SIMULATED, and the
assumed h must be recorded beside it, because the number is a statement about
an assumed cooling condition rather than about the part.

## What will NOT be claimed

- No structural capability. CalculiX has that, is already verified here, and
  a second answer to the same question is a cross-check, not a capability.
- No nonlinear magnetics in the first cut. Linear mu only, with the saturation
  limit stated and a refusal rather than a silent extrapolation past it.
- No free-surface or turbulent flow. OpenFOAM has the flow, and Elmer's is
  not obviously better.

## How it gets verified

The same ladder as every other node, and the same rule: an analytic case
whose answer is known independently of the solver.

1. **Infinite solenoid.** B = mu0 n I on the axis, uniform inside. A closed
   form with no fitted constant. The finite model must be run long enough that
   end effects are below the tolerance, and that length is itself a measured
   quantity, not an assumption.
2. **Coaxial cable.** B = mu0 I / (2 pi r) between the conductors, zero
   outside. Tests that the solver gets the field to fall off correctly and
   that the return current cancels.
3. **Slab with uniform heat generation.** T_max - T_wall = q L^2 / (2 k),
   parabolic profile. Tests the conduction and the source term together.
4. **Coupled check.** A bar carrying known current: compute the Joule heating
   analytically from I^2 R, and require the solver's integrated source to
   match it. This is the one that would catch a coupling wired up backwards,
   which the two solves passing separately would not.

Every one of these is a hand-checkable number. None of them is a tolerance
tuned until the solver passes.
