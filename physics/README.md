# physics — GPU physics via NVIDIA Warp

Kernels (`warp_kernels/`), structural evaluation (`structural/`), batch
orchestration against the active GPU profile (`solver/`). `rigid_body`,
`thermal` and `collision` are still stubs.

## Model fidelity — Phase 2

The current structural model is **Euler–Bernoulli beam theory**: a root-fixed
cantilever with a transverse point load at the free tip and a uniform hollow
rectangular section. It is differentiable and runs a whole population in one
GPU launch, which is what an optimizer needs — but it is a *beam* model, and it
is honest about what that costs:

- **No stress concentration.** The root fillet, bolt holes and any fixture
  detail are invisible to it. Real peak stress at the root will be **higher**
  than `max_bending_stress_pa` reports. Treat the reported stress as a lower
  bound on the true peak.
- **No transverse shear deformation.** Deflection is bending-only. That is
  accurate for a slender link (L/h large) and increasingly wrong as the link
  gets stubby; Timoshenko theory or FEM is needed there.
- **No buckling check.** A thin-walled section can fail by local wall buckling
  well before the bending stress reaches yield. Nothing here detects that.
- **Uniform prismatic section, perfect material.** No taper, no joints, no end
  caps, no weld or heat-affected zones, no residual stress.
- **Static only.** `first_natural_frequency_hz` is the analytic first bending
  mode of a bare cantilever under self-weight, with no tip mass and no damping.
- **Mean shear stress is reported for awareness only.** Bending governs a
  slender link; `mean_transverse_shear_stress_pa` is `P/A`, not a peak value.

Consequently a design that passes here is a *candidate*, not a verified part.
3D FEM in a later phase is what resolves stress concentration, shear and
buckling, and it is what a design must clear before it is treated as real.
This mirrors the analytical→FEA split: cheap differentiable models to search
the space, expensive high-fidelity models to confirm the winner.

## Verification

Both the forward metrics and the autodiff gradients are checked against
`tests/reference_beam.py`, an independent float64 numpy implementation that
imports neither Warp nor any project module — so a bug shared between the
kernel and `core.design_genome` cannot pass both.
