"""physics.warp_kernels.kernels - GPU kernels for beam evaluation.

Euler-Bernoulli cantilever with a transverse point load at the free tip and a
uniform hollow-rectangular section. One thread per design candidate, so a
whole population is evaluated in a single launch.

Written to be differentiable: every operation is one Warp records on its tape,
so d(metric)/d(b, h, t) comes straight from `wp.Tape().backward()`.

MODEL FIDELITY - read before trusting a number out of here:
  Euler-Bernoulli beam theory. It ignores the root stress concentration where
  the link meets its fixture, ignores transverse shear deformation (fine for
  slender beams, wrong for stubby ones), and does not check buckling. Peak
  real stress at the root will be HIGHER than sigma_max reported here. 3D FEM
  in a later phase is what resolves those.
"""

import warp as wp

# First eigenvalue of the clamped-free beam: beta_1 * L = 1.875104
BETA1 = 1.875104
BETA1_SQ = wp.constant(BETA1 * BETA1)
TWO_PI = wp.constant(6.283185307179586)


@wp.kernel
def cantilever_hollow_rect_metrics(
    # --- design variables, one entry per candidate ---
    outer_width: wp.array(dtype=wp.float32),      # b [m]
    outer_height: wp.array(dtype=wp.float32),     # h [m], vertical == load axis
    wall_thickness: wp.array(dtype=wp.float32),   # t [m]
    # --- load case, shared by the whole batch ---
    length: float,            # L     [m]
    tip_load: float,          # P     [N]
    youngs_modulus: float,    # E     [Pa]
    density: float,           # rho   [kg/m^3]
    yield_strength: float,    # sigma_y [Pa]
    # --- outputs ---
    mass: wp.array(dtype=wp.float32),               # [kg]
    max_bending_stress: wp.array(dtype=wp.float32),  # [Pa]
    tip_deflection: wp.array(dtype=wp.float32),      # [m]
    safety_factor: wp.array(dtype=wp.float32),       # [-]
    first_natural_frequency: wp.array(dtype=wp.float32),  # [Hz]
    mean_shear_stress: wp.array(dtype=wp.float32),   # [Pa]
):
    i = wp.tid()

    b = outer_width[i]
    h = outer_height[i]
    t = wall_thickness[i]

    # --- section geometry (same closed form as core.design_genome.section) ---
    b_i = b - 2.0 * t
    h_i = h - 2.0 * t

    area = b * h - b_i * h_i
    inertia = (b * h * h * h - b_i * h_i * h_i * h_i) / 12.0
    c = h * 0.5                       # extreme fibre distance

    # --- metrics ---
    m = area * length * density                       # mass
    moment = tip_load * length                        # root bending moment
    sigma = moment * c / inertia                      # max bending stress
    delta = tip_load * length * length * length / (3.0 * youngs_modulus * inertia)
    sf = yield_strength / sigma

    # f1 = (beta1^2 / 2pi) * sqrt(E*I / (rho*A*L^4))
    l2 = length * length
    l4 = l2 * l2
    f1 = (BETA1_SQ / TWO_PI) * wp.sqrt(
        youngs_modulus * inertia / (density * area * l4)
    )

    # Secondary check only. Bending governs a slender link; this average shear
    # is reported for awareness, not used as the design driver.
    tau = tip_load / area

    mass[i] = m
    max_bending_stress[i] = sigma
    tip_deflection[i] = delta
    safety_factor[i] = sf
    first_natural_frequency[i] = f1
    mean_shear_stress[i] = tau
