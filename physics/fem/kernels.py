"""physics.fem.kernels - Warp GPU kernels for the matrix-free FEM solve.

No global stiffness matrix is ever assembled. On a structured grid every
element is identical, so a single 24x24 Ke is uploaded once and the
matrix-vector product is an element-by-element gather / multiply /
scatter-add. That is what fits a 3D solve into a 4 GB card: memory scales with
the number of DOFs, not with the number of matrix non-zeros.

float64 throughout. The condition number of a slender-beam stiffness matrix is
large, and CG in float32 stalls on it well before the residual is small enough
to trust the deflection.
"""

import warp as wp

ELEM_DOFS = 24


@wp.kernel
def stiffness_matvec(
    u: wp.array(dtype=wp.float64),
    ke: wp.array(dtype=wp.float64),          # flattened 24x24
    conn: wp.array2d(dtype=wp.int32),        # (n_elements, 8)
    scale: wp.array(dtype=wp.float64),       # per-element stiffness multiplier
    y: wp.array(dtype=wp.float64),
):
    """y += K @ u, accumulated element by element.

    `scale` multiplies each element's contribution. It is 1 for a uniform part
    and carries the SIMP density interpolation E(x)/E0 during topology
    optimization: stiffness is linear in E, so one shared Ke still serves every
    element and the matrix-free structure is unchanged.
    """
    e = wp.tid()
    s = scale[e]
    for a in range(8):
        na = conn[e, a]
        for di in range(3):
            i = a * 3 + di
            acc = wp.float64(0.0)
            for b in range(8):
                nb = conn[e, b]
                for dj in range(3):
                    acc += ke[i * ELEM_DOFS + b * 3 + dj] * u[nb * 3 + dj]
            wp.atomic_add(y, na * 3 + di, s * acc)


@wp.kernel
def stiffness_diagonal(
    ke: wp.array(dtype=wp.float64),
    conn: wp.array2d(dtype=wp.int32),
    scale: wp.array(dtype=wp.float64),
    diag: wp.array(dtype=wp.float64),
):
    """diag(K), for Jacobi preconditioning."""
    e = wp.tid()
    s = scale[e]
    for a in range(8):
        na = conn[e, a]
        for di in range(3):
            i = a * 3 + di
            wp.atomic_add(diag, na * 3 + di, s * ke[i * ELEM_DOFS + i])


@wp.kernel
def apply_dirichlet(
    v: wp.array(dtype=wp.float64),
    fixed: wp.array(dtype=wp.int32),         # 1 where the DOF is constrained
):
    """Zero the constrained DOFs. Homogeneous constraints only."""
    i = wp.tid()
    if fixed[i] == 1:
        v[i] = wp.float64(0.0)


@wp.kernel
def axpy(a: wp.float64, x: wp.array(dtype=wp.float64),
         y: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = y[i] + a * x[i]


@wp.kernel
def xpay(a: wp.float64, x: wp.array(dtype=wp.float64),
         y: wp.array(dtype=wp.float64)):
    """y = x + a*y"""
    i = wp.tid()
    y[i] = x[i] + a * y[i]


@wp.kernel
def elementwise_multiply(x: wp.array(dtype=wp.float64),
                         y: wp.array(dtype=wp.float64),
                         out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = x[i] * y[i]


@wp.kernel
def reciprocal_safe(x: wp.array(dtype=wp.float64),
                    out: wp.array(dtype=wp.float64)):
    """1/x, leaving zeros as zero so constrained DOFs stay inert."""
    i = wp.tid()
    if x[i] > wp.float64(0.0):
        out[i] = wp.float64(1.0) / x[i]
    else:
        out[i] = wp.float64(0.0)


@wp.kernel
def dot_partial(x: wp.array(dtype=wp.float64),
                y: wp.array(dtype=wp.float64),
                out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    wp.atomic_add(out, 0, x[i] * y[i])


@wp.kernel
def zero(v: wp.array(dtype=wp.float64)):
    v[wp.tid()] = wp.float64(0.0)


@wp.kernel
def element_stress(
    u: wp.array(dtype=wp.float64),
    db: wp.array(dtype=wp.float64),          # flattened 6x24 (D @ B at centre)
    conn: wp.array2d(dtype=wp.int32),
    stress: wp.array2d(dtype=wp.float64),    # (n_elements, 6) Voigt
):
    """Stress at each element centre - the superconvergent point for this
    element, and one value per element rather than per Gauss point."""
    e = wp.tid()
    for c in range(6):
        acc = wp.float64(0.0)
        for b in range(8):
            nb = conn[e, b]
            for dj in range(3):
                acc += db[c * ELEM_DOFS + b * 3 + dj] * u[nb * 3 + dj]
        stress[e, c] = acc


@wp.kernel
def element_strain_energy(
    u: wp.array(dtype=wp.float64),
    ke: wp.array(dtype=wp.float64),
    conn: wp.array2d(dtype=wp.int32),
    energy: wp.array(dtype=wp.float64),
):
    """u_e^T Ke0 u_e for each element, at the UNSCALED base stiffness.

    This is the quantity the SIMP sensitivity needs. Keeping it unscaled means
    the density interpolation and its derivative live in one place in Python
    rather than being baked into the kernel.
    """
    e = wp.tid()
    total = wp.float64(0.0)
    for a in range(8):
        na = conn[e, a]
        for di in range(3):
            i = a * 3 + di
            row = wp.float64(0.0)
            for b in range(8):
                nb = conn[e, b]
                for dj in range(3):
                    row += ke[i * ELEM_DOFS + b * 3 + dj] * u[nb * 3 + dj]
            total += u[na * 3 + di] * row
    energy[e] = total
