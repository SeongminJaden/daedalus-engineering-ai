"""physics.fem.solver - preconditioned conjugate gradient on the GPU.

Homogeneous Dirichlet constraints are handled by projection: constrained DOFs
are zeroed in the residual and search direction every iteration, so the solve
happens entirely in the free subspace without touching the operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .element import (
    element_stiffness_from_c, element_stress_operator_from_c, von_mises,
)
from .mesh import Mesh


@dataclass
class SolveReport:
    iterations: int
    residual: float
    converged: bool
    n_dofs: int

    def as_dict(self) -> dict:
        return {"iterations": self.iterations, "residual": self.residual,
                "converged": self.converged, "n_dofs": self.n_dofs}


@dataclass
class FemSolution:
    displacements: np.ndarray        # (n_nodes, 3)
    element_stress: np.ndarray       # (n_elements, 6) Voigt
    element_von_mises: np.ndarray    # (n_elements,)
    report: SolveReport
    mesh: Mesh
    # u_e^T Ke0 u_e per element, at the unscaled stiffness. The SIMP
    # sensitivity is built from this.
    element_strain_energy: np.ndarray | None = None
    load_vector: np.ndarray | None = None

    def tip_deflection(self, direction: int = 1) -> float:
        """Mean displacement of the tip face along `direction` (default y)."""
        tip = self.mesh.nodes_at_x(self.mesh.nx * self.mesh.dx)
        return float(np.mean(self.displacements[tip, direction]))

    def max_displacement_magnitude(self) -> float:
        return float(np.linalg.norm(self.displacements, axis=1).max())

    def compliance(self) -> float:
        """c = U^T K U = F^T U. Using F^T U avoids a second matvec."""
        if self.load_vector is None:
            raise ValueError("solution carries no load vector")
        return float(self.load_vector @ self.displacements.reshape(-1))


def _resolve_device(device: str | None) -> str:
    import warp as wp
    if device is not None:
        return device
    cuda = [d for d in wp.get_devices() if d.is_cuda]
    return str(cuda[0]) if cuda else "cpu"


def solve_linear_elasticity(
    mesh: Mesh,
    youngs_modulus: float,
    poisson_ratio: float,
    fixed_nodes: np.ndarray,
    load_nodes: np.ndarray,
    total_load_n: float,
    load_direction: int = 1,
    # Relative residual. 1e-8 is not a compromise: measured on the MVP section,
    # the tip deflection agreed to five significant figures between residual
    # 1.7e-7 and 9.8e-10, so tightening further buys precision the model does
    # not have while costing thousands of iterations.
    stiffness_matrix: np.ndarray | None = None,
    element_scale: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iterations: int | None = None,
    device: str | None = None,
) -> FemSolution:
    """Solve K u = f for a structured mesh with a fixed face and a loaded face.

    `total_load_n` is distributed equally over `load_nodes`.
    """
    import warp as wp

    dev = _resolve_device(device)
    n_dofs = mesh.n_dofs
    # CG on an elasticity operator needs O(sqrt(condition number)) steps, and
    # a slender beam is badly conditioned. A cap that is too low silently
    # returns an under-converged (too stiff) deflection that still looks
    # plausible, so it is set generously and convergence is reported.
    # High aspect ratio elements (a long thin-walled beam meshed to resolve
    # a 1 mm wall) are badly conditioned, and CG needs O(sqrt(cond))
    # steps. A cap that is too low returns an under-converged, and so
    # too stiff, deflection that still looks plausible. Convergence is
    # reported either way, but the cap is set generously enough that
    # realistic meshes reach tolerance.
    max_iterations = max_iterations or max(5000, 200 * int(np.sqrt(n_dofs)))

    if stiffness_matrix is None:
        from core.materials import isotropic_stiffness
        stiffness_matrix = isotropic_stiffness(youngs_modulus, poisson_ratio)
    ke = element_stiffness_from_c(mesh.dx, mesh.dy, mesh.dz, stiffness_matrix)
    db = element_stress_operator_from_c(mesh.dx, mesh.dy, mesh.dz,
                                        stiffness_matrix)

    # --- device arrays ---
    ke_d = wp.array(ke.reshape(-1), dtype=wp.float64, device=dev)
    db_d = wp.array(db.reshape(-1), dtype=wp.float64, device=dev)
    conn_d = wp.array(mesh.connectivity.astype(np.int32), dtype=wp.int32,
                      device=dev)

    # Per-element stiffness multiplier. Ones for a uniform part; the SIMP
    # density interpolation during topology optimization.
    if element_scale is None:
        scale_host = np.ones(mesh.n_elements, dtype=np.float64)
    else:
        scale_host = np.asarray(element_scale, dtype=np.float64).reshape(-1)
        if scale_host.shape[0] != mesh.n_elements:
            raise ValueError(
                f"element_scale has {scale_host.shape[0]} entries for "
                f"{mesh.n_elements} elements")
        if np.any(scale_host <= 0):
            raise ValueError("element_scale must be strictly positive; a zero "
                             "makes the stiffness matrix singular")
    scale_d = wp.array(scale_host, dtype=wp.float64, device=dev)

    fixed = np.zeros(n_dofs, dtype=np.int32)
    for node in np.asarray(fixed_nodes, dtype=np.int64):
        fixed[3 * node:3 * node + 3] = 1
    fixed_d = wp.array(fixed, dtype=wp.int32, device=dev)

    load_nodes = np.asarray(load_nodes, dtype=np.int64)
    if load_nodes.size == 0:
        raise ValueError("no loaded nodes")
    f = np.zeros(n_dofs, dtype=np.float64)
    f[3 * load_nodes + load_direction] = total_load_n / load_nodes.size
    f[fixed == 1] = 0.0

    from .kernels import (
        apply_dirichlet, axpy, dot_partial, elementwise_multiply,
        element_strain_energy, element_stress, reciprocal_safe,
        stiffness_diagonal, stiffness_matvec,
        xpay, zero,
    )

    def new_vec(init=None):
        if init is None:
            return wp.zeros(n_dofs, dtype=wp.float64, device=dev)
        return wp.array(init, dtype=wp.float64, device=dev)

    u = new_vec()
    r = new_vec(f)
    ap = new_vec()
    scratch = wp.zeros(1, dtype=wp.float64, device=dev)

    # Jacobi preconditioner
    diag = new_vec()
    wp.launch(stiffness_diagonal, dim=mesh.n_elements,
              inputs=[ke_d, conn_d, scale_d], outputs=[diag], device=dev)
    m_inv = new_vec()
    wp.launch(reciprocal_safe, dim=n_dofs, inputs=[diag], outputs=[m_inv],
              device=dev)
    wp.launch(apply_dirichlet, dim=n_dofs, inputs=[m_inv, fixed_d], device=dev)

    def dot(a, b) -> float:
        wp.launch(zero, dim=1, inputs=[scratch], device=dev)
        wp.launch(dot_partial, dim=n_dofs, inputs=[a, b], outputs=[scratch],
                  device=dev)
        return float(scratch.numpy()[0])

    def matvec(x, out):
        wp.launch(zero, dim=n_dofs, inputs=[out], device=dev)
        wp.launch(stiffness_matvec, dim=mesh.n_elements,
                  inputs=[x, ke_d, conn_d, scale_d], outputs=[out], device=dev)
        wp.launch(apply_dirichlet, dim=n_dofs, inputs=[out, fixed_d], device=dev)

    wp.launch(apply_dirichlet, dim=n_dofs, inputs=[r, fixed_d], device=dev)

    z = new_vec()
    wp.launch(elementwise_multiply, dim=n_dofs, inputs=[m_inv, r], outputs=[z],
              device=dev)
    p = new_vec(z.numpy())

    rz = dot(r, z)
    f_norm = float(np.linalg.norm(f))
    if f_norm == 0.0:
        raise ValueError("zero load vector")

    iterations, residual, converged = 0, float("inf"), False
    for iterations in range(1, max_iterations + 1):
        matvec(p, ap)
        pap = dot(p, ap)
        if pap <= 0.0:
            break                                # loss of positive definiteness
        alpha = rz / pap
        wp.launch(axpy, dim=n_dofs, inputs=[wp.float64(alpha), p], outputs=[u],
                  device=dev)
        wp.launch(axpy, dim=n_dofs, inputs=[wp.float64(-alpha), ap], outputs=[r],
                  device=dev)

        residual = float(np.linalg.norm(r.numpy())) / f_norm
        if residual < tol:
            converged = True
            break

        wp.launch(elementwise_multiply, dim=n_dofs, inputs=[m_inv, r],
                  outputs=[z], device=dev)
        rz_new = dot(r, z)
        wp.launch(xpay, dim=n_dofs, inputs=[wp.float64(rz_new / rz), z],
                  outputs=[p], device=dev)
        rz = rz_new

    stress_d = wp.zeros((mesh.n_elements, 6), dtype=wp.float64, device=dev)
    wp.launch(element_stress, dim=mesh.n_elements,
              inputs=[u, db_d, conn_d], outputs=[stress_d], device=dev)
    wp.synchronize_device(dev)

    energy_d = wp.zeros(mesh.n_elements, dtype=wp.float64, device=dev)
    wp.launch(element_strain_energy, dim=mesh.n_elements,
              inputs=[u, ke_d, conn_d], outputs=[energy_d], device=dev)
    wp.synchronize_device(dev)

    stress = stress_d.numpy()
    return FemSolution(
        displacements=u.numpy().reshape(-1, 3),
        element_stress=stress,
        element_von_mises=von_mises(stress),
        report=SolveReport(iterations=iterations, residual=residual,
                           converged=converged, n_dofs=n_dofs),
        mesh=mesh,
        element_strain_energy=energy_d.numpy(),
        load_vector=f,
    )
