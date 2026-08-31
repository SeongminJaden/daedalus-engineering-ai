"""optimization.topology.stress: stress-constrained topology optimization.

Phase 13 minimised compliance and said nothing about stress. This minimises
**volume subject to a stress limit**, which is a harder problem for two reasons
that both have to be handled explicitly.

THE SINGULARITY PHENOMENON. The stress *in the material* of an element does not
vanish as its density does: a nearly-void element still reports a finite
microscopic stress, so its constraint stays active and the optimizer can never
remove material from a stressed region. The fix used here is **qp-relaxation**:
the stress measure is scaled by x^q with q < p, so it goes to zero with density
and the constraint releases. p = 3 for stiffness, q = 2.5 for stress.

    sigma_e = x_e^q * vonMises(C0 B u_e)

THE AGGREGATION. The stress limit is local, one constraint per element, which is
thousands of constraints. They are aggregated into a single P-norm:

    sigma_PN = ( sum_e (sigma_e / sigma_lim)^P )^(1/P)

A P-norm is a **lower bound on the true maximum** and approaches it from below as
P grows. So satisfying the aggregated constraint does not guarantee every element
is under the limit, and the gap is reported rather than hidden.

NOT SELF-ADJOINT. Unlike compliance, this needs a second (adjoint) solve:

    K lambda = dg/du
    dg/dx_j = dg/dx_j|explicit - lambda_j^T (dK/dx_j) u
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.materials import isotropic_stiffness
from physics.fem.element import element_stress_operator_from_c
from physics.fem.mesh import Mesh
from physics.fem.solver import solve_linear_elasticity

from .simp import (
    MIN_DENSITY, PENALTY, SimpProblem, VOID_STIFFNESS_RATIO,
    apply_sensitivity_filter, build_filter_weights, stiffness_scale,
    stiffness_scale_derivative,
)

# Stress relaxation exponent. Must be strictly below the stiffness penalty for
# the relaxation to remove the singular optima.
STRESS_PENALTY = 2.5

# von Mises quadratic form in Voigt order [xx, yy, zz, xy, yz, zx]:
#   vm^2 = s^T V s
VON_MISES_MATRIX = np.array([
    [1.0, -0.5, -0.5, 0.0, 0.0, 0.0],
    [-0.5, 1.0, -0.5, 0.0, 0.0, 0.0],
    [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
], dtype=np.float64)


@dataclass
class StressProblem:
    """A SIMP domain plus a stress limit."""

    base: SimpProblem
    stress_limit_pa: float
    p_norm: float = 8.0
    stress_penalty: float = STRESS_PENALTY
    # Residual for every solve, the adjoint included. The default matches the
    # solver's. The finite-difference check of the adjoint needs it tighter:
    # at 1e-8 the near-zero sensitivity entries showed up to 60% relative
    # error, which was the difference quotient drowning in solver noise, not a
    # wrong gradient. At 1e-12 the worst scaled error is 3.9e-08.
    solver_tolerance: float = 1e-8

    @property
    def mesh(self) -> Mesh:
        return self.base.mesh

    def n_elements(self) -> int:
        return self.base.n_elements()


def element_dofs(mesh: Mesh) -> np.ndarray:
    """(n_elements, 24) global DOF indices."""
    return (mesh.connectivity[:, :, None] * 3
            + np.arange(3)[None, None, :]).reshape(mesh.n_elements, 24)


def solid_stress_operator(problem: StressProblem) -> np.ndarray:
    """C0 @ B at the element centre, for the SOLID material.

    The stress measure uses the solid constitutive law and carries the density
    dependence in the explicit x^q factor, which is what qp-relaxation means.
    """
    mesh = problem.mesh
    c0 = isotropic_stiffness(problem.base.youngs_modulus_pa,
                             problem.base.poisson_ratio)
    return element_stress_operator_from_c(mesh.dx, mesh.dy, mesh.dz, c0)


def relaxed_stress(problem: StressProblem, density: np.ndarray,
                   displacements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relaxed von Mises per element, and the unrelaxed microscopic value."""
    mesh = problem.mesh
    db = solid_stress_operator(problem)
    dofs = element_dofs(mesh)
    u = displacements.reshape(-1)

    stress = (db @ u[dofs].T).T                       # (n_elements, 6)
    micro = np.sqrt(np.einsum("ij,jk,ik->i", stress, VON_MISES_MATRIX, stress))
    x = np.clip(density, MIN_DENSITY, 1.0)
    return x ** problem.stress_penalty * micro, micro


def p_norm_stress(problem: StressProblem, relaxed: np.ndarray) -> float:
    ratio = relaxed / problem.stress_limit_pa
    return float(np.sum(ratio ** problem.p_norm) ** (1.0 / problem.p_norm))


@dataclass
class StressEvaluation:
    volume_fraction: float
    p_norm: float
    max_relaxed_stress_pa: float
    max_micro_stress_pa: float
    relaxed_stress_pa: np.ndarray
    displacements: np.ndarray
    converged: bool

    @property
    def aggregation_gap(self) -> float:
        """Ratio of the aggregated stress to the true elementwise maximum.

        `sigma_PN * sigma_limit` against `max(sigma_e)`. Above 1 means the
        aggregate is conservative, and for the plain-sum P-norm used here it is
        always above 1: the sum over every element exceeds its largest term.
        Measured on the L-bracket it falls from 3.92 at P=2 to 1.009 at P=32,
        approaching the true maximum from above.

        That direction is worth stating because the more familiar averaged form
        of the P-norm under-estimates the maximum, which is the unsafe
        direction. This one over-estimates, so a design that satisfies the
        aggregate satisfies the real limit. It is still reported rather than
        assumed, since the gap is what a low P costs in conservatism.
        """
        true_max_ratio = self.max_relaxed_stress_pa
        return self.p_norm_scaled / true_max_ratio if true_max_ratio > 0 else 1.0

    p_norm_scaled: float = 0.0


def evaluate(problem: StressProblem, density: np.ndarray,
             device: str | None = None) -> StressEvaluation:
    """Solve, then compute the stress measures."""
    base = problem.base
    solution = solve_linear_elasticity(
        base.mesh, base.youngs_modulus_pa, base.poisson_ratio,
        fixed_nodes=base.fixed_nodes, load_nodes=base.load_nodes,
        total_load_n=base.total_load_n, load_direction=base.load_direction,
        element_scale=stiffness_scale(density, base.penalty), device=device,
        tol=problem.solver_tolerance)

    relaxed, micro = relaxed_stress(problem, density, solution.displacements)
    pn = p_norm_stress(problem, relaxed)
    return StressEvaluation(
        volume_fraction=float(np.mean(density)),
        p_norm=pn,
        max_relaxed_stress_pa=float(relaxed.max()),
        max_micro_stress_pa=float(micro.max()),
        relaxed_stress_pa=relaxed,
        displacements=solution.displacements,
        converged=solution.report.converged,
        p_norm_scaled=pn * problem.stress_limit_pa,
    )


def p_norm_sensitivity(problem: StressProblem, density: np.ndarray,
                       device: str | None = None):
    """d(sigma_PN)/dx by the adjoint method.

    Three pieces, and the middle one is why a second solve is needed:

      explicit   dsigma_PN/dx_e through the x^q relaxation factor
      implicit   through u, which depends on x via K(x) u = f
      adjoint    K lambda = dsigma_PN/du solves the implicit part in one go
    """
    base = problem.base
    mesh = base.mesh
    n = mesh.n_elements

    solution = solve_linear_elasticity(
        mesh, base.youngs_modulus_pa, base.poisson_ratio,
        fixed_nodes=base.fixed_nodes, load_nodes=base.load_nodes,
        total_load_n=base.total_load_n, load_direction=base.load_direction,
        element_scale=stiffness_scale(density, base.penalty), device=device,
        tol=problem.solver_tolerance)
    u = solution.displacements.reshape(-1)

    db = solid_stress_operator(problem)
    dofs = element_dofs(mesh)
    stress = (db @ u[dofs].T).T
    micro_sq = np.einsum("ij,jk,ik->i", stress, VON_MISES_MATRIX, stress)
    micro = np.sqrt(np.maximum(micro_sq, 1e-300))

    x = np.clip(density, MIN_DENSITY, 1.0)
    q = problem.stress_penalty
    relaxed = x ** q * micro

    limit = problem.stress_limit_pa
    ratio = relaxed / limit
    total = float(np.sum(ratio ** problem.p_norm))
    pn = total ** (1.0 / problem.p_norm)

    # d(sigma_PN)/d(sigma_e)
    d_pn_d_sigma = (total ** (1.0 / problem.p_norm - 1.0)
                    * ratio ** (problem.p_norm - 1.0) / limit)

    # --- explicit part: through the x^q factor ---
    explicit = d_pn_d_sigma * q * x ** (q - 1.0) * micro

    # --- adjoint right-hand side: d(sigma_PN)/du ---
    # d(micro)/du_e = (1/micro) * (V s)^T (C0 B)
    weights = (d_pn_d_sigma * x ** q / micro)[:, None]      # (n, 1)
    vs = stress @ VON_MISES_MATRIX                           # (n, 6)
    element_rhs = weights * (vs @ db)                        # (n, 24)

    rhs = np.zeros(mesh.n_dofs, dtype=np.float64)
    np.add.at(rhs, dofs.reshape(-1), element_rhs.reshape(-1))

    adjoint = solve_linear_elasticity(
        mesh, base.youngs_modulus_pa, base.poisson_ratio,
        fixed_nodes=base.fixed_nodes, force_vector=rhs,
        element_scale=stiffness_scale(density, base.penalty), device=device,
        tol=problem.solver_tolerance)
    lam = adjoint.displacements.reshape(-1)

    # --- implicit part: -lambda_e^T (dK/dx_e) u_e ---
    from physics.fem.element import element_stiffness_from_c
    c = isotropic_stiffness(base.youngs_modulus_pa, base.poisson_ratio)
    ke0 = element_stiffness_from_c(mesh.dx, mesh.dy, mesh.dz, c)
    dscale = stiffness_scale_derivative(density, base.penalty)

    ue = u[dofs]                                             # (n, 24)
    le = lam[dofs]
    bilinear = np.einsum("ij,jk,ik->i", le, ke0, ue)
    implicit = -dscale * bilinear

    return pn, explicit + implicit, solution, adjoint


@dataclass
class StressResult:
    """The outcome of a stress-minimising run at fixed volume.

    Every history list is indexed by iterate and the final entry describes
    `density`, so the reported stress belongs to the reported design.
    """

    density: np.ndarray
    volume_history: list[float] = field(default_factory=list)
    p_norm_history: list[float] = field(default_factory=list)
    max_stress_history: list[float] = field(default_factory=list)
    iterations: int = 0

    @property
    def feasible(self) -> bool:
        """Whether the final design is under the stress limit."""
        return bool(self.p_norm_history) and self.p_norm_history[-1] <= 1.0

    @property
    def final_p_norm(self) -> float:
        return self.p_norm_history[-1]

    @property
    def final_max_stress_pa(self) -> float:
        return self.max_stress_history[-1]

    @property
    def volume_fraction(self) -> float:
        return self.volume_history[-1]


def _project_to_volume(density: np.ndarray, direction: np.ndarray,
                       move_limit: float, target_volume: float) -> np.ndarray:
    """Take a move-limited step along `direction`, projected onto the volume.

    `x(lam) = clip(x - move * (direction + lam))` is monotone decreasing in
    `lam`, so a bisection finds the multiplier that lands the mean exactly on
    the target. This is the same Lagrange-multiplier bisection the compliance
    OC update uses, but written as a projected gradient step so it does not
    require the sensitivity to be negative everywhere. The stress sensitivity
    is not: removing material can lower the stress in one place and raise it in
    another, and the OC exponent form has no meaning for a positive entry.
    """
    lo, hi = -10.0, 10.0
    lower = np.maximum(density - 2.0 * move_limit, MIN_DENSITY)
    upper = np.minimum(density + 2.0 * move_limit, 1.0)

    def at(lam: float) -> np.ndarray:
        return np.clip(density - move_limit * (direction + lam), lower, upper)

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if at(mid).mean() > target_volume:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return at(0.5 * (lo + hi))


def optimize_stress(problem: StressProblem, max_iterations: int = 60,
                    move_limit: float = 0.05, device: str | None = None,
                    use_filter: bool = True, callback=None) -> StressResult:
    """Minimise the aggregated stress at a fixed volume fraction.

    The volume comes from `problem.base.volume_fraction`, so this is the same
    design space the compliance minimiser works in and the two are directly
    comparable: same mesh, same load, same amount of material, different
    objective. That comparison is the point of the method, and it is only a
    comparison if the volume is held equal.

    **The other formulation was tried first and abandoned.** Minimising volume
    subject to a penalised stress constraint degenerates here. The volume
    gradient is the constant 1/n, so a normalised step moves every element by
    the same amount and the density field shrinks uniformly without ever
    developing a shape: a 50-iteration run returned a field that was exactly
    0.0500 in every element, which is not a structure. Recovering a design from
    it would have needed a genuine constrained step (MMA or similar), so the
    formulation here is the one that is well posed with the machinery present.

    The sensitivity driving this is the adjoint from `p_norm_sensitivity`,
    checked against finite differences to a worst scaled error of 3.9e-08.
    """
    mesh = problem.mesh
    n = mesh.n_elements
    target = problem.base.volume_fraction
    density = np.full(n, target)
    rows = weights = None
    if use_filter:
        rows, weights = build_filter_weights(mesh,
                                             problem.base.filter_radius_elements)

    result = StressResult(density=density)
    for iteration in range(1, max_iterations + 1):
        pn, d_pn, solution, _ = p_norm_sensitivity(problem, density, device)
        relaxed, _ = relaxed_stress(problem, density, solution.displacements)

        # Record before stepping so the reported numbers belong to the density
        # they were computed from.
        result.volume_history.append(float(density.mean()))
        result.p_norm_history.append(pn)
        result.max_stress_history.append(float(relaxed.max()))
        result.iterations = iteration
        if callback is not None:
            callback(iteration, float(density.mean()), pn, float(relaxed.max()))

        gradient = d_pn
        if use_filter:
            gradient = apply_sensitivity_filter(gradient, density, rows, weights,
                                                MIN_DENSITY)
        direction = gradient / max(np.abs(gradient).max(), 1e-30)
        density = _project_to_volume(density, direction, move_limit, target)

    # The loop steps after evaluating, so the last density has not been checked.
    pn, _, solution, _ = p_norm_sensitivity(problem, density, device)
    relaxed, _ = relaxed_stress(problem, density, solution.displacements)
    result.volume_history.append(float(density.mean()))
    result.p_norm_history.append(pn)
    result.max_stress_history.append(float(relaxed.max()))
    result.density = density
    return result


@dataclass
class ConstrainedResult:
    """A compliance minimisation carrying an aggregated stress constraint."""

    density: np.ndarray
    best_feasible_density: np.ndarray | None = None
    best_feasible_compliance: float | None = None
    best_feasible_p_norm: float | None = None
    best_feasible_max_stress_pa: float | None = None
    compliance_history: list[float] = field(default_factory=list)
    p_norm_history: list[float] = field(default_factory=list)
    max_stress_history: list[float] = field(default_factory=list)
    penalty_history: list[float] = field(default_factory=list)
    iterations: int = 0

    @property
    def feasible(self) -> bool:
        """Whether the FINAL iterate satisfies the constraint.

        A penalty method oscillates about the constraint boundary, so the last
        iterate can land marginally outside it even when the run went well: a
        17 MPa case finished at p-norm 1.0036. Use `best_feasible_density` for
        the design to build, and this only to see where the run stopped.
        """
        return bool(self.p_norm_history) and self.p_norm_history[-1] <= 1.0

    @property
    def found_feasible(self) -> bool:
        return self.best_feasible_density is not None

    @property
    def final_compliance(self) -> float:
        return self.compliance_history[-1]

    @property
    def final_p_norm(self) -> float:
        return self.p_norm_history[-1]

    @property
    def final_max_stress_pa(self) -> float:
        return self.max_stress_history[-1]


def _record_constrained(result: "ConstrainedResult", density: np.ndarray,
                        compliance: float, pn: float, max_stress: float,
                        weight: float) -> None:
    """Append one iterate and keep the best feasible design seen so far."""
    result.compliance_history.append(compliance)
    result.p_norm_history.append(pn)
    result.max_stress_history.append(max_stress)
    result.penalty_history.append(weight)
    if pn <= 1.0 and (result.best_feasible_compliance is None
                      or compliance < result.best_feasible_compliance):
        result.best_feasible_density = density.copy()
        result.best_feasible_compliance = compliance
        result.best_feasible_p_norm = pn
        result.best_feasible_max_stress_pa = max_stress


def optimize_constrained(problem: StressProblem, max_iterations: int = 80,
                         move_limit: float = 0.05, penalty_weight: float = 1.0,
                         penalty_growth: float = 1.15, device: str | None = None,
                         use_filter: bool = True,
                         callback=None) -> ConstrainedResult:
    """Minimise compliance at fixed volume subject to the stress constraint.

    This is the formulation that produces a topology. Compliance minimisation
    drives the design towards solid and void, which pure stress minimisation
    does not: minimising the P-norm alone returns a graded field with no solid
    and no void, because the `q < p` relaxation makes intermediate density
    stress-efficient. Adding the stress term as a constraint on a compliance
    objective keeps the black-and-white behaviour and changes the design only
    where the stress limit actually binds, which is the re-entrant corner.

    Both gradients are normalised by their own largest entry before being
    combined, so `penalty_weight` means the same thing across problems rather
    than depending on the units of the objective. The weight grows while the
    constraint is violated, because a fixed exterior penalty leaves the
    optimum outside the feasible set.

    **The penalty schedule trades design quality against how fast feasibility
    arrives, and the default chooses quality.** A hard schedule reaches the
    feasible set in fewer iterations by steering almost entirely on the stress
    gradient, and a design built that way is grey: on the 20x20x2 bracket at a
    16 MPa limit, starting from weight 10 with growth 1.3 costs 212% compliance
    and leaves a 99.6% grey design, while the gentle default costs 46% and
    leaves 79.7% grey. Callers that only need a feasible point quickly, tests
    among them, should pass the harder schedule explicitly.
    """
    from .simp import compliance_and_sensitivity

    base = problem.base
    mesh = problem.mesh
    target = base.volume_fraction
    density = np.full(mesh.n_elements, target)
    rows = weights = None
    if use_filter:
        rows, weights = build_filter_weights(mesh, base.filter_radius_elements)

    result = ConstrainedResult(density=density)
    weight = float(penalty_weight)

    for iteration in range(1, max_iterations + 1):
        compliance, d_compliance, _ = compliance_and_sensitivity(base, density,
                                                                 device=device)
        pn, d_pn, solution, _ = p_norm_sensitivity(problem, density, device)
        relaxed, _ = relaxed_stress(problem, density, solution.displacements)

        _record_constrained(result, density, float(compliance), pn,
                            float(relaxed.max()), weight)
        result.iterations = iteration
        if callback is not None:
            callback(iteration, float(compliance), pn, float(relaxed.max()))

        violation = max(0.0, pn - 1.0)
        dc = d_compliance / max(np.abs(d_compliance).max(), 1e-30)
        ds = d_pn / max(np.abs(d_pn).max(), 1e-30)
        gradient = dc + weight * 2.0 * violation * ds
        if use_filter:
            gradient = apply_sensitivity_filter(gradient, density, rows, weights,
                                                MIN_DENSITY)
        direction = gradient / max(np.abs(gradient).max(), 1e-30)
        density = _project_to_volume(density, direction, move_limit, target)
        if violation > 0:
            weight *= penalty_growth

    compliance, _, _ = compliance_and_sensitivity(base, density, device=device)
    pn, _, solution, _ = p_norm_sensitivity(problem, density, device)
    relaxed, _ = relaxed_stress(problem, density, solution.displacements)
    _record_constrained(result, density, float(compliance), pn,
                        float(relaxed.max()), weight)
    result.density = density
    return result
