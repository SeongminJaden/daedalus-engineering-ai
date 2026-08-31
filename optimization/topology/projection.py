"""Three-field SIMP: design variable, filtered density, projected density.

SIMP on its own leaves grey. Phase 13 reported a grey fraction around 0.59 on
the cantilever and Phase 14a around 0.80 on the stress-constrained bracket, and
an element at density 0.4 is not a material anyone can make. Thresholding the
result afterwards gives a shape that was never the one the optimizer evaluated.

The three-field formulation makes the optimizer itself produce a black and
white design. The design variable is filtered, the filtered field is pushed
towards 0 or 1 by a smoothed Heaviside, and the projected field is what the
physics sees:

    x  --filter-->  x_tilde  --project-->  x_bar  -->  stiffness

`beta` sets how sharp the projection is. It has to start low and rise: a sharp
projection from the beginning has almost zero gradient nearly everywhere, and
the optimizer cannot move. That is what continuation is for.

**The sensitivity now travels through the projection**, so the adjoint
expressions verified in Phases 13 and 14a give the derivative with respect to
`x_bar`, not with respect to the design variable. The chain rule back to the
design variable is `H^T (dx_bar/dx_tilde * dc/dx_bar)`, and it is finite
difference checked exactly like the underlying adjoints were, because an
un-checked chain rule would silently optimize the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from physics.fem.mesh import Mesh

from .simp import MIN_DENSITY, build_filter_weights

# The projection threshold. 0.5 keeps the filtered volume roughly unchanged by
# the projection, which matters because the volume constraint is enforced on
# the projected field.
DEFAULT_ETA = 0.5

# Below this, tanh(beta*eta) + tanh(beta*(1-eta)) underflows towards zero and
# the projection formula divides by nothing. The projection is essentially the
# identity there anyway, so it is returned directly rather than computed.
MIN_BETA = 1e-6


def build_density_filter(mesh: Mesh, radius_elements: float) -> sp.csr_matrix:
    """Row-normalised neighbour weights as a sparse matrix.

    A matrix rather than the per-element loop the sensitivity filter uses,
    because the chain rule needs the exact transpose. Writing the transpose out
    by hand as a second loop is how the forward and backward passes drift apart.
    """
    rows, weights = build_filter_weights(mesh, radius_elements)
    n = mesh.n_elements
    row_index, col_index, values = [], [], []
    for e, (neighbours, w) in enumerate(zip(rows, weights)):
        total = float(np.sum(w))
        if total <= 0:
            row_index.append(e); col_index.append(e); values.append(1.0)
            continue
        for target, weight in zip(neighbours, w):
            row_index.append(e)
            col_index.append(int(target))
            values.append(float(weight) / total)
    return sp.csr_matrix((values, (row_index, col_index)), shape=(n, n))


def project(filtered: np.ndarray, beta: float,
            eta: float = DEFAULT_ETA) -> np.ndarray:
    """Smoothed Heaviside, the standard tanh form.

    Maps [0, 1] onto [0, 1] with a step at `eta` whose sharpness is `beta`.
    """
    if beta < MIN_BETA:
        return np.asarray(filtered, dtype=float).copy()
    numerator = np.tanh(beta * eta) + np.tanh(beta * (filtered - eta))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return numerator / denominator


def projection_derivative(filtered: np.ndarray, beta: float,
                          eta: float = DEFAULT_ETA) -> np.ndarray:
    """d(x_bar)/d(x_tilde), elementwise."""
    if beta < MIN_BETA:
        return np.ones_like(np.asarray(filtered, dtype=float))
    denominator = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    sech2 = 1.0 - np.tanh(beta * (filtered - eta)) ** 2
    return beta * sech2 / denominator


@dataclass
class DesignTransform:
    """The map from design variable to the density the physics sees.

    `beta` is mutable so a continuation schedule can sharpen it in place
    between iterations without rebuilding the filter matrix.
    """

    filter_matrix: sp.csr_matrix
    beta: float = 1.0
    eta: float = DEFAULT_ETA

    @classmethod
    def for_mesh(cls, mesh: Mesh, radius_elements: float, beta: float = 1.0,
                 eta: float = DEFAULT_ETA) -> "DesignTransform":
        return cls(filter_matrix=build_density_filter(mesh, radius_elements),
                   beta=beta, eta=eta)

    def filtered(self, design: np.ndarray) -> np.ndarray:
        return self.filter_matrix @ np.asarray(design, dtype=float)

    def physical(self, design: np.ndarray) -> np.ndarray:
        """The density the solver uses, clipped away from exact zero.

        The floor is the same one SIMP already relies on: a truly zero density
        makes the stiffness matrix singular, so the solve fails rather than
        producing a soft region.
        """
        projected = project(self.filtered(design), self.beta, self.eta)
        return np.clip(projected, MIN_DENSITY, 1.0)

    def chain(self, gradient_physical: np.ndarray,
              design: np.ndarray) -> np.ndarray:
        """Pull a gradient with respect to x_bar back to the design variable.

        `H^T (dx_bar/dx_tilde * dc/dx_bar)`. The clip in `physical` is not
        differentiated: it is inactive wherever the projection lands inside the
        bounds, and treating it as active would zero out gradients along the
        whole void region, which is exactly where material has to be able to
        come back.
        """
        filtered = self.filtered(design)
        slope = projection_derivative(filtered, self.beta, self.eta)
        return self.filter_matrix.T @ (slope * np.asarray(gradient_physical,
                                                          dtype=float))


@dataclass
class BetaSchedule:
    """Continuation on the projection sharpness.

    `beta` doubles every `every` iterations up to `maximum`. Starting sharp
    stalls the optimizer, because a steep Heaviside has near-zero derivative
    almost everywhere and there is no gradient to descend.
    """

    start: float = 1.0
    maximum: float = 16.0
    every: int = 15
    factor: float = 2.0
    history: list[float] = field(default_factory=list)

    def beta_at(self, iteration: int) -> float:
        steps = max(0, (iteration - 1) // self.every)
        return float(min(self.maximum, self.start * self.factor ** steps))

    def apply(self, transform: DesignTransform, iteration: int) -> float:
        transform.beta = self.beta_at(iteration)
        self.history.append(transform.beta)
        return transform.beta
