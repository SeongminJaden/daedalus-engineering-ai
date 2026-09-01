"""Verification of the general isoparametric hex.

Every check here has an answer known independently of this code: it either
compares against the already verified structured element, or against a closed
form, or asserts a refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.fem import element as structured
from physics.fem import isoparametric as iso

YOUNGS = 210e9
POISSON = 0.3


def stiffness():
    return structured.elasticity_matrix(YOUNGS, POISSON)


def distorted_nodes(amount: float = 0.18, seed: int = 7):
    rng = np.random.default_rng(seed)
    return iso.box_nodes(1.0, 1.0, 1.0) + rng.uniform(-amount, amount, (8, 3))


def element_volume(nodes):
    return sum(w * iso.jacobian(nodes, x, y, z)[1]
               for x, y, z, w in structured.gauss_points())


# ----------------------------------------------------- reduces to the special

@pytest.mark.parametrize("incompatible", [False, True])
def test_a_box_gives_exactly_the_structured_stiffness(incompatible):
    """The strongest check available: on a box the two paths must agree.

    The structured element is already verified, so any error in the general
    element's Jacobian, node ordering or quadrature shows up here at once.
    """
    dx, dy, dz = 0.7, 1.3, 2.1
    expected = structured.element_stiffness_from_c(
        dx, dy, dz, stiffness(), incompatible_modes=incompatible)
    got = iso.element_stiffness_from_c(
        iso.box_nodes(dx, dy, dz), stiffness(), incompatible_modes=incompatible)
    assert np.abs(got - expected).max() / np.abs(expected).max() < 1e-14


def test_the_jacobian_of_a_box_is_the_structured_one():
    """det J = dx dy dz / 8, constant, which is the assumption element.py makes."""
    dx, dy, dz = 0.7, 1.3, 2.1
    nodes = iso.box_nodes(dx, dy, dz)
    for xi, eta, zeta, _ in structured.gauss_points():
        _, det = iso.jacobian(nodes, xi, eta, zeta)
        assert det == pytest.approx(dx * dy * dz / 8.0, rel=1e-14)


# ------------------------------------------------------------- rigid body

def test_a_distorted_element_has_exactly_six_zero_eigenvalues():
    """Six rigid body modes: three translations and three rotations.

    Fewer means the element is spuriously constrained and will read as too
    stiff. More means a zero energy hourglass mode, which pollutes a solve
    with displacement that costs nothing.
    """
    ke = iso.element_stiffness_from_c(distorted_nodes(), stiffness())
    eigenvalues = np.sort(np.linalg.eigvalsh(ke))
    scale = np.abs(eigenvalues).max()
    assert np.sum(np.abs(eigenvalues) < 1e-9 * scale) == 6


def test_rigid_body_motion_produces_no_internal_force():
    nodes = distorted_nodes()
    ke = iso.element_stiffness_from_c(nodes, stiffness())
    translation = np.array([0.31, -0.2, 0.5])
    rotation = np.array([0.02, -0.03, 0.05])
    u = np.concatenate([translation + np.cross(rotation, p) for p in nodes])
    assert np.abs(ke @ u).max() / (np.abs(ke).max() * np.abs(u).max()) < 1e-14


# ------------------------------------------------------------- patch test

def patch_energy_relative_error(nodes, ke):
    """Twice the strain energy under a linear displacement field.

    For a constant strain state the answer is V eps^T C eps exactly, whatever
    the element's shape. An element that cannot reproduce constant strain does
    not converge to the right answer under refinement, it converges to a
    different one.
    """
    gradient = np.array([[3e-4, 1e-4, -2e-4],
                         [1e-4, -5e-4, 4e-4],
                         [-2e-4, 4e-4, 2e-4]])
    u = np.concatenate([gradient @ p for p in nodes])
    strain = np.array([gradient[0, 0], gradient[1, 1], gradient[2, 2],
                       gradient[0, 1] + gradient[1, 0],
                       gradient[1, 2] + gradient[2, 1],
                       gradient[0, 2] + gradient[2, 0]])
    exact = element_volume(nodes) * float(strain @ stiffness() @ strain)
    return abs(float(u @ ke @ u) - exact) / exact


@pytest.mark.parametrize("incompatible", [False, True])
def test_the_patch_test_passes_on_a_distorted_element(incompatible):
    nodes = distorted_nodes()
    ke = iso.element_stiffness_from_c(nodes, stiffness(),
                                      incompatible_modes=incompatible)
    assert patch_energy_relative_error(nodes, ke) < 1e-12


def test_without_the_taylor_correction_the_patch_test_fails():
    """The correction is load bearing, and this measures by how much.

    Wilson's bubbles are derived for a rectangle. Taking their gradients from
    the LOCAL Jacobian instead of the element centre, and dropping the det
    ratio, is the natural looking implementation, and it is wrong. Measured
    here at roughly 8 parts in a thousand on a mildly distorted element, which
    is large enough to matter and small enough to be mistaken for
    discretisation error.
    """
    nodes = distorted_nodes()
    c = stiffness()
    ke = np.zeros((24, 24))
    k_ua = np.zeros((24, 9))
    k_aa = np.zeros((9, 9))
    for xi, eta, zeta, weight in structured.gauss_points():
        j, det = iso.jacobian(nodes, xi, eta, zeta)
        bu = structured.strain_displacement(
            structured.shape_derivatives(xi, eta, zeta) @ np.linalg.inv(j))
        ba = iso._incompatible_b(xi, eta, zeta, np.linalg.inv(j))
        w = weight * det
        ke += w * (bu.T @ c @ bu)
        k_ua += w * (bu.T @ c @ ba)
        k_aa += w * (ba.T @ c @ ba)
    uncorrected = ke - k_ua @ np.linalg.solve(k_aa, k_ua.T)

    error = patch_energy_relative_error(nodes, uncorrected)
    assert error > 1e-4, "the correction would not be needed if this passed"
    assert error < 1e-1


# --------------------------------------------------------------- refusals

def test_a_folded_element_is_refused():
    """A negative Jacobian still produces numbers. Those numbers look like an
    answer, so the element must refuse rather than return them."""
    nodes = iso.box_nodes(1.0, 1.0, 1.0).copy()
    nodes[6] = [-0.5, -0.5, -0.5]
    with pytest.raises(iso.DegenerateElement, match="folded or inverted"):
        iso.element_stiffness_from_c(nodes, stiffness())


def test_a_flat_element_is_refused():
    """Zero thickness, so det J is zero rather than negative."""
    nodes = iso.box_nodes(1.0, 1.0, 1.0).copy()
    nodes[:, 2] = 0.0
    with pytest.raises(iso.DegenerateElement):
        iso.element_stiffness_from_c(nodes, stiffness())


def test_the_degeneracy_test_is_relative_to_element_size():
    """A part modelled in metres is not degenerate for being small.

    det J for a 1 mm cube in metre units is about 1e-10, which an absolute
    threshold would reject. The same cube in millimetres has det J of 0.125.
    They are the same element and must both be accepted.
    """
    for scale in (1e-3, 1.0, 1e3):
        nodes = iso.box_nodes(scale, scale, scale)
        ke = iso.element_stiffness_from_c(nodes, stiffness())
        assert np.isfinite(ke).all()


def test_wrong_node_count_is_refused():
    with pytest.raises(ValueError, match=r"nodes must be \(8, 3\)"):
        iso.element_stiffness_from_c(np.zeros((6, 3)), stiffness())


# ------------------------------------------------- the GPU kernels

def gpu_or_skip():
    wp = pytest.importorskip("warp")
    wp.init()
    if not wp.get_cuda_device_count():
        pytest.skip("no CUDA device")
    return wp


def test_the_per_element_kernel_matches_the_shared_one_on_boxes():
    """On a mesh of identical boxes the two kernels must agree exactly.

    Not approximately: they multiply the same numbers in the same order, so
    any difference is an indexing error, and the per-element kernel's only
    change is the offset into the Ke array.
    """
    import numpy as np

    wp = gpu_or_skip()
    from physics.fem import kernels

    device = wp.get_device("cuda:0")
    n = 512
    ke = structured.element_stiffness_from_c(1.0, 1.0, 1.0, stiffness())
    conn = np.tile(np.arange(8, dtype=np.int32), (n, 1))
    u = np.random.default_rng(0).random(24)

    d_u = wp.array(u, dtype=wp.float64, device=device)
    d_conn = wp.array2d(conn, dtype=wp.int32, device=device)
    d_scale = wp.array(np.ones(n), dtype=wp.float64, device=device)

    shared = wp.zeros(24, dtype=wp.float64, device=device)
    wp.launch(kernels.stiffness_matvec, dim=n,
              inputs=[d_u, wp.array(ke.flatten(), dtype=wp.float64,
                                    device=device),
                      d_conn, d_scale, shared], device=device)

    per = wp.zeros(24, dtype=wp.float64, device=device)
    wp.launch(kernels.stiffness_matvec_per_element, dim=n,
              inputs=[d_u, wp.array(np.tile(ke.flatten(), n),
                                    dtype=wp.float64, device=device),
                      d_conn, d_scale, per], device=device)
    wp.synchronize()
    assert np.array_equal(shared.numpy(), per.numpy())


def test_the_per_element_diagonal_matches_the_shared_one():
    import numpy as np

    wp = gpu_or_skip()
    from physics.fem import kernels

    device = wp.get_device("cuda:0")
    n = 512
    ke = structured.element_stiffness_from_c(1.0, 1.0, 1.0, stiffness())
    conn = np.tile(np.arange(8, dtype=np.int32), (n, 1))
    d_conn = wp.array2d(conn, dtype=wp.int32, device=device)
    d_scale = wp.array(np.ones(n), dtype=wp.float64, device=device)

    shared = wp.zeros(24, dtype=wp.float64, device=device)
    wp.launch(kernels.stiffness_diagonal, dim=n,
              inputs=[wp.array(ke.flatten(), dtype=wp.float64, device=device),
                      d_conn, d_scale, shared], device=device)
    per = wp.zeros(24, dtype=wp.float64, device=device)
    wp.launch(kernels.stiffness_diagonal_per_element, dim=n,
              inputs=[wp.array(np.tile(ke.flatten(), n), dtype=wp.float64,
                               device=device),
                      d_conn, d_scale, per], device=device)
    wp.synchronize()
    assert np.array_equal(shared.numpy(), per.numpy())
