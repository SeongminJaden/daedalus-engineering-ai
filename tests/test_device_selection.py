"""Choosing the solve device by problem size.

Measured on this machine, the GPU is slower than the CPU below roughly 10,000
degrees of freedom: kernel launch overhead dominates a small solve, and this
consumer part runs the float64 the solver needs at a fraction of its float32
rate. The meshes this project actually runs sit below that crossover.

The gate is that the choice is **numerically neutral**. Picking a device for
speed is only acceptable if it cannot change an answer, so the same problem is
solved on both and the results are required to agree.
"""

import numpy as np
import pytest
import warp as wp

from optimization.topology.simp import stiffness_scale
from physics.fem.mesh import solid_box_mesh
from physics.fem.solver import (CPU_DOF_THRESHOLD, _resolve_device,
                                solve_linear_elasticity)

E, NU, LOAD = 71.7e9, 0.33, 200.0

CUDA_AVAILABLE = any(d.is_cuda for d in wp.get_devices())
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE,
                                   reason="no CUDA device on this machine")


@requires_cuda
def test_small_problems_choose_the_cpu_and_large_ones_the_gpu():
    assert _resolve_device(None, CPU_DOF_THRESHOLD - 1) == "cpu"
    assert _resolve_device(None, CPU_DOF_THRESHOLD).startswith("cuda")
    assert _resolve_device(None, 10 * CPU_DOF_THRESHOLD).startswith("cuda")


def test_an_explicit_device_is_never_overridden():
    """The heuristic is a default, not a policy. Tests and callers pin devices."""
    assert _resolve_device("cpu", 10 ** 9) == "cpu"
    assert _resolve_device("cuda:0", 1) == "cuda:0"


def test_an_unstated_size_does_not_silently_pick_the_cpu():
    """Callers that do not pass a size keep the previous behaviour."""
    expected = "cuda:0" if CUDA_AVAILABLE else "cpu"
    assert _resolve_device(None, None) == expected


@requires_cuda
def test_the_two_devices_agree():
    """The gate: device selection must not move a number.

    Same kernels, same float64, same algorithm. The two differ only in
    floating-point reduction order, so they agree to near the solver's own
    convergence tolerance rather than exactly.
    """
    mesh = solid_box_mesh(0.16, 0.05, 0.02, 12, 5, 2)
    density = np.linspace(0.3, 0.9, mesh.n_elements)
    common = dict(fixed_nodes=mesh.nodes_at_x(0.0),
                  load_nodes=mesh.nodes_at_x(0.16), total_load_n=-LOAD,
                  load_direction=1,
                  element_scale=stiffness_scale(density, 3.0))
    on_cpu = solve_linear_elasticity(mesh, E, NU, device="cpu", **common)
    on_gpu = solve_linear_elasticity(mesh, E, NU, device="cuda:0", **common)

    assert on_cpu.report.converged and on_gpu.report.converged
    scale = np.abs(on_cpu.displacements).max()
    assert scale > 0
    difference = np.abs(on_cpu.displacements - on_gpu.displacements).max() / scale
    assert difference < 1e-6, f"devices disagree by {difference:.2e}"

    stress_scale = np.abs(on_cpu.element_von_mises).max()
    stress_difference = (np.abs(on_cpu.element_von_mises
                                - on_gpu.element_von_mises).max() / stress_scale)
    assert stress_difference < 1e-6


@requires_cuda
def test_the_default_matches_an_explicit_choice_of_the_same_device():
    """Letting the heuristic choose gives what pinning that device gives."""
    mesh = solid_box_mesh(0.16, 0.05, 0.02, 10, 4, 2)
    assert mesh.n_dofs < CPU_DOF_THRESHOLD
    common = dict(fixed_nodes=mesh.nodes_at_x(0.0),
                  load_nodes=mesh.nodes_at_x(0.16), total_load_n=-LOAD,
                  load_direction=1)
    chosen = solve_linear_elasticity(mesh, E, NU, **common)
    pinned = solve_linear_elasticity(mesh, E, NU, device="cpu", **common)
    assert chosen.displacements == pytest.approx(pinned.displacements, rel=1e-12)
