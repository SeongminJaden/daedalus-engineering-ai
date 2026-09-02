"""Re-solving a smoothed shape, and the reason there is no smoothing verdict.

The obvious design was to solve the blocky shape, solve the smoothed one, and
reject smoothing when the peak stress rose. Measurement killed it: peak stress
on linear tetrahedra at these densities is not converged, so the difference is
dominated by remeshing. These tests pin that finding, because it is the reason
the module's shape is what it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from geometry.surfacing import surface_from_density
from geometry.surfacing.revalidate import (mesh_sensitivity, revalidate,
                                           tet_mesh_from_surface)
from nodes import calculix

requires_calculix = pytest.mark.skipif(
    not calculix.is_available(), reason="CalculiX is not installed")

SPACING = 0.002
ALUMINIUM_E = 71.7e9
ALUMINIUM_NU = 0.33


def bracket_field() -> np.ndarray:
    """An L with a sharp re-entrant corner, which is where stress concentrates."""
    field = np.zeros((40, 20, 20))
    field[2:38, 2:8, 2:18] = 1.0
    field[2:12, 2:18, 2:18] = 1.0
    return field


def block_field() -> np.ndarray:
    field = np.zeros((30, 14, 14))
    field[2:28, 2:12, 2:12] = 1.0
    return field


# ------------------------------------------------------------- volume mesh

def test_a_smoothed_surface_fills_with_tetrahedra():
    report = surface_from_density(block_field(), SPACING, smoothing_passes=8)
    mesh = tet_mesh_from_surface(report.vertices, report.faces)
    assert mesh.connectivity.shape[1] == 4
    assert mesh.connectivity.shape[0] > 100
    assert mesh.node_coords.shape[1] == 3


def test_an_open_surface_is_refused():
    """An open surface can still be filled with something, and that something
    is not the part."""
    report = surface_from_density(block_field(), SPACING, smoothing_passes=0)
    torn = report.faces[:-40]
    with pytest.raises(ValueError, match="not watertight"):
        tet_mesh_from_surface(report.vertices, torn)


# ------------------------------------------------------------- the re-solve

@requires_calculix
def test_the_smoothed_shape_still_carries_the_load():
    """The question worth answering: does it pass on its own terms."""
    report = surface_from_density(block_field(), SPACING, smoothing_passes=8)
    result = revalidate(report.vertices, report.faces, ALUMINIUM_E,
                        ALUMINIUM_NU, total_load_n=200.0)

    assert result.tetrahedra > 100
    assert result.peak_von_mises_pa > 0.0
    assert result.max_displacement_m > 0.0
    assert result.safety_factor(480e6) > 1.0


@requires_calculix
def test_a_zero_stress_result_refuses_to_give_a_safety_factor():
    """Dividing by nothing would report infinite margin, which is the most
    dangerous possible answer to return silently."""
    from geometry.surfacing.revalidate import Revalidation

    empty = Revalidation(peak_von_mises_pa=0.0, max_displacement_m=0.0,
                         tetrahedra=1, nodes=4, enclosed_volume_m3=1.0)
    with pytest.raises(ValueError, match="meaningless rather than infinite"):
        empty.safety_factor(480e6)


# ------------------------------------- why there is no before/after verdict

@requires_calculix
def test_peak_stress_is_more_mesh_sensitive_than_displacement():
    """The measurement that removed a feature from this module.

    On ONE fixed geometry, refining the mesh moves the peak stress far more
    than the displacement, because a peak is an extremum of a derivative while
    a displacement is an integral of the solution. A before and after stress
    comparison across smoothing therefore measures the remesh, not the shape.
    """
    report = surface_from_density(bracket_field(), SPACING, smoothing_passes=8)
    sensitivity = mesh_sensitivity(report.vertices, report.faces, ALUMINIUM_E,
                                   ALUMINIUM_NU, total_load_n=400.0,
                                   coarse=1.0, fine=0.5)

    assert sensitivity["fine"].tetrahedra > sensitivity["coarse"].tetrahedra * 2
    assert sensitivity["stress_change"] > sensitivity["displacement_change"]
    assert sensitivity["stress_change"] > 0.10, (
        "if the peak stress had settled, a before and after comparison would "
        "be defensible and this module should grow one")
