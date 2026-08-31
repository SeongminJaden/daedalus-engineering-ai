"""physics.dynamics.inertia: link inertia derived from the section geometry.

A hollow rectangular prism is a solid box minus a concentric smaller box, so its
inertia tensor is the difference of the two closed forms. Concentricity is what
makes the subtraction valid: both boxes share a centroid, so no parallel-axis
term is needed.

Axes match the rest of the project: x along the link, y the section height, z
the section width.
"""

from __future__ import annotations

import numpy as np


def box_inertia_tensor(mass: float, length_x: float, length_y: float,
                       length_z: float) -> np.ndarray:
    """Inertia of a solid rectangular box about its own centroid.

        Ixx = m (Ly^2 + Lz^2) / 12
        Iyy = m (Lx^2 + Lz^2) / 12
        Izz = m (Lx^2 + Ly^2) / 12
    """
    if mass < 0:
        raise ValueError("mass must be >= 0")
    for value in (length_x, length_y, length_z):
        if value <= 0:
            raise ValueError(f"box dimensions must be > 0, got {value}")
    return np.diag([
        mass * (length_y ** 2 + length_z ** 2) / 12.0,
        mass * (length_x ** 2 + length_z ** 2) / 12.0,
        mass * (length_x ** 2 + length_y ** 2) / 12.0,
    ])


def hollow_rect_inertia(length_m: float, outer_width_m: float,
                        outer_height_m: float, wall_thickness_m: float,
                        density_kg_m3: float) -> np.ndarray:
    """Inertia tensor of a hollow rectangular prism about its centroid.

    Outer box minus the cavity, both centred on the same point.
    """
    inner_w = outer_width_m - 2.0 * wall_thickness_m
    inner_h = outer_height_m - 2.0 * wall_thickness_m
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("wall thickness leaves no cavity")

    outer_mass = length_m * outer_height_m * outer_width_m * density_kg_m3
    inner_mass = length_m * inner_h * inner_w * density_kg_m3
    outer = box_inertia_tensor(outer_mass, length_m, outer_height_m,
                               outer_width_m)
    inner = box_inertia_tensor(inner_mass, length_m, inner_h, inner_w)
    return outer - inner


def link_inertia(link, density_kg_m3: float) -> np.ndarray:
    """Inertia tensor of an assembly Link about its own centre of mass."""
    section = link.genome.section
    return hollow_rect_inertia(link.length_m, section.outer_width_m,
                               section.outer_height_m,
                               section.wall_thickness_m, density_kg_m3)


def parallel_axis(inertia: np.ndarray, mass: float, offset) -> np.ndarray:
    """Shift an inertia tensor by `offset` from the centre of mass.

        I' = I + m ((d.d) E - d d^T)
    """
    d = np.asarray(offset, dtype=np.float64).reshape(3)
    return (np.asarray(inertia, dtype=np.float64)
            + mass * (float(d @ d) * np.eye(3) - np.outer(d, d)))


def is_valid_inertia(inertia: np.ndarray, tol: float = 1e-12) -> bool:
    """Symmetric, positive definite, and satisfying the triangle inequality.

    The triangle inequality on the principal moments is what distinguishes a
    physically realisable rigid body from an arbitrary positive definite matrix.
    """
    i = np.asarray(inertia, dtype=np.float64)
    if i.shape != (3, 3):
        return False
    if not np.allclose(i, i.T, atol=tol * max(np.abs(i).max(), 1.0)):
        return False
    eigenvalues = np.sort(np.linalg.eigvalsh(0.5 * (i + i.T)))
    if eigenvalues[0] <= 0:
        return False
    return eigenvalues[0] + eigenvalues[1] >= eigenvalues[2] * (1.0 - 1e-9)
