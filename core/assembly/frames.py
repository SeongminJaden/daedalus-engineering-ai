"""core.assembly.frames: coordinate conventions and homogeneous transforms.

CONVENTIONS, FIXED HERE SO NOTHING GUESSES

  * Right-handed frames throughout. SI units: metres, radians, newtons.
  * World frame: x forward, **y up**, z out of the plane.
  * Gravity acts along **-y**, matching `configs/default.yaml` and the beam
    model, where the section height h and the tip load both run along y. Using
    z-up here would have silently flipped the load direction relative to every
    structural result already in the project.
  * A revolute joint rotates about its own axis by the joint value, positive by
    the right-hand rule. A prismatic joint translates along its axis.
  * `T_parent_child` reads "the pose of child expressed in parent", so
    transforms compose left to right down the chain.
"""

from __future__ import annotations

import numpy as np

GRAVITY_DIRECTION = np.array([0.0, -1.0, 0.0])
STANDARD_GRAVITY = 9.80665


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def translation(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    t = identity()
    t[:3, 3] = (x, y, z)
    return t


def normalize_axis(axis) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(a)
    if norm < 1e-12:
        raise ValueError(f"axis must be non-zero, got {axis}")
    return a / norm


def rotation_about_axis(axis, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation as a 4x4 homogeneous transform."""
    a = normalize_axis(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    skew = np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]], dtype=np.float64)
    r = np.eye(3) + s * skew + (1.0 - c) * (skew @ skew)
    t = identity()
    t[:3, :3] = r
    return t


def translation_along_axis(axis, distance: float) -> np.ndarray:
    return translation(*(normalize_axis(axis) * float(distance)))


def compose(*transforms: np.ndarray) -> np.ndarray:
    out = identity()
    for t in transforms:
        out = out @ np.asarray(t, dtype=np.float64)
    return out


def inverse(transform: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, using R^T rather than a general inverse."""
    t = np.asarray(transform, dtype=np.float64)
    r, p = t[:3, :3], t[:3, 3]
    out = identity()
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


def position(transform: np.ndarray) -> np.ndarray:
    return np.asarray(transform, dtype=np.float64)[:3, 3].copy()


def rotation(transform: np.ndarray) -> np.ndarray:
    return np.asarray(transform, dtype=np.float64)[:3, :3].copy()


def transform_point(transform: np.ndarray, point) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    t = np.asarray(transform, dtype=np.float64)
    return t[:3, :3] @ p + t[:3, 3]


def is_rigid_transform(transform: np.ndarray, tol: float = 1e-9) -> bool:
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4):
        return False
    r = t[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=tol):
        return False
    if not np.isclose(np.linalg.det(r), 1.0, atol=tol):
        return False
    return np.allclose(t[3, :], [0.0, 0.0, 0.0, 1.0], atol=tol)
