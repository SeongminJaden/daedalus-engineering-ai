"""Classical laminate theory: ABD, ply stresses and first-ply failure.

Phase 8 gave the solver orthotropic elasticity, which lets a composite be
analysed once someone has decided what it is made of. CLT is how that decision
gets made: a laminate is designed as a STACKING SEQUENCE, and the same plies in
a different order or at different angles give a different stiffness, a
different strength and a different first ply to fail.

VALIDITY, written before the implementation:

* **Thin plate, Kirchhoff kinematics.** Straight lines normal to the midplane
  stay straight and normal, which requires the laminate to be thin compared
  with its in-plane dimensions. A thick laminate shears through its thickness
  and CLT cannot see that.

* **Plane stress in every ply.** sigma_3, tau_13 and tau_23 are taken as zero.
  This is the assumption that fails hardest AT A FREE EDGE, where interlaminar
  stresses are genuinely three-dimensional and are what actually delaminates
  real laminates. CLT will report a comfortable laminate that delaminates at
  its edges, and nothing here will warn about it.

* **Perfect bonding.** Plies do not slip relative to one another; there is no
  interlaminar compliance and no delamination.

* **Linear elastic to failure.** No matrix plasticity, no progressive damage.

* **FIRST-PLY failure, which is not ultimate failure.** The laminate is
  reported as failed when its first ply fails. A real laminate usually carries
  more load after that: the failed ply sheds its share to the others and the
  stack continues. So this is CONSERVATIVE as a strength prediction, sometimes
  by a large margin, and using it as the ultimate strength wastes material.
  The opposite error is also available: if the first failure is a fibre failure
  in the primary load direction, the laminate really is finished, and treating
  first-ply as conservative there would be wrong.

* **No hygrothermal terms.** Cure-induced residual stresses and moisture
  swelling are real, are locked into every laminate, and are not modelled.

Notation follows the usual convention: 1 is along the fibres, 2 across them in
the ply plane, and 6 is in-plane shear. x and y are the laminate axes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.materials.db import MaterialSpec


@dataclass(frozen=True)
class Lamina:
    """One ply's material properties in its own coordinates."""

    e1_pa: float
    e2_pa: float
    g12_pa: float
    nu12: float
    thickness_m: float

    def __post_init__(self) -> None:
        if min(self.e1_pa, self.e2_pa, self.g12_pa, self.thickness_m) <= 0.0:
            raise ValueError("moduli and thickness must be positive")
        if not 0.0 < self.nu12 < 1.0:
            raise ValueError("nu12 must lie in (0, 1)")

    @property
    def nu21(self) -> float:
        """The minor Poisson ratio, fixed by reciprocity: nu21 = nu12 E2 / E1.

        Not an independent property. Supplying it separately is a common way to
        end up with a non-symmetric compliance matrix, which is unphysical.
        """
        return self.nu12 * self.e2_pa / self.e1_pa


def reduced_stiffness(lamina: Lamina) -> np.ndarray:
    """The plane-stress reduced stiffness Q, 3x3, in material coordinates.

        Q11 = E1 / (1 - nu12 nu21)      Q12 = nu12 E2 / (1 - nu12 nu21)
        Q22 = E2 / (1 - nu12 nu21)      Q66 = G12

    VALIDITY: plane stress. These are REDUCED stiffnesses, not the 3D ones, and
    they already encode sigma_3 = 0. Using them where the through-thickness
    stress matters gives the wrong answer with no warning.
    """
    denominator = 1.0 - lamina.nu12 * lamina.nu21
    if denominator <= 0.0:
        raise ValueError(
            "1 - nu12 nu21 is not positive, so this ply is not physically "
            "admissible")
    q = np.zeros((3, 3))
    q[0, 0] = lamina.e1_pa / denominator
    q[1, 1] = lamina.e2_pa / denominator
    q[0, 1] = q[1, 0] = lamina.nu12 * lamina.e2_pa / denominator
    q[2, 2] = lamina.g12_pa
    return q


def transformed_stiffness(q: np.ndarray, angle_deg: float) -> np.ndarray:
    """Q-bar: the reduced stiffness rotated into laminate coordinates.

    The off-axis terms Q16 and Q26 are zero only at 0 and 90 degrees. Nonzero
    they couple extension to shear, which is why an unbalanced laminate twists
    when you pull it.
    """
    theta = math.radians(angle_deg)
    m, n = math.cos(theta), math.sin(theta)
    q11, q12, q22, q66 = q[0, 0], q[0, 1], q[1, 1], q[2, 2]

    bar = np.zeros((3, 3))
    bar[0, 0] = (q11 * m ** 4 + 2.0 * (q12 + 2.0 * q66) * m ** 2 * n ** 2
                 + q22 * n ** 4)
    bar[0, 1] = bar[1, 0] = ((q11 + q22 - 4.0 * q66) * m ** 2 * n ** 2
                             + q12 * (m ** 4 + n ** 4))
    bar[1, 1] = (q11 * n ** 4 + 2.0 * (q12 + 2.0 * q66) * m ** 2 * n ** 2
                 + q22 * m ** 4)
    bar[0, 2] = bar[2, 0] = ((q11 - q12 - 2.0 * q66) * m ** 3 * n
                             + (q12 - q22 + 2.0 * q66) * m * n ** 3)
    bar[1, 2] = bar[2, 1] = ((q11 - q12 - 2.0 * q66) * m * n ** 3
                             + (q12 - q22 + 2.0 * q66) * m ** 3 * n)
    bar[2, 2] = ((q11 + q22 - 2.0 * q12 - 2.0 * q66) * m ** 2 * n ** 2
                 + q66 * (m ** 4 + n ** 4))
    return bar


@dataclass
class Laminate:
    """A stack of plies, bottom to top, with their orientations."""

    plies: list[Lamina]
    angles_deg: list[float]

    def __post_init__(self) -> None:
        if len(self.plies) != len(self.angles_deg):
            raise ValueError(
                f"{len(self.plies)} plies but {len(self.angles_deg)} angles")
        if not self.plies:
            raise ValueError("a laminate needs at least one ply")

    @classmethod
    def from_material(cls, material: MaterialSpec, angles_deg: list[float],
                      ply_thickness_m: float) -> "Laminate":
        """Build a laminate of identical plies from an orthotropic material."""
        for field in ("e1_pa", "e2_pa", "g12_pa", "nu12"):
            if getattr(material, field) is None:
                raise ValueError(
                    f"{material.id} has no {field}; CLT needs orthotropic ply "
                    f"properties and an isotropic entry does not carry them")
        lamina = Lamina(e1_pa=material.e1_pa, e2_pa=material.e2_pa,
                        g12_pa=material.g12_pa, nu12=material.nu12,
                        thickness_m=ply_thickness_m)
        return cls(plies=[lamina] * len(angles_deg),
                   angles_deg=list(angles_deg))

    @property
    def thickness_m(self) -> float:
        return sum(ply.thickness_m for ply in self.plies)

    def interfaces_m(self) -> np.ndarray:
        """Ply boundaries measured from the midplane, bottom to top.

        Measuring from the MIDPLANE and not from the bottom face is what makes
        B vanish for a symmetric stack. From any other datum it would not, and
        the coupling would look real.
        """
        edges = np.concatenate([[0.0], np.cumsum(
            [ply.thickness_m for ply in self.plies])])
        return edges - 0.5 * self.thickness_m

    def is_symmetric(self) -> bool:
        """Whether the stack mirrors about its midplane in angle and thickness."""
        angles = self.angles_deg
        thicknesses = [ply.thickness_m for ply in self.plies]
        return (angles == angles[::-1]
                and np.allclose(thicknesses, thicknesses[::-1]))

    def is_balanced(self) -> bool:
        """Whether every off-axis angle has an equal and opposite partner.

        A balanced laminate has A16 = A26 = 0, so pulling it does not shear it.
        Symmetry and balance are different properties and a laminate can have
        either without the other.
        """
        off_axis = [a for a in self.angles_deg
                    if abs(a % 180.0) > 1e-9 and abs(a % 180.0 - 90.0) > 1e-9]
        counts: dict[float, int] = {}
        for angle in off_axis:
            key = round(angle % 180.0, 9)
            counts[key] = counts.get(key, 0) + 1
        for key, count in counts.items():
            mirror = round((-key) % 180.0, 9)
            if counts.get(mirror, 0) != count:
                return False
        return True


@dataclass(frozen=True)
class AbdMatrices:
    """The laminate constitutive matrices.

    A couples in-plane force to in-plane strain, D couples moment to curvature,
    and B couples the two together. A nonzero B means pulling the laminate
    bends it, which is usually a defect of the stacking sequence rather than an
    intention.
    """

    a: np.ndarray
    b: np.ndarray
    d: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        """The assembled 6x6 [[A, B], [B, D]]."""
        return np.block([[self.a, self.b], [self.b, self.d]])

    @property
    def couples_extension_to_bending(self) -> bool:
        scale = max(np.abs(self.a).max(), 1.0)
        return bool(np.abs(self.b).max() > 1e-9 * scale)


def abd_matrices(laminate: Laminate) -> AbdMatrices:
    """A, B and D by integrating Q-bar through the thickness.

        A_ij = sum Qbar_ij (z_k - z_k-1)
        B_ij = 1/2 sum Qbar_ij (z_k^2 - z_k-1^2)
        D_ij = 1/3 sum Qbar_ij (z_k^3 - z_k-1^3)
    """
    z = laminate.interfaces_m()
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    d = np.zeros((3, 3))
    for index, (ply, angle) in enumerate(zip(laminate.plies,
                                             laminate.angles_deg)):
        bar = transformed_stiffness(reduced_stiffness(ply), angle)
        lower, upper = z[index], z[index + 1]
        a += bar * (upper - lower)
        b += bar * (upper ** 2 - lower ** 2) / 2.0
        d += bar * (upper ** 3 - lower ** 3) / 3.0
    return AbdMatrices(a=a, b=b, d=d)


@dataclass(frozen=True)
class PlyState:
    """One ply's strain and stress, in laminate and in material coordinates."""

    index: int
    angle_deg: float
    z_m: float
    strain_laminate: np.ndarray       # [eps_x, eps_y, gamma_xy]
    stress_laminate: np.ndarray       # [sigma_x, sigma_y, tau_xy]
    stress_material: np.ndarray       # [sigma_1, sigma_2, tau_12]


def stress_transformation(angle_deg: float) -> np.ndarray:
    """T taking laminate stresses to material coordinates.

    Stress and engineering strain transform differently because engineering
    shear strain carries a factor of two. Using the stress matrix on strains,
    or the reverse, is a standard way to get answers that are wrong by exactly
    that factor in the shear term alone.
    """
    theta = math.radians(angle_deg)
    m, n = math.cos(theta), math.sin(theta)
    return np.array([
        [m ** 2, n ** 2, 2.0 * m * n],
        [n ** 2, m ** 2, -2.0 * m * n],
        [-m * n, m * n, m ** 2 - n ** 2],
    ])


def ply_states(laminate: Laminate, force_resultant: np.ndarray,
               moment_resultant: np.ndarray | None = None,
               at_ply_midplane: bool = False) -> list[PlyState]:
    """Solve the laminate for midplane strain and curvature, then per-ply state.

    `force_resultant` is [Nx, Ny, Nxy] in N/m and `moment_resultant` is
    [Mx, My, Mxy] in N. Resultants, per unit width, not forces.

    Each ply is evaluated at its most highly strained surface unless
    `at_ply_midplane`, because under bending the extreme fibre of a ply governs
    and the ply midplane understates it.
    """
    abd = abd_matrices(laminate)
    n = np.asarray(force_resultant, dtype=float).reshape(3)
    m = (np.zeros(3) if moment_resultant is None
         else np.asarray(moment_resultant, dtype=float).reshape(3))
    try:
        response = np.linalg.solve(abd.matrix, np.concatenate([n, m]))
    except np.linalg.LinAlgError as singular:
        raise ValueError(
            "the ABD matrix is singular, so this stack has no unique "
            "response") from singular
    midplane_strain, curvature = response[:3], response[3:]

    z = laminate.interfaces_m()
    states: list[PlyState] = []
    for index, (ply, angle) in enumerate(zip(laminate.plies,
                                             laminate.angles_deg)):
        if at_ply_midplane:
            height = 0.5 * (z[index] + z[index + 1])
        else:
            # Whichever face of this ply is further from the midplane sees the
            # larger bending strain.
            height = (z[index] if abs(z[index]) > abs(z[index + 1])
                      else z[index + 1])
        strain = midplane_strain + height * curvature
        bar = transformed_stiffness(reduced_stiffness(ply), angle)
        stress_laminate = bar @ strain
        stress_material = stress_transformation(angle) @ stress_laminate
        states.append(PlyState(index=index, angle_deg=angle, z_m=height,
                               strain_laminate=strain,
                               stress_laminate=stress_laminate,
                               stress_material=stress_material))
    return states
