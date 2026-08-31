"""Gear tooth capacity: Lewis bending and Hertzian contact.

A gear tooth has two independent ways of failing and they are sized by
different things. **Bending** breaks the tooth at its root, and is governed by
the module and the face width. **Pitting** is surface fatigue from Hertzian
contact stress, and is governed by the pitch diameter and the surface hardness.
Which one binds depends on the gear: a small, soft gear usually breaks its
teeth, and a hard, case-carburised one usually pits first. Checking only one is
how a gear set passes review and then fails in the way that was not checked.

**This is Lewis and elementary Hertz, not AGMA.** The real standard multiplies
both stresses by factors for dynamic load, load distribution across the face,
application shock, rim thickness, size and surface condition, and every one of
those is at least 1.0. The numbers here therefore run OPTIMISTIC against a real
gear, and the correction factors are exposed as inputs so a caller can supply
measured ones rather than have this pretend they equal one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Lewis form factor Y for 20 degree full-depth teeth, by tooth count. The
# standard tabulation. A tooth with fewer teeth has a more undercut, weaker
# root, which is why Y falls away sharply below about 18 teeth.
_LEWIS_Y: tuple[tuple[int, float], ...] = (
    (12, 0.245), (13, 0.261), (14, 0.277), (15, 0.290), (16, 0.296),
    (17, 0.303), (18, 0.309), (19, 0.314), (20, 0.322), (21, 0.328),
    (22, 0.331), (24, 0.337), (26, 0.346), (28, 0.353), (30, 0.359),
    (34, 0.371), (38, 0.384), (43, 0.397), (50, 0.409), (60, 0.422),
    (75, 0.435), (100, 0.447), (150, 0.460), (300, 0.472), (400, 0.480),
)

# Pressure angle of the tooth form these factors describe.
PRESSURE_ANGLE_DEG = 20.0


def lewis_form_factor(tooth_count: int) -> float:
    """Y at this tooth count, interpolated between tabulated entries.

    Clamped at both ends rather than extrapolated. Below 12 teeth a 20 degree
    tooth is undercut badly enough that the Lewis model stops describing it,
    and inventing a trend past the table would be inventing the table.
    """
    if tooth_count < 1:
        raise ValueError("a gear needs at least one tooth")
    if tooth_count <= _LEWIS_Y[0][0]:
        return _LEWIS_Y[0][1]
    if tooth_count >= _LEWIS_Y[-1][0]:
        return _LEWIS_Y[-1][1]
    for (low, y_low), (high, y_high) in zip(_LEWIS_Y, _LEWIS_Y[1:]):
        if low <= tooth_count <= high:
            span = high - low
            weight = 0.0 if span == 0 else (tooth_count - low) / span
            return y_low + weight * (y_high - y_low)
    return _LEWIS_Y[-1][1]      # pragma: no cover - covered by the clamps


def pitch_diameter_m(module_m: float, tooth_count: int) -> float:
    """d = m N."""
    if module_m <= 0.0:
        raise ValueError("module must be positive")
    return module_m * tooth_count


def tangential_load_n(torque_nm: float, pitch_diameter: float) -> float:
    """W_t = 2 T / d, the force transmitted at the pitch circle."""
    if pitch_diameter <= 0.0:
        raise ValueError("pitch diameter must be positive")
    return 2.0 * torque_nm / pitch_diameter


def lewis_bending_stress_pa(tangential_load: float, face_width_m: float,
                            module_m: float, form_factor: float,
                            correction: float = 1.0) -> float:
    """sigma = W_t / (b m Y), times whatever correction the caller supplies.

    `correction` stands for the product of the AGMA factors this model does not
    compute: overload, dynamic, load distribution and size. It defaults to 1.0,
    which is the OPTIMISTIC choice and is why it is a visible argument rather
    than absent. A real gear in a robot joint would use something around 1.5 to
    2.5.
    """
    if min(face_width_m, module_m, form_factor) <= 0.0:
        raise ValueError("face width, module and form factor must be positive")
    return correction * tangential_load / (face_width_m * module_m * form_factor)


def elastic_coefficient(e1_pa: float, nu1: float, e2_pa: float,
                        nu2: float) -> float:
    """Z_E = sqrt(1 / (pi ((1-nu1^2)/E1 + (1-nu2^2)/E2))), in sqrt(Pa).

    A material pairing term: two steel gears give a higher contact stress than
    a steel gear running against a softer one, because the softer material
    spreads the contact.
    """
    if min(e1_pa, e2_pa) <= 0.0:
        raise ValueError("moduli must be positive")
    compliance = (1.0 - nu1 ** 2) / e1_pa + (1.0 - nu2 ** 2) / e2_pa
    return math.sqrt(1.0 / (math.pi * compliance))


def geometry_factor_i(gear_ratio: float,
                      pressure_angle_deg: float = PRESSURE_ANGLE_DEG) -> float:
    """I = (cos phi sin phi / 2) (m_G / (m_G + 1)) for external gears.

    The pitting geometry factor. It rises with gear ratio, so a pinion driving
    a much larger wheel has a more favourable contact geometry than a
    one-to-one pair.
    """
    if gear_ratio <= 0.0:
        raise ValueError("gear ratio must be positive")
    phi = math.radians(pressure_angle_deg)
    return (math.cos(phi) * math.sin(phi) / 2.0) * (gear_ratio
                                                    / (gear_ratio + 1.0))


def hertz_contact_stress_pa(tangential_load: float, face_width_m: float,
                            pitch_diameter: float, elastic_coefficient_pa: float,
                            geometry_factor: float,
                            correction: float = 1.0) -> float:
    """sigma_c = Z_E sqrt(W_t / (b d I)), times the caller's correction.

    Note the square root: contact stress goes as the square root of load, so
    doubling the torque raises it by only 41 percent. Bending stress is linear
    in load. That difference is why which mode governs can flip as a design is
    scaled.
    """
    if min(face_width_m, pitch_diameter, geometry_factor) <= 0.0:
        raise ValueError("face width, diameter and geometry factor must be "
                         "positive")
    if tangential_load < 0.0:
        raise ValueError("tangential load is a magnitude here")
    return (correction * elastic_coefficient_pa
            * math.sqrt(tangential_load
                        / (face_width_m * pitch_diameter * geometry_factor)))


@dataclass(frozen=True)
class GearMesh:
    """One meshing pair, described by the things that set its capacity."""

    module_m: float
    pinion_teeth: int
    gear_teeth: int
    face_width_m: float
    torque_nm: float                     # at the PINION
    pressure_angle_deg: float = PRESSURE_ANGLE_DEG

    def __post_init__(self) -> None:
        if self.pinion_teeth < 1 or self.gear_teeth < 1:
            raise ValueError("tooth counts must be positive")
        if self.face_width_m <= 0.0:
            raise ValueError("face width must be positive")

    @property
    def ratio(self) -> float:
        return self.gear_teeth / self.pinion_teeth

    @property
    def pinion_pitch_diameter_m(self) -> float:
        return pitch_diameter_m(self.module_m, self.pinion_teeth)

    @property
    def tangential_load_n(self) -> float:
        return tangential_load_n(self.torque_nm, self.pinion_pitch_diameter_m)


@dataclass(frozen=True)
class GearResult:
    """Both failure modes, and which one binds."""

    mesh: GearMesh
    tangential_load_n: float
    bending_stress_pa: float
    contact_stress_pa: float
    bending_allowable_pa: float
    contact_allowable_pa: float
    bending_safety_factor: float
    contact_safety_factor: float
    governing_mode: str

    @property
    def governing_safety_factor(self) -> float:
        return min(self.bending_safety_factor, self.contact_safety_factor)

    @property
    def passes(self) -> bool:
        return self.governing_safety_factor >= 1.0


def analyze_mesh(mesh: GearMesh, bending_allowable_pa: float,
                 contact_allowable_pa: float,
                 e_pinion_pa: float = 207e9, nu_pinion: float = 0.29,
                 e_gear_pa: float = 207e9, nu_gear: float = 0.29,
                 bending_correction: float = 1.0,
                 contact_correction: float = 1.0) -> GearResult:
    """Check a mesh against both tooth bending and surface pitting.

    The two allowable stresses are separate inputs and are NOT the same number.
    A through-hardened steel might allow 200 MPa in bending and 700 MPa in
    contact; case carburising raises the contact allowable far more than the
    bending one. Passing one number for both would make the comparison between
    the modes meaningless.
    """
    load = mesh.tangential_load_n
    bending = lewis_bending_stress_pa(
        load, mesh.face_width_m, mesh.module_m,
        lewis_form_factor(mesh.pinion_teeth), bending_correction)
    coefficient = elastic_coefficient(e_pinion_pa, nu_pinion, e_gear_pa,
                                      nu_gear)
    contact = hertz_contact_stress_pa(
        load, mesh.face_width_m, mesh.pinion_pitch_diameter_m, coefficient,
        geometry_factor_i(mesh.ratio, mesh.pressure_angle_deg),
        contact_correction)

    bending_safety = (math.inf if bending <= 0.0
                      else bending_allowable_pa / bending)
    contact_safety = (math.inf if contact <= 0.0
                      else contact_allowable_pa / contact)
    return GearResult(
        mesh=mesh, tangential_load_n=load, bending_stress_pa=bending,
        contact_stress_pa=contact, bending_allowable_pa=bending_allowable_pa,
        contact_allowable_pa=contact_allowable_pa,
        bending_safety_factor=bending_safety,
        contact_safety_factor=contact_safety,
        governing_mode=("bending" if bending_safety < contact_safety
                        else "pitting"))
