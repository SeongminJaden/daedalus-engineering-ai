"""Parametric shape families, each with the closed form that checks it.

A family is a small number of parameters, a rule for turning them into a
B-rep, and the exact volume that B-rep must have. The volume is not
decoration. It is what lets the STEP analyzer be checked against arithmetic
for every part the engine makes, so a broken export or a misread unit fails
loudly on the first record instead of poisoning ten thousand of them.

Where a family carries features, the parameters also say what the feature
recogniser must find: this many holes at this diameter, this many fillets at
this radius. A label is therefore checked against the parameters that made the
part before it is stored. Synthetic data is only worth having because its
ground truth is known; a generator that does not check its own ground truth
has thrown that advantage away.

VALIDITY DOMAIN
===============
Stated before implementing.

    Five families, all single solids, all built in millimetres from
    parameters given in metres, all with their long axis along x so that one
    cantilever load case applies to every one of them: clamp the x-minimum
    face, load the x-maximum face.

    These are analysis shapes. They have no threads, no tolerances, no
    chamfers except where a parameter says so, and no resemblance to any
    catalogue part. A model trained on them has seen five kinds of prism, and
    its opinion of a gearbox housing is worth exactly that.

    The plate's holes are placed on a fixed pattern so that they can never
    intersect a fillet or an edge for any parameter the sampler admits; the
    admissibility rule is written down rather than left to the kernel to
    discover.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from geometry.cad_export.hollow_rect import (METRES_TO_MM, analytic_volume,
                                             build_solid as hollow_solid)
from geometry.cad_export.kernel import Kernel, require_kernel


@dataclass(frozen=True)
class ExpectedFeatures:
    """What the recogniser must report for a part, from its parameters."""

    hole_count: int = 0
    hole_diameter_m: float | None = None
    fillet_count: int = 0
    fillet_radius_m: float | None = None


@dataclass(frozen=True)
class Family:
    name: str
    #: Parameter name to (low, high), in metres or dimensionless.
    bounds: dict[str, tuple[float, float]]
    build: Callable[[dict[str, float], Kernel], object]
    volume_m3: Callable[[dict[str, float]], float]
    admissible: Callable[[dict[str, float]], bool]
    expected_features: Callable[[dict[str, float]], ExpectedFeatures]
    #: Index of the load direction for the cantilever case: 1 is y, 2 is z.
    load_direction: int = 1
    description: str = ""
    integer_parameters: tuple[str, ...] = field(default_factory=tuple)


def _mm(value_m: float) -> float:
    return value_m * METRES_TO_MM


# ------------------------------------------------------------------- box

def _box_build(p, kernel):
    b = kernel.module
    return b.Box(_mm(p["length_m"]), _mm(p["height_m"]), _mm(p["width_m"]))


def _box_volume(p):
    return p["length_m"] * p["height_m"] * p["width_m"]


BOX = Family(
    name="box",
    bounds={"length_m": (0.05, 0.30), "height_m": (0.01, 0.08),
            "width_m": (0.01, 0.08)},
    build=_box_build, volume_m3=_box_volume,
    admissible=lambda p: True,
    expected_features=lambda p: ExpectedFeatures(),
    description="a solid rectangular prism, the one shape every mesher covers")


# ----------------------------------------------------------- hollow rect

def _hollow_build(p, kernel):
    return hollow_solid(p["length_m"], p["width_m"], p["height_m"],
                        p["wall_m"], kernel)


def _hollow_volume(p):
    return analytic_volume(p["length_m"], p["width_m"], p["height_m"],
                           p["wall_m"])


def _hollow_admissible(p):
    return p["wall_m"] <= 0.45 * min(p["height_m"], p["width_m"])


HOLLOW_RECT = Family(
    name="hollow_rect",
    bounds={"length_m": (0.05, 0.30), "height_m": (0.02, 0.08),
            "width_m": (0.02, 0.08), "wall_m": (0.001, 0.010)},
    build=_hollow_build, volume_m3=_hollow_volume,
    admissible=_hollow_admissible,
    expected_features=lambda p: ExpectedFeatures(),
    description="the project's own link section, an open-ended hollow prism")


# ------------------------------------------------------------- L bracket

def _l_build(p, kernel):
    b = kernel.module
    s, t, w = _mm(p["size_m"]), _mm(p["thickness_m"]), _mm(p["width_m"])
    outer = b.Box(s, s, w)
    # The cut is longer than the part in z so that no face pair coincides.
    cut = b.Pos(t / 2.0, t / 2.0, 0.0) * b.Box(s - t, s - t, w * 1.1)
    return outer - cut


def _l_volume(p):
    s, t, w = p["size_m"], p["thickness_m"], p["width_m"]
    return w * (s * s - (s - t) * (s - t))


def _l_admissible(p):
    return 0.08 * p["size_m"] <= p["thickness_m"] <= 0.45 * p["size_m"]


L_BRACKET = Family(
    name="l_bracket",
    bounds={"size_m": (0.04, 0.15), "thickness_m": (0.004, 0.06),
            "width_m": (0.01, 0.06)},
    build=_l_build, volume_m3=_l_volume,
    admissible=_l_admissible,
    expected_features=lambda p: ExpectedFeatures(),
    description="two equal arms at a right angle, a reentrant corner")


# ------------------------------------------------------ plate with holes

def _plate_hole_centres(p):
    L, W = p["length_m"], p["width_m"]
    if int(p["hole_count"]) == 4:
        return [(L / 4, W / 4), (-L / 4, W / 4), (L / 4, -W / 4),
                (-L / 4, -W / 4)]
    return [(L / 4, 0.0), (-L / 4, 0.0)]


def _plate_build(p, kernel):
    b = kernel.module
    L, W, T = _mm(p["length_m"]), _mm(p["width_m"]), _mm(p["thickness_m"])
    with b.BuildPart() as part:
        b.Box(L, W, T)
        b.fillet(part.edges().filter_by(b.Axis.Z),
                 radius=_mm(p["fillet_radius_m"]))
        with b.Locations(*[(_mm(x), _mm(y), 0.0)
                           for x, y in _plate_hole_centres(p)]):
            b.Hole(radius=_mm(p["hole_radius_m"]))
    return part.part


def _plate_volume(p):
    L, W, T = p["length_m"], p["width_m"], p["thickness_m"]
    r, rf, n = p["hole_radius_m"], p["fillet_radius_m"], int(p["hole_count"])
    corners = 4.0 * (1.0 - math.pi / 4.0) * rf * rf * T
    return L * W * T - n * math.pi * r * r * T - corners


def _plate_admissible(p):
    L, W = p["length_m"], p["width_m"]
    r, rf, n = p["hole_radius_m"], p["fillet_radius_m"], int(p["hole_count"])
    if n not in (2, 4):
        return False
    if rf > 0.2 * min(L, W):
        return False
    # every hole centre sits at L/4 from the near x edge and, for four holes,
    # W/4 from the near y edge; it must clear both the edge and the fillet
    clear_x = L / 4 - r - rf
    clear_y = (W / 4 if n == 4 else W / 2) - r - rf
    if clear_x <= 0.002 or clear_y <= 0.002:
        return False
    # holes must not touch each other
    return 2 * r < min(L / 2, W / 2 if n == 4 else L) - 0.002


def _plate_features(p):
    return ExpectedFeatures(hole_count=int(p["hole_count"]),
                            hole_diameter_m=2.0 * p["hole_radius_m"],
                            fillet_count=4,
                            fillet_radius_m=p["fillet_radius_m"])


PLATE_WITH_HOLES = Family(
    name="plate_with_holes",
    bounds={"length_m": (0.06, 0.20), "width_m": (0.04, 0.12),
            "thickness_m": (0.004, 0.020), "hole_radius_m": (0.002, 0.010),
            "fillet_radius_m": (0.002, 0.012), "hole_count": (2, 4)},
    build=_plate_build, volume_m3=_plate_volume,
    admissible=_plate_admissible,
    expected_features=_plate_features,
    load_direction=2,
    description="a filleted plate with two or four through holes",
    integer_parameters=("hole_count",))


# --------------------------------------------------------- stepped shaft

def _shaft_build(p, kernel):
    b = kernel.module
    r1, l1 = _mm(p["radius_m"]), _mm(p["length_1_m"])
    r2, l2 = _mm(p["radius_m"] * p["step_ratio"]), _mm(p["length_2_m"])
    first = b.Pos(0.0, 0.0, l1 / 2.0) * b.Cylinder(r1, l1)
    second = b.Pos(0.0, 0.0, l1 + l2 / 2.0) * b.Cylinder(r2, l2)
    shaft = first + second
    # built along z, turned so that the axis is x like every other family
    return b.Rotation(0.0, 90.0, 0.0) * shaft


def _shaft_volume(p):
    r1, r2 = p["radius_m"], p["radius_m"] * p["step_ratio"]
    return math.pi * (r1 * r1 * p["length_1_m"] + r2 * r2 * p["length_2_m"])


STEPPED_SHAFT = Family(
    name="stepped_shaft",
    bounds={"radius_m": (0.005, 0.030), "length_1_m": (0.02, 0.10),
            "step_ratio": (0.4, 0.9), "length_2_m": (0.02, 0.10)},
    build=_shaft_build, volume_m3=_shaft_volume,
    admissible=lambda p: True,
    expected_features=lambda p: ExpectedFeatures(),
    description="two coaxial cylinders of different radius, a shoulder")


# =========================================================================== #
# industrial part classes, added for the dataset specification
# =========================================================================== #
#
# Each of these keeps the rule the first five set: a closed form volume that
# the analyzer must reproduce, and a statement of the holes and fillets the
# parameters put there. They are analysis shapes named after part classes,
# not catalogue parts; a flange here is a disc with a bore and a bolt circle
# and nothing more.


def _segment_area(radius: float, depth: float) -> float:
    """Area of a circular segment of the given depth."""
    return (radius * radius * math.acos((radius - depth) / radius)
            - (radius - depth) * math.sqrt(2.0 * radius * depth - depth * depth))


def _bracket_build(p, kernel):
    b = kernel.module
    L, W, T, H = (_mm(p["length_m"]), _mm(p["width_m"]), _mm(p["thickness_m"]),
                  _mm(p["height_m"]))
    r = _mm(p["hole_radius_m"])
    base = b.Pos(L / 2, T / 2, 0) * b.Box(L, T, W)
    upright = b.Pos(L - T / 2, H / 2, 0) * b.Box(T, H, W)
    part = base + upright
    for z in (-W / 4, W / 4):
        part = part - (b.Pos(L / 4, T / 2, z) * b.Rotation(90, 0, 0)
                       * b.Cylinder(r, T * 3))
        part = part - (b.Pos(L - T / 2, 3 * H / 4, z) * b.Rotation(0, 90, 0)
                       * b.Cylinder(r, T * 3))
    return part


def _bracket_volume(p):
    L, W, T, H, r = (p["length_m"], p["width_m"], p["thickness_m"],
                     p["height_m"], p["hole_radius_m"])
    return L * W * T + H * W * T - T * T * W - 4.0 * math.pi * r * r * T


def _bracket_admissible(p):
    L, W, T, H, r = (p["length_m"], p["width_m"], p["thickness_m"],
                     p["height_m"], p["hole_radius_m"])
    if T >= 0.4 * min(L, H, W):
        return False
    # base holes at x = L/4, z = +-W/4; upright holes at y = 3H/4, z = +-W/4
    clear = 0.002
    return (r + clear < W / 4 and 2 * r + clear < W / 2 and r + clear < L / 4
            and r + clear < min(3 * H / 4 - T, H / 4) and 2 * r + clear < W / 2)


BRACKET = Family(
    name="bracket",
    bounds={"length_m": (0.06, 0.20), "width_m": (0.04, 0.12),
            "thickness_m": (0.004, 0.015), "height_m": (0.04, 0.15),
            "hole_radius_m": (0.003, 0.008)},
    build=_bracket_build, volume_m3=_bracket_volume,
    admissible=_bracket_admissible,
    expected_features=lambda p: ExpectedFeatures(
        hole_count=4, hole_diameter_m=2.0 * p["hole_radius_m"]),
    description="an angle bracket: a base plate and an upright, two holes in each")


def _flange_build(p, kernel):
    b = kernel.module
    R, T = _mm(p["outer_radius_m"]), _mm(p["thickness_m"])
    rb, r, Rb = (_mm(p["bore_radius_m"]), _mm(p["bolt_radius_m"]),
                 _mm(p["bolt_circle_radius_m"]))
    n = int(p["bolt_count"])
    disc = b.Rotation(0, 90, 0) * (b.Pos(0, 0, T / 2) * b.Cylinder(R, T))
    disc = disc - (b.Rotation(0, 90, 0) * (b.Pos(0, 0, T / 2) * b.Cylinder(rb, T * 3)))
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        y, z = Rb * math.cos(angle), Rb * math.sin(angle)
        disc = disc - (b.Pos(T / 2, y, z) * b.Rotation(0, 90, 0) * b.Cylinder(r, T * 3))
    return disc


def _flange_volume(p):
    R, T, rb, r, n = (p["outer_radius_m"], p["thickness_m"], p["bore_radius_m"],
                      p["bolt_radius_m"], int(p["bolt_count"]))
    return math.pi * (R * R - rb * rb - n * r * r) * T


def _flange_admissible(p):
    R, rb, r, Rb, n = (p["outer_radius_m"], p["bore_radius_m"], p["bolt_radius_m"],
                       p["bolt_circle_radius_m"], int(p["bolt_count"]))
    if n not in (4, 6, 8):
        return False
    clear = 0.002
    if not (rb + clear < Rb - r and Rb + r + clear < R):
        return False
    # neighbouring bolt holes must not touch
    return 2.0 * Rb * math.sin(math.pi / n) > 2.0 * r + clear


FLANGE = Family(
    name="flange",
    bounds={"outer_radius_m": (0.04, 0.12), "thickness_m": (0.006, 0.025),
            "bore_radius_m": (0.008, 0.04), "bolt_radius_m": (0.003, 0.007),
            "bolt_circle_radius_m": (0.025, 0.10), "bolt_count": (4, 8)},
    build=_flange_build, volume_m3=_flange_volume,
    admissible=_flange_admissible,
    expected_features=lambda p: ExpectedFeatures(hole_count=int(p["bolt_count"]) + 1),
    description="a disc with a central bore and a bolt circle; the axis is x",
    integer_parameters=("bolt_count",))


def _housing_build(p, kernel):
    b = kernel.module
    L, W, H, t = (_mm(p["length_m"]), _mm(p["width_m"]), _mm(p["height_m"]),
                  _mm(p["wall_m"]))
    outer = b.Box(L, H, W)
    # the cavity is as tall as the housing and shifted up by one wall, so it
    # leaves a floor of thickness t and runs out through the top face
    cavity = b.Pos(0, t, 0) * b.Box(L - 2 * t, H, W - 2 * t)
    return outer - cavity


def _housing_volume(p):
    L, W, H, t = p["length_m"], p["width_m"], p["height_m"], p["wall_m"]
    return L * W * H - (L - 2 * t) * (W - 2 * t) * (H - t)


HOUSING = Family(
    name="housing",
    bounds={"length_m": (0.06, 0.25), "width_m": (0.04, 0.15),
            "height_m": (0.03, 0.12), "wall_m": (0.002, 0.012)},
    build=_housing_build, volume_m3=_housing_volume,
    admissible=lambda p: p["wall_m"] <= 0.3 * min(p["length_m"], p["width_m"],
                                                    p["height_m"]),
    expected_features=lambda p: ExpectedFeatures(),
    description="an open box: five walls, the top open")


def _keyed_shaft_build(p, kernel):
    b = kernel.module
    R, L, d, w = (_mm(p["radius_m"]), _mm(p["length_m"]), _mm(p["flat_depth_m"]),
                  _mm(p["flat_width_m"]))
    shaft = b.Rotation(0, 90, 0) * (b.Pos(0, 0, L / 2) * b.Cylinder(R, L))
    # a flat: everything above y = R - d within |z| < w/2 is removed along the
    # whole length. w is chosen so the cut is inside the circle.
    cut = b.Pos(L / 2, R - d + R, 0) * b.Box(L * 1.1, 2 * R, w)
    return shaft - cut


def _keyed_shaft_volume(p):
    R, L, d, w = p["radius_m"], p["length_m"], p["flat_depth_m"], p["flat_width_m"]
    # removed section: the part of the circle above the chord y = R - d,
    # limited to |z| < w/2. With the chord half-width c = sqrt(2Rd - d^2) and
    # w/2 >= c the removed area is the full segment; the admissibility rule
    # requires exactly that, so the closed form is the segment.
    return L * (math.pi * R * R - _segment_area(R, d))


def _keyed_shaft_admissible(p):
    R, d, w = p["radius_m"], p["flat_depth_m"], p["flat_width_m"]
    if not 0.05 * R < d < 0.5 * R:
        return False
    chord_half = math.sqrt(2 * R * d - d * d)
    return w / 2 >= chord_half * 1.05 and w / 2 < R * 1.5


KEYED_SHAFT = Family(
    name="keyed_shaft",
    bounds={"radius_m": (0.006, 0.03), "length_m": (0.05, 0.25),
            "flat_depth_m": (0.001, 0.012), "flat_width_m": (0.006, 0.09)},
    build=_keyed_shaft_build, volume_m3=_keyed_shaft_volume,
    admissible=_keyed_shaft_admissible,
    expected_features=lambda p: ExpectedFeatures(),
    description="a shaft with a flat along its length, the seat a key would sit on")


def _gear_blank_build(p, kernel):
    b = kernel.module
    R, T, rh, Lh, rb = (_mm(p["outer_radius_m"]), _mm(p["thickness_m"]),
                        _mm(p["hub_radius_m"]), _mm(p["hub_length_m"]),
                        _mm(p["bore_radius_m"]))
    disc = b.Pos(0, 0, T / 2) * b.Cylinder(R, T)
    hub = b.Pos(0, 0, T + Lh / 2) * b.Cylinder(rh, Lh)
    blank = disc + hub
    blank = blank - (b.Pos(0, 0, (T + Lh) / 2) * b.Cylinder(rb, (T + Lh) * 3))
    return b.Rotation(0, 90, 0) * blank


def _gear_blank_volume(p):
    R, T, rh, Lh, rb = (p["outer_radius_m"], p["thickness_m"], p["hub_radius_m"],
                        p["hub_length_m"], p["bore_radius_m"])
    return math.pi * (R * R * T + rh * rh * Lh - rb * rb * (T + Lh))


def _gear_blank_admissible(p):
    R, rh, rb = p["outer_radius_m"], p["hub_radius_m"], p["bore_radius_m"]
    return rb + 0.002 < rh and rh + 0.004 < R


GEAR_BLANK = Family(
    name="gear_blank",
    bounds={"outer_radius_m": (0.02, 0.10), "thickness_m": (0.006, 0.03),
            "hub_radius_m": (0.01, 0.05), "hub_length_m": (0.01, 0.06),
            "bore_radius_m": (0.004, 0.025)},
    build=_gear_blank_build, volume_m3=_gear_blank_volume,
    admissible=_gear_blank_admissible,
    expected_features=lambda p: ExpectedFeatures(hole_count=1,
                                                 hole_diameter_m=2.0 * p["bore_radius_m"]),
    description="a disc with a hub and a through bore, before any teeth are cut")


def _link_build(p, kernel):
    b = kernel.module
    L, W, T, r = (_mm(p["length_m"]), _mm(p["width_m"]), _mm(p["thickness_m"]),
                  _mm(p["eye_radius_m"]))
    body = b.Box(L - W, W, T)
    ends = (b.Pos(-(L - W) / 2, 0, 0) * b.Cylinder(W / 2, T)
            + b.Pos((L - W) / 2, 0, 0) * b.Cylinder(W / 2, T))
    link = body + ends
    for x in (-(L - W) / 2, (L - W) / 2):
        link = link - (b.Pos(x, 0, 0) * b.Cylinder(r, T * 3))
    return link


def _link_volume(p):
    L, W, T, r = p["length_m"], p["width_m"], p["thickness_m"], p["eye_radius_m"]
    return ((L - W) * W + math.pi * (W / 2) ** 2 - 2 * math.pi * r * r) * T


LINK = Family(
    name="link",
    bounds={"length_m": (0.06, 0.25), "width_m": (0.012, 0.05),
            "thickness_m": (0.004, 0.02), "eye_radius_m": (0.003, 0.015)},
    build=_link_build, volume_m3=_link_volume,
    admissible=lambda p: (p["eye_radius_m"] + 0.002 < p["width_m"] / 2
                          and p["length_m"] > 2.0 * p["width_m"]),
    # The recogniser reports the two rounded ends as fillets: each half
    # cylinder is tangent to both long faces, which is its rule. Measured, not
    # assumed; the first version expected none and the engine refused it.
    expected_features=lambda p: ExpectedFeatures(
        hole_count=2, hole_diameter_m=2.0 * p["eye_radius_m"],
        fillet_count=2, fillet_radius_m=p["width_m"] / 2.0),
    load_direction=2,
    description="a flat link with rounded ends and an eye at each")


def _mount_build(p, kernel):
    b = kernel.module
    L, W, T = _mm(p["length_m"]), _mm(p["width_m"]), _mm(p["thickness_m"])
    rb, hb, r = _mm(p["boss_radius_m"]), _mm(p["boss_height_m"]), _mm(p["hole_radius_m"])
    plate = b.Box(L, T, W)
    boss = b.Pos(0, T / 2 + hb / 2, 0) * b.Rotation(90, 0, 0) * b.Cylinder(rb, hb)
    mount = plate + boss
    mount = mount - (b.Rotation(90, 0, 0) * b.Cylinder(r, (T + hb) * 3))
    return mount


def _mount_volume(p):
    L, W, T = p["length_m"], p["width_m"], p["thickness_m"]
    rb, hb, r = p["boss_radius_m"], p["boss_height_m"], p["hole_radius_m"]
    return L * W * T + math.pi * rb * rb * hb - math.pi * r * r * (T + hb)


MOUNT = Family(
    name="mount",
    bounds={"length_m": (0.05, 0.20), "width_m": (0.03, 0.12),
            "thickness_m": (0.004, 0.02), "boss_radius_m": (0.008, 0.04),
            "boss_height_m": (0.005, 0.04), "hole_radius_m": (0.003, 0.02)},
    build=_mount_build, volume_m3=_mount_volume,
    admissible=lambda p: (p["hole_radius_m"] + 0.002 < p["boss_radius_m"]
                          and p["boss_radius_m"] + 0.003 < min(p["length_m"],
                                                             p["width_m"]) / 2),
    expected_features=lambda p: ExpectedFeatures(hole_count=1,
                                                 hole_diameter_m=2.0 * p["hole_radius_m"]),
    description="a base plate with a bossed through hole in the middle")


def _ribbed_build(p, kernel):
    b = kernel.module
    L, W, T = _mm(p["length_m"]), _mm(p["width_m"]), _mm(p["thickness_m"])
    tr, hr, n = _mm(p["rib_thickness_m"]), _mm(p["rib_height_m"]), int(p["rib_count"])
    plate = b.Box(L, T, W)
    part = plate
    for i in range(n):
        z = -W / 2 + W * (i + 0.5) / n
        part = part + (b.Pos(0, T / 2 + hr / 2, z) * b.Box(L, hr, tr))
    return part


def _ribbed_volume(p):
    L, W, T = p["length_m"], p["width_m"], p["thickness_m"]
    return L * W * T + int(p["rib_count"]) * L * p["rib_thickness_m"] * p["rib_height_m"]


RIBBED_PLATE = Family(
    name="ribbed_plate",
    bounds={"length_m": (0.06, 0.25), "width_m": (0.04, 0.15),
            "thickness_m": (0.003, 0.012), "rib_thickness_m": (0.002, 0.008),
            "rib_height_m": (0.005, 0.03), "rib_count": (2, 4)},
    build=_ribbed_build, volume_m3=_ribbed_volume,
    admissible=lambda p: (p["rib_thickness_m"] + 0.002
                          < p["width_m"] / int(p["rib_count"])),
    expected_features=lambda p: ExpectedFeatures(),
    description="a plate stiffened by parallel ribs along its length",
    integer_parameters=("rib_count",))


FAMILIES: dict[str, Family] = {f.name: f for f in (
    BOX, HOLLOW_RECT, L_BRACKET, PLATE_WITH_HOLES, STEPPED_SHAFT,
    BRACKET, FLANGE, HOUSING, KEYED_SHAFT, GEAR_BLANK, LINK, MOUNT, RIBBED_PLATE)}

#: The first five, which the classifier rules and the CAD executor were
#: built and measured on. The classifier says UNKNOWN for the rest, which is
#: correct until its rules are extended and measured.
ORIGINAL_FAMILIES = ("box", "hollow_rect", "l_bracket", "plate_with_holes",
                     "stepped_shaft")


def family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"no family {name!r}; the families are "
                       f"{sorted(FAMILIES)}") from None


# --------------------------------------------------------------- sampling

def sample_parameters(fam: Family, rng: np.random.Generator,
                      max_attempts: int = 1000) -> dict[str, float]:
    """One admissible parameter set, uniform within the bounds.

    Rejection sampling against the family's admissibility rule, so the
    distribution is uniform on the admissible region rather than on the box
    around it. Deterministic for a given generator state.
    """
    for _ in range(max_attempts):
        params: dict[str, float] = {}
        for name, (low, high) in fam.bounds.items():
            if name in fam.integer_parameters:
                params[name] = float(rng.integers(int(low), int(high) + 1))
            else:
                params[name] = float(rng.uniform(low, high))
        if fam.admissible(params):
            return params
    raise RuntimeError(
        f"{fam.name}: no admissible parameters in {max_attempts} draws; the "
        f"bounds and the admissibility rule disagree")


def part_id_for(fam: Family, params: dict[str, float]) -> str:
    """A stable id from the parameters, so a re-run consolidates rather than
    duplicates. Rounded to a nanometre first, because a float that differs in
    its last bit is the same part."""
    canonical = json.dumps({k: round(v, 9) for k, v in sorted(params.items())},
                           sort_keys=True)
    digest = hashlib.sha1(canonical.encode()).hexdigest()[:10]
    return f"{fam.name}-{digest}"


def build(fam: Family, params: dict[str, float], kernel: Kernel | None = None):
    """The B-rep for one parameter set, in the kernel's millimetres."""
    if not fam.admissible(params):
        raise ValueError(f"{fam.name}: parameters {params} are not admissible")
    return fam.build(params, kernel or require_kernel())
