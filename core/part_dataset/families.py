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


FAMILIES: dict[str, Family] = {f.name: f for f in (
    BOX, HOLLOW_RECT, L_BRACKET, PLATE_WITH_HOLES, STEPPED_SHAFT)}


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
