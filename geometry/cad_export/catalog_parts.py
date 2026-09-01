"""Solids for catalogue parts, where the geometry exists and where it does not.

The catalogues split cleanly in two, and this module keeps that split visible
rather than papering over it.

Bearings carry REAL geometry. A 6206 is 30 by 62 by 16 mm from any
manufacturer, so those numbers are a standard rather than an invention, and
the solid built from them is a genuine envelope.

Motors and gearboxes carry NO geometry at all. Their catalogue entries hold
torque, speed, inertia and mass, and nothing about shape. A solid for one of
them is therefore made up, and this module makes that impossible to forget:
the part is named PLACEHOLDER, and the only physical content it has is its
MASS. The dimensions follow from the mass and an assumed density that the
caller has to state, so the block displaces the right amount of material and
claims nothing else.

WHY NOT JUST PICK A PLAUSIBLE SIZE
==================================
Because a plausible size is the dangerous one. A block that looks like a real
motor invites someone to check clearance against it, and the answer would be
fiction with no way to tell. A block that is merely mass-correct, and labelled,
can still be used for what it is good for, which is packaging arithmetic and
seeing that something occupies space.

VALIDITY DOMAIN
===============
Stated before implementing.

    A bearing solid is the outer envelope: an annular ring of the ISO bore,
    outer diameter and width. It is NOT the internal geometry. There are no
    balls, no cage, no seals and no raceway, so it is right for clearance and
    for mounting, and says nothing about load paths inside the bearing.

    A placeholder is right for mass and for volume at the density stated, and
    for nothing else. Its aspect ratio is a convention chosen here, its
    mounting features do not exist, and its centre of mass is its centroid
    rather than a measured one.

    Vendor CAD replaces all of this. Until it does, an interference check
    against a placeholder is a check against an assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .hollow_rect import METRES_TO_MM, solid_bounding_box_m, solid_volume_m3
from .kernel import Kernel, require_kernel

#: Marker put in the name of every solid whose shape was invented. Grep for it.
PLACEHOLDER = "PLACEHOLDER_NO_GEOMETRY"

#: Density assumed when turning a catalogue mass into a placeholder volume.
#: A motor is mostly steel laminations and copper with air between, so the bulk
#: figure is well below either. Stated here because the caller is entitled to
#: disagree, and because a number this arbitrary must not be buried.
ASSUMED_PLACEHOLDER_DENSITY_KG_M3 = 4000.0

#: Length over diameter for a placeholder motor, and the proportions of a
#: placeholder gearbox block. Conventions, not measurements.
MOTOR_ASPECT = 1.5
GEARBOX_ASPECT = (1.0, 1.0, 0.8)


@dataclass(frozen=True)
class PartSolid:
    """A built solid, and an honest label for what it represents."""

    name: str
    kind: str
    solid: object
    mass_kg: float
    volume_m3: float
    bounding_box_m: tuple[float, float, float]
    geometry_is_real: bool
    source: str

    @property
    def is_placeholder(self) -> bool:
        return not self.geometry_is_real

    def describe(self) -> str:
        tag = "" if self.geometry_is_real else f" [{PLACEHOLDER}]"
        return (f"{self.name}{tag}: {self.kind}, {self.mass_kg:.4f} kg, "
                f"envelope {tuple(round(v, 4) for v in self.bounding_box_m)} m")


def bearing_solid(spec, kernel: Kernel | None = None) -> PartSolid:
    """An annular ring from the bearing's ISO boundary dimensions.

    Real geometry: the bore, outer diameter and width of a designated bearing
    are standardised, so this envelope is a fact rather than a guess. The
    INSIDE is not modelled, which the validity domain above spells out.
    """
    kernel = kernel or require_kernel()
    outer_r = spec.outer_diameter_m / 2.0 * METRES_TO_MM
    bore_r = spec.bore_m / 2.0 * METRES_TO_MM
    width = spec.width_m * METRES_TO_MM
    if bore_r >= outer_r:
        raise ValueError(
            f"bearing {spec.designation}: bore is not smaller than the outer "
            f"diameter")

    if kernel.name == "build123d":
        b = kernel.module
        outer = b.Cylinder(radius=outer_r, height=width)
        # Longer than the part so the subtraction leaves no coincident faces.
        bore = b.Cylinder(radius=bore_r, height=width * 1.2)
        solid = outer - bore
    else:
        cq = kernel.module
        solid = (cq.Workplane("XY").circle(outer_r).extrude(width)
                 .cut(cq.Workplane("XY").circle(bore_r).extrude(width * 1.2)))

    return PartSolid(
        name=spec.designation, kind="bearing", solid=solid,
        mass_kg=spec.mass_kg, volume_m3=solid_volume_m3(solid, kernel),
        bounding_box_m=solid_bounding_box_m(solid, kernel),
        geometry_is_real=True,
        source="ISO boundary dimensions, standard for the designation")


def _placeholder_volume_m3(mass_kg: float, density_kg_m3: float) -> float:
    if mass_kg <= 0.0:
        raise ValueError("a placeholder needs a positive catalogue mass")
    if density_kg_m3 <= 0.0:
        raise ValueError("assumed density must be positive")
    return mass_kg / density_kg_m3


def motor_placeholder(spec, kernel: Kernel | None = None,
                      density_kg_m3: float = ASSUMED_PLACEHOLDER_DENSITY_KG_M3
                      ) -> PartSolid:
    """A cylinder with the motor's mass and an invented shape.

    The catalogue has no dimensions for a motor, so the diameter and length
    here are derived from the mass and the assumed density. They carry no
    information about the real part beyond how much space its mass occupies.
    """
    kernel = kernel or require_kernel()
    volume = _placeholder_volume_m3(spec.mass_kg, density_kg_m3)
    # V = pi r^2 (2r * aspect)  ->  r = (V / (2 pi aspect))^(1/3)
    radius = (volume / (2.0 * math.pi * MOTOR_ASPECT)) ** (1.0 / 3.0)
    length = 2.0 * radius * MOTOR_ASPECT

    if kernel.name == "build123d":
        solid = kernel.module.Cylinder(radius=radius * METRES_TO_MM,
                                       height=length * METRES_TO_MM)
    else:
        solid = (kernel.module.Workplane("XY")
                 .circle(radius * METRES_TO_MM)
                 .extrude(length * METRES_TO_MM))

    return PartSolid(
        name=f"MOTOR_{spec.id}_{PLACEHOLDER}", kind="motor", solid=solid,
        mass_kg=spec.mass_kg, volume_m3=solid_volume_m3(solid, kernel),
        bounding_box_m=solid_bounding_box_m(solid, kernel),
        geometry_is_real=False,
        source=(f"shape invented; mass is the catalogue value and the volume "
                f"follows from an assumed {density_kg_m3:g} kg/m3"))


def gearbox_placeholder(spec, kernel: Kernel | None = None,
                        density_kg_m3: float = ASSUMED_PLACEHOLDER_DENSITY_KG_M3
                        ) -> PartSolid:
    """A block with the gearbox's mass and an invented shape."""
    kernel = kernel or require_kernel()
    volume = _placeholder_volume_m3(spec.mass_kg, density_kg_m3)
    ratios = GEARBOX_ASPECT
    scale = (volume / (ratios[0] * ratios[1] * ratios[2])) ** (1.0 / 3.0)
    dims = [scale * r * METRES_TO_MM for r in ratios]

    if kernel.name == "build123d":
        solid = kernel.module.Box(*dims)
    else:
        solid = (kernel.module.Workplane("XY").rect(dims[0], dims[1])
                 .extrude(dims[2]))

    return PartSolid(
        name=f"GEARBOX_{spec.id}_{PLACEHOLDER}", kind="gearbox", solid=solid,
        mass_kg=spec.mass_kg, volume_m3=solid_volume_m3(solid, kernel),
        bounding_box_m=solid_bounding_box_m(solid, kernel),
        geometry_is_real=False,
        source=(f"shape invented; mass is the catalogue value and the volume "
                f"follows from an assumed {density_kg_m3:g} kg/m3"))


def place(part: PartSolid, position_m: tuple[float, float, float],
          kernel: Kernel | None = None) -> PartSolid:
    """Move a part to a position in the assembly frame.

    Returns a new PartSolid rather than mutating, so the catalogue solid stays
    reusable and two placements of the same bearing cannot alias.
    """
    kernel = kernel or require_kernel()
    offset = [v * METRES_TO_MM for v in position_m]
    if kernel.name == "build123d":
        b = kernel.module
        moved = b.Pos(*offset) * part.solid
    else:
        moved = part.solid.translate(tuple(offset))
    return PartSolid(name=part.name, kind=part.kind, solid=moved,
                     mass_kg=part.mass_kg, volume_m3=part.volume_m3,
                     bounding_box_m=solid_bounding_box_m(moved, kernel),
                     geometry_is_real=part.geometry_is_real,
                     source=part.source)


def interference_m3(first: PartSolid, second: PartSolid,
                    kernel: Kernel | None = None) -> float:
    """Overlapping volume between two placed parts, zero when they clear.

    Volume rather than a boolean because "they touch" and "one is buried
    inside the other" are different problems and a yes or no cannot tell them
    apart.
    """
    kernel = kernel or require_kernel()
    if kernel.name == "build123d":
        common = first.solid & second.solid
    else:
        common = first.solid.intersect(second.solid)
    try:
        return solid_volume_m3(common, kernel)
    except (ValueError, RuntimeError, AttributeError):
        # An empty intersection is not an error, it is the good outcome, and
        # the kernels disagree about how to represent one.
        return 0.0


@dataclass(frozen=True)
class InterferenceReport:
    """Which placed parts overlap, and by how much.

    Carries `involves_placeholder` because an interference against an invented
    shape is a statement about an assumption, not about the hardware, and the
    two must not be read the same way.
    """

    clashes: tuple[tuple[str, str, float], ...]
    involves_placeholder: bool

    @property
    def is_clear(self) -> bool:
        return not self.clashes

    def summary(self) -> str:
        if self.is_clear:
            return "no interference"
        worst = max(volume for _, _, volume in self.clashes)
        caveat = (" (at least one is a placeholder, so this is a check "
                  "against an assumed shape)" if self.involves_placeholder
                  else "")
        return (f"{len(self.clashes)} interference(s), worst {worst:.3e} m3"
                + caveat)


def check_interference(parts: "list[PartSolid]", tolerance_m3: float = 1e-12,
                       kernel: Kernel | None = None) -> InterferenceReport:
    """Every pair, so a third part buried in the first two is not missed."""
    kernel = kernel or require_kernel()
    clashes = []
    placeholder = False
    for i, first in enumerate(parts):
        for second in parts[i + 1:]:
            volume = interference_m3(first, second, kernel)
            if volume > tolerance_m3:
                clashes.append((first.name, second.name, volume))
                placeholder |= first.is_placeholder or second.is_placeholder
    return InterferenceReport(clashes=tuple(clashes),
                              involves_placeholder=placeholder)
