"""Standard purchased parts as parametric solids, each dimension with its source.

The catalogue split in `catalog_parts.py` was between parts whose geometry is
a standard (bearings) and parts whose geometry nobody recorded (motors,
gearboxes). This module adds the parts whose geometry IS a standard or a
manufacturer's table and was read on the retrieval date:

    socket head cap screws   ISO 4762, head diameter dk and height k per size
    hexagon nuts             ISO 4032, width across flats s and height m
    GT2 timing pulleys       2 mm pitch: pitch diameter 2 N / pi, outside
                             diameter 0.508 mm smaller (a pitch line
                             differential of 0.254 mm, read off a
                             manufacturer's table where PD minus OD is 0.020
                             inch on every row)
    GT2 belts                closed loop length is pitch times tooth count,
                             exactly; the belt itself is not built as a solid
    heat-set inserts         outer diameter and length per size, from the
                             Ruthex and CNC Kitchen family of inserts as
                             tabulated by a third party

WHAT IS AN ENVELOPE AND WHAT IS REAL
====================================
A screw is built as a shank cylinder at the nominal diameter, a head cylinder
at dk by k, and a hexagonal socket of width s cut to a depth of k/2. The
thread is not modelled: the shank is the nominal diameter, which is what a
clearance hole is designed against, and the socket depth is a convention of
this module (ISO 4762 gives a minimum t that was not read). A nut is a
hexagonal prism with a bore at the nominal diameter, again unthreaded. A
pulley is a cylinder at the OUTSIDE diameter with a hub; the teeth are not
cut, so it is an envelope for clearance and a correct pitch diameter for
belt length, and nothing more. An insert is a plain cylinder at its outer
diameter with a bore at the nominal thread; the knurl is not modelled.

Every solid says which of these it is through `geometry_is_real`: True for
the nut and screw envelopes whose outer dimensions are the standard's, False
for the pulley whose teeth are missing. Volumes are checked against closed
forms in the tests, as every solid in this project is.

MATERIAL LINKS
==============
Each part names the material entry its envelope should be given for mass and
clearance work, and says when the database has none: a property class 12.9
screw is an alloy steel that this database has only as steel_scm440
(quenched and tempered), and an insert is brass, which the database does not
hold at all, so its `material_id` is None and its mass is None. A None here
is a missing datum, not a zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .catalog_parts import PartSolid
from .hollow_rect import METRES_TO_MM, solid_bounding_box_m, solid_volume_m3
from .kernel import Kernel, require_kernel


@dataclass(frozen=True)
class DimensionSource:
    id: str
    title: str
    publisher: str
    url: str
    retrieved: str
    note: str = ""


SOURCES = {s.id: s for s in (
    DimensionSource("iso4762_ee", "ISO Socket Head Screw Size Data Table Chart ISO DIN BS EN 4762",
                    "Engineers Edge", "https://www.engineersedge.com/iso_socket_head_screw.htm",
                    "2026-09-03", "restates ISO 4762 dk, k, s and pitch; the standard itself was not read"),
    DimensionSource("iso4032_ames", "Metric Hex Nut Dimensions (ISO 4032) Size Chart (mm)",
                    "AmesWeb", "https://amesweb.info/Fasteners/Nut/Metric-Hex-Nut-Sizes-Dimensions-Chart.aspx",
                    "2026-09-03", "restates ISO 4032 s max, e min, m max"),
    DimensionSource("powerdrive_gt2", "GT2 Timing Belt Pulley 2MM Pitch, pages G-85 and G-86",
                    "PowerDrive", "https://www.powerdrive.com/Downloads/ECatalogs/TimingBeltDrive/GT2%20Timing%20Pulley%20-%202mm%20Pitch.pdf",
                    "2026-09-03", "P.D. and O.D. in inches for 12 to 120 grooves; P.D. minus O.D. is 0.020 inch on every row, which is a pitch line differential of 0.254 mm"),
    DimensionSource("creative3dp_inserts", "Heat-Set Insert Hole Size Chart: M2 to M8 Complete Reference",
                    "Creative3DP Tools", "https://tools.creative3dp.com/blog/heat-set-insert-hole-size-chart/",
                    "2026-09-03", "third party table for the Ruthex and CNC Kitchen inserts; outer diameter and pocket depth (insert length plus 0.5 mm)"),
)}

#: ISO 4762 per size: (pitch mm, head diameter dk mm, head height k mm, socket s mm)
ISO_4762 = {
    "M3": (0.5, 5.5, 3.0, 2.5), "M4": (0.7, 7.0, 4.0, 3.0), "M5": (0.8, 8.5, 5.0, 4.0),
    "M6": (1.0, 10.0, 6.0, 5.0), "M8": (1.25, 13.0, 8.0, 6.0), "M10": (1.5, 16.0, 10.0, 8.0),
}
#: The socket width in the table is a maximum with tolerance (2.58, 3.08,
#: 4.095, 5.14, 6.14, 8.175); the nominal hex key sizes above are what a
#: modeller uses and the difference is the socket tolerance band.

#: ISO 4032 per size: (width across flats s max mm, height m max mm)
ISO_4032 = {
    "M3": (5.5, 2.4), "M4": (7.0, 3.2), "M5": (8.0, 4.7),
    "M6": (10.0, 5.2), "M8": (13.0, 6.8), "M10": (16.0, 8.4),
}

#: Heat-set inserts per size: (outer diameter mm, length mm)
HEAT_SET_INSERTS = {
    "M2": (3.2, 4.0), "M2.5": (3.6, 5.7), "M3": (4.0, 5.7), "M4": (5.6, 8.1),
    "M5": (6.4, 9.5), "M6": (8.0, 12.7), "M8": (10.0, 14.5),
}
#: Lengths: Ruthex product designations M2x4, M2.5x5.7, M3x5.7, M4x8.1,
#: M5x9.5, M6x12.7 (from the Ruthex CAD data page, read 2026-09-03). The M8
#: length is the Creative3DP pocket depth 15.0 mm minus its stated 0.5 mm
#: allowance, and is the least certain number in this table.

GT2_PITCH_MM = 2.0
GT2_PLD_MM = 0.254            # derived from the PowerDrive table, see SOURCES

#: Which material entry an envelope is given, or None when the database has
#: no entry for what the part is made of.
MATERIAL_LINKS = {
    "screw": ("steel_scm440", "property class 8.8 to 12.9 alloy steel; the database holds "
                              "quenched and tempered SCM440 (AISI 4140) as the nearest entry"),
    "nut": ("steel_scm440", "same steel family as the screw"),
    "pulley": ("al_6061_t6", "the PowerDrive pulleys are anodised aluminium; alloy not stated, "
                             "6061-T6 assumed as the common pulley alloy and marked as such"),
    "insert": (None, "brass, which the material database does not hold"),
}


def nominal_diameter_m(size: str) -> float:
    return float(size[1:]) * 1e-3


def _mm(x_m: float) -> float:
    return x_m * METRES_TO_MM


def socket_head_screw_volume_m3(size: str, length_m: float) -> float:
    """Shank plus head minus the hexagonal socket, all as built."""
    pitch, dk, k, s = ISO_4762[size]
    d = nominal_diameter_m(size)
    shank = math.pi * (d / 2) ** 2 * length_m
    head = math.pi * (dk * 1e-3 / 2) ** 2 * (k * 1e-3)
    hexagon_area = (math.sqrt(3) / 2) * (s * 1e-3) ** 2   # width across flats s
    socket = hexagon_area * (k * 1e-3 / 2)
    return shank + head - socket


def socket_head_screw(size: str, length_m: float, kernel: Kernel | None = None) -> PartSolid:
    """ISO 4762 socket head cap screw envelope: unthreaded shank, cylindrical
    head, hexagonal socket half the head deep. Axis along z, head at z > 0."""
    if size not in ISO_4762:
        raise KeyError(f"no ISO 4762 row for {size}; have {sorted(ISO_4762)}")
    if length_m <= 0.0:
        raise ValueError("screw length must be positive")
    kernel = kernel or require_kernel()
    b = kernel.module
    pitch, dk, k, s = ISO_4762[size]
    d = _mm(nominal_diameter_m(size))
    L = _mm(length_m)
    shank = b.Pos(0, 0, -L / 2) * b.Cylinder(d / 2, L)
    head = b.Pos(0, 0, k / 2) * b.Cylinder(dk / 2, k)
    # a regular hexagon of width across flats s has circumradius s / sqrt(3)
    # the socket starts half way up the head and runs out through the top;
    # extrude builds from the polygon's plane in +z, so it is placed at k/2
    socket = b.Pos(0, 0, k / 2) * b.extrude(b.RegularPolygon(s / math.sqrt(3), 6), k / 2 + 1.0)
    solid = (shank + head) - socket
    return PartSolid(
        name=f"ISO4762_{size}x{length_m * 1e3:g}", kind="screw", solid=solid,
        mass_kg=0.0, volume_m3=solid_volume_m3(solid, kernel),
        bounding_box_m=solid_bounding_box_m(solid, kernel), geometry_is_real=True,
        source=f"ISO 4762 dk {dk} mm, k {k} mm, s {s} mm via {SOURCES['iso4762_ee'].publisher}; "
               f"thread not modelled, shank at nominal {size}")


def hex_nut_volume_m3(size: str) -> float:
    s, m = ISO_4032[size]
    area = (math.sqrt(3) / 2) * (s * 1e-3) ** 2
    return area * (m * 1e-3) - math.pi * (nominal_diameter_m(size) / 2) ** 2 * (m * 1e-3)


def hex_nut(size: str, kernel: Kernel | None = None) -> PartSolid:
    """ISO 4032 hexagon nut envelope: hexagonal prism, unthreaded bore at the
    nominal diameter, no chamfers. Axis along z."""
    if size not in ISO_4032:
        raise KeyError(f"no ISO 4032 row for {size}; have {sorted(ISO_4032)}")
    kernel = kernel or require_kernel()
    b = kernel.module
    s, m = ISO_4032[size]
    d = _mm(nominal_diameter_m(size))
    prism = b.extrude(b.RegularPolygon(s / math.sqrt(3), 6), m)
    bore = b.Pos(0, 0, m / 2) * b.Cylinder(d / 2, m * 3)
    solid = prism - bore
    return PartSolid(
        name=f"ISO4032_{size}", kind="nut", solid=solid, mass_kg=0.0,
        volume_m3=solid_volume_m3(solid, kernel),
        bounding_box_m=solid_bounding_box_m(solid, kernel), geometry_is_real=True,
        source=f"ISO 4032 s {s} mm, m {m} mm via {SOURCES['iso4032_ames'].publisher}; "
               f"thread and chamfers not modelled")


def gt2_pitch_diameter_m(teeth: int) -> float:
    return GT2_PITCH_MM * teeth / math.pi * 1e-3


def gt2_outside_diameter_m(teeth: int) -> float:
    return gt2_pitch_diameter_m(teeth) - 2.0 * GT2_PLD_MM * 1e-3


def gt2_belt_length_m(teeth: int) -> float:
    """A closed GT2 belt of this many teeth. Exact: pitch times teeth."""
    return GT2_PITCH_MM * teeth * 1e-3


def gt2_pulley_volume_m3(teeth: int, width_m: float, hub_diameter_m: float,
                         hub_length_m: float, bore_m: float) -> float:
    od = gt2_outside_diameter_m(teeth)
    return (math.pi * (od / 2) ** 2 * width_m
            + math.pi * (hub_diameter_m / 2) ** 2 * hub_length_m
            - math.pi * (bore_m / 2) ** 2 * (width_m + hub_length_m))


def gt2_pulley(teeth: int, width_m: float, hub_diameter_m: float,
               hub_length_m: float, bore_m: float, kernel: Kernel | None = None
               ) -> PartSolid:
    """GT2 2 mm pitch pulley ENVELOPE: a cylinder at the outside diameter, a
    hub, a bore. Teeth are not cut; the outside and pitch diameters are the
    standard's, so clearance and belt length are right and the tooth form
    is absent, which `geometry_is_real=False` records."""
    if teeth < 12:
        raise ValueError("the PowerDrive table starts at 12 grooves")
    if bore_m >= hub_diameter_m or hub_diameter_m > gt2_outside_diameter_m(teeth):
        raise ValueError("bore must be inside the hub and the hub inside the teeth")
    kernel = kernel or require_kernel()
    b = kernel.module
    od = _mm(gt2_outside_diameter_m(teeth))
    W, Dh, Lh, Db = _mm(width_m), _mm(hub_diameter_m), _mm(hub_length_m), _mm(bore_m)
    disc = b.Pos(0, 0, W / 2) * b.Cylinder(od / 2, W)
    hub = b.Pos(0, 0, W + Lh / 2) * b.Cylinder(Dh / 2, Lh)
    solid = (disc + hub) - (b.Pos(0, 0, (W + Lh) / 2) * b.Cylinder(Db / 2, (W + Lh) * 3))
    return PartSolid(
        name=f"GT2_2MGT_{teeth}T", kind="pulley", solid=solid, mass_kg=0.0,
        volume_m3=solid_volume_m3(solid, kernel),
        bounding_box_m=solid_bounding_box_m(solid, kernel), geometry_is_real=False,
        source=f"pitch diameter 2 x {teeth} / pi mm, outside diameter 2 x 0.254 mm smaller, "
               f"per {SOURCES['powerdrive_gt2'].publisher} pages G-85 and G-86; teeth NOT modelled")


def heat_set_insert_volume_m3(size: str) -> float:
    od, length = HEAT_SET_INSERTS[size]
    return (math.pi * (od * 1e-3 / 2) ** 2 - math.pi * (nominal_diameter_m(size) / 2) ** 2) * length * 1e-3


def heat_set_insert(size: str, kernel: Kernel | None = None) -> PartSolid:
    """Heat-set threaded insert envelope: plain cylinder at the outer
    diameter with an unthreaded bore at the nominal thread; knurl not modelled."""
    if size not in HEAT_SET_INSERTS:
        raise KeyError(f"no insert row for {size}; have {sorted(HEAT_SET_INSERTS)}")
    kernel = kernel or require_kernel()
    b = kernel.module
    od, length = HEAT_SET_INSERTS[size]
    d = _mm(nominal_diameter_m(size))
    solid = (b.Pos(0, 0, length / 2) * b.Cylinder(od / 2, length)
             - b.Pos(0, 0, length / 2) * b.Cylinder(d / 2, length * 3))
    return PartSolid(
        name=f"INSERT_{size}x{length:g}", kind="insert", solid=solid, mass_kg=0.0,
        volume_m3=solid_volume_m3(solid, kernel),
        bounding_box_m=solid_bounding_box_m(solid, kernel), geometry_is_real=True,
        source=f"outer diameter {od} mm and length {length} mm per "
               f"{SOURCES['creative3dp_inserts'].publisher} (Ruthex / CNC Kitchen family); "
               f"knurl and thread not modelled")


def material_for(part: PartSolid):
    """(material_id or None, note) for the part kind, and the mass the
    envelope would have at that material's density, or None."""
    from core.materials import get_material

    material_id, note = MATERIAL_LINKS[part.kind]
    if material_id is None:
        return None, None, note
    density = get_material(material_id).density_kg_m3
    return material_id, part.volume_m3 * density, note
