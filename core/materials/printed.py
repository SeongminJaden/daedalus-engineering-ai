"""Printed parts are not the bulk material, and the difference is a table.

The material database stores isotropic bulk values with a note saying a
printed part is not isotropic and that the bulk numbers are an upper bound.
This module makes that enforceable: direction dependent values where a data
sheet prints them, a refusal where none does, and a check that a design using
bulk values for a printed part is told it is optimistic.

WHAT THE DATA SHEETS ACTUALLY SAY
=================================
Two are read here, both direction resolved, and they disagree about how much
anisotropy is normal, which is the point.

    Stratasys ABS-M30, F900 with a T16 tip: yield 30.8 MPa on edge (XZ) and
    27.5 upright (ZX), modulus 2.40 and 2.30 GPa, elongation at break 8.1 and
    1.8 percent. The SAME material on an F770: 32.5 and 23.1 MPa, 2.00 and
    1.78 GPa. So the strength ratio between orientations is 0.89 on one
    machine and 0.71 on the other, and a single anisotropy factor for "FDM
    ABS" would be wrong on at least one of them.

    EOS PA 2200 Speed 1.0, laser sintered: tensile modulus 1600, 1600 and 1550 MPa in
    X, Y and Z, strength 48, 48 and 42 MPa, strain at break 18, 18 and 4
    percent. The strength falls 12 percent across the layers and the ductility
    falls by four fifths, which is the number that matters for a part that
    must survive an impact.

WHAT THIS MODULE REFUSES
========================
Everything not printed. There is no interpolation between orientations, no
factor applied to a material that has no measured table, and no default
anisotropy. `printed_strength_pa` for a material with no entry raises with the
material name, and `bulk_is_an_upper_bound` states in one place what the
database note says in prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BuildOrientation(str, Enum):
    """How the test bar sat on the plate.

    The two naming conventions in the sheets read here are kept apart rather
    than merged: Stratasys names a plane (XZ is on edge, ZX is upright) and
    EOS names an axis (X and Y in plane, Z across the layers). Merging them
    would lose which one the number came from.
    """

    FLAT_XY = "flat_xy"
    ON_EDGE_XZ = "on_edge_xz"
    UPRIGHT_ZX = "upright_zx"
    AXIS_X = "axis_x"
    AXIS_Y = "axis_y"
    AXIS_Z = "axis_z"


@dataclass(frozen=True)
class PrintedProperty:
    """One direction of one printed material, as its sheet prints it."""

    orientation: BuildOrientation
    tensile_strength_pa: float | None = None
    yield_strength_pa: float | None = None
    modulus_pa: float | None = None
    elongation_at_break_percent: float | None = None
    printed_as: str = ""


@dataclass
class PrintedMaterial:
    """A printed material, its process, its machine, and its source."""

    id: str
    base_material_id: str
    process: str
    machine: str
    document_title: str
    document_url: str
    read_on: str
    standard: str
    directions: dict[BuildOrientation, PrintedProperty] = field(default_factory=dict)
    notes: str = ""

    def orientations(self) -> list[BuildOrientation]:
        return sorted(self.directions, key=lambda o: o.value)

    def strength_pa(self, orientation: BuildOrientation) -> float:
        entry = self.directions.get(orientation)
        if entry is None or entry.tensile_strength_pa is None:
            raise MissingPrintedData(
                f"{self.id} has no printed tensile strength for "
                f"{orientation.value}; the sheet prints "
                f"{[o.value for o in self.orientations()]}")
        return entry.tensile_strength_pa

    def weakest(self) -> PrintedProperty:
        """The orientation a part should be designed against unless the build
        direction is fixed and known."""
        entries = [e for e in self.directions.values()
                   if e.tensile_strength_pa is not None]
        if not entries:
            raise MissingPrintedData(f"{self.id} prints no tensile strengths")
        return min(entries, key=lambda e: e.tensile_strength_pa)

    def anisotropy_ratio(self) -> float:
        """Weakest over strongest printed tensile strength."""
        values = [e.tensile_strength_pa for e in self.directions.values()
                  if e.tensile_strength_pa is not None]
        if len(values) < 2:
            raise MissingPrintedData(
                f"{self.id} prints fewer than two orientations, so it states "
                f"no anisotropy")
        return min(values) / max(values)


class MissingPrintedData(ValueError):
    """A printed value that no sheet in this module prints."""


STRATASYS_ABS_M30_F900 = PrintedMaterial(
    id="abs_m30_fdm_f900_t16",
    base_material_id="abs",
    process="FDM", machine="Stratasys F900 with a T16 tip",
    document_title="ABS-M30 Data Sheet, table 5, mechanical properties (F900, T16 tip)",
    document_url="https://www.stratasys.com/siteassets/materials/materials-catalog/fdm-materials/abs-m30/mds_fdm_abs-m30_0921a.pdf",
    read_on="2026-09-03",
    standard="ASTM D638 tensile, ASTM D790 procedure A flexural",
    directions={
        BuildOrientation.ON_EDGE_XZ: PrintedProperty(
            orientation=BuildOrientation.ON_EDGE_XZ,
            yield_strength_pa=30.8e6, tensile_strength_pa=28.1e6,
            modulus_pa=2.40e9, elongation_at_break_percent=8.1,
            printed_as="yield 30.8 (0.85) MPa, break 28.1 (0.58) MPa, "
                       "modulus 2.40 (0.080) GPa, elongation at break 8.1 (1.5) %"),
        BuildOrientation.UPRIGHT_ZX: PrintedProperty(
            orientation=BuildOrientation.UPRIGHT_ZX,
            yield_strength_pa=27.5e6, tensile_strength_pa=26.8e6,
            modulus_pa=2.30e9, elongation_at_break_percent=1.8,
            printed_as="yield 27.5 (0.28) MPa, break 26.8 (0.84) MPa, "
                       "modulus 2.30 (0.16) GPa, elongation at break 1.8 (0.31) %"),
    },
    notes="Values in the sheet carry standard deviations, kept in printed_as.")

STRATASYS_ABS_M30_F770 = PrintedMaterial(
    id="abs_m30_fdm_f770",
    base_material_id="abs",
    process="FDM", machine="Stratasys F770",
    document_title="ABS-M30 Data Sheet, table 6, mechanical properties (F770)",
    document_url="https://www.stratasys.com/siteassets/materials/materials-catalog/fdm-materials/abs-m30/mds_fdm_abs-m30_0921a.pdf",
    read_on="2026-09-03",
    standard="ASTM D638 tensile, ASTM D790 procedure A flexural",
    directions={
        BuildOrientation.ON_EDGE_XZ: PrintedProperty(
            orientation=BuildOrientation.ON_EDGE_XZ,
            yield_strength_pa=32.5e6, tensile_strength_pa=27.6e6,
            modulus_pa=2.00e9, elongation_at_break_percent=4.5,
            printed_as="yield 32.5 (1.7) MPa, break 27.6 (2.4) MPa, "
                       "modulus 2.00 GPa, elongation at break 4.5 (1.2) %"),
        BuildOrientation.UPRIGHT_ZX: PrintedProperty(
            orientation=BuildOrientation.UPRIGHT_ZX,
            yield_strength_pa=23.1e6, tensile_strength_pa=22.9e6,
            modulus_pa=1.78e9, elongation_at_break_percent=1.6,
            printed_as="yield 23.1 (1.3) MPa, break 22.9 (1.2) MPa, "
                       "modulus 1.78 GPa, elongation at break 1.6 (0.2) %"),
    },
    notes="The same material as the F900 entry on a different machine. The "
          "yield ratio between orientations is 0.71 here and 0.89 there, "
          "which is why this module stores machines and not materials.")

EOS_PA2200 = PrintedMaterial(
    id="pa2200_sls_eos",
    base_material_id="pa12",
    process="SLS", machine="EOS laser sintering, PA 2200",
    document_title="EOS material data sheet, PA 2200",
    document_url="https://www.metcompany.eu/media/Tecnologia-SLS-PA2200.pdf",
    read_on="2026-09-03",
    standard="ISO 527 tensile, ISO 178 flexural",
    directions={
        BuildOrientation.AXIS_X: PrintedProperty(
            orientation=BuildOrientation.AXIS_X, tensile_strength_pa=48e6,
            modulus_pa=1600e6, elongation_at_break_percent=18.0,
            printed_as="X direction: modulus 1600 MPa, strength 48 MPa, "
                       "strain at break 18 %"),
        BuildOrientation.AXIS_Y: PrintedProperty(
            orientation=BuildOrientation.AXIS_Y, tensile_strength_pa=48e6,
            modulus_pa=1600e6, elongation_at_break_percent=18.0,
            printed_as="Y direction: modulus 1600 MPa, strength 48 MPa, "
                       "strain at break 18 %"),
        BuildOrientation.AXIS_Z: PrintedProperty(
            orientation=BuildOrientation.AXIS_Z, tensile_strength_pa=42e6,
            modulus_pa=1550e6, elongation_at_break_percent=4.0,
            printed_as="Z direction: modulus 1550 MPa, strength 42 MPa, "
                       "strain at break 4 %"),
    },
    notes="Strength falls 12 percent across the layers and strain at break "
          "falls by four fifths, which is the number that matters for impact.")

PRINTED_MATERIALS: dict[str, PrintedMaterial] = {
    m.id: m for m in (STRATASYS_ABS_M30_F900, STRATASYS_ABS_M30_F770, EOS_PA2200)}


def printed_material(part_id: str) -> PrintedMaterial:
    if part_id not in PRINTED_MATERIALS:
        raise MissingPrintedData(
            f"no printed data for {part_id!r}; have "
            f"{sorted(PRINTED_MATERIALS)}")
    return PRINTED_MATERIALS[part_id]


def printed_strength_pa(part_id: str, orientation: BuildOrientation) -> float:
    return printed_material(part_id).strength_pa(orientation)


def bulk_is_an_upper_bound(material_id: str) -> str:
    """The sentence the database note makes, in one place a check can use.

    A printed part designed against bulk properties is designed against
    numbers no printed specimen reached. The break strength ratios measured
    here are 0.954 and 0.83 for the same ABS on two machines, and 0.875 for
    PA 2200 across the layers; on yield the ABS ratios are 0.89 and 0.71.
    """
    return (f"{material_id}: the stored values are bulk material and are an "
            f"UPPER BOUND on a printed part. Measured ratios between printed "
            f"orientations are 0.83 to 0.954 on break for ABS-M30 and 0.875 "
            f"for PA 2200; against bulk the gap is larger. Designing a printed "
            f"part against these numbers is optimistic, and optimism is not a "
            f"safety factor")
