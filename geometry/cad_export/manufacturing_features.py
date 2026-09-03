"""Turning an analysis shape into something a shop could quote.

The part families produce clean prismatic solids with sharp interior corners,
no fastener features and no tolerances, which is the right input for a solver
and the wrong output for a machinist. This module adds the three things a
drawing needs and measures what each one does.

WHAT IS ADDED
=============
fillets at the re-entrant edges
    A sharp interior corner is a stress singularity, so the peak stress the
    labeller reports there is not converged and no fillet radius can be
    justified from it. What CAN be measured is the change: solve the same part
    at several radii and report the peak against the sharp case at a fixed
    mesh. That is a comparison, not a stress concentration factor, and the
    docstring of `fillet_study` says so.

fastener features
    A clearance hole and a counterbore sized from the same ISO 4762 table the
    catalogue screws come from, so the hole and the screw cannot disagree.
    Sizes come from `standard_parts`, not from a second table.

tolerances
    ISO 2768 general tolerances by length band and ISO 286 fits for a named
    feature, attached to the part as a `DrawingNotes` record. STEP AP203, which
    this project writes, carries no tolerance information at all, so the notes
    are written beside the STEP file as JSON and the limitation is stated
    rather than worked around.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: ISO 2768-1 general tolerances for linear dimensions, millimetres.
#: Bands are (over, up to] in mm; the columns are the fine, medium, coarse and
#: very coarse classes. Values are the printed table.
ISO_2768_LINEAR: list[tuple[float, float, dict[str, float | None]]] = [
    (0.5, 3.0, {"f": 0.05, "m": 0.1, "c": 0.2, "v": None}),
    (3.0, 6.0, {"f": 0.05, "m": 0.1, "c": 0.3, "v": 0.5}),
    (6.0, 30.0, {"f": 0.1, "m": 0.2, "c": 0.5, "v": 1.0}),
    (30.0, 120.0, {"f": 0.15, "m": 0.3, "c": 0.8, "v": 1.5}),
    (120.0, 400.0, {"f": 0.2, "m": 0.5, "c": 1.2, "v": 2.5}),
    (400.0, 1000.0, {"f": 0.3, "m": 0.8, "c": 2.0, "v": 4.0}),
    (1000.0, 2000.0, {"f": 0.5, "m": 1.2, "c": 3.0, "v": 6.0}),
]

ISO_2768_SOURCE = (
    "ISO 2768-1 general tolerances for linear dimensions, standard table, "
    "classes f (fine), m (medium), c (coarse), v (very coarse)")


class ToleranceOutOfRange(ValueError):
    """A dimension outside the range the general tolerance table covers."""


def general_tolerance_mm(nominal_mm: float, tolerance_class: str = "m") -> float:
    """The ISO 2768 general tolerance for a linear dimension.

    Refuses below 0.5 mm and above 2000 mm, which is where the table stops,
    and refuses the very coarse class under 3 mm, where the table has no
    entry rather than a large one.
    """
    if tolerance_class not in ("f", "m", "c", "v"):
        raise ValueError(f"unknown tolerance class {tolerance_class!r}")
    for low, high, values in ISO_2768_LINEAR:
        if low < nominal_mm <= high:
            value = values[tolerance_class]
            if value is None:
                raise ToleranceOutOfRange(
                    f"ISO 2768 class {tolerance_class} has no entry for "
                    f"{nominal_mm} mm; the table starts at 3 mm for that class")
            return value
    raise ToleranceOutOfRange(
        f"{nominal_mm} mm is outside the ISO 2768 linear table "
        f"(0.5 mm to 2000 mm)")


# --------------------------------------------------------- fastener features

@dataclass(frozen=True)
class FastenerFeature:
    """A hole for one screw, sized from the screw's own table."""

    size: str
    clearance_diameter_mm: float
    counterbore_diameter_mm: float
    counterbore_depth_mm: float
    fit: str                    # close, normal or free clearance
    source: str

    def as_dict(self) -> dict:
        return asdict(self)


#: ISO 273 clearance holes, millimetres, for the sizes the catalogue holds.
#: (close, normal, free)
ISO_273_CLEARANCE: dict[str, tuple[float, float, float]] = {
    "M3": (3.2, 3.4, 3.6), "M4": (4.3, 4.5, 4.8), "M5": (5.3, 5.5, 5.8),
    "M6": (6.4, 6.6, 7.0), "M8": (8.4, 9.0, 10.0), "M10": (10.5, 11.0, 12.0),
}

ISO_273_SOURCE = "ISO 273 clearance holes for metric bolts, standard table"


def fastener_feature(size: str, fit: str = "normal",
                     counterbore_clearance_mm: float = 0.4) -> FastenerFeature:
    """Clearance hole and counterbore for one ISO 4762 screw.

    The counterbore diameter is the screw's head diameter from the same table
    the catalogue screw is built from, plus a clearance, and its depth is the
    head height plus the same clearance. Taking the head size from anywhere
    else would let the hole and the screw disagree.
    """
    from geometry.cad_export.standard_parts import ISO_4762

    if size not in ISO_273_CLEARANCE or size not in ISO_4762:
        raise KeyError(f"no clearance data for {size!r}; have "
                       f"{sorted(set(ISO_273_CLEARANCE) & set(ISO_4762))}")
    index = {"close": 0, "normal": 1, "free": 2}
    if fit not in index:
        raise ValueError(f"fit must be close, normal or free, not {fit!r}")
    _pitch, head_diameter, head_height, _socket = ISO_4762[size]
    return FastenerFeature(
        size=size,
        clearance_diameter_mm=ISO_273_CLEARANCE[size][index[fit]],
        counterbore_diameter_mm=head_diameter + counterbore_clearance_mm,
        counterbore_depth_mm=head_height + counterbore_clearance_mm,
        fit=fit,
        source=f"{ISO_273_SOURCE}; head sizes from the ISO 4762 table in "
               f"geometry/cad_export/standard_parts.py")


# ------------------------------------------------------------ drawing notes

@dataclass
class DrawingNotes:
    """What a drawing would carry and a STEP AP203 file cannot.

    Written beside the STEP file rather than into it. AP203 has no tolerance
    entity; AP242 does, and this project does not write AP242. Saying so is
    the honest form of the limitation.
    """

    part_id: str
    general_tolerance_class: str
    general_tolerance_source: str = ISO_2768_SOURCE
    dimensions_mm: dict[str, float] = field(default_factory=dict)
    fits: list[dict] = field(default_factory=list)
    fasteners: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def tolerance_for(self, name: str) -> float:
        return general_tolerance_mm(self.dimensions_mm[name],
                                    self.general_tolerance_class)

    def with_fit(self, feature: str, nominal_mm: float, hole_grade: int,
                 shaft_letter: str, shaft_grade: int) -> "DrawingNotes":
        """Attach an ISO 286 fit to a named feature, computed by the fits
        module rather than restated here."""
        from physics.elements.fits import fit as iso_fit

        result = iso_fit(nominal_mm, hole_grade, shaft_letter, shaft_grade)
        self.fits.append({
            "feature": feature, "nominal_mm": nominal_mm,
            "designation": f"H{hole_grade}/{shaft_letter}{shaft_grade}",
            "type": result.fit_type.value,
            "min_clearance_mm": result.min_clearance_mm,
            "max_clearance_mm": result.max_clearance_mm,
            "source": "ISO 286 limits computed in physics/elements/fits.py"})
        return self

    def as_dict(self) -> dict:
        data = asdict(self)
        data["step_limitation"] = (
            "written beside the STEP file: this project writes AP203, which "
            "carries no tolerance entity. Reading these values requires this "
            "JSON, not the STEP.")
        return data

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path


# --------------------------------------------------------- the fillet study

@dataclass
class FilletMeasurement:
    radius_m: float
    peak_von_mises_pa: float
    max_displacement_m: float
    mass_kg: float
    elements: int
    mesh_size_m: float

    def as_dict(self) -> dict:
        return asdict(self)


def fillet_study(build, radii, material, mesh_size_m: float = 0.004,
                 total_load_n: float = -500.0, load_direction: int = 1,
                 directory: str | Path | None = None) -> list[FilletMeasurement]:
    """Solve the same part at several fillet radii and report the peak stress.

    `build` takes a radius in metres and returns a solid, so the caller owns
    the geometry. Every case is meshed at the SAME target size, because the
    comparison is between radii and a changing mesh would confound it.

    WHAT THIS IS NOT. A stress concentration factor. The sharp corner case is
    a singularity whose peak stress rises without bound as the mesh is
    refined, so the ratio between a filleted case and a sharp one is a
    property of the mesh as much as of the geometry. Between two FILLETED
    radii the comparison is meaningful, which is why the sharp case is
    reported and marked rather than used as the denominator.
    """
    import tempfile

    import numpy as np

    from nodes import calculix as ccx
    from nodes import gmsh_node as gm
    from geometry.cad_export.kernel import require_kernel

    kernel = require_kernel()
    root = Path(directory) if directory else Path(tempfile.mkdtemp())
    root.mkdir(parents=True, exist_ok=True)
    out: list[FilletMeasurement] = []
    for radius in radii:
        solid = build(radius)
        path = root / f"fillet_{radius * 1e3:.2f}mm.step"
        kernel.module.export_step(solid, str(path))
        mesh = gm.tetrahedral_mesh_from_step(str(path), mesh_size_m, order=2)
        fixed = mesh.nodes_at_extreme(0, "min")
        loaded = mesh.nodes_at_extreme(0, "max")
        result = ccx.solve(mesh, material.youngs_modulus_pa,
                           material.poisson_ratio, fixed, loaded,
                           total_load_n=total_load_n,
                           load_direction=load_direction,
                           element_type=ccx.ElementType.C3D10)
        volume = float(solid.volume) * 1e-9      # build123d works in mm
        out.append(FilletMeasurement(
            radius_m=float(radius),
            peak_von_mises_pa=float(result.max_von_mises_pa()),
            max_displacement_m=float(np.abs(result.displacements).max()),
            mass_kg=volume * material.density_kg_m3,
            elements=mesh.n_elements, mesh_size_m=mesh_size_m))
    return out


def format_fillet_table(rows: list[FilletMeasurement]) -> str:
    lines = ["| fillet radius mm | peak von Mises MPa | against the sharpest | "
             "max displacement m | mass kg | elements |", "|" + "---|" * 6]
    reference = rows[0].peak_von_mises_pa if rows else 1.0
    for row in rows:
        lines.append(
            f"| {row.radius_m * 1e3:.2f} | {row.peak_von_mises_pa / 1e6:.2f} | "
            f"{row.peak_von_mises_pa / reference:.2f} | "
            f"{row.max_displacement_m:.3e} | {row.mass_kg:.4f} | {row.elements} |")
    return "\n".join(lines)
