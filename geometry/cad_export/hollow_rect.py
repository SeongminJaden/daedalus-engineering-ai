"""geometry.cad_export.hollow_rect: parametric solid and STEP output.

The design is a hollow rectangular prism: an outer box minus an inner cavity.
That is a parametric solid, which is the case where clean STEP output is
guaranteed. Organic and topology-optimized shapes are a different problem, and
this module does not pretend otherwise (see `stl_fallback` and the README).

THE CONSISTENCY REQUIREMENT: the exported solid must be the part that was
analysed. The volume of the B-rep, times the material density, has to match the
mass the physics used. If it does not, either the CAD is wrong or the analysis
was, and shipping the file anyway would mean manufacturing a part nobody
simulated. `export_step` checks it and refuses on a mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .kernel import Kernel, require_kernel

# The analysis works in metres. STEP has no unit of its own in the sense that
# matters here: exporters write millimetres by convention, and every CAD system
# expects that. Converting on the way out keeps the file usable and keeps the
# analysis in SI.
METRES_TO_MM = 1000.0

# Volume agreement between the B-rep and the analytic section. The B-rep is
# exact for a box minus a box, so this is a transcription check rather than a
# tolerance on a real approximation.
VOLUME_TOLERANCE = 1e-9


@dataclass
class ExportReport:
    """What was written, and the checks that were run on it."""

    path: Path
    kernel: str
    volume_m3: float
    analytic_volume_m3: float
    volume_relative_error: float
    bounding_box_m: tuple[float, float, float]
    mass_kg: float | None = None
    analytic_mass_kg: float | None = None
    mass_relative_error: float | None = None
    solid_count: int = 1

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "kernel": self.kernel,
            "volume_m3": self.volume_m3,
            "analytic_volume_m3": self.analytic_volume_m3,
            "volume_relative_error": self.volume_relative_error,
            "bounding_box_m": list(self.bounding_box_m),
            "mass_kg": self.mass_kg,
            "analytic_mass_kg": self.analytic_mass_kg,
            "mass_relative_error": self.mass_relative_error,
            "solid_count": self.solid_count,
        }


def analytic_volume(length_m: float, outer_width_m: float, outer_height_m: float,
                    wall_thickness_m: float) -> float:
    """Cross-sectional area times length, the same quantity the physics uses."""
    inner_w = outer_width_m - 2.0 * wall_thickness_m
    inner_h = outer_height_m - 2.0 * wall_thickness_m
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("wall thickness leaves no cavity")
    area = outer_width_m * outer_height_m - inner_w * inner_h
    return area * length_m


def build_solid(length_m: float, outer_width_m: float, outer_height_m: float,
                wall_thickness_m: float, kernel: Kernel | None = None):
    """Build the hollow prism B-rep. Dimensions in metres, model in mm.

    Axes match the analysis: x along the link, y the section height (the load
    direction), z the section width.
    """
    kernel = kernel or require_kernel()
    inner_w = outer_width_m - 2.0 * wall_thickness_m
    inner_h = outer_height_m - 2.0 * wall_thickness_m
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError(
            f"wall thickness {wall_thickness_m} leaves no cavity in a "
            f"{outer_width_m} x {outer_height_m} section")

    length = length_m * METRES_TO_MM
    outer_w = outer_width_m * METRES_TO_MM
    outer_h = outer_height_m * METRES_TO_MM
    cavity_w = inner_w * METRES_TO_MM
    cavity_h = inner_h * METRES_TO_MM

    if kernel.name == "build123d":
        b = kernel.module
        outer = b.Box(length, outer_h, outer_w)
        # The cavity runs the full length, so it is made longer than the part
        # and trimmed by the subtraction; that avoids a coincident face pair at
        # the ends, which is a classic source of invalid B-rep.
        cavity = b.Box(length * 1.1, cavity_h, cavity_w)
        return outer - cavity

    cq = kernel.module
    outer = cq.Workplane("YZ").rect(outer_h, outer_w).extrude(length)
    cavity = cq.Workplane("YZ").rect(cavity_h, cavity_w).extrude(length * 1.1)
    return outer.cut(cavity)


def solid_volume_m3(solid, kernel: Kernel) -> float:
    """Volume of the B-rep, converted back to cubic metres."""
    if kernel.name == "build123d":
        volume_mm3 = solid.volume
    else:
        volume_mm3 = solid.val().Volume()
    return float(volume_mm3) / (METRES_TO_MM ** 3)


def solid_bounding_box_m(solid, kernel: Kernel) -> tuple[float, float, float]:
    if kernel.name == "build123d":
        bb = solid.bounding_box()
        size = (bb.size.X, bb.size.Y, bb.size.Z)
    else:
        bb = solid.val().BoundingBox()
        size = (bb.xlen, bb.ylen, bb.zlen)
    return tuple(float(v) / METRES_TO_MM for v in size)


def count_solids(solid, kernel: Kernel) -> int:
    try:
        if kernel.name == "build123d":
            return len(solid.solids())
        return len(solid.val().Solids())
    except Exception:      # noqa: BLE001 - a kernel that cannot count is not fatal
        return 1


def export_step(
    length_m: float,
    outer_width_m: float,
    outer_height_m: float,
    wall_thickness_m: float,
    path: str | Path,
    density_kg_m3: float | None = None,
    analytic_mass_kg: float | None = None,
    mass_tolerance: float = 1e-6,
    kernel: Kernel | None = None,
) -> ExportReport:
    """Write a STEP file and verify it describes the analysed part.

    Raises if the B-rep volume disagrees with the analytic section, or if the
    implied mass disagrees with the mass the physics used. A file that passes
    those checks is the same part the analysis saw.
    """
    kernel = kernel or require_kernel()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    solid = build_solid(length_m, outer_width_m, outer_height_m,
                        wall_thickness_m, kernel)
    volume = solid_volume_m3(solid, kernel)
    expected = analytic_volume(length_m, outer_width_m, outer_height_m,
                               wall_thickness_m)
    volume_error = abs(volume - expected) / expected
    if volume_error > VOLUME_TOLERANCE:
        raise ValueError(
            f"CAD volume {volume:.9g} m^3 disagrees with the analytic section "
            f"volume {expected:.9g} m^3 (relative {volume_error:.3e}); the "
            "exported solid is not the analysed part")

    mass = mass_error = None
    if density_kg_m3 is not None:
        mass = volume * density_kg_m3
        if analytic_mass_kg is not None:
            mass_error = abs(mass - analytic_mass_kg) / analytic_mass_kg
            if mass_error > mass_tolerance:
                raise ValueError(
                    f"CAD mass {mass:.9g} kg disagrees with the analysed mass "
                    f"{analytic_mass_kg:.9g} kg (relative {mass_error:.3e}); "
                    "refusing to export a part that was not the one simulated")

    if kernel.name == "build123d":
        kernel.module.export_step(solid, str(path))
    else:
        kernel.module.exporters.export(solid, str(path), exportType="STEP")

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"STEP export produced no file at {path}")

    return ExportReport(
        path=path,
        kernel=kernel.name,
        volume_m3=volume,
        analytic_volume_m3=expected,
        volume_relative_error=volume_error,
        bounding_box_m=solid_bounding_box_m(solid, kernel),
        mass_kg=mass,
        analytic_mass_kg=analytic_mass_kg,
        mass_relative_error=mass_error,
        solid_count=count_solids(solid, kernel),
    )


def import_step(path: str | Path, kernel: Kernel | None = None):
    """Read a STEP file back. Used to verify what was written."""
    kernel = kernel or require_kernel()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if kernel.name == "build123d":
        return kernel.module.import_step(str(path))
    return kernel.module.importers.importStep(str(path))
