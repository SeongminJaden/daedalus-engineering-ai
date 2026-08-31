"""geometry.cad_export.assembly: multi-part STEP for a posed assembly.

Each link is the same parametric solid Phase 9 exports, placed at the pose
forward kinematics computed for it, and combined into one compound. The Phase 9
consistency gate is reused per part, so every component in the file is still the
part that was analysed, and the assembly mass is checked against the sum of the
link masses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .hollow_rect import METRES_TO_MM, analytic_volume, build_solid
from .kernel import Kernel, require_kernel


@dataclass
class AssemblyExportReport:
    path: Path
    kernel: str
    part_count: int
    total_volume_m3: float
    analytic_volume_m3: float
    volume_relative_error: float
    total_mass_kg: float
    analytic_mass_kg: float
    mass_relative_error: float
    part_volumes_m3: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "path": str(self.path), "kernel": self.kernel,
            "part_count": self.part_count,
            "total_volume_m3": self.total_volume_m3,
            "analytic_volume_m3": self.analytic_volume_m3,
            "volume_relative_error": self.volume_relative_error,
            "total_mass_kg": self.total_mass_kg,
            "analytic_mass_kg": self.analytic_mass_kg,
            "mass_relative_error": self.mass_relative_error,
            "part_volumes_m3": dict(self.part_volumes_m3),
        }


def _place(solid, transform: np.ndarray, kernel: Kernel):
    """Move a solid to a world pose. Translation converted to millimetres."""
    t = np.asarray(transform, dtype=np.float64)
    rotation, offset = t[:3, :3], t[:3, 3] * METRES_TO_MM

    if kernel.name == "build123d":
        b = kernel.module
        location = b.Location(
            b.Plane(origin=(float(offset[0]), float(offset[1]), float(offset[2])),
                    x_dir=tuple(float(v) for v in rotation[:, 0]),
                    z_dir=tuple(float(v) for v in rotation[:, 2])))
        return location * solid

    cq = kernel.module
    # cadquery: rotate then translate using the same basis.
    matrix = cq.Matrix([
        [float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2]),
         float(offset[0])],
        [float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2]),
         float(offset[1])],
        [float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2]),
         float(offset[2])],
    ])
    return solid.val().transformShape(matrix)


def export_assembly_step(
    assembly,
    q,
    density_kg_m3: float,
    path: str | Path,
    mass_tolerance: float = 1e-6,
    kernel: Kernel | None = None,
) -> AssemblyExportReport:
    """Write the posed assembly as a single STEP compound, mass-checked."""
    from core.assembly.kinematics import forward_kinematics

    kernel = kernel or require_kernel()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pose = forward_kinematics(assembly, q)
    placed, volumes = [], {}
    total_volume = analytic_total = 0.0

    for link in assembly.links:
        section = link.genome.section
        solid = build_solid(link.length_m, section.outer_width_m,
                            section.outer_height_m, section.wall_thickness_m,
                            kernel)
        expected = analytic_volume(link.length_m, section.outer_width_m,
                                   section.outer_height_m,
                                   section.wall_thickness_m)
        # Same per-part gate as Phase 9: each component must be the analysed one.
        from .hollow_rect import solid_volume_m3
        volume = solid_volume_m3(solid, kernel)
        if abs(volume - expected) / expected > 1e-9:
            raise ValueError(
                f"link {link.name}: CAD volume {volume:.9g} disagrees with the "
                f"analytic section volume {expected:.9g}")
        volumes[link.name] = volume
        total_volume += volume
        analytic_total += expected
        placed.append(_place(solid, pose.link_transforms[link.name], kernel))

    total_mass = total_volume * density_kg_m3
    analytic_mass = assembly.total_mass_kg(density_kg_m3)
    mass_error = abs(total_mass - analytic_mass) / analytic_mass
    if mass_error > mass_tolerance:
        raise ValueError(
            f"assembly mass {total_mass:.9g} kg disagrees with the sum of link "
            f"masses {analytic_mass:.9g} kg (relative {mass_error:.3e})")

    if kernel.name == "build123d":
        compound = kernel.module.Compound(children=placed)
        kernel.module.export_step(compound, str(path))
    else:
        cq = kernel.module
        compound = cq.Compound.makeCompound(placed)
        cq.exporters.export(cq.Workplane(obj=compound), str(path),
                            exportType="STEP")

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"assembly STEP export produced no file at {path}")

    return AssemblyExportReport(
        path=path, kernel=kernel.name, part_count=len(assembly.links),
        total_volume_m3=total_volume, analytic_volume_m3=analytic_total,
        volume_relative_error=abs(total_volume - analytic_total) / analytic_total,
        total_mass_kg=total_mass, analytic_mass_kg=analytic_mass,
        mass_relative_error=mass_error, part_volumes_m3=volumes,
    )
