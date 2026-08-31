"""geometry.cad_export: parametric B-rep and STEP output (optional dependency).

Parametric solids export to clean STEP, exactly. Organic and topology-optimized
shapes do not, and the mesh fallback says so rather than pretending.
"""

from .assembly import AssemblyExportReport, export_assembly_step
from .hollow_rect import (
    ExportReport,
    analytic_volume,
    build_solid,
    export_step,
    import_step,
    solid_bounding_box_m,
    solid_volume_m3,
)
from .kernel import (
    INSTALL_HINT,
    Kernel,
    find_kernel,
    kernel_available,
    require_kernel,
)
from .mesh_fallback import export_stl, mesh_from_density_field, stl_volume_m3

__all__ = [
    "AssemblyExportReport", "ExportReport", "INSTALL_HINT", "Kernel", "analytic_volume", "build_solid",
    "export_assembly_step", "export_step", "export_stl", "find_kernel", "import_step",
    "kernel_available", "mesh_from_density_field", "require_kernel",
    "solid_bounding_box_m", "solid_volume_m3", "stl_volume_m3",
]
