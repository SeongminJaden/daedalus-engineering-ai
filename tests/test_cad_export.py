"""Phase 9 verification: parametric CAD and STEP output.

The point of these tests is not that a file was written. It is that the file
describes **the part that was analysed**. A STEP file whose volume disagrees
with the section the physics integrated would mean manufacturing something
nobody simulated, so that agreement is checked rather than assumed.

CAD is an optional dependency. Every test here skips cleanly when no kernel is
installed, and the rest of the suite must still pass in that state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.materials import get_material  # noqa: E402
from geometry.cad_export import INSTALL_HINT, find_kernel, kernel_available  # noqa: E402
from optimization.constraints import build_optimization_problem, evaluate_design  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402

cad = pytest.mark.skipif(not kernel_available(),
                         reason="no CAD kernel installed (optional dependency)")

# The Phase 7.5 optimum, the design that actually passed the 3D FEM gate.
LENGTH_M = 0.5
SECTION = (0.010, 0.0816185, 0.001)      # b, h, t


def analytic_area(b, h, t):
    return b * h - (b - 2 * t) * (h - 2 * t)


# =========================================================================== #
# the optional dependency behaves like one
# =========================================================================== #
def test_kernel_detection_never_raises():
    """Import-time detection must be safe whether or not CAD is installed."""
    assert isinstance(kernel_available(), bool)


def test_install_hint_is_actionable():
    assert "requirements-cad.txt" in INSTALL_HINT
    assert "does not require it" in INSTALL_HINT


def test_missing_kernel_raises_a_helpful_error(monkeypatch):
    import geometry.cad_export.kernel as kernel_mod
    monkeypatch.setattr(kernel_mod, "find_kernel", lambda: None)
    with pytest.raises(ModuleNotFoundError, match="requirements-cad.txt"):
        kernel_mod.require_kernel()


# =========================================================================== #
# the analytic volume, independent of any CAD kernel
# =========================================================================== #
def test_analytic_volume_matches_section_times_length():
    from geometry.cad_export import analytic_volume
    b, h, t = SECTION
    assert analytic_volume(LENGTH_M, b, h, t) == pytest.approx(
        analytic_area(b, h, t) * LENGTH_M, rel=1e-15)


def test_analytic_volume_rejects_an_impossible_wall():
    from geometry.cad_export import analytic_volume
    with pytest.raises(ValueError, match="cavity"):
        analytic_volume(0.5, 0.02, 0.02, 0.02)


# =========================================================================== #
# B-rep and STEP
# =========================================================================== #
@cad
def test_solid_volume_matches_the_analytic_section():
    """A box minus a box is exact, so this is a transcription check: any real
    disagreement means the CAD is not the analysed geometry."""
    from geometry.cad_export import build_solid, solid_volume_m3
    kernel = find_kernel()
    b, h, t = SECTION
    volume = solid_volume_m3(build_solid(LENGTH_M, b, h, t, kernel), kernel)
    assert volume == pytest.approx(analytic_area(b, h, t) * LENGTH_M, rel=1e-12)


@cad
def test_solid_bounding_box_matches_the_design_dimensions():
    from geometry.cad_export import build_solid, solid_bounding_box_m
    kernel = find_kernel()
    b, h, t = SECTION
    bbox = solid_bounding_box_m(build_solid(LENGTH_M, b, h, t, kernel), kernel)
    assert bbox[0] == pytest.approx(LENGTH_M, rel=1e-9)
    assert bbox[1] == pytest.approx(h, rel=1e-9)
    assert bbox[2] == pytest.approx(b, rel=1e-9)


@cad
def test_step_round_trip(tmp_path):
    """Write, read back, and confirm it is still one solid of the same volume."""
    from geometry.cad_export import (
        export_step, import_step, solid_bounding_box_m, solid_volume_m3,
    )
    kernel = find_kernel()
    b, h, t = SECTION
    path = tmp_path / "link.step"
    report = export_step(LENGTH_M, b, h, t, path)

    assert path.exists() and path.stat().st_size > 0
    assert report.solid_count == 1

    reimported = import_step(path, kernel)
    assert solid_volume_m3(reimported, kernel) == pytest.approx(
        report.volume_m3, rel=1e-9)
    bbox = solid_bounding_box_m(reimported, kernel)
    assert bbox[0] == pytest.approx(LENGTH_M, rel=1e-6)
    assert bbox[1] == pytest.approx(h, rel=1e-6)


@cad
def test_step_file_declares_itself_as_step(tmp_path):
    from geometry.cad_export import export_step
    b, h, t = SECTION
    path = tmp_path / "link.step"
    export_step(LENGTH_M, b, h, t, path)
    head = path.read_text(errors="ignore")[:200]
    assert "ISO-10303" in head, "file does not identify as a STEP part 21 file"


# =========================================================================== #
# THE CONSISTENCY GATE: the exported part is the analysed part
# =========================================================================== #
@cad
def test_exported_mass_matches_the_analysed_mass(tmp_path):
    """The gate. CAD volume times density must equal the mass the physics used.

    Tolerance note: the analysis mass comes from the fp32 GPU kernel, which
    carries ~1e-7 relative precision. The CAD volume is float64 and exact for a
    box. So the residual here is set by the PHYSICS precision, not by the
    geometry, and a tolerance tighter than fp32 would be measuring the wrong
    thing.
    """
    from geometry.cad_export import export_step
    problem = build_mvp_problem()
    op = build_optimization_problem(problem)
    material = get_material(problem.material_id)
    b, h, t = SECTION

    analysis = evaluate_design(op, np.array([b, h, t]))
    report = export_step(LENGTH_M, b, h, t, tmp_path / "link.step",
                         density_kg_m3=material.density_kg_m3,
                         analytic_mass_kg=analysis.mass_kg)
    assert report.mass_relative_error < 1e-6
    assert report.mass_kg == pytest.approx(analysis.mass_kg, rel=1e-6)


@cad
def test_export_refuses_a_mass_mismatch(tmp_path):
    """If the geometry and the analysis disagree, the file must not be written.

    Shipping it anyway would mean handing a manufacturer a part that nobody
    simulated.
    """
    from geometry.cad_export import export_step
    b, h, t = SECTION
    path = tmp_path / "wrong.step"
    with pytest.raises(ValueError, match="refusing to export"):
        export_step(LENGTH_M, b, h, t, path,
                    density_kg_m3=2810.0,
                    analytic_mass_kg=0.5)        # deliberately wrong
    assert not path.exists(), "a rejected export must leave no file behind"


@cad
def test_export_rejects_impossible_geometry(tmp_path):
    from geometry.cad_export import export_step
    with pytest.raises(ValueError, match="cavity"):
        export_step(0.5, 0.02, 0.02, 0.02, tmp_path / "bad.step")


# =========================================================================== #
# determinism
# =========================================================================== #
@cad
def test_export_is_deterministic(tmp_path):
    from geometry.cad_export import export_step
    b, h, t = SECTION
    a = export_step(LENGTH_M, b, h, t, tmp_path / "a.step")
    c = export_step(LENGTH_M, b, h, t, tmp_path / "b.step")
    assert a.volume_m3 == c.volume_m3
    assert a.bounding_box_m == c.bounding_box_m


@cad
def test_different_designs_give_different_volumes(tmp_path):
    from geometry.cad_export import export_step
    b, h, t = SECTION
    thin = export_step(LENGTH_M, b, h, t, tmp_path / "thin.step")
    thick = export_step(LENGTH_M, b, h, 0.002, tmp_path / "thick.step")
    assert thick.volume_m3 > thin.volume_m3


# =========================================================================== #
# STL fallback, and the boundary of the STEP guarantee
# =========================================================================== #
@cad
def test_stl_tessellation_is_watertight_and_close(tmp_path):
    from geometry.cad_export import build_solid, export_stl, stl_volume_m3
    kernel = find_kernel()
    b, h, t = SECTION
    solid = build_solid(LENGTH_M, b, h, t, kernel)
    path = export_stl(solid, tmp_path / "link.stl", kernel)
    assert path.stat().st_size > 0
    # stl_volume_m3 raises if the mesh is not watertight
    volume = stl_volume_m3(path)
    assert volume == pytest.approx(analytic_area(b, h, t) * LENGTH_M, rel=1e-6)


def test_topology_meshing_is_an_honest_stub():
    """The organic path is not implemented, and says so rather than emitting
    geometry that looks manufacturable and is not."""
    from geometry.cad_export import mesh_from_density_field
    with pytest.raises(NotImplementedError, match="surface reconstruction"):
        mesh_from_density_field(np.zeros((4, 4, 4)), 1e-3)
