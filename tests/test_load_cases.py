"""The five load cases, each checked against a closed form before it labels.

A load case that has not been checked on a shape with a known answer would
label ten thousand parts with a number nobody has verified. So each case is
run here on a slender square bar where a hand calculation exists, and the
floors are set above the errors measured (axial 0.36 percent, torsion 1.5
percent against an approximate Roark constant, thermal gradient 2.2 percent,
combined agreeing with its two components to 0.1 and 1.6 percent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.materials import get_material
from core.part_dataset import FAMILIES, make_part
from core.part_dataset.labeller import (LoadCase, LoadKind, PRIMARY_LABEL,
                                        cantilever_labels, labelling_available,
                                        torque_forces)
from geometry.cad_export.kernel import kernel_available
from nodes import step_analyzer as sa

pytestmark = pytest.mark.slow
requires_all = pytest.mark.skipif(
    not (kernel_available() and sa.is_available() and labelling_available()),
    reason="needs build123d, OCP, gmsh and CalculiX")

L, H, W = 0.2, 0.02, 0.02


@pytest.fixture(scope="module")
def bar(tmp_path_factory):
    if not (kernel_available() and sa.is_available() and labelling_available()):
        pytest.skip("needs build123d, OCP, gmsh and CalculiX")
    root = tmp_path_factory.mktemp("bar")
    record, _ = make_part(FAMILIES["box"], dict(length_m=L, height_m=H, width_m=W),
                          root, labelled=False)
    material = get_material("al_7075_t6")

    def run(case: LoadCase):
        return cantilever_labels(root / f"{record.part_id}.step",
                                 record.geometry.volume_m3,
                                 record.geometry.bounding_box_m, material,
                                 case).labels
    return run, material


def test_torque_forces_sum_to_the_torque_and_to_no_net_force():
    rng = np.random.default_rng(0)
    points = np.column_stack([np.full(40, 0.2), rng.uniform(-0.01, 0.01, 40),
                              rng.uniform(-0.01, 0.01, 40)])
    forces, _, _ = torque_forces(points, 3.0, axis=0)
    r = points - points.mean(axis=0)
    r[:, 0] = 0.0
    assert np.cross(r, forces)[:, 0].sum() == pytest.approx(3.0)
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-9 * 3.0 / 0.01)


def test_every_kind_names_its_primary_label_and_scaling():
    for kind in LoadKind:
        name, unit, scaling = PRIMARY_LABEL[kind]
        assert name.endswith("_m") or name.endswith("_rad")
        assert scaling in ("inverse_modulus", "inverse_shear_modulus",
                           "expansion")
        assert LoadCase(kind=kind).as_dict()["load_kind"] == kind.value


@requires_all
def test_axial_elongation_is_f_l_over_e_a(bar):
    """Measured 0.36 percent, mesh sensitivity 0.08 percent."""
    run, material = bar
    force = 1000.0
    labels = run(LoadCase(total_load_n=force, kind=LoadKind.AXIAL))
    exact = force * L / (material.youngs_modulus_pa * H * W)
    assert labels["elongation_m"]["value"] == pytest.approx(exact, rel=0.01)
    assert labels["elongation_m"]["scaling"] == "inverse_modulus"
    assert labels["load_case"]["load_kind"] == "axial"


@requires_all
def test_torsion_of_a_square_bar_matches_roark(bar):
    """theta = T L / (G J) with J = 0.1406 a^4 for a square, which is itself a
    series approximation; measured 1.5 percent, mesh sensitivity 2.5 percent."""
    run, material = bar
    torque = 2.0
    labels = run(LoadCase(kind=LoadKind.TORSION, torque_nm=torque))
    g = material.youngs_modulus_pa / (2.0 * (1.0 + material.poisson_ratio))
    exact = torque * L / (g * 0.1406 * H ** 4)
    assert labels["twist_rad"]["value"] == pytest.approx(exact, rel=0.03)
    assert labels["twist_rad"]["scaling"] == "inverse_shear_modulus"
    assert labels["load_case"]["torque_nm"] == torque
    assert "total_load_n" not in labels["load_case"]


@requires_all
def test_the_combined_case_is_the_sum_of_its_parts(bar):
    """Linear superposition, measured: the tip deflection under bending plus
    torque equals bending alone to 0.1 percent, and the twist, with the rigid
    translation of the face removed first, equals torsion alone to 1.6
    percent. The first version read the bending translation as a twist of
    0.12 rad; removing the mean displacement fixed it."""
    run, _ = bar
    bending = run(LoadCase(total_load_n=-100.0, kind=LoadKind.BENDING))
    torsion = run(LoadCase(kind=LoadKind.TORSION, torque_nm=2.0))
    combined = run(LoadCase(total_load_n=-100.0, kind=LoadKind.COMBINED,
                            torque_nm=2.0))
    assert combined["tip_deflection_m"]["value"] == pytest.approx(
        bending["tip_deflection_m"]["value"], rel=0.005)
    assert combined["twist_rad"]["value"] == pytest.approx(
        torsion["twist_rad"]["value"], rel=0.03)
    assert combined["load_case"]["load_kind"] == "combined"


@requires_all
def test_a_thermal_gradient_bends_the_bar_by_alpha_g_l_squared_over_two(bar):
    """Curvature alpha times gradient, tip deflection half of that times L
    squared. Measured 2.2 percent, the clamped face resisting the free
    curvature near the root; mesh sensitivity 0.6 percent."""
    run, material = bar
    gradient = 1000.0
    labels = run(LoadCase(kind=LoadKind.THERMAL_GRADIENT,
                          gradient_k_per_m=gradient, direction=1))
    exact = material.thermal_expansion_1_k * gradient * L ** 2 / 2.0
    assert abs(labels["thermal_tip_deflection_m"]["value"]) == pytest.approx(
        exact, rel=0.04)
    assert labels["thermal_tip_deflection_m"]["scaling"] == "expansion"
    assert labels["max_von_mises_pa"]["scaling"] == "modulus_times_expansion"
    assert "total_load_n" not in labels["load_case"]


@requires_all
def test_the_thermal_case_refuses_a_material_without_an_expansion_coefficient(
        tmp_path):
    """PLA 4043D's sheet gives no coefficient; the database stores none; the
    case is refused rather than run with a guess."""
    record, _ = make_part(FAMILIES["box"], dict(length_m=L, height_m=H, width_m=W),
                          tmp_path, labelled=False)
    pla = get_material("pla")
    assert pla.thermal_expansion_1_k is None
    with pytest.raises(RuntimeError, match="no sourced thermal expansion"):
        cantilever_labels(tmp_path / f"{record.part_id}.step",
                          record.geometry.volume_m3,
                          record.geometry.bounding_box_m, pla,
                          LoadCase(kind=LoadKind.THERMAL_GRADIENT))
