"""The MVP problem: Python definition and YAML must agree, values as specified."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.engineering_ir import (  # noqa: E402
    BoundaryConditionType, BoundaryLocation, LoadApplication, LoadType,
    ObjectiveQuantity, ObjectiveSense,
)
from core.materials import load_materials  # noqa: E402
from core.units import MM, MPA  # noqa: E402
from projects.robotic_link.problem import (  # noqa: E402
    build_mvp_problem, load_mvp_problem,
)


def test_python_and_yaml_are_equivalent():
    """The YAML must not drift from the Python definition."""
    assert build_mvp_problem() == load_mvp_problem()


def test_mvp_values():
    p = build_mvp_problem()
    assert p.name == "mvp_cantilever_link"
    assert p.geometry.length_m == 0.5
    assert p.geometry.max_width_m == 0.1
    assert p.geometry.max_height_m == 0.1
    assert p.material_id == "al_7075_t6"


def test_mvp_load():
    load = build_mvp_problem().loads[0]
    assert load.type is LoadType.POINT_FORCE
    assert load.magnitude_n == pytest.approx(196.2)
    assert load.direction.as_tuple() == (0.0, -1.0, 0.0)
    assert load.direction.is_unit()
    assert load.application is LoadApplication.TIP


def test_mvp_load_matches_a_20kg_payload():
    """196.2 N is 20 kg under 9.81 m/s^2 - documents where the number came from."""
    assert build_mvp_problem().loads[0].magnitude_n == pytest.approx(20.0 * 9.81)


def test_mvp_boundary_is_cantilever():
    bc = build_mvp_problem().boundary_conditions[0]
    assert bc.type is BoundaryConditionType.FIXED
    assert bc.location is BoundaryLocation.ROOT


def test_mvp_constraints():
    c = build_mvp_problem().constraints
    assert c.max_stress_pa == 120.0 * MPA
    assert c.max_deflection_m == 1.0 * MM
    assert c.min_safety_factor == 2.0


def test_mvp_objective():
    obj = build_mvp_problem().objectives[0]
    assert obj.sense is ObjectiveSense.MINIMIZE
    assert obj.quantity is ObjectiveQuantity.MASS


def test_mvp_material_exists_in_db():
    p = build_mvp_problem()
    assert load_materials().get(p.material_id).name == "Aluminium 7075-T6"


def test_stress_constraint_is_below_material_yield():
    """120 MPa against 503 MPa yield: the constraint must be the binding limit,
    otherwise the safety factor is not doing anything."""
    p = build_mvp_problem()
    material = load_materials().get(p.material_id)
    assert p.constraints.max_stress_pa < material.yield_strength_pa
    assert (p.constraints.max_stress_pa
            <= material.allowable_stress_pa(p.constraints.min_safety_factor))
