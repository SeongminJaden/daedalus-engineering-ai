"""Engineering IR: schema validation and YAML round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.engineering_ir import (  # noqa: E402
    EngineeringProblem, Geometry, Load, LoadApplication, LoadType, Objective,
    ObjectiveQuantity, ObjectiveSense, Vec3, load_problem, save_problem,
)
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402


# --- rejection -------------------------------------------------------------
def test_negative_length_rejected():
    with pytest.raises(ValidationError):
        Geometry(length_m=-0.5)


def test_zero_length_rejected():
    with pytest.raises(ValidationError):
        Geometry(length_m=0.0)


def test_unknown_enum_rejected():
    with pytest.raises(ValidationError):
        Load(type="magnetic_field", magnitude_n=1.0,
             direction=Vec3(x=0, y=-1, z=0), application=LoadApplication.TIP)


def test_unknown_objective_quantity_rejected():
    with pytest.raises(ValidationError):
        Objective(sense=ObjectiveSense.MINIMIZE, quantity="beauty")


def test_non_unit_direction_rejected():
    with pytest.raises(ValidationError, match="unit vector"):
        Load(type=LoadType.POINT_FORCE, magnitude_n=10.0,
             direction=Vec3(x=0.0, y=-2.0, z=0.0))


def test_zero_direction_rejected():
    with pytest.raises(ValidationError, match="unit vector"):
        Load(type=LoadType.POINT_FORCE, magnitude_n=10.0,
             direction=Vec3(x=0.0, y=0.0, z=0.0))


def test_negative_load_magnitude_rejected():
    with pytest.raises(ValidationError):
        Load(type=LoadType.POINT_FORCE, magnitude_n=-1.0,
             direction=Vec3(x=0, y=-1, z=0))


def test_unknown_field_rejected():
    """extra='forbid': a typo in YAML must fail rather than be ignored."""
    with pytest.raises(ValidationError):
        Geometry(length_m=0.5, lenght_m=0.5)


def test_empty_loads_rejected():
    p = build_mvp_problem().model_dump(mode="json")
    p["loads"] = []
    with pytest.raises(ValidationError):
        EngineeringProblem.model_validate(p)


# --- acceptance ------------------------------------------------------------
def test_unit_direction_accepted():
    load = Load(type=LoadType.POINT_FORCE, magnitude_n=196.2,
                direction=Vec3(x=0.0, y=-1.0, z=0.0))
    assert load.direction.is_unit()


def test_diagonal_unit_direction_accepted():
    r = 3.0**-0.5
    load = Load(type=LoadType.POINT_FORCE, magnitude_n=1.0,
                direction=Vec3(x=r, y=-r, z=r))
    assert load.direction.is_unit()


def test_constraints_all_optional():
    from core.engineering_ir import Constraints
    c = Constraints()
    assert c.max_stress_pa is None
    assert c.min_natural_frequency_hz is None
    assert c.no_collision is False


# --- round-trip ------------------------------------------------------------
def test_yaml_round_trip(tmp_path):
    original = build_mvp_problem()
    path = save_problem(original, tmp_path / "p.yaml")
    assert load_problem(path) == original


def test_dict_round_trip():
    from core.engineering_ir import problem_from_dict, problem_to_dict
    original = build_mvp_problem()
    assert problem_from_dict(problem_to_dict(original)) == original


def test_multi_objective_allowed():
    """The schema takes a list so Phase 3 multi-objective needs no change."""
    p = build_mvp_problem().model_dump(mode="json")
    p["objectives"].append(
        {"sense": ObjectiveSense.MAXIMIZE.value,
         "quantity": ObjectiveQuantity.STIFFNESS.value, "weight": 0.5}
    )
    assert len(EngineeringProblem.model_validate(p).objectives) == 2
