"""core.engineering_ir.schema - the Engineering IR (the *problem*).

The IR describes what is being asked for and never how it is shaped. Length,
loads, material choice, constraints and objectives are fixed inputs; the
cross-section dimensions that a search is free to vary live in the Design
Genome instead (see core.design_genome).

All quantities are SI: metres, newtons, pascals, radians.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

UNIT_VECTOR_TOLERANCE = 1e-9


class _Strict(BaseModel):
    """Reject unknown keys everywhere - a typo in YAML must fail loudly."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# enums
# --------------------------------------------------------------------------- #
class SectionType(str, Enum):
    HOLLOW_RECTANGLE = "hollow_rectangle"


class LoadType(str, Enum):
    POINT_FORCE = "point_force"


class LoadApplication(str, Enum):
    TIP = "tip"
    ROOT = "root"
    MID_SPAN = "mid_span"
    DISTRIBUTED = "distributed"


class BoundaryConditionType(str, Enum):
    FIXED = "fixed"
    PINNED = "pinned"


class BoundaryLocation(str, Enum):
    ROOT = "root"
    TIP = "tip"


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveQuantity(str, Enum):
    MASS = "mass"
    STRESS = "stress"
    DEFLECTION = "deflection"
    STIFFNESS = "stiffness"
    COST = "cost"


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
class Vec3(_Strict):
    """A 3-vector. Dimensionless here; meaning comes from the field using it."""

    x: float
    y: float
    z: float

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def is_unit(self, tol: float = UNIT_VECTOR_TOLERANCE) -> bool:
        return abs(self.norm() - 1.0) <= tol

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


# --------------------------------------------------------------------------- #
# problem parts
# --------------------------------------------------------------------------- #
class Geometry(_Strict):
    """Fixed geometry of the problem plus the design-space envelope.

    Cross-section dimensions are deliberately absent: those are design
    variables and belong to the genome.
    """

    length_m: float = Field(gt=0.0)
    max_width_m: float | None = Field(default=None, gt=0.0)
    max_height_m: float | None = Field(default=None, gt=0.0)
    section_type: SectionType = SectionType.HOLLOW_RECTANGLE


class Load(_Strict):
    type: LoadType = LoadType.POINT_FORCE
    magnitude_n: float = Field(ge=0.0)
    direction: Vec3
    application: LoadApplication = LoadApplication.TIP

    @field_validator("direction")
    @classmethod
    def _direction_must_be_unit(cls, v: Vec3) -> Vec3:
        if not v.is_unit():
            raise ValueError(
                f"load direction must be a unit vector, got norm={v.norm():.12g}"
            )
        return v


class BoundaryCondition(_Strict):
    type: BoundaryConditionType = BoundaryConditionType.FIXED
    location: BoundaryLocation = BoundaryLocation.ROOT


class Constraints(_Strict):
    """All optional - an absent constraint is simply not applied."""

    max_stress_pa: float | None = Field(default=None, gt=0.0)
    max_deflection_m: float | None = Field(default=None, gt=0.0)
    min_safety_factor: float | None = Field(default=None, gt=0.0)
    min_natural_frequency_hz: float | None = Field(default=None, gt=0.0)
    no_collision: bool = False


class Objective(_Strict):
    sense: ObjectiveSense
    quantity: ObjectiveQuantity
    weight: float = Field(default=1.0, gt=0.0)


class EngineeringProblem(_Strict):
    """Top level of the Engineering IR."""

    name: str = Field(min_length=1)
    geometry: Geometry
    material_id: str = Field(min_length=1)
    loads: list[Load] = Field(min_length=1)
    boundary_conditions: list[BoundaryCondition] = Field(min_length=1)
    constraints: Constraints = Field(default_factory=Constraints)
    objectives: list[Objective] = Field(min_length=1)
