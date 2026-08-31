"""core.materials.db - material property database loader.

Materials are referenced by id from the IR and never inlined into it, so a
property correction happens in one place and every stored problem picks it up.

`source` and `status` are first-class schema fields, not comments: a design
decision needs to know whether a number is a handbook typical or a certified
datasheet value. Phase 5's Brain is expected to widen this into a confidence
model; keeping the provenance now means nothing has to be back-filled later.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "materials.yaml"


class MaterialStatus(str, Enum):
    """How much a value can be trusted."""

    REFERENCE_TYPICAL = "reference_typical"   # handbook/database typical
    SUPPLIER_DATASHEET = "supplier_datasheet"  # named supplier, named product
    MEASURED = "measured"                      # measured on the actual stock
    ASSUMED = "assumed"                        # placeholder, not defensible


class MaterialSpec(BaseModel):
    """One material. All values SI."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    density_kg_m3: float = Field(gt=0.0)
    youngs_modulus_pa: float = Field(gt=0.0)
    yield_strength_pa: float = Field(gt=0.0)
    ultimate_strength_pa: float = Field(gt=0.0)
    poisson_ratio: float = Field(gt=0.0, lt=0.5)
    shear_modulus_pa: float = Field(gt=0.0)
    fatigue_strength_pa: float = Field(gt=0.0)
    source: str = Field(min_length=1)
    status: MaterialStatus

    @model_validator(mode="after")
    def _yield_below_ultimate(self) -> "MaterialSpec":
        if self.yield_strength_pa >= self.ultimate_strength_pa:
            raise ValueError(
                f"{self.id}: yield ({self.yield_strength_pa:.4g} Pa) must be "
                f"below ultimate ({self.ultimate_strength_pa:.4g} Pa)"
            )
        return self

    def allowable_stress_pa(self, safety_factor: float) -> float:
        """Yield divided by a safety factor. Phase 2 uses this for checks."""
        if safety_factor <= 0:
            raise ValueError("safety_factor must be > 0")
        return self.yield_strength_pa / safety_factor


class MaterialDB(BaseModel):
    model_config = ConfigDict(extra="forbid")

    materials: list[MaterialSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> "MaterialDB":
        ids = [m.id for m in self.materials]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate material ids: {sorted(dupes)}")
        return self

    def get(self, material_id: str) -> MaterialSpec:
        for m in self.materials:
            if m.id == material_id:
                return m
        raise KeyError(
            f"unknown material {material_id!r}; available: {self.ids()}"
        )

    def ids(self) -> list[str]:
        return [m.id for m in self.materials]


def load_materials(path: str | Path | None = None) -> MaterialDB:
    """Load and validate the material database."""
    p = Path(path) if path is not None else DEFAULT_DB_PATH
    with p.open() as fh:
        return MaterialDB.model_validate(yaml.safe_load(fh))


def get_material(material_id: str, path: str | Path | None = None) -> MaterialSpec:
    return load_materials(path).get(material_id)
