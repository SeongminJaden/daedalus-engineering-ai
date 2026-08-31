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


class MaterialClass(str, Enum):
    """How many independent elastic constants the material needs.

    Isotropic is stored as the special case it is: E and nu expand to the full
    orthotropic set (Ei = E, Gij = E/2(1+nu), nu_ij = nu), so there is exactly
    ONE constitutive path and the isotropic results cannot drift away from the
    general ones.
    """

    ISOTROPIC = "isotropic"
    ORTHOTROPIC = "orthotropic"
    ANISOTROPIC = "anisotropic"      # reserved; not yet constructed


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

    # --- isotropic constants. For an orthotropic entry these are the
    #     representative in-plane / axis-1 values kept for reporting; the
    #     direction-dependent fields below are what the physics uses. ---
    youngs_modulus_pa: float = Field(gt=0.0)
    yield_strength_pa: float = Field(gt=0.0)
    ultimate_strength_pa: float = Field(gt=0.0)
    poisson_ratio: float = Field(gt=0.0, lt=0.5)
    shear_modulus_pa: float = Field(gt=0.0)
    fatigue_strength_pa: float = Field(gt=0.0)
    source: str = Field(min_length=1)
    status: MaterialStatus

    material_class: MaterialClass = MaterialClass.ISOTROPIC

    # --- orthotropic elastic constants (required when class is orthotropic) ---
    e1_pa: float | None = Field(default=None, gt=0.0)
    e2_pa: float | None = Field(default=None, gt=0.0)
    e3_pa: float | None = Field(default=None, gt=0.0)
    g12_pa: float | None = Field(default=None, gt=0.0)
    g13_pa: float | None = Field(default=None, gt=0.0)
    g23_pa: float | None = Field(default=None, gt=0.0)
    nu12: float | None = Field(default=None, gt=0.0, lt=0.999)
    nu13: float | None = Field(default=None, gt=0.0, lt=0.999)
    nu23: float | None = Field(default=None, gt=0.0, lt=0.999)

    # --- direction-dependent strengths. A single "yield" number is meaningless
    #     for a composite: 1500 MPa along the fibres and 50 MPa across them. ---
    strength_long_pa: float | None = Field(default=None, gt=0.0)
    strength_trans_pa: float | None = Field(default=None, gt=0.0)
    strength_shear_pa: float | None = Field(default=None, gt=0.0)

    # --- composite layup. Single orientation for now, list-shaped so a real
    #     stack does not need a schema change later. ---
    orientation_deg: list[float] | None = None
    layup: list[str] | None = None

    # --- honesty fields ---
    # True when a value was DERIVED from others (exact, e.g. G = E/2(1+nu))
    # rather than taken from a source. Never used to dress an estimate up.
    derived_fields: list[str] = Field(default_factory=list)
    # Free-text caveats that travel with the material.
    notes: str = ""

    @model_validator(mode="after")
    def _yield_below_ultimate(self) -> "MaterialSpec":
        if self.yield_strength_pa >= self.ultimate_strength_pa:
            raise ValueError(
                f"{self.id}: yield ({self.yield_strength_pa:.4g} Pa) must be "
                f"below ultimate ({self.ultimate_strength_pa:.4g} Pa)"
            )
        return self

    @model_validator(mode="after")
    def _orthotropic_is_complete_and_consistent(self) -> "MaterialSpec":
        if self.material_class is not MaterialClass.ORTHOTROPIC:
            return self
        required = ("e1_pa", "e2_pa", "e3_pa", "g12_pa", "g13_pa", "g23_pa",
                    "nu12", "nu13", "nu23")
        missing = [f for f in required if getattr(self, f) is None]
        if missing:
            raise ValueError(
                f"{self.id}: orthotropic material is missing {missing}")
        if self.strength_long_pa is None or self.strength_trans_pa is None:
            raise ValueError(
                f"{self.id}: orthotropic material needs direction-dependent "
                "strengths; a single yield value would be misleading")
        return self

    # --- expansion to the single constitutive path -------------------------- #
    def elastic_constants(self) -> dict[str, float]:
        """The nine orthotropic constants, with isotropy as the special case.

        An isotropic entry expands to Ei = E, Gij = E/2(1+nu), nu_ij = nu. That
        expansion is EXACT, not an estimate: it is the definition of isotropy.
        """
        if self.material_class is MaterialClass.ISOTROPIC:
            e = self.youngs_modulus_pa
            nu = self.poisson_ratio
            g = e / (2.0 * (1.0 + nu))
            return {"E1": e, "E2": e, "E3": e,
                    "G12": g, "G13": g, "G23": g,
                    "nu12": nu, "nu13": nu, "nu23": nu}
        return {"E1": self.e1_pa, "E2": self.e2_pa, "E3": self.e3_pa,
                "G12": self.g12_pa, "G13": self.g13_pa, "G23": self.g23_pa,
                "nu12": self.nu12, "nu13": self.nu13, "nu23": self.nu23}

    def reciprocal_poisson(self) -> dict[str, float]:
        """nu_ji from the Maxwell-Betti reciprocity nu_ij/Ei = nu_ji/Ej.

        Derived exactly, never stored, so the compliance matrix cannot be
        made non-symmetric by a typo in a data file.
        """
        c = self.elastic_constants()
        return {
            "nu21": c["nu12"] * c["E2"] / c["E1"],
            "nu31": c["nu13"] * c["E3"] / c["E1"],
            "nu32": c["nu23"] * c["E3"] / c["E2"],
        }

    def axial_modulus_pa(self) -> float:
        """Modulus along the part axis, for the 1D beam model.

        For an orthotropic material this ASSUMES the material 1-axis is aligned
        with the beam axis. Off-axis loading couples extension and shear and
        needs the 3D model; the beam model cannot represent it.
        """
        return self.elastic_constants()["E1"]

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
