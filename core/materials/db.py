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


class SourceGrade(str, Enum):
    """How close a cited document is to a measurement.

    PRIMARY    the producer's own datasheet or a standard, read in full
    SECONDARY  a database or article that restates values without attributing
               them to a measurement; named so it can be checked, not trusted
    DERIVED    computed exactly from other stored values
    """

    PRIMARY = "primary"
    SECONDARY = "secondary"
    DERIVED = "derived"


class SourceDocument(BaseModel):
    """One document a value was read from, identified well enough to find."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    grade: SourceGrade
    url: str = ""
    document_date: str = ""       # as printed on the document, or "not printed"
    retrieved: str = ""           # ISO date the document was read
    notes: str = ""


class ValueSource(BaseModel):
    """Where ONE stored value came from, by field name."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)    # a SourceDocument id in `sources`
    grade: SourceGrade
    location: str = ""                   # table, page or row in the document
    #: The document's own number and unit, before conversion to SI, so the
    #: conversion can be checked.
    as_printed: str = ""
    condition: str = ""                  # temper, product form, orientation
    note: str = ""


class MissingMaterialValue(ValueError):
    """A property this database does not have a sourced value for.

    Raised instead of returning a guess. A method that needs the value has to
    say so, and the caller has to find a source or choose another material.
    """


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
    #: None means NO SOURCED VALUE EXISTS in this database, not zero. The
    #: fatigue methods refuse such a material through
    #: `require_fatigue_strength_pa` rather than inventing an endurance limit.
    fatigue_strength_pa: float | None = Field(default=None, gt=0.0)
    source: str = Field(min_length=1)
    status: MaterialStatus

    # --- provenance per document and per value ------------------------------
    #
    # `source` above is the one-line summary that every consumer prints.
    # `sources` are the documents themselves and `value_sources` says which
    # field came from which document, at which table, in which condition,
    # with the number as printed so the SI conversion can be checked. A value
    # with no entry in `value_sources` has no better provenance than the
    # one-line summary, and the tests count those.
    sources: list[SourceDocument] = Field(default_factory=list)
    value_sources: dict[str, ValueSource] = Field(default_factory=dict)
    #: Service temperature range the source states, in kelvin, or None when
    #: the source states none. Not a limit anything enforces yet; recorded so
    #: that a thermal problem can be refused before it is solved wrongly.
    service_temperature_k: tuple[float, float] | None = None
    #: Sourced temperature curves, field name to a list of (T_K, value)
    #: pairs, only where a document tabulates them. Nothing here is
    #: interpolated from a room temperature value.
    temperature_dependence: dict[str, list[tuple[float, float]]] = Field(
        default_factory=dict)
    temperature_dependence_sources: dict[str, ValueSource] = Field(
        default_factory=dict)

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
    # These three are TENSILE. The name predates the compressive fields below
    # and is kept so existing callers do not break, but the distinction matters
    # enormously for a composite and is why the compressive values are separate
    # rather than assumed equal.
    strength_long_pa: float | None = Field(default=None, gt=0.0)
    strength_trans_pa: float | None = Field(default=None, gt=0.0)
    strength_shear_pa: float | None = Field(default=None, gt=0.0)

    # Compressive strengths, which are NOT the tensile ones.
    #
    # A unidirectional composite is strongly asymmetric and in opposite
    # directions along its two axes. Along the fibres it is WEAKER in
    # compression, because the failure is fibre microbuckling rather than fibre
    # fracture. Across the fibres it is several times STRONGER in compression,
    # because transverse tension simply pulls the matrix apart while
    # compression does not. Assuming symmetry would be wrong by a factor of
    # four transversely, and in the unconservative direction for the load case
    # that matters.
    strength_long_compressive_pa: float | None = Field(default=None, gt=0.0)
    strength_trans_compressive_pa: float | None = Field(default=None, gt=0.0)

    # --- thermal expansion, 1/K ---
    #
    # NOT constrained positive. Carbon fibre CONTRACTS along its length when
    # heated, so a unidirectional laminate has a small negative alpha_1, and a
    # gt=0 constraint here would reject the physically correct value. That is
    # the whole reason this field is written out rather than assumed.
    #
    # For an orthotropic material a single number is as meaningless as a single
    # yield strength: CFRP is about -0.5e-6 along the fibres and 25e-6 across
    # them, a factor of fifty apart and of opposite sign. The directional
    # fields below carry that and the validator requires them.
    thermal_expansion_1_k: float | None = None
    cte1_1_k: float | None = None
    cte2_1_k: float | None = None
    cte3_1_k: float | None = None

    # --- raw material price, USD per kg ---
    #
    # VALIDITY: raw stock only. This is NOT a part cost, and multiplying it by
    # a mass gives a material bill, not a price. For a small machined bracket
    # the machining dominates the material several times over, so a design
    # chosen to minimise this number is not necessarily the cheaper part. It is
    # here to RANK materials against each other under a fixed process, which is
    # a question it can answer.
    price_per_kg_usd: float | None = Field(default=None, gt=0.0)

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
    def _provenance_is_consistent(self) -> "MaterialSpec":
        """Every cited document exists, every cited field exists, and a value
        graded DERIVED is listed among the derived fields."""
        ids = {d.id for d in self.sources}
        if len(ids) != len(self.sources):
            raise ValueError(f"{self.id}: duplicate source ids")
        fields = set(type(self).model_fields)
        for name, vs in {**self.value_sources,
                         **self.temperature_dependence_sources}.items():
            if name not in fields:
                raise ValueError(f"{self.id}: value source names unknown field "
                                 f"{name!r}")
            if vs.source not in ids:
                raise ValueError(f"{self.id}: field {name!r} cites source "
                                 f"{vs.source!r}, which is not in sources")
        for name, vs in self.value_sources.items():
            if vs.grade is SourceGrade.DERIVED and name not in self.derived_fields:
                raise ValueError(f"{self.id}: {name!r} is graded derived but not "
                                 f"listed in derived_fields")
        for name, curve in self.temperature_dependence.items():
            if name not in fields:
                raise ValueError(f"{self.id}: temperature curve names unknown "
                                 f"field {name!r}")
            if name not in self.temperature_dependence_sources:
                raise ValueError(f"{self.id}: temperature curve for {name!r} "
                                 f"has no source; a curve without one is a guess")
            temps = [t for t, _ in curve]
            if len(curve) < 2 or temps != sorted(temps):
                raise ValueError(f"{self.id}: temperature curve for {name!r} "
                                 f"needs at least two points in rising order")
        if self.service_temperature_k is not None:
            low, high = self.service_temperature_k
            if not 0.0 < low < high:
                raise ValueError(f"{self.id}: service temperature range must be "
                                 f"positive and rising")
        return self

    def require_fatigue_strength_pa(self) -> float:
        """The endurance value, or a refusal naming the material."""
        if self.fatigue_strength_pa is None:
            raise MissingMaterialValue(
                f"{self.id} has no sourced fatigue strength; the database "
                f"records none rather than a guess, so a fatigue check cannot "
                f"run on it")
        return self.fatigue_strength_pa

    def unsourced_fields(self) -> list[str]:
        """Stored numeric values with no per-value source entry."""
        numeric = ("density_kg_m3", "youngs_modulus_pa", "yield_strength_pa",
                   "ultimate_strength_pa", "poisson_ratio", "shear_modulus_pa",
                   "fatigue_strength_pa", "thermal_expansion_1_k")
        return [f for f in numeric if getattr(self, f) is not None
                and f not in self.value_sources]

    def modulus_at_temperature_pa(self, temperature_k: float) -> float:
        """Young's modulus from the sourced curve, linearly interpolated
        INSIDE its range. Refuses to extrapolate and refuses when no curve is
        stored; the room temperature value is not a curve."""
        curve = self.temperature_dependence.get("youngs_modulus_pa")
        if not curve:
            raise MissingMaterialValue(
                f"{self.id} has no sourced modulus versus temperature curve")
        temps = [t for t, _ in curve]
        if not temps[0] <= temperature_k <= temps[-1]:
            raise MissingMaterialValue(
                f"{self.id}: {temperature_k:.0f} K is outside the sourced "
                f"curve {temps[0]:.0f} to {temps[-1]:.0f} K; no extrapolation")
        for (t0, v0), (t1, v1) in zip(curve, curve[1:]):
            if t0 <= temperature_k <= t1:
                frac = 0.0 if t1 == t0 else (temperature_k - t0) / (t1 - t0)
                return v0 + frac * (v1 - v0)
        return curve[-1][1]

    def quasi_isotropic_modulus_estimate_pa(self) -> float:
        """For an orthotropic lamina, the in-plane modulus a quasi-isotropic
        laminate of it would have, by the common estimate 3/8 E1 + 5/8 E2.

        DERIVED and approximate: the exact laminate result needs G12 and nu12
        through classical lamination theory, and this estimate is what is
        usually quoted before that is done. It is a stiffness only. No
        strength follows from it, which is why no isotropic CFRP entry exists
        in this database: a strength for it would have to be invented.
        """
        if self.material_class is not MaterialClass.ORTHOTROPIC:
            raise MissingMaterialValue(
                f"{self.id} is not orthotropic; the estimate is for laminae")
        return 3.0 / 8.0 * self.e1_pa + 5.0 / 8.0 * self.e2_pa

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
        if self.thermal_expansion_1_k is not None and (
                self.cte1_1_k is None or self.cte2_1_k is None):
            raise ValueError(
                f"{self.id}: orthotropic material declares a thermal "
                f"expansion coefficient but not the directional ones. A single "
                f"alpha is meaningless here for the same reason a single yield "
                f"strength is, and worse: the two directions can differ in "
                f"SIGN")
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
