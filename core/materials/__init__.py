"""core.materials - material property database (referenced by id from the IR)."""

from .constitutive import (
    check_stiffness,
    compliance_matrix,
    isotropic_stiffness,
    stiffness_from_constants,
    stiffness_matrix,
)
from .inference import (
    Estimate,
    apply_estimate,
    check_derived_fields,
    estimate_fatigue_strength,
    shear_modulus_from_isotropic,
)
from .db import (
    DEFAULT_DB_PATH,
    MaterialClass,
    MaterialDB,
    MaterialSpec,
    MaterialStatus,
    MissingMaterialValue,
    SourceDocument,
    SourceGrade,
    ValueSource,
    get_material,
    load_materials,
)

__all__ = [
    "DEFAULT_DB_PATH", "MaterialClass", "MaterialDB", "MaterialSpec", "MaterialStatus",
    "MissingMaterialValue", "SourceDocument", "SourceGrade", "ValueSource",
    "Estimate", "apply_estimate", "check_derived_fields",
    "check_stiffness", "compliance_matrix", "estimate_fatigue_strength",
    "get_material",
    "isotropic_stiffness", "load_materials", "stiffness_from_constants",
    "shear_modulus_from_isotropic", "stiffness_matrix",
]
