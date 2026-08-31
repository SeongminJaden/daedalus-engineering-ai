"""core.materials - material property database (referenced by id from the IR)."""

from .db import (
    DEFAULT_DB_PATH,
    MaterialDB,
    MaterialSpec,
    MaterialStatus,
    get_material,
    load_materials,
)

__all__ = [
    "DEFAULT_DB_PATH", "MaterialDB", "MaterialSpec", "MaterialStatus",
    "get_material", "load_materials",
]
