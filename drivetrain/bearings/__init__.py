"""drivetrain.bearings: rolling bearing archetypes and L10 life."""

from .catalog import (
    LIFE_EXPONENT,
    BEARINGS,
    BearingSpec,
    BearingType,
    all_bearings,
    get_bearing,
)
from .life import (
    BearingLifeResult,
    EquivalentLoad,
    equivalent_dynamic_load,
    l10_hours,
    l10_revolutions,
    rate_bearing,
)

__all__ = [
    "BEARINGS", "BearingLifeResult", "BearingSpec", "BearingType",
    "EquivalentLoad", "LIFE_EXPONENT", "all_bearings",
    "equivalent_dynamic_load", "get_bearing", "l10_hours", "l10_revolutions",
    "rate_bearing",
]
