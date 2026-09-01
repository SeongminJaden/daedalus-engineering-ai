"""physics.fatigue: high-cycle stress-life analysis."""

from .miner import (
    DAMAGE_AT_FAILURE,
    KNEE_CYCLES,
    LOW_CYCLE_LIMIT,
    STRENGTH_FRACTION_AT_1000,
    BlockDamage,
    LoadBlock,
    LowCycleRegime,
    MinerResult,
    cumulative_damage,
    cycles_to_failure,
    equivalent_alternating_stress,
)
from .sn import (
    ENDURANCE_LIMIT_MATERIALS,
    REFERENCE_LIFE_CYCLES,
    FatigueResult,
    MeanStressCriterion,
    StressCycle,
    fatigue_safety_factor,
    governing_failure_mode,
    has_endurance_limit,
)

__all__ = [
    "BlockDamage", "DAMAGE_AT_FAILURE", "KNEE_CYCLES", "LOW_CYCLE_LIMIT",
    "LoadBlock", "LowCycleRegime", "MinerResult", "STRENGTH_FRACTION_AT_1000",
    "cumulative_damage", "cycles_to_failure", "equivalent_alternating_stress",
    "ENDURANCE_LIMIT_MATERIALS", "FatigueResult", "MeanStressCriterion",
    "REFERENCE_LIFE_CYCLES", "StressCycle", "fatigue_safety_factor",
    "governing_failure_mode", "has_endurance_limit",
]
