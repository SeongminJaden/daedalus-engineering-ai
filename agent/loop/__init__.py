"""agent.loop - the autonomous design loop state machine."""

from .engine import (
    PHASE_CYCLE,
    DesignLoop,
    LoopConfig,
    LoopPhase,
    LoopResult,
    TerminationReason,
)

__all__ = [
    "PHASE_CYCLE", "DesignLoop", "LoopConfig", "LoopPhase", "LoopResult",
    "TerminationReason",
]
