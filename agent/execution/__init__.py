"""agent.execution: running the design method the registry selected."""

from .dispatch import EXECUTORS, NoExecutor, execute, executable_methods
from .outcome import DesignOutcome, OutcomeVerdict

__all__ = ["EXECUTORS", "DesignOutcome", "NoExecutor", "OutcomeVerdict", "execute",
           "executable_methods"]
