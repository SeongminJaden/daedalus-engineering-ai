"""surrogate.models - the surrogate network, scalers and training."""

from .mlp import (
    Standardizer,
    SurrogateBundle,
    SurrogateMLP,
    TrainingReport,
    evaluate_predictions,
    resolve_device,
    train_surrogate,
)

__all__ = [
    "Standardizer", "SurrogateBundle", "SurrogateMLP", "TrainingReport",
    "evaluate_predictions", "resolve_device", "train_surrogate",
]
