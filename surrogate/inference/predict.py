"""surrogate.inference.predict - predictions, with their expected error attached.

A surrogate prediction is never returned bare. Every call also reports the
held-out error the model showed on data it did not train on, so a caller cannot
treat a prediction as a measurement by accident.

Safety factor is derived (yield / predicted stress), never predicted, so it can
never disagree with the stress the model produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from surrogate.datasets import INPUT_NAMES, OUTPUT_NAMES
from surrogate.models import SurrogateBundle


@dataclass
class Prediction:
    """Predicted metrics plus the model's known error on held-out data."""

    values: dict[str, np.ndarray]
    expected_relative_error: dict[str, float]  # p95 held-out relative error
    verified: bool = False                     # never true for a prediction
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(next(iter(self.values.values()))))

    def candidate(self, i: int) -> dict[str, float]:
        return {k: float(v[i]) for k, v in self.values.items()}

    def interval(self, metric: str, i: int) -> tuple[float, float]:
        """A plain +/- band from the p95 held-out error. Not a calibrated
        confidence interval - it is an honest order-of-magnitude guard."""
        value = float(self.values[metric][i])
        err = self.expected_relative_error.get(metric, 0.0)
        return value * (1.0 - err), value * (1.0 + err)


def build_inputs(
    b: np.ndarray, h: np.ndarray, t: np.ndarray, case,
) -> np.ndarray:
    """Assemble the model's input matrix in the canonical column order."""
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    h = np.atleast_1d(np.asarray(h, dtype=np.float64))
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    n = b.shape[0]
    context = np.tile(
        [case.length_m, case.tip_load_n, case.youngs_modulus_pa,
         case.density_kg_m3, case.yield_strength_pa], (n, 1))
    return np.column_stack([context, b, h, t])


class SurrogatePredictor:
    """Wraps a trained bundle with derived quantities and error reporting."""

    def __init__(self, bundle: SurrogateBundle):
        self.bundle = bundle

    @property
    def expected_relative_error(self) -> dict[str, float]:
        return {
            name: float(stats.get("p95_rel_err", 0.0))
            for name, stats in (self.bundle.test_metrics or {}).items()
        }

    def predict(self, inputs: np.ndarray) -> Prediction:
        raw = self.bundle.predict_array(inputs)
        values = {name: raw[:, i] for i, name in enumerate(self.bundle.output_names)}

        # Derived, not predicted: SF is exactly yield / stress.
        yield_col = list(INPUT_NAMES).index("yield_strength_pa")
        yield_strength = np.atleast_2d(inputs)[:, yield_col]
        values["safety_factor"] = yield_strength / np.maximum(
            values["max_bending_stress_pa"], 1e-30)

        errors = self.expected_relative_error
        # SF inherits the stress error, since it is a function of it.
        errors["safety_factor"] = errors.get("max_bending_stress_pa", 0.0)
        return Prediction(values=values, expected_relative_error=errors,
                          verified=False,
                          meta={"n": int(np.atleast_2d(inputs).shape[0])})

    def predict_designs(self, b, h, t, case) -> Prediction:
        return self.predict(build_inputs(b, h, t, case))
