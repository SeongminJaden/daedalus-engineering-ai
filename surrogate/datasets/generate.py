"""surrogate.datasets.generate - build training data from the Phase 2 solver.

WHAT THE SURROGATE IS APPROXIMATING: the Phase 2 Euler-Bernoulli evaluator, not
3D FEM. There is no higher-fidelity model in the system yet, so the surrogate's
error stacks *on top of* beam theory's own error. See surrogate/README.md.

The problem context is sampled too - length, tip load and material properties -
not just the section. Training on one fixed problem would produce a surrogate
that only knows that problem; sweeping the context gives a model that
generalizes across the family.

Sampling is grouped by context: the kernel takes one load case per launch, so a
context becomes one launch over many designs. That keeps generation to a few
hundred GPU launches instead of one per row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.profile import load_profile
from physics.structural import BeamLoadCase, evaluate_beam_case

# Input layout. Order is part of the saved artifact - a model trained on one
# ordering must not be fed another.
INPUT_NAMES = (
    "length_m", "tip_load_n", "youngs_modulus_pa", "density_kg_m3",
    "yield_strength_pa", "outer_width_m", "outer_height_m", "wall_thickness_m",
)

# Predicted outputs. safety_factor is deliberately NOT among them: it is
# yield / stress exactly, so predicting it separately would let the model
# return an SF inconsistent with its own stress. It is derived at inference.
OUTPUT_NAMES = (
    "mass_kg", "max_bending_stress_pa", "tip_deflection_m",
    "first_natural_frequency_hz",
)


@dataclass
class SamplingRanges:
    """Design and context ranges to sample. SI.

    Material ranges span the aluminium alloys in the database and a little
    beyond, so the model is interpolating rather than extrapolating for them.
    """

    length_m: tuple[float, float] = (0.1, 1.5)
    tip_load_n: tuple[float, float] = (10.0, 2000.0)
    youngs_modulus_pa: tuple[float, float] = (60.0e9, 80.0e9)
    density_kg_m3: tuple[float, float] = (2500.0, 3000.0)
    yield_strength_pa: tuple[float, float] = (200.0e6, 600.0e6)
    outer_width_m: tuple[float, float] = (0.010, 0.100)
    outer_height_m: tuple[float, float] = (0.010, 0.100)
    wall_thickness_m: tuple[float, float] = (0.001, 0.020)
    # Keep the wall clear of the degenerate t = min(b,h)/2 boundary, where the
    # section inertia collapses and the metrics blow up.
    max_thickness_fraction: float = 0.45

    # Sampled log-uniformly rather than uniformly. Length and tip load each
    # span roughly two orders of magnitude; drawing them uniformly puts almost
    # every sample in the top decade and leaves the low-load, short-link corner
    # nearly unvisited - exactly where the MVP problem sits (196 N over a range
    # to 2000 N). Measured effect at the MVP context: stress error 8.9% -> 3.1%,
    # deflection 11.1% -> 2.9%.
    log_sampled: tuple[str, ...] = ("length_m", "tip_load_n")

    def as_dict(self) -> dict:
            return {k: list(v) if isinstance(v, tuple) else v
                for k, v in self.__dict__.items()}


@dataclass
class Dataset:
    """Inputs, outputs and the provenance needed to reproduce them.

    `context_ids` records which sampled problem context each row came from.
    It exists so the split can hold out whole contexts - see `split`.
    """

    inputs: np.ndarray            # (n, len(INPUT_NAMES))
    outputs: np.ndarray           # (n, len(OUTPUT_NAMES))
    input_names: tuple[str, ...] = INPUT_NAMES
    output_names: tuple[str, ...] = OUTPUT_NAMES
    meta: dict = field(default_factory=dict)
    context_ids: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            inputs=self.inputs,
            outputs=self.outputs,
            context_ids=(np.array([]) if self.context_ids is None
                         else self.context_ids),
            input_names=np.array(self.input_names),
            output_names=np.array(self.output_names),
            meta=np.array([repr(self.meta)]),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        with np.load(Path(path), allow_pickle=False) as f:
            ids = f["context_ids"] if "context_ids" in f else np.array([])
            return cls(
                inputs=f["inputs"],
                outputs=f["outputs"],
                input_names=tuple(str(x) for x in f["input_names"]),
                output_names=tuple(str(x) for x in f["output_names"]),
                meta={"loaded_from": str(path)},
                context_ids=None if ids.size == 0 else ids,
            )

    def split(self, val_fraction: float = 0.15, test_fraction: float = 0.15,
              seed: int = 0, group_by_context: bool = True):
        """Deterministic train/val/test split.

        **Splits by problem context by default, not by row.** Rows are
        generated in groups that share a context (one kernel launch per
        context), so a random row split puts designs from the *same* problem in
        both train and test. The resulting score then measures interpolation
        between designs of a problem the model already saw - which flatters it
        and says nothing about a new problem.

        Grouping whole contexts into a single split makes the test set an
        honest measure of generalization to unseen problems, which is what the
        surrogate is actually for. Pass group_by_context=False for the naive
        row split (useful only for diagnosis).
        """
        if not 0 < val_fraction + test_fraction < 1:
            raise ValueError("val + test fractions must be in (0, 1)")
        rng = np.random.default_rng(seed)

        if group_by_context and self.context_ids is not None:
            groups = np.unique(self.context_ids)
            if len(groups) >= 3:
                order = rng.permutation(len(groups))
                n_test = max(1, int(round(len(groups) * test_fraction)))
                n_val = max(1, int(round(len(groups) * val_fraction)))
                if n_test + n_val >= len(groups):
                    n_test = n_val = 1
                pick = {
                    "test": groups[order[:n_test]],
                    "val": groups[order[n_test:n_test + n_val]],
                    "train": groups[order[n_test + n_val:]],
                }
                return tuple(
                    self._subset(np.flatnonzero(np.isin(self.context_ids,
                                                        pick[name])), name)
                    for name in ("train", "val", "test")
                )

        order = rng.permutation(len(self))
        n_test = int(round(len(self) * test_fraction))
        n_val = int(round(len(self) * val_fraction))
        return tuple(
            self._subset(i, name) for name, i in (
                ("train", order[n_test + n_val:]),
                ("val", order[n_test:n_test + n_val]),
                ("test", order[:n_test]),
            )
        )

    def _subset(self, idx: np.ndarray, name: str) -> "Dataset":
        return Dataset(
            inputs=self.inputs[idx],
            outputs=self.outputs[idx],
            input_names=self.input_names,
            output_names=self.output_names,
            meta={"split": name, **self.meta},
            context_ids=(None if self.context_ids is None
                         else self.context_ids[idx]),
        )


def resolve_dataset_size(profile: str | None = None,
                         n_samples: int | None = None) -> int:
    if n_samples is not None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        return n_samples
    cfg = load_profile(profile)
    return int(cfg["surrogate"].get("dataset_samples", 20000))


def _sample_designs(rng, n: int, r: SamplingRanges):
    b = rng.uniform(*r.outer_width_m, size=n)
    h = rng.uniform(*r.outer_height_m, size=n)
    t_max = np.minimum(r.wall_thickness_m[1],
                       r.max_thickness_fraction * np.minimum(b, h))
    t = rng.uniform(r.wall_thickness_m[0], t_max, size=n)
    return b, h, t


def generate_dataset(
    n_samples: int | None = None,
    n_contexts: int | None = None,
    ranges: SamplingRanges | None = None,
    seed: int = 0,
    profile: str | None = None,
    device: str | None = None,
) -> Dataset:
    """Sample designs and contexts, evaluate on the GPU, return the pairs."""
    ranges = ranges or SamplingRanges()
    total = resolve_dataset_size(profile, n_samples)
    # Context coverage is what limits generalization to a NEW problem, and it
    # is much thinner than the row count suggests when many designs share one
    # context. Default to ~50 designs per context rather than a fixed count.
    if n_contexts is None:
        n_contexts = max(1, total // 50)
    n_contexts = max(1, min(n_contexts, total))
    per_context = max(1, total // n_contexts)

    rng = np.random.default_rng(seed)
    all_inputs, all_outputs, all_context_ids = [], [], []

    def draw(name: str) -> float:
        lo, hi = getattr(ranges, name)
        if name in ranges.log_sampled:
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(rng.uniform(lo, hi))

    for context_index in range(n_contexts):
        case = BeamLoadCase(
            length_m=draw("length_m"),
            tip_load_n=draw("tip_load_n"),
            youngs_modulus_pa=draw("youngs_modulus_pa"),
            density_kg_m3=draw("density_kg_m3"),
            yield_strength_pa=draw("yield_strength_pa"),
        )
        b, h, t = _sample_designs(rng, per_context, ranges)
        metrics = evaluate_beam_case(b, h, t, case, device=device)

        context = np.tile(
            [case.length_m, case.tip_load_n, case.youngs_modulus_pa,
             case.density_kg_m3, case.yield_strength_pa], (per_context, 1))
        all_inputs.append(np.column_stack([context, b, h, t]))
        all_context_ids.append(np.full(per_context, context_index, dtype=np.int64))
        all_outputs.append(np.column_stack([
            metrics.mass_kg, metrics.max_bending_stress_pa,
            metrics.tip_deflection_m, metrics.first_natural_frequency_hz]))

    inputs = np.vstack(all_inputs)
    outputs = np.vstack(all_outputs)
    context_ids = np.concatenate(all_context_ids)

    # Any non-finite row means the kernel was handed something degenerate.
    finite = np.all(np.isfinite(inputs), axis=1) & np.all(np.isfinite(outputs), axis=1)
    positive = np.all(outputs > 0, axis=1)
    keep = finite & positive
    return Dataset(
        inputs=inputs[keep],
        outputs=outputs[keep],
        context_ids=context_ids[keep],
        meta={
            "seed": seed,
            "n_contexts": n_contexts,
            "per_context": per_context,
            "requested": total,
            "dropped": int((~keep).sum()),
            "ranges": ranges.as_dict(),
            "source": "physics.structural Euler-Bernoulli beam evaluator",
            "fidelity_note": (
                "Targets are Phase 2 beam-theory outputs, NOT 3D FEM. "
                "Surrogate error adds to beam-theory error."
            ),
        },
    )
