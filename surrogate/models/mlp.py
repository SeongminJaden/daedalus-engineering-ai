"""surrogate.models.mlp - the surrogate network and its training pipeline.

A small MLP is the right size here: the mapping is smooth and low-dimensional
(8 inputs, 4 outputs), so capacity is not the limiting factor and a big network
would only make inference slower for no accuracy.

Targets are learned in **log space**. Deflection alone spans about seven orders
of magnitude across the sampled space; a plain MSE on raw values would be
dominated entirely by the largest samples and the model would be useless in the
small-deflection regime that actually matters for a stiff design.

Safety factor is never predicted - it is yield / stress exactly, and deriving
it keeps it consistent with the stress the model actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from surrogate.datasets import INPUT_NAMES, OUTPUT_NAMES, Dataset


@dataclass
class Standardizer:
    """Zero-mean unit-variance scaling, fitted on train data only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        # A constant column would divide by zero; leave it untouched instead.
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std + self.mean

    def as_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(mean=np.array(d["mean"]), std=np.array(d["std"]))


class SurrogateMLP(nn.Module):
    def __init__(self, n_inputs: int = len(INPUT_NAMES),
                 n_outputs: int = len(OUTPUT_NAMES),
                 hidden: tuple[int, ...] = (128, 128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_inputs
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.SiLU()]
            prev = width
        layers.append(nn.Linear(prev, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainingReport:
    """What happened during training, and how wrong the model is on held-out
    data. The error table is carried into inference as the uncertainty a
    prediction should be quoted with."""

    epochs_run: int
    best_epoch: int
    train_loss: float
    val_loss: float
    test_metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "test_metrics": self.test_metrics,
        }


def _to_log(y: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(y, dtype=np.float64), 1e-30, None))


def _from_log(z: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(z, dtype=np.float64))


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_predictions(pred: np.ndarray, true: np.ndarray,
                         names: tuple[str, ...] = OUTPUT_NAMES) -> dict:
    """Per-metric relative error and R^2 on raw (not log) values."""
    out = {}
    for i, name in enumerate(names):
        p, t = pred[:, i], true[:, i]
        rel = np.abs(p - t) / np.maximum(np.abs(t), 1e-30)
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        out[name] = {
            "mean_rel_err": float(rel.mean()),
            "median_rel_err": float(np.median(rel)),
            "p95_rel_err": float(np.percentile(rel, 95)),
            "max_rel_err": float(rel.max()),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }
    return out


def train_surrogate(
    dataset: Dataset,
    seed: int = 0,
    hidden: tuple[int, ...] = (128, 128, 128),
    epochs: int = 400,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
    patience: int = 40,
    device: str | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    profile: str | None = None,
) -> tuple["SurrogateBundle", TrainingReport]:
    """Train with a deterministic split, early stopping and a held-out test set."""
    from core.profile import load_profile

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = resolve_device(device)

    if batch_size is None:
        cfg = load_profile(profile)
        batch_size = int(cfg["surrogate"].get("train_batch", 256))

    train, val, test = dataset.split(val_fraction, test_fraction, seed=seed)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"split produced an empty part (train={len(train)}, val={len(val)}, "
            f"test={len(test)}); the dataset is too small"
        )

    x_scaler = Standardizer.fit(train.inputs)
    y_scaler = Standardizer.fit(_to_log(train.outputs))

    def tensors(d: Dataset):
        x = torch.tensor(x_scaler.transform(d.inputs), dtype=torch.float32,
                         device=dev)
        y = torch.tensor(y_scaler.transform(_to_log(d.outputs)),
                         dtype=torch.float32, device=dev)
        return x, y

    x_tr, y_tr = tensors(train)
    x_va, y_va = tensors(val)

    model = SurrogateMLP(len(dataset.input_names), len(dataset.output_names),
                         hidden).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val, best_epoch, since_improved = float("inf"), 0, 0
    train_loss = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(x_tr.shape[0], generator=generator).to(dev)
        running, seen = 0.0, 0
        for start in range(0, x_tr.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * idx.numel()
            seen += idx.numel()
        train_loss = running / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_va), y_va).item())

        if val_loss < best_val - 1e-9:
            best_val, best_epoch, since_improved = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_improved += 1
            if since_improved >= patience:
                break

    model.load_state_dict(best_state)
    bundle = SurrogateBundle(model=model, x_scaler=x_scaler, y_scaler=y_scaler,
                             input_names=dataset.input_names,
                             output_names=dataset.output_names, device=dev)

    predictions = bundle.predict_array(test.inputs)
    report = TrainingReport(
        epochs_run=epoch,
        best_epoch=best_epoch,
        train_loss=train_loss,
        val_loss=best_val,
        test_metrics=evaluate_predictions(predictions, test.outputs,
                                          dataset.output_names),
    )
    bundle.test_metrics = report.test_metrics
    return bundle, report


@dataclass
class SurrogateBundle:
    """Model plus everything needed to use it: scalers, names, known error."""

    model: SurrogateMLP
    x_scaler: Standardizer
    y_scaler: Standardizer
    input_names: tuple[str, ...] = INPUT_NAMES
    output_names: tuple[str, ...] = OUTPUT_NAMES
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    test_metrics: dict = field(default_factory=dict)

    def predict_array(self, inputs: np.ndarray) -> np.ndarray:
        """Raw-space predictions for an (n, n_inputs) array."""
        inputs = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
        if inputs.shape[1] != len(self.input_names):
            raise ValueError(
                f"expected {len(self.input_names)} inputs "
                f"({self.input_names}), got {inputs.shape[1]}"
            )
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(self.x_scaler.transform(inputs),
                             dtype=torch.float32, device=self.device)
            z = self.model(x).cpu().numpy().astype(np.float64)
        return _from_log(self.y_scaler.inverse(z))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "hidden": [m.out_features for m in self.model.net
                       if isinstance(m, nn.Linear)][:-1],
            "x_scaler": self.x_scaler.as_dict(),
            "y_scaler": self.y_scaler.as_dict(),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "test_metrics": self.test_metrics,
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "SurrogateBundle":
        dev = resolve_device(device)
        blob = torch.load(Path(path), map_location=dev, weights_only=False)
        model = SurrogateMLP(len(blob["input_names"]), len(blob["output_names"]),
                             tuple(blob["hidden"])).to(dev)
        model.load_state_dict(blob["state_dict"])
        return cls(
            model=model,
            x_scaler=Standardizer.from_dict(blob["x_scaler"]),
            y_scaler=Standardizer.from_dict(blob["y_scaler"]),
            input_names=tuple(blob["input_names"]),
            output_names=tuple(blob["output_names"]),
            device=dev,
            test_metrics=blob.get("test_metrics", {}),
        )
