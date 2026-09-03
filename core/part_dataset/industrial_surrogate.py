"""A surrogate over the industrial dataset: thirteen families, five load cases,
every isotropic material in the table. It ranks, and that is all.

The Phase 7 surrogate learned one load case on five families and one
material from forty parts. This module is the same idea applied to a run of
`scripts/generate_industrial_dataset.py`: solved records for every cell,
their scaled copies for the other materials, and a held-out fifth of every
cell chosen by draw order, so that every family and every load case is in
the test set and a part's scaled copies stay on the same side as the part.

WHAT IS BEING APPROXIMATED
==========================
The labeller's primary response for each load kind (tip deflection,
elongation, twist, thermal tip deflection), the maximum displacement and the
peak von Mises stress, as CalculiX reported them with quadratic tetrahedra.
The stress peak is mesh dependent (mesh_sensitivity is stored with every
label, and on holed parts it reaches half) and no surrogate can be better
than its labels; the primary response is the number that is asked to rank.

WHAT THE PROXIES ARE
====================
One closed form per load kind on the bounding box, with the section scaled
by the fill ratio: Euler-Bernoulli bending, F L / (E A) extension, T L / (G J)
twist with the rectangular polar moment, and alpha g L^2 / 2 thermal
curvature. Crude on purpose, as in shape_surrogate.py: a good proxy would be
the answer, and a crude one leaves the residual to learn. Each proxy is
computed for every record whose load case has the quantity it needs and is
zero otherwise; the load kind is also given to the model one-hot.

WHAT WAS MEASURED
=================
The numbers for run 1 are in docs/dataset_spec.md, under "Surrogate on run
1", and in DESIGN.md. They are a measurement on one run, not a property of
the method, and the tests here check the mechanics on a small corpus, not
the accuracy on the run, because the run is not in the repository.

VALIDITY DOMAIN
===============
Parts of the thirteen families inside the sampled parameter ranges, the
five load cases at the loads the run used, the isotropic materials the run
scaled to. Everything outside is extrapolation the model cannot detect.
Every prediction is SURROGATE, below SIMULATED; the integration layer
refuses a verdict built on it, and `screened_check` is the only CheckResult
a prediction may become.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from brain.semantic.evidence import EvidenceKind, EvidenceLevel
from core.materials.db import MaterialSpec, get_material
from integration.checks import CheckResult, CheckStatus
from surrogate.models.mlp import (Standardizer, SurrogateMLP,
                                  evaluate_predictions, resolve_device)

from .descriptors import DESCRIPTOR_NAMES, describe_step
from .labeller import PRIMARY_LABEL, LoadKind
from .schema import PartRecord, validate_record
from .shape_surrogate import ranking_metrics

KINDS: tuple[LoadKind, ...] = (LoadKind.BENDING, LoadKind.AXIAL, LoadKind.TORSION,
                               LoadKind.COMBINED, LoadKind.THERMAL_GRADIENT)

FEATURE_NAMES: tuple[str, ...] = DESCRIPTOR_NAMES + (
    "log_volume_m3", "log_length_m", "log_height_m", "log_width_m",
    "log_youngs_modulus_pa", "log_shear_modulus_pa", "log_expansion_1_k",
    "log_load_n", "log_torque_nm", "log_gradient_k_per_m",
    "load_dir_y", "load_dir_z",
) + tuple(f"kind_{k.value}" for k in KINDS) + (
    "log_bending_proxy_m", "log_axial_proxy_m", "log_torsion_proxy_rad",
    "log_thermal_proxy_m")

TARGET_NAMES: tuple[str, ...] = ("primary_response", "max_displacement_m",
                                 "max_von_mises_pa")

HOLDOUT_FRACTION = 0.2


# ------------------------------------------------------------------ cases

@dataclass(frozen=True)
class CaseValues:
    """What a record's load_case label says, as numbers."""
    kind: LoadKind
    direction: int
    total_load_n: float
    torque_nm: float
    gradient_k_per_m: float

    @classmethod
    def from_record(cls, record: PartRecord) -> "CaseValues":
        c = record.labels["load_case"]
        return cls(kind=LoadKind(c["load_kind"]), direction=int(c["direction"]),
                   total_load_n=float(c.get("total_load_n", 0.0)),
                   torque_nm=float(c.get("torque_nm", 0.0)),
                   gradient_k_per_m=float(c.get("gradient_k_per_m", 0.0)))


def _section(record: PartRecord, direction: int) -> tuple[float, float, float, float]:
    """Length, height (along the load direction), width, fill ratio."""
    length, a, b = record.geometry.bounding_box_m
    height = (a, b)[direction - 1]
    width = (a, b)[2 - direction]
    fill = record.geometry.volume_m3 / (length * a * b)
    return length, height, width, fill


def bending_proxy_m(record: PartRecord, material: MaterialSpec, case: CaseValues) -> float:
    length, height, width, fill = _section(record, case.direction)
    second_moment = width * height ** 3 / 12.0 * fill
    return abs(case.total_load_n) * length ** 3 / (3.0 * material.youngs_modulus_pa * second_moment)


def axial_proxy_m(record: PartRecord, material: MaterialSpec, case: CaseValues) -> float:
    length, height, width, fill = _section(record, case.direction)
    return abs(case.total_load_n) * length / (material.youngs_modulus_pa * height * width * fill)


def torsion_proxy_rad(record: PartRecord, material: MaterialSpec, case: CaseValues) -> float:
    length, height, width, fill = _section(record, case.direction)
    polar = height * width * (height ** 2 + width ** 2) / 12.0 * fill
    return abs(case.torque_nm) * length / (material.shear_modulus_pa * polar)


def thermal_proxy_m(record: PartRecord, material: MaterialSpec, case: CaseValues) -> float:
    if material.thermal_expansion_1_k is None:
        return 0.0
    length = record.geometry.bounding_box_m[0]
    return material.thermal_expansion_1_k * abs(case.gradient_k_per_m) * length ** 2 / 2.0


def _log_or_zero(value: float) -> float:
    return float(np.log(value)) if value > 0.0 else 0.0


def features_for(record: PartRecord, descriptor: Sequence[float],
                 material: MaterialSpec, case: CaseValues) -> np.ndarray:
    length, height, width, _fill = _section(record, case.direction)
    uses_force = case.kind in (LoadKind.BENDING, LoadKind.AXIAL, LoadKind.COMBINED)
    uses_torque = case.kind in (LoadKind.TORSION, LoadKind.COMBINED)
    thermal = case.kind is LoadKind.THERMAL_GRADIENT
    row = list(descriptor)
    row += [np.log(record.geometry.volume_m3), np.log(length), np.log(height),
            np.log(width), np.log(material.youngs_modulus_pa),
            np.log(material.shear_modulus_pa),
            _log_or_zero(material.thermal_expansion_1_k or 0.0),
            _log_or_zero(abs(case.total_load_n)) if uses_force else 0.0,
            _log_or_zero(abs(case.torque_nm)) if uses_torque else 0.0,
            _log_or_zero(abs(case.gradient_k_per_m)) if thermal else 0.0,
            1.0 if case.direction == 1 else 0.0,
            1.0 if case.direction == 2 else 0.0]
    row += [1.0 if case.kind is k else 0.0 for k in KINDS]
    row += [_log_or_zero(bending_proxy_m(record, material, case)) if uses_force and case.kind is not LoadKind.AXIAL else 0.0,
            _log_or_zero(axial_proxy_m(record, material, case)) if case.kind is LoadKind.AXIAL else 0.0,
            _log_or_zero(torsion_proxy_rad(record, material, case)) if uses_torque else 0.0,
            _log_or_zero(thermal_proxy_m(record, material, case)) if thermal else 0.0]
    return np.asarray(row, dtype=np.float64)


def targets_for(record: PartRecord, case: CaseValues) -> np.ndarray:
    primary = PRIMARY_LABEL[case.kind][0]
    return np.array([abs(float(record.labels[primary]["value"])),
                     abs(float(record.labels["max_displacement_m"]["value"])),
                     abs(float(record.labels["max_von_mises_pa"]["value"]))])


# ---------------------------------------------------------------- corpus

@dataclass
class IndustrialSet:
    x: np.ndarray
    y: np.ndarray
    part_ids: list[str]
    base_ids: list[str]
    families: list[str]
    kinds: list[str]
    materials: list[str]

    def __len__(self) -> int:
        return len(self.part_ids)

    def subset(self, mask: np.ndarray) -> "IndustrialSet":
        idx = np.flatnonzero(mask)
        pick = lambda xs: [xs[i] for i in idx]  # noqa: E731
        return IndustrialSet(x=self.x[idx], y=self.y[idx], part_ids=pick(self.part_ids),
                             base_ids=pick(self.base_ids), families=pick(self.families),
                             kinds=pick(self.kinds), materials=pick(self.materials))


def base_id_of(record: PartRecord) -> str:
    """The solved part behind a scaled copy: the copy's id is the base id with
    the target material appended."""
    suffix = f"-{record.material_id}"
    if record.part_id.endswith(suffix) and any(
            isinstance(v, dict) and v.get("derived") for v in record.labels.values()):
        return record.part_id[: -len(suffix)]
    return record.part_id


def _describe(args: tuple[str, str]) -> tuple[str, list[float]]:
    part_id, path = args
    d = describe_step(path)[0]
    return part_id, [float(d[n]) for n in DESCRIPTOR_NAMES]


def cache_descriptors(root: str | Path, workers: int = 1) -> dict[str, list[float]]:
    """Descriptors for every solved part in a run, computed once and kept in
    `descriptors.json` next to the cells. 0.12 s per part measured."""
    root = Path(root)
    cache_path = root / "descriptors.json"
    cache: dict[str, list[float]] = (json.loads(cache_path.read_text())
                                     if cache_path.exists() else {})
    todo = []
    for cell_dir in sorted(p for p in root.iterdir() if (p / "records.jsonl").exists()):
        for step in sorted((cell_dir / "step").glob("*.step")):
            if step.stem not in cache:
                todo.append((step.stem, str(step)))
    if todo:
        if workers > 1:
            with get_context("spawn").Pool(workers) as pool:
                for part_id, values in pool.imap_unordered(_describe, todo, chunksize=16):
                    cache[part_id] = values
        else:
            for item in todo:
                part_id, values = _describe(item)
                cache[part_id] = values
        cache_path.write_text(json.dumps(cache))
    return cache


def draw_order(root: str | Path) -> dict[str, int]:
    """Position of every part in its cell's draw order, from the done files,
    refused parts included so the positions are the sampler's."""
    order: dict[str, int] = {}
    for done in Path(root).glob("*/done"):
        for i, part_id in enumerate(done.read_text().split()):
            order[part_id] = i
    return order


def _iter_records(path: Path) -> Iterable[PartRecord]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield validate_record(json.loads(line))


def load_run(root: str | Path, descriptors: dict[str, list[float]],
             include_scaled: bool = True,
             materials: dict[str, MaterialSpec] | None = None) -> IndustrialSet:
    """Every solved record in a run, and its scaled copies, as rows."""
    root = Path(root)
    materials = materials or {}
    xs, ys, ids, bases, fams, kinds, mats = [], [], [], [], [], [], []
    paths = sorted(root.glob("*/records.jsonl"))
    if include_scaled:
        paths += sorted((root / "scaled").glob("*.jsonl"))
    for path in paths:
        for record in _iter_records(path):
            base = base_id_of(record)
            if base not in descriptors:
                continue
            case = CaseValues.from_record(record)
            if PRIMARY_LABEL[case.kind][0] not in record.labels:
                continue
            material = materials.get(record.material_id)
            if material is None:
                material = materials[record.material_id] = get_material(record.material_id)
            xs.append(features_for(record, descriptors[base], material, case))
            ys.append(targets_for(record, case))
            ids.append(record.part_id)
            bases.append(base)
            fams.append(record.provenance.generator)
            kinds.append(case.kind.value)
            mats.append(record.material_id)
    if not xs:
        raise ValueError(f"no usable records under {root}")
    return IndustrialSet(x=np.array(xs), y=np.array(ys), part_ids=ids, base_ids=bases,
                         families=fams, kinds=kinds, materials=mats)


def holdout_mask(data: IndustrialSet, order: dict[str, int], samples_per_cell: int,
                 fraction: float = HOLDOUT_FRACTION) -> np.ndarray:
    """True for rows whose base part was drawn in the last `fraction` of its
    cell. Scaled copies follow their base part."""
    cut = int(round(samples_per_cell * (1.0 - fraction)))
    return np.array([order.get(b, 0) >= cut for b in data.base_ids])


# ----------------------------------------------------------------- model

@dataclass
class IndustrialPrediction:
    values: dict[str, np.ndarray]
    expected_relative_error: dict[str, float]
    verified: bool = False

    @property
    def evidence_kind(self) -> EvidenceKind:
        return EvidenceKind.SURROGATE

    @property
    def evidence_level(self) -> EvidenceLevel:
        return EvidenceLevel.SURROGATE

    def screened_check(self, component: str, failure_mode: str, i: int,
                       limit: float, quantity: str = "primary_response") -> CheckResult:
        predicted = float(self.values[quantity][i])
        err = self.expected_relative_error.get(quantity, float("nan"))
        return CheckResult(
            component=component, failure_mode=failure_mode,
            status=CheckStatus.SCREENED, method="industrial_surrogate",
            detail=(f"surrogate predicts {quantity} {predicted:.3e} against a limit "
                    f"of {limit:.3e}, p95 held-out error {err:.0%}; not a verdict, "
                    f"run the solver"),
            evidence_kind=EvidenceKind.SURROGATE)


@dataclass
class IndustrialSurrogate:
    model: SurrogateMLP
    x_scaler: Standardizer
    y_scaler: Standardizer
    test_metrics: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    device: str = "cpu"

    @property
    def expected_relative_error(self) -> dict[str, float]:
        return {name: float(stats.get("p95_rel_err", float("nan")))
                for name, stats in self.test_metrics.items()}

    def predict_array(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            z = torch.as_tensor(self.x_scaler.transform(x), dtype=torch.float32,
                                device=self.device)
            out = self.model(z).cpu().numpy()
        return np.exp(self.y_scaler.inverse(out))

    def predict(self, x: np.ndarray) -> IndustrialPrediction:
        raw = self.predict_array(np.atleast_2d(x))
        return IndustrialPrediction(
            values={name: raw[:, i] for i, name in enumerate(TARGET_NAMES)},
            expected_relative_error=self.expected_relative_error)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        (directory / "meta.json").write_text(json.dumps({
            "x_scaler": self.x_scaler.as_dict(), "y_scaler": self.y_scaler.as_dict(),
            "test_metrics": self.test_metrics, "training": self.training,
            "feature_names": list(FEATURE_NAMES), "target_names": list(TARGET_NAMES),
            "evidence": EvidenceLevel.SURROGATE.value}, indent=2))
        return directory

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None) -> "IndustrialSurrogate":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        n_features = meta["training"].get("n_features", len(meta["feature_names"]))
        if n_features == len(FEATURE_NAMES) and meta["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("saved surrogate was trained on different features")
        dev = resolve_device(device)
        model = SurrogateMLP(n_features, len(TARGET_NAMES),
                             hidden=tuple(meta["training"]["hidden"]))
        model.load_state_dict(torch.load(directory / "model.pt", map_location=dev))
        model.to(dev)
        return cls(model=model, x_scaler=Standardizer.from_dict(meta["x_scaler"]),
                   y_scaler=Standardizer.from_dict(meta["y_scaler"]),
                   test_metrics=meta["test_metrics"], training=meta["training"],
                   device=str(dev))


def train_industrial_surrogate(train: IndustrialSet, test: IndustrialSet,
                               epochs: int = 3000, seed: int = 0, lr: float = 3e-3,
                               hidden: tuple[int, ...] = (64, 64),
                               weight_decay: float = 1e-4, batch: int | None = None,
                               device: str | None = None) -> IndustrialSurrogate:
    """Fit in log space; quote the error on parts the model never saw."""
    dev = resolve_device(device)
    torch.manual_seed(seed)
    x_scaler = Standardizer.fit(train.x)
    y_log = np.log(np.clip(train.y, 1e-30, None))
    y_scaler = Standardizer.fit(y_log)
    x = torch.as_tensor(x_scaler.transform(train.x), dtype=torch.float32, device=dev)
    y = torch.as_tensor(y_scaler.transform(y_log), dtype=torch.float32, device=dev)
    # The width comes from the data, not from FEATURE_NAMES: the ablation
    # trains the same model on the columns left after the proxies are removed.
    model = SurrogateMLP(train.x.shape[1], len(TARGET_NAMES), hidden).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    losses = []
    n = len(train)
    for _ in range(epochs):
        model.train()
        if batch is None or batch >= n:
            loss = nn.functional.mse_loss(model(x), y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        else:
            perm = torch.randperm(n, generator=generator).to(dev)
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                loss = nn.functional.mse_loss(model(x[idx]), y[idx])
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        scheduler.step()
        losses.append(loss.item())
    surrogate = IndustrialSurrogate(model=model, x_scaler=x_scaler, y_scaler=y_scaler,
                                    device=str(dev))
    surrogate.training = {"epochs": epochs, "hidden": list(hidden), "lr": lr,
                          "batch": batch, "n_features": int(train.x.shape[1]),
                          "n_train": len(train), "n_test": len(test),
                          "first_loss": losses[0], "final_loss": losses[-1],
                          "seconds": time.perf_counter() - started}
    predicted = surrogate.predict_array(test.x)
    surrogate.test_metrics = evaluate_predictions(predicted, test.y, TARGET_NAMES)
    for i, name in enumerate(TARGET_NAMES):
        surrogate.test_metrics[name].update(ranking_metrics(predicted[:, i], test.y[:, i]))
    return surrogate


# --------------------------------------------------------------- metrics

def metrics_by_group(surrogate: IndustrialSurrogate, data: IndustrialSet,
                     keys: Sequence[str] = ("families", "kinds"),
                     target: str = "primary_response") -> list[dict]:
    """Held-out ranking and error per group: Spearman, log R2, median and p95
    relative error, and the count, for the given target."""
    i = TARGET_NAMES.index(target)
    predicted = surrogate.predict_array(data.x)[:, i]
    true = data.y[:, i]
    groups = list(zip(*[getattr(data, k) for k in keys]))
    rows = []
    for g in sorted(set(groups)):
        mask = np.array([x == g for x in groups])
        p, t = predicted[mask], true[mask]
        rel = np.abs(p - t) / np.maximum(np.abs(t), 1e-30)
        r = ranking_metrics(p, t)
        rows.append({**dict(zip(keys, g)), "n": int(mask.sum()),
                     "spearman": r["spearman"], "r2_log": r["r2_log"],
                     "median_rel_err": float(np.median(rel)),
                     "p95_rel_err": float(np.percentile(rel, 95))})
    return rows


def format_table(rows: list[dict], keys: Sequence[str]) -> str:
    head = "| " + " | ".join(list(keys) + ["n", "Spearman", "log R2", "median err", "p95 err"]) + " |"
    sep = "|" + "---|" * (len(keys) + 5)
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join([str(r[k]) for k in keys] + [
            str(r["n"]), f"{r['spearman']:.2f}", f"{r['r2_log']:.2f}",
            f"{r['median_rel_err']:.2f}", f"{r['p95_rel_err']:.2f}"]) + " |")
    return "\n".join(lines)


# -------------------------------------------------------------- baselines

PROXY_COLUMNS: tuple[str, ...] = ("log_bending_proxy_m", "log_axial_proxy_m",
                                  "log_torsion_proxy_rad", "log_thermal_proxy_m")

#: Which proxy is the closed form for which load kind. The combined case uses
#: the bending proxy, because its primary response is the tip deflection.
PROXY_FOR_KIND = {LoadKind.BENDING: "log_bending_proxy_m",
                  LoadKind.AXIAL: "log_axial_proxy_m",
                  LoadKind.TORSION: "log_torsion_proxy_rad",
                  LoadKind.COMBINED: "log_bending_proxy_m",
                  LoadKind.THERMAL_GRADIENT: "log_thermal_proxy_m"}


def proxy_prediction(data: IndustrialSet) -> np.ndarray:
    """The closed form alone as a prediction of the primary response.

    This is the baseline every learned number has to beat. Inside one family
    and one load case the parts differ only in size, and a beam formula
    already orders those, so a Spearman near one on such a group says nothing
    about the model until this row is next to it.
    """
    idx = {name: FEATURE_NAMES.index(name) for name in PROXY_COLUMNS}
    out = np.empty(len(data))
    for i, kind in enumerate(data.kinds):
        out[i] = np.exp(data.x[i, idx[PROXY_FOR_KIND[LoadKind(kind)]]])
    return out


def baseline_metrics(data: IndustrialSet, keys: Sequence[str] = ("kinds",)) -> list[dict]:
    """The proxy alone, grouped, on the same rows and the same measures as the
    model, so the two tables can be read side by side."""
    predicted = proxy_prediction(data)
    true = data.y[:, TARGET_NAMES.index("primary_response")]
    groups = list(zip(*[getattr(data, k) for k in keys]))
    rows = []
    for g in sorted(set(groups)):
        mask = np.array([x == g for x in groups])
        p, t = predicted[mask], true[mask]
        rel = np.abs(p - t) / np.maximum(np.abs(t), 1e-30)
        r = ranking_metrics(p, t)
        rows.append({**dict(zip(keys, g)), "n": int(mask.sum()),
                     "spearman": r["spearman"], "r2_log": r["r2_log"],
                     "median_rel_err": float(np.median(rel)),
                     "p95_rel_err": float(np.percentile(rel, 95))})
    return rows


def without_proxies(data: IndustrialSet) -> IndustrialSet:
    """The same rows with the proxy columns removed, for the ablation that
    says whether the closed forms are carrying the model."""
    keep = [i for i, name in enumerate(FEATURE_NAMES) if name not in PROXY_COLUMNS]
    return IndustrialSet(x=data.x[:, keep], y=data.y, part_ids=data.part_ids,
                         base_ids=data.base_ids, families=data.families,
                         kinds=data.kinds, materials=data.materials)
