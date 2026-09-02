"""A surrogate over CAD shapes: predicts solver labels from descriptors, and
is allowed to do exactly one thing with them, which is to rank.

This is Phase 7 of the knowledge layer, and it is the phase the SURROGATE
evidence gate was built for. The model here learns the map from a part's
descriptors, size, material and load to what CalculiX said about it. Every
prediction it makes grades SURROGATE, carries the held-out error the model
showed on parts it did not train on, and cannot become a verdict: the
integration layer refuses a PASSED or FAILED built on it, and the screening
routine below returns only parts the solver actually ran on.

WHAT IS BEING APPROXIMATED
==========================
The labeller's cantilever case on the five synthetic families, solved by
CalculiX with quadratic tetrahedra. So the surrogate's error stacks on top of
the solver's own, and the solver's own peak stress does not converge at the
clamped edge (see labeller.py). A prediction of that peak is a prediction of
a number that was already mesh dependent, and the tests do not pretend
otherwise: the deflection is the target that is asked to be accurate.

WHY THIS CAN PAY WHERE THE BEAM SURROGATE COULD NOT
===================================================
The Phase 6 beam surrogate was slower than the closed-form kernel it
replaced. Here the evaluator is a mesh and a solve, 0.2 to 6 seconds per
part measured, and a forward pass is microseconds. The ratio has inverted,
which was the condition stated in surrogate/README.md for the machinery to
be worth anything. Whether the accuracy is good enough to rank on is the
question the measurements answer, and the answer depends on how many
labelled parts exist, which is a cost paid in solver time.

WHAT WAS MEASURED, 40 training parts and 15 held out, deflection spanning
three decades
==============================================================================
    beam theory proxy alone, no fit              R2 in log space   0.15
    proxy with a fitted exponent                                   0.53
    ridge on the 22 descriptors and sizes                          0.68
    ridge on those plus the proxy                                  0.91
    MLP on those plus the proxy, one layer of 32                   0.94
        relative error on held-out deflection    median 0.07   p95 0.47

    The same model on the same 40 parts, judged as a ranker: log-space R2
    0.97, Spearman rank correlation 0.99. On 20 training parts and 8 held
    out, over eight random draws: Spearman never below 0.79, log R2 never
    below 0.41, while the raw R2 fell to 0.29 because two large parts decide
    it. Raw R2 is reported and is the wrong number to read here.

    MLP on the descriptors and sizes WITHOUT the proxy: R2 below zero, median
    relative error 0.8. Forty parts across three decades are not enough to
    learn a cubic in length from scratch, and they are enough to learn the
    residual on top of one. So the proxy is a feature: a cantilever's
    deflection from beam theory using the bounding box as a solid section
    scaled by fill. It is deliberately crude, because a good proxy would be
    the answer and a crude one leaves the model something to learn. Peak
    von Mises reaches R2 0.96 the same way, but see the labeller on why that
    peak is not a converged number to begin with.

    A p95 error near fifty percent is what makes this a ranker and not a
    verdict. It sorts candidates well enough that the solver's time goes to
    the promising ones, and the gate keeps it from doing anything else.

VALIDITY DOMAIN
===============
    Parts of the five families, aluminium 7075 unless trained otherwise, the
    one cantilever load case. A query outside the training ranges is
    extrapolation and the model does not know it; the descriptor classifier
    can at least say UNKNOWN for a shape outside the families, and a caller
    who wants that guard has to ask for it. Nothing here is physically
    validated. SURROGATE, below SIMULATED, by construction.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from brain.semantic.evidence import Evidence, EvidenceKind, EvidenceLevel
from core.materials.db import MaterialSpec, get_material
from integration.checks import CheckResult, CheckStatus
from surrogate.models.mlp import (Standardizer, SurrogateMLP,
                                  evaluate_predictions, resolve_device)

from .descriptors import DESCRIPTOR_NAMES, ShapeDescriptor, describe_step
from .engine import GroundTruthMismatch, make_part
from .families import FAMILIES, Family, part_id_for
from .labeller import LoadCase
from .schema import PartRecord

FEATURE_NAMES: tuple[str, ...] = DESCRIPTOR_NAMES + (
    "log_volume_m3", "log_longest_m", "log_youngs_modulus_pa",
    "log_load_n", "load_dir_y", "load_dir_z", "log_beam_proxy_m")
TARGET_NAMES: tuple[str, ...] = ("tip_deflection_m", "max_displacement_m",
                                 "max_von_mises_pa")


def beam_proxy_m(record: PartRecord, material: MaterialSpec,
                 case: LoadCase) -> float:
    """Euler-Bernoulli tip deflection of the bounding box as a solid
    cantilever, with its second moment scaled by the fill ratio.

    Crude on purpose. It knows nothing about where the material sits in the
    section, so a hollow tube and a solid bar of the same fill get the same
    proxy; that is exactly the residual left for the model to learn. Its own
    R2 against the solver in log space was 0.15.
    """
    length, a, b = record.geometry.bounding_box_m
    height = (a, b)[case.direction - 1]
    width = (a, b)[2 - case.direction]
    fill = record.geometry.volume_m3 / (length * a * b)
    second_moment = width * height ** 3 / 12.0 * fill
    return abs(case.total_load_n) * length ** 3 / (
        3.0 * material.youngs_modulus_pa * second_moment)


def features_for(record: PartRecord, descriptor: ShapeDescriptor,
                 material: MaterialSpec, case: LoadCase) -> np.ndarray:
    """The input row for one part. Sizes are logged because the targets span
    decades and a linear size feature would let the largest parts set the
    scale for all of them."""
    row = [descriptor[n] for n in DESCRIPTOR_NAMES]
    row += [np.log(record.geometry.volume_m3),
            np.log(max(record.geometry.bounding_box_m)),
            np.log(material.youngs_modulus_pa),
            np.log(abs(case.total_load_n)),
            1.0 if case.direction == 1 else 0.0,
            1.0 if case.direction == 2 else 0.0,
            np.log(beam_proxy_m(record, material, case))]
    return np.asarray(row, dtype=np.float64)


def targets_for(record: PartRecord) -> np.ndarray:
    """Magnitudes of the three solver labels. The sign of the tip deflection
    is the sign of the load and carries no information about the part."""
    return np.array([abs(float(record.labels[n]["value"]))
                     for n in TARGET_NAMES])


@dataclass
class ShapeTrainingSet:
    x: np.ndarray
    y: np.ndarray
    part_ids: list[str]
    families: list[str]

    def __len__(self) -> int:
        return len(self.part_ids)


def training_set_from(records: Sequence[PartRecord], step_dir: str | Path,
                      material: MaterialSpec) -> ShapeTrainingSet:
    """Rows from labelled records whose STEP files are in `step_dir`."""
    xs, ys, ids, fams = [], [], [], []
    for record in records:
        if "tip_deflection_m" not in record.labels:
            continue
        case_dict = record.labels["load_case"]
        case = LoadCase(total_load_n=case_dict["total_load_n"],
                        direction=case_dict["direction"])
        descriptor = describe_step(Path(step_dir) / f"{record.part_id}.step")[0]
        xs.append(features_for(record, descriptor, material, case))
        ys.append(targets_for(record))
        ids.append(record.part_id)
        fams.append(record.provenance.generator)
    if not xs:
        raise ValueError("no labelled records to train on")
    return ShapeTrainingSet(x=np.array(xs), y=np.array(ys), part_ids=ids,
                            families=fams)


# ------------------------------------------------------------------ model

@dataclass
class ShapePrediction:
    """Predicted magnitudes for a batch of parts, graded SURROGATE."""

    values: dict[str, np.ndarray]
    expected_relative_error: dict[str, float]   # p95 on held-out parts
    verified: bool = False                      # never true for a prediction

    @property
    def evidence_kind(self) -> EvidenceKind:
        return EvidenceKind.SURROGATE

    @property
    def evidence_level(self) -> EvidenceLevel:
        return EvidenceLevel.SURROGATE

    def as_evidence(self, ref: str, run_id: str | None = None) -> Evidence:
        return Evidence(kind=EvidenceKind.SURROGATE, ref=ref, run_id=run_id,
                        note="shape surrogate prediction; not a solve")

    def screened_check(self, component: str, failure_mode: str, i: int,
                       limit_m: float) -> CheckResult:
        """The only CheckResult a prediction may become: SCREENED, a gap."""
        predicted = float(self.values["tip_deflection_m"][i])
        err = self.expected_relative_error.get("tip_deflection_m", 0.0)
        return CheckResult(
            component=component, failure_mode=failure_mode,
            status=CheckStatus.SCREENED, method="shape_surrogate",
            detail=(f"surrogate predicts deflection {predicted:.3e} m against "
                    f"a limit of {limit_m:.3e} m, p95 held-out error "
                    f"{err:.0%}; not a verdict, run the solver"),
            evidence_kind=EvidenceKind.SURROGATE)


@dataclass
class ShapeSurrogate:
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

    def predict(self, x: np.ndarray) -> ShapePrediction:
        raw = self.predict_array(np.atleast_2d(x))
        return ShapePrediction(
            values={name: raw[:, i] for i, name in enumerate(TARGET_NAMES)},
            expected_relative_error=self.expected_relative_error)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        (directory / "meta.json").write_text(json.dumps({
            "x_scaler": self.x_scaler.as_dict(),
            "y_scaler": self.y_scaler.as_dict(),
            "test_metrics": self.test_metrics, "training": self.training,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "evidence": EvidenceLevel.SURROGATE.value}, indent=2))
        return directory

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None
             ) -> "ShapeSurrogate":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        if meta["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("saved surrogate was trained on different features")
        dev = resolve_device(device)
        model = SurrogateMLP(len(FEATURE_NAMES), len(TARGET_NAMES),
                             hidden=tuple(meta["training"]["hidden"]))
        model.load_state_dict(torch.load(directory / "model.pt",
                                         map_location=dev))
        model.to(dev)
        return cls(model=model, x_scaler=Standardizer.from_dict(meta["x_scaler"]),
                   y_scaler=Standardizer.from_dict(meta["y_scaler"]),
                   test_metrics=meta["test_metrics"], training=meta["training"],
                   device=str(dev))


def train_shape_surrogate(train: ShapeTrainingSet, test: ShapeTrainingSet,
                          epochs: int = 2000, seed: int = 0, lr: float = 3e-3,
                          hidden: tuple[int, ...] = (32,),
                          weight_decay: float = 1e-3,
                          device: str | None = None) -> ShapeSurrogate:
    """Fit in log space on the training set; report error on the test set.

    The test set is what the quoted error comes from, so it must hold parts
    the model never saw. Callers split by generation seed, which is the
    honest split here: the same seed makes the same parts.
    """
    dev = resolve_device(device)
    torch.manual_seed(seed)
    x_scaler = Standardizer.fit(train.x)
    y_log = np.log(np.clip(train.y, 1e-30, None))
    y_scaler = Standardizer.fit(y_log)
    x = torch.as_tensor(x_scaler.transform(train.x), dtype=torch.float32,
                        device=dev)
    y = torch.as_tensor(y_scaler.transform(y_log), dtype=torch.float32,
                        device=dev)
    model = SurrogateMLP(len(FEATURE_NAMES), len(TARGET_NAMES), hidden).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    started = time.perf_counter()
    losses = []
    for _ in range(epochs):
        model.train()
        loss = nn.functional.mse_loss(model(x), y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    surrogate = ShapeSurrogate(model=model, x_scaler=x_scaler,
                               y_scaler=y_scaler, device=str(dev))
    surrogate.training = {"epochs": epochs, "hidden": list(hidden), "lr": lr,
                          "n_train": len(train), "n_test": len(test),
                          "first_loss": losses[0], "final_loss": losses[-1],
                          "seconds": time.perf_counter() - started}
    predicted = surrogate.predict_array(test.x)
    surrogate.test_metrics = evaluate_predictions(predicted, test.y, TARGET_NAMES)
    for i, name in enumerate(TARGET_NAMES):
        surrogate.test_metrics[name].update(
            ranking_metrics(predicted[:, i], test.y[:, i]))
    return surrogate


def ranking_metrics(predicted: np.ndarray, true: np.ndarray) -> dict:
    """What a ranker is judged on, which the raw R2 is not.

    Targets here span three decades, so an R2 on raw values is decided by
    the one or two largest parts and says nothing about whether the small
    ones are in the right order. The log-space R2 weights every decade
    alike, and the Spearman correlation is the order itself: the fraction of
    the ranking a screening pass would get right.
    """
    p = np.log(np.clip(predicted, 1e-30, None))
    t = np.log(np.clip(true, 1e-30, None))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2_log = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rank_p = np.argsort(np.argsort(p)).astype(float)
    rank_t = np.argsort(np.argsort(t)).astype(float)
    if len(t) > 1 and rank_p.std() > 0 and rank_t.std() > 0:
        spearman = float(np.corrcoef(rank_p, rank_t)[0, 1])
    else:
        spearman = float("nan")
    return {"r2_log": r2_log, "spearman": spearman}


# ------------------------------------------------- screen, then verify

@dataclass
class ShapeScreeningResult:
    """A solver-verified winner, or nothing. Never a predicted one."""

    winner: PartRecord | None
    verified: bool
    n_screened: int
    n_verified: int
    predicted_deflection_m: float | None = None
    solver_deflection_m: float | None = None
    shortlist: list[str] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)

    @property
    def evidence_level(self) -> EvidenceLevel:
        return (EvidenceLevel.SIMULATED if self.verified
                else EvidenceLevel.SURROGATE)

    @property
    def surrogate_error_on_winner(self) -> float | None:
        if self.predicted_deflection_m is None or not self.solver_deflection_m:
            return None
        return abs(self.predicted_deflection_m - abs(self.solver_deflection_m)) \
            / abs(self.solver_deflection_m)


def screen_and_verify_parts(surrogate: ShapeSurrogate,
                            candidates: Sequence[tuple[str, dict[str, float]]],
                            deflection_limit_m: float, step_dir: str | Path,
                            top_k: int = 4, material: MaterialSpec | None = None,
                            total_load_n: float = -100.0) -> ShapeScreeningResult:
    """Rank candidates by the surrogate, verify the top few with CalculiX,
    and return the lightest SOLVER-verified part inside the deflection limit.

    Stage 1 builds every candidate unlabelled (cheap: a B-rep and a STEP
    read) and predicts its deflection. Candidates predicted feasible are
    ordered lightest first, then the predicted-infeasible ones, since the
    prediction is itself approximate and must not hard-filter. Stage 2
    labels the top_k with the real solver. The winner's numbers are the
    solver's, and the surrogate's prediction for it is kept alongside only
    so the two can be compared.
    """
    material = material or get_material("al_7075_t6")
    step_dir = Path(step_dir)
    rows, records, refused = [], [], []
    for name, params in candidates:
        fam = FAMILIES[name]
        try:
            record, _ = make_part(fam, params, step_dir, labelled=False)
        except (GroundTruthMismatch, ValueError) as exc:
            refused.append((part_id_for(fam, params), str(exc)))
            continue
        case = LoadCase(total_load_n=total_load_n, direction=fam.load_direction)
        descriptor = describe_step(step_dir / f"{record.part_id}.step")[0]
        rows.append(features_for(record, descriptor, material, case))
        records.append((fam, params, record))
    if not rows:
        return ShapeScreeningResult(winner=None, verified=False,
                                    n_screened=len(candidates), n_verified=0,
                                    refused=refused)

    prediction = surrogate.predict(np.array(rows))
    predicted = prediction.values["tip_deflection_m"]
    masses = np.array([r.geometry.volume_m3 * material.density_kg_m3
                       for _, _, r in records])
    feasible = predicted <= deflection_limit_m
    order = sorted(range(len(records)),
                   key=lambda i: (not bool(feasible[i]), float(masses[i])))
    chosen = order[:min(top_k, len(order))]

    best, best_mass, best_pred, best_solver = None, np.inf, None, None
    for i in chosen:
        fam, params, _ = records[i]
        record, _ = make_part(fam, params, step_dir, material,
                              LoadCase(total_load_n=total_load_n,
                                       direction=fam.load_direction),
                              labelled=True)
        solved = abs(float(record.labels["tip_deflection_m"]["value"]))
        if solved <= deflection_limit_m and masses[i] < best_mass:
            best, best_mass = record, float(masses[i])
            best_pred, best_solver = float(predicted[i]), solved
    return ShapeScreeningResult(
        winner=best, verified=best is not None, n_screened=len(records),
        n_verified=len(chosen), predicted_deflection_m=best_pred,
        solver_deflection_m=best_solver,
        shortlist=[records[i][2].part_id for i in chosen], refused=refused)
