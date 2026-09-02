"""Resumable dataset generation, one cell at a time.

A cell is a family and a load case. Each cell has its own JSONL file of
labelled records, a `done` file of part ids already labelled, and a
`refused.jsonl` of parts that failed a ground truth check or a solve, with
the reason. A run that is interrupted loses at most the part it was on: the
next run reads the done set and skips those ids. Nothing is dropped silently;
the report counts refusals, and a run whose refusals rise is one whose
families or load cases have a problem, not one to be quietly retried.

Material scaling happens after labelling, not during it: the solver runs once
per part with the reference material, and `expand_materials` writes the
scaled records for every isotropic material whose load case has a measured
Poisson residual. Scaled records are graded derived and name their reference.

VALIDITY DOMAIN
===============
    Sampling is deterministic per cell from (seed, family, load case), so a
    resumed run draws the same parts it would have drawn uninterrupted. The
    cost estimate in `plan` is the measured per-part mean; it is a plan, and
    the run reports what it actually took.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from core.materials import MaterialClass, MaterialSpec, get_material, load_materials

from .engine import GroundTruthMismatch, LabellingFailed, make_part
from .families import FAMILIES, part_id_for, sample_parameters
from .labeller import LoadCase, LoadKind
from .scaling import POISSON_RESIDUAL_BOUND, UnscalableLabel, scale_record
from .schema import PartRecord, validate_record

#: Measured mean over 55 parts, two meshes and two solves each, 2026-09-02.
MEASURED_SECONDS_PER_PART = 3.2


@dataclass(frozen=True)
class Cell:
    family: str
    case: LoadCase

    @property
    def name(self) -> str:
        return f"{self.family}__{self.case.kind.value}"

    def seed(self, base: int) -> int:
        return (base * 1000003 + hash(self.name) % 1000003) % (2 ** 32)


@dataclass
class CellProgress:
    cell: Cell
    labelled: int
    refused: int
    seconds: float


@dataclass
class BatchReport:
    cells: list[CellProgress] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def labelled(self) -> int:
        return sum(c.labelled for c in self.cells)

    @property
    def refused(self) -> int:
        return sum(c.refused for c in self.cells)

    def summary(self) -> str:
        lines = [f"{self.labelled} labelled, {self.refused} refused, "
                 f"{self.seconds:.0f} s"]
        for c in self.cells:
            lines.append(f"  {c.cell.name}: {c.labelled} labelled, "
                         f"{c.refused} refused, {c.seconds:.0f} s")
        return "\n".join(lines)


def default_cases() -> tuple[LoadCase, ...]:
    return (LoadCase(total_load_n=-100.0, kind=LoadKind.BENDING),
            LoadCase(total_load_n=1000.0, kind=LoadKind.AXIAL),
            LoadCase(kind=LoadKind.TORSION, torque_nm=2.0),
            LoadCase(total_load_n=-100.0, kind=LoadKind.COMBINED, torque_nm=2.0),
            LoadCase(kind=LoadKind.THERMAL_GRADIENT, gradient_k_per_m=1000.0))


def cells_for(families: Sequence[str], cases: Sequence[LoadCase]) -> list[Cell]:
    return [Cell(f, c) for f in families for c in cases]


def plan(cells: Sequence[Cell], samples_per_cell: int,
         seconds_per_part: float = MEASURED_SECONDS_PER_PART,
         workers: int = 1) -> dict:
    """The bill before it is paid, from the measured per-part cost."""
    solves = len(cells) * samples_per_cell
    return {"cells": len(cells), "samples_per_cell": samples_per_cell,
            "solves": solves,
            "hours_one_worker": solves * seconds_per_part / 3600.0,
            "hours_at_workers": solves * seconds_per_part / 3600.0 / max(workers, 1),
            "seconds_per_part_assumed": seconds_per_part}


def _read_done(path: Path) -> set[str]:
    return set(path.read_text().split()) if path.exists() else set()


def run_cell(cell: Cell, samples: int, root: str | Path, seed: int = 0,
             material: MaterialSpec | None = None,
             stop_after_seconds: float | None = None) -> CellProgress:
    """Label up to `samples` parts for one cell, resuming from what exists.

    Parameters are drawn in a fixed order from the cell's seed, so the i-th
    part is the same on every run; parts whose ids are in the done file are
    skipped without being rebuilt. `stop_after_seconds` lets a caller bound
    one invocation and come back.
    """
    root = Path(root)
    directory = root / cell.name
    directory.mkdir(parents=True, exist_ok=True)
    records_path = directory / "records.jsonl"
    refused_path = directory / "refused.jsonl"
    done_path = directory / "done"
    step_dir = directory / "step"
    material = material or get_material("al_7075_t6")
    fam = FAMILIES[cell.family]
    rng = np.random.default_rng(cell.seed(seed))
    done = _read_done(done_path)
    started = time.perf_counter()
    labelled = refused = 0
    case = LoadCase(**{**cell.case.__dict__, "direction": fam.load_direction})
    for _ in range(samples):
        params = sample_parameters(fam, rng)
        part_id = part_id_for(fam, params)
        if part_id in done:
            continue
        if stop_after_seconds is not None and \
                time.perf_counter() - started > stop_after_seconds:
            break
        try:
            record, _ = make_part(fam, params, step_dir, material, case,
                                  labelled=True)
        except (GroundTruthMismatch, LabellingFailed) as exc:
            with refused_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"part_id": part_id, "family": fam.name,
                                         "load_kind": case.kind.value,
                                         "reason": str(exc)}) + "\n")
            refused += 1
        else:
            with records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.model_dump(mode="json"),
                                        sort_keys=True) + "\n")
            labelled += 1
        with done_path.open("a", encoding="utf-8") as handle:
            handle.write(part_id + "\n")
    return CellProgress(cell=cell, labelled=labelled, refused=refused,
                        seconds=time.perf_counter() - started)


def run_batch(cells: Sequence[Cell], samples_per_cell: int, root: str | Path,
              seed: int = 0, material: MaterialSpec | None = None,
              stop_after_seconds: float | None = None) -> BatchReport:
    """Every cell in turn, each resuming from its own files."""
    started = time.perf_counter()
    report = BatchReport()
    for cell in cells:
        remaining = (None if stop_after_seconds is None else
                     stop_after_seconds - (time.perf_counter() - started))
        if remaining is not None and remaining <= 0.0:
            break
        report.cells.append(run_cell(cell, samples_per_cell, root, seed,
                                     material, remaining))
    report.seconds = time.perf_counter() - started
    return report


def read_cell(root: str | Path, cell: Cell) -> list[PartRecord]:
    path = Path(root) / cell.name / "records.jsonl"
    if not path.exists():
        return []
    return [validate_record(json.loads(line))
            for line in path.read_text().splitlines() if line.strip()]


def expand_materials(records: Iterable[PartRecord], reference: MaterialSpec,
                     targets: Sequence[MaterialSpec] | None = None
                     ) -> tuple[list[PartRecord], list[tuple[str, str]]]:
    """Scaled copies of each record for every isotropic target material whose
    load case has a measured Poisson residual. Returns the records and the
    (part_id, reason) pairs that could not be scaled."""
    if targets is None:
        targets = [m for m in load_materials().materials
                   if m.material_class is MaterialClass.ISOTROPIC
                   and m.id != reference.id]
    out: list[PartRecord] = []
    skipped: list[tuple[str, str]] = []
    for record in records:
        for target in targets:
            try:
                out.append(scale_record(record, reference, target))
            except (UnscalableLabel, Exception) as exc:  # noqa: BLE001 - reported
                skipped.append((f"{record.part_id}-{target.id}", str(exc)))
    return out, skipped
