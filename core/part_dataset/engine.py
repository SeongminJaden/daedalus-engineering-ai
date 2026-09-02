"""The synthetic data engine: parameters in, checked and labelled records out.

For each part:

    1. sample admissible parameters for a family, deterministically
    2. build the B-rep and write it to STEP
    3. read the STEP back through the analyzer, and REFUSE the part if the
       analyzer's volume disagrees with the family's closed form
    4. run the feature recogniser, and REFUSE the part if it does not find
       exactly the features the parameters put there
    5. label through Gmsh and CalculiX, every label graded SIMULATED
    6. store the record with synthetic provenance under Apache-2.0

Steps 3 and 4 are the reason to generate data rather than scrape it. The
ground truth is known, so every record is checked against it, and a record
that fails the check is reported and dropped rather than stored with a note.
The report lists every refusal with its reason, because a generator that
silently drops parts is hiding a bug in itself.

Nothing here is registered as a capability. Making parts is not analysing
them, and the registry is for methods that answer a physical question.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from core.materials.db import MaterialSpec, get_material

from .families import (FAMILIES, Family, build, part_id_for,
                       sample_parameters)
from .labeller import LabelReport, LoadCase, cantilever_labels
from .schema import Licence, PartRecord, Provenance, ProvenanceKind
from .store import write_jsonl

#: Every part the engine makes carries this. The generator field is filled in
#: per family so that a record says which rule produced it.
SYNTHETIC_PROVENANCE = Provenance(
    kind=ProvenanceKind.SYNTHETIC_PARAMETRIC,
    source="daedalus synthetic engine",
    licence=Licence(identifier="Apache-2.0",
                    url="https://www.apache.org/licenses/LICENSE-2.0",
                    redistributable=True))

#: The analyzer must agree with the closed form to this. It is a B-rep of
#: planes and cylinders read back by the kernel that wrote it, so the
#: agreement is arithmetic and anything looser would hide a real fault.
VOLUME_TOLERANCE = 1e-6
FEATURE_TOLERANCE_M = 1e-9


class GroundTruthMismatch(ValueError):
    """The analyzer or the recogniser disagreed with the parameters."""


@dataclass
class GenerationReport:
    requested: int
    generated: int
    refused: list[tuple[str, str]] = field(default_factory=list)
    seconds: float = 0.0
    labelling_seconds: float = 0.0
    path: str | None = None
    per_family: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{self.generated} of {self.requested} parts in "
                 f"{self.seconds:.1f} s ({self.labelling_seconds:.1f} s "
                 f"labelling)"]
        for name, count in sorted(self.per_family.items()):
            lines.append(f"  {name}: {count}")
        for part_id, reason in self.refused:
            lines.append(f"  refused {part_id}: {reason}")
        return "\n".join(lines)


def _provenance_for(fam: Family) -> Provenance:
    return SYNTHETIC_PROVENANCE.model_copy(update={"generator": fam.name})


def _check_features(fam: Family, params, report) -> None:
    expected = fam.expected_features(params)
    if report.hole_count != expected.hole_count:
        raise GroundTruthMismatch(
            f"recogniser found {report.hole_count} holes, parameters put "
            f"{expected.hole_count}")
    if report.fillet_count != expected.fillet_count:
        raise GroundTruthMismatch(
            f"recogniser found {report.fillet_count} fillets, parameters put "
            f"{expected.fillet_count}")
    if expected.hole_diameter_m is not None:
        for d in report.hole_diameters_m():
            if abs(d - expected.hole_diameter_m) > FEATURE_TOLERANCE_M:
                raise GroundTruthMismatch(
                    f"hole diameter {d} differs from {expected.hole_diameter_m}")
    if expected.fillet_radius_m is not None:
        for r in report.fillet_radii_m():
            if abs(r - expected.fillet_radius_m) > FEATURE_TOLERANCE_M:
                raise GroundTruthMismatch(
                    f"fillet radius {r} differs from {expected.fillet_radius_m}")


def make_part(fam: Family, params: dict[str, float], step_dir: Path,
              material: MaterialSpec | None = None,
              load: LoadCase | None = None,
              labelled: bool = True) -> tuple[PartRecord, LabelReport | None]:
    """One part, checked against its own parameters and labelled.

    Raises GroundTruthMismatch when the analyzer or the recogniser disagrees
    with the parameters; the caller decides whether that stops the run.
    """
    from geometry.cad_export.kernel import require_kernel
    from nodes import feature_recognizer as fr
    from nodes import step_analyzer as sa

    kernel = require_kernel()
    part_id = part_id_for(fam, params)
    Path(step_dir).mkdir(parents=True, exist_ok=True)
    step_path = Path(step_dir) / f"{part_id}.step"
    solid = build(fam, params, kernel)
    kernel.module.export_step(solid, str(step_path))

    records = sa.analyse_step(step_path, _provenance_for(fam), part_id=part_id)
    if len(records) != 1:
        raise GroundTruthMismatch(f"STEP holds {len(records)} solids, not one")
    record = records[0]

    expected_volume = fam.volume_m3(params)
    error = abs(record.geometry.volume_m3 - expected_volume) / expected_volume
    if error > VOLUME_TOLERANCE:
        raise GroundTruthMismatch(
            f"analyzer volume {record.geometry.volume_m3:.9g} m3 differs from "
            f"the closed form {expected_volume:.9g} m3 by {error:.2e}")

    contents = sa.read_step(step_path)
    features = fr.recognise(contents.shapes[0], contents.unit_to_metres)
    _check_features(fam, params, features)
    feature_list = (
        [{"kind": "hole", "diameter_m": h.diameter_m, "axis": list(h.axis),
          "point_on_axis_m": list(h.point_on_axis_m)} for h in features.holes]
        + [{"kind": "fillet", "radius_m": f.radius_m,
            "surface_kind": f.surface_kind} for f in features.fillets])

    labels: dict = {}
    label_report = None
    if labelled:
        material = material or get_material("al_7075_t6")
        case = load or LoadCase(direction=fam.load_direction)
        label_report = cantilever_labels(
            step_path, record.geometry.volume_m3, record.geometry.bounding_box_m,
            material, case)
        labels = label_report.labels

    record = record.model_copy(update={
        "material_id": material.id if labelled else "",
        "features": feature_list,
        "labels": {**labels, "parameters": {
            **params, "family": fam.name,
            "note": "the parameters that built the part; exact by construction"}},
        "notes": (f"{fam.name}: {fam.description}. {record.notes}; "
                  f"analyzer volume agrees with the closed form to "
                  f"{error:.1e}"),
    })
    return PartRecord.model_validate(record.model_dump()), label_report


def generate_dataset(n: int, seed: int, out_path: str | Path | None = None,
                     families: Sequence[str] | None = None,
                     material_id: str = "al_7075_t6",
                     total_load_n: float = -100.0,
                     step_dir: str | Path | None = None,
                     labelled: bool = True,
                     stop_on_mismatch: bool = False,
                     ) -> tuple[list[PartRecord], GenerationReport]:
    """Make n parts, cycling through the families, and write them.

    Deterministic for a seed: the same call makes the same parts with the
    same ids. A mismatch against ground truth is recorded in the report and
    the part is dropped, unless `stop_on_mismatch`, in which case it raises,
    which is the right setting for a test and the wrong one for a long run.
    """
    names = list(families or FAMILIES)
    fams = [FAMILIES[name] for name in names]
    rng = np.random.default_rng(seed)
    material = get_material(material_id)
    started = time.perf_counter()
    report = GenerationReport(requested=n, generated=0)
    records: list[PartRecord] = []

    context = tempfile.TemporaryDirectory() if step_dir is None else None
    directory = Path(context.name) if context else Path(step_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for index in range(n):
            fam = fams[index % len(fams)]
            params = sample_parameters(fam, rng)
            part_id = part_id_for(fam, params)
            try:
                record, labels = make_part(
                    fam, params, directory, material,
                    LoadCase(total_load_n=total_load_n,
                             direction=fam.load_direction),
                    labelled=labelled)
            except GroundTruthMismatch as exc:
                if stop_on_mismatch:
                    raise
                report.refused.append((part_id, str(exc)))
                continue
            records.append(record)
            report.generated += 1
            report.per_family[fam.name] = report.per_family.get(fam.name, 0) + 1
            if labels is not None:
                report.labelling_seconds += labels.seconds
        if out_path is not None:
            write_jsonl(out_path, records)
            report.path = str(out_path)
    finally:
        if context is not None:
            context.cleanup()
    report.seconds = time.perf_counter() - started
    return records, report
