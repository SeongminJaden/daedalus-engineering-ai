"""Executing a generative CAD strategy: search part families, emit STEP.

The parametric strategy moves within one section. The topology strategy
grows a density field that no CAD kernel can read back cleanly. This one
searches the synthetic part families of `core.part_dataset`, which are real
B-reps, so the design it returns is a STEP file whose volume the analyzer has
checked against a closed form, labelled by CalculiX, and stored with
provenance. That is what "an autonomous loop that emits CAD" means here, and
it is all it means.

HOW THE SURROGATE IS ALLOWED IN
===============================
Every candidate is built as a B-rep and read back, which is cheap. A RANKER
then predicts its deflection: the beam-theory proxy by default, or a trained
shape surrogate when one is supplied. The ranker sorts; it decides nothing.
The top few are labelled by the solver, and the winner is the lightest part
the SOLVER found inside the deflection limit. A ranker that is wrong costs
solver time on the wrong parts and can miss a better one; it cannot put an
unverified part into the episode log, because the outcome is built from the
solver's labels and nothing else. That is the Phase 6 discipline, and the
SURROGATE evidence level enforces it below this layer.

WHAT "GENERATIVE" DOES NOT MEAN HERE
====================================
    Three parametric families with the problem's length fixed: a solid box, a
    hollow rectangle, a plate with holes. Nothing free-form, nothing learned
    proposes geometry, no topology change beyond what those families span.
    A method that could invent shapes would be a different method with a
    different name, and calling this one generative design without this
    paragraph would be the overclaim the project exists to avoid.

    Stress is NOT a constraint here. The labeller's peak von Mises sits at a
    clamped-edge singularity and does not converge, so this executor reports
    it in the detail and does not judge on it. Deflection is the constraint,
    exactly as in the topology executor.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from core.materials import get_material
from core.part_dataset import FAMILIES, make_part, sample_parameters
from core.part_dataset.descriptors import describe_step
from core.part_dataset.engine import GroundTruthMismatch, LabellingFailed
from core.part_dataset.labeller import LoadCase
from core.part_dataset.shape_surrogate import beam_proxy_m, features_for

from .outcome import DesignOutcome

METHOD = "generative_cad"

#: Families with a `length_m` parameter, so the problem's length can be
#: imposed. The bracket and the shaft are not cantilevers of a stated length.
DEFAULT_FAMILIES = ("box", "hollow_rect", "plate_with_holes")

#: How many parts are built and ranked, and how many the solver labels.
DEFAULT_CANDIDATES = 12
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class Candidate:
    family: str
    parameters: dict[str, float]
    record: object                 # PartRecord, unlabelled
    predicted_deflection_m: float
    mass_kg: float


Ranker = Callable[[Sequence[Candidate], object, LoadCase], np.ndarray]


def _case_for(candidate: Candidate, case: LoadCase) -> LoadCase:
    """The load case in the candidate's own family convention: a plate is
    loaded through its thickness, a beam through its height."""
    return LoadCase(total_load_n=case.total_load_n,
                    direction=FAMILIES[candidate.family].load_direction)


def proxy_ranker(candidates: Sequence[Candidate], material, case: LoadCase
                 ) -> np.ndarray:
    """Beam theory on the bounding box, scaled by fill. Analytical, crude, and
    the default because it needs no training."""
    return np.array([beam_proxy_m(c.record, material, _case_for(c, case))
                     for c in candidates])


def surrogate_ranker(surrogate, step_dir: Path) -> Ranker:
    """A trained ShapeSurrogate as the ranker. Its predictions are SURROGATE
    and never leave this function as anything but an ordering."""
    def rank(candidates: Sequence[Candidate], material, case: LoadCase
             ) -> np.ndarray:
        rows = []
        for c in candidates:
            descriptor = describe_step(step_dir / f"{c.record.part_id}.step")[0]
            rows.append(features_for(c.record, descriptor, material,
                                     _case_for(c, case)))
        return surrogate.predict(np.array(rows)).values["tip_deflection_m"]
    return rank


def _envelope(problem) -> dict[str, tuple[float, float] | None]:
    """The problem's design envelope, where it states one."""
    geometry = problem.geometry
    return {"height": (None if geometry.max_height_m is None
                       else (0.0, geometry.max_height_m)),
            "width": (None if geometry.max_width_m is None
                      else (0.0, geometry.max_width_m))}


def _sample(family, rng, length_m: float, envelope) -> dict[str, float]:
    """A parameter set with the problem's length imposed and the section
    inside the envelope, by rejection."""
    for _ in range(200):
        params = sample_parameters(family, rng)
        params["length_m"] = length_m
        height_key = "height_m" if "height_m" in params else "thickness_m"
        if envelope["height"] and params[height_key] > envelope["height"][1]:
            continue
        if envelope["width"] and params["width_m"] > envelope["width"][1]:
            continue
        if family.admissible(params):
            return params
    raise RuntimeError(f"{family.name}: no admissible candidate inside the "
                       f"problem envelope in 200 draws")


def run(op, candidates: int = DEFAULT_CANDIDATES, top_k: int = DEFAULT_TOP_K,
        seed: int = 0, families: Sequence[str] = DEFAULT_FAMILIES,
        ranker: Ranker | None = None, step_dir: str | Path | None = None,
        **_: object) -> DesignOutcome:
    """Build, rank, verify, and return the lightest solver-verified part."""
    began = time.monotonic()
    material = get_material(op.problem.material_id)
    load = op.problem.loads[0]
    length_m = float(op.problem.geometry.length_m)
    limit = op.max_deflection_m
    rng = np.random.default_rng(seed)
    envelope = _envelope(op.problem)

    context = tempfile.TemporaryDirectory() if step_dir is None else None
    directory = Path(context.name) if context else Path(step_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        # --- stage 1: build every candidate, cheaply, and rank -----------
        built: list[Candidate] = []
        refused: list[tuple[str, str]] = []
        fams = [FAMILIES[name] for name in families]
        for index in range(candidates):
            fam = fams[index % len(fams)]
            params = _sample(fam, rng, length_m, envelope)
            try:
                record, _ = make_part(fam, params, directory, labelled=False)
            except (GroundTruthMismatch, ValueError) as exc:
                refused.append((fam.name, str(exc)))
                continue
            built.append(Candidate(
                family=fam.name, parameters=params, record=record,
                predicted_deflection_m=float("nan"),
                mass_kg=record.geometry.volume_m3 * material.density_kg_m3))
        if not built:
            raise RuntimeError("no candidate could be built: "
                               + "; ".join(r for _, r in refused))

        case = LoadCase(total_load_n=-float(load.magnitude_n), direction=1)
        rank = ranker or proxy_ranker
        predicted = np.asarray(rank(built, material, case), dtype=float)
        built = [Candidate(c.family, c.parameters, c.record, float(p), c.mass_kg)
                 for c, p in zip(built, predicted)]
        looks_feasible = [limit is None or c.predicted_deflection_m <= limit
                          for c in built]
        order = sorted(range(len(built)),
                       key=lambda i: (not looks_feasible[i], built[i].mass_kg))
        shortlist = order[:min(top_k, len(order))]

        # --- stage 2: the solver decides ---------------------------------
        verified = []
        for i in shortlist:
            c = built[i]
            fam = FAMILIES[c.family]
            try:
                record, _ = make_part(fam, c.parameters, directory, material,
                                      LoadCase(total_load_n=case.total_load_n,
                                               direction=fam.load_direction),
                                      labelled=True)
            except LabellingFailed as exc:
                refused.append((c.record.part_id, str(exc)))
                continue
            deflection = abs(float(record.labels["tip_deflection_m"]["value"]))
            verified.append((c, record, deflection))
        if not verified:
            raise RuntimeError("the solver returned nothing for any shortlisted "
                               "part: " + "; ".join(r for _, r in refused))

        feasible = [v for v in verified
                    if limit is None or v[2] <= limit]
        if feasible:
            chosen, record, deflection = min(feasible, key=lambda v: v[1].labels
                                             ["mass_kg"]["value"])
            ok = True
        else:
            # Nothing passed. Report the closest, and say it failed.
            chosen, record, deflection = min(verified, key=lambda v: v[2])
            ok = False
        margin = 0.0 if limit is None else float(limit - deflection)
        mass = float(record.labels["mass_kg"]["value"])
        step_path = directory / f"{record.part_id}.step"
        return DesignOutcome(
            method=METHOD, mass_kg=mass, feasible=ok,
            constraints={"deflection": margin},
            evaluations=len(verified), seconds=time.monotonic() - began,
            converged=True, cad_record=record,
            detail={
                "family": chosen.family, "parameters": dict(chosen.parameters),
                "part_id": record.part_id,
                "step_path": str(step_path) if context is None else None,
                "tip_deflection_m": deflection,
                "predicted_deflection_m": chosen.predicted_deflection_m,
                "ranker": "shape_surrogate" if ranker is not None else "beam_proxy",
                "n_screened": len(built), "n_verified": len(verified),
                "shortlist": [built[i].record.part_id for i in shortlist],
                "refused": refused,
                "max_von_mises_pa": float(record.labels["max_von_mises_pa"]["value"]),
                "stress_note": ("not assessed as a constraint: the peak sits at "
                                "a clamped-edge singularity and does not "
                                "converge under refinement"),
                "evidence": record.labels["tip_deflection_m"]["evidence"],
            })
    finally:
        if context is not None:
            context.cleanup()
