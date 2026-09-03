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
    Eleven parametric families with the problem's length imposed. Nothing
    free-form, nothing learned proposes geometry, no topology change beyond
    what those families span. A method that could invent shapes would be a
    different method with a different name, and calling this one generative
    design without this paragraph would be the overclaim the project exists
    to avoid.

    Eleven, not thirteen: the flange and the gear blank are discs whose axial
    extent is a thickness, not a span, so a cantilever of stated length is not
    a thing they can be. They are refused by name with that reason rather than
    stretched into a shape they are not. The l bracket takes the length on its
    leg and the stepped shaft splits it between its two sections, keeping
    their sampled ratio.

    Stress is NOT a constraint here. The labeller's peak von Mises sits at a
    clamped-edge singularity and does not converge, so this executor reports
    it in the detail and does not judge on it. The constraint is the primary
    response of the load case, which for bending is the deflection, exactly
    as in the topology executor.

FIVE LOAD CASES, AND WHERE THEIR LIMITS COME FROM
=================================================
    Bending, axial, torsion, combined and thermal gradient, each with the
    closed-form check the labeller already carries. The engineering problem
    states one limit, `max_deflection_m`, which is the bending limit; a
    caller who searches under another load case has to supply the limit for
    its primary response, because inventing one would be inventing the
    requirement. Without a limit the search still runs and returns the
    lightest solver-verified part, with the margin reported as absent.

NINETEEN MATERIALS FROM ONE SOLVE
=================================
    A material change does not change the geometry, and for an isotropic
    material every label the labeller writes carries a scaling tag whose
    residual against a direct solve was measured (see core/part_dataset/
    scaling.py). So the shortlist is solved once in the reference material
    and read across the candidate materials by exact scaling, which costs
    nothing. The winner is then SOLVED AGAIN in its own material, and that
    solve is what the outcome carries: a scaled label is derived evidence and
    good enough to choose with, not to report as the answer.

MANUFACTURABILITY AS A PREFERENCE, NEVER AS A VERDICT
=====================================================
    When a process is named, every solver-verified part is measured against
    that process's rules (geometry/manufacturability) and the fraction of
    measurable rules it passes becomes a preference among parts that already
    satisfy the physical constraint. It cannot reject a part and it cannot
    rescue one: the ordering is (feasible, then rule failures, then mass), and
    the grade stays rule_based_dfm_guideline, which is not an evidence level.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from core.materials import MaterialSpec, get_material
from core.part_dataset import FAMILIES, make_part, sample_parameters
from core.part_dataset.descriptors import describe_step
from core.part_dataset.engine import GroundTruthMismatch, LabellingFailed
from core.part_dataset.labeller import PRIMARY_LABEL, LoadCase, LoadKind
from core.part_dataset.scaling import UnscalableLabel, scale_record
from core.part_dataset.shape_surrogate import beam_proxy_m, features_for

from .outcome import DesignOutcome

METHOD = "generative_cad"

#: The three families the executor was first measured on. Kept as a name
#: because the older tests pin the behaviour of exactly these.
DEFAULT_FAMILIES = ("box", "hollow_rect", "plate_with_holes")

#: How a family takes the problem's length. A family absent from this map
#: cannot be a cantilever of a stated length and is refused by name.
LENGTH_PARAMETER: dict[str, str] = {
    name: "length_m" for name, fam in FAMILIES.items() if "length_m" in fam.bounds}
LENGTH_PARAMETER["l_bracket"] = "size_m"        # the leg length is the span
LENGTH_PARAMETER["stepped_shaft"] = "split"     # shared by the two sections

#: Discs. Their axial extent is a thickness, not a span.
NO_LENGTH_REASON = {
    "flange": "a flange's axial extent is its thickness, not a cantilever span",
    "gear_blank": "a gear blank's axial extent is its hub length, not a span"}

#: Every family that can take an imposed length, in a fixed order.
SEARCHABLE_FAMILIES: tuple[str, ...] = tuple(
    n for n in FAMILIES if n in LENGTH_PARAMETER)

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
    loaded through its thickness, a beam through its height. Everything else
    about the case, the kind included, is carried through unchanged."""
    return LoadCase(total_load_n=case.total_load_n,
                    direction=FAMILIES[candidate.family].load_direction,
                    kind=case.kind, torque_nm=case.torque_nm,
                    gradient_k_per_m=case.gradient_k_per_m,
                    delta_k=case.delta_k)


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


def impose_length(family, params: dict[str, float], length_m: float) -> dict[str, float]:
    """The sampled parameters with the problem's span written into whichever
    parameter carries it for this family.

    The stepped shaft has two sections and no single length, so the span is
    divided between them in the ratio the sampler drew, which preserves the
    shape the sampler intended and changes only the scale.
    """
    how = LENGTH_PARAMETER.get(family.name)
    if how is None:
        raise UnsearchableFamily(
            f"{family.name}: {NO_LENGTH_REASON.get(family.name, 'no length parameter')}")
    if how == "split":
        first, second = params["length_1_m"], params["length_2_m"]
        share = first / (first + second)
        params["length_1_m"] = length_m * share
        params["length_2_m"] = length_m * (1.0 - share)
        return params
    params[how] = length_m
    return params


def _sample(family, rng, length_m: float, envelope) -> dict[str, float]:
    """A parameter set with the problem's length imposed and the section
    inside the envelope, by rejection."""
    for _ in range(200):
        params = impose_length(family, sample_parameters(family, rng), length_m)
        height_key = next((k for k in ("height_m", "thickness_m", "radius_m",
                                       "outer_radius_m", "size_m") if k in params), None)
        width_key = next((k for k in ("width_m", "radius_m", "outer_radius_m",
                                      "size_m") if k in params), None)
        if envelope["height"] and height_key and params[height_key] > envelope["height"][1]:
            continue
        if envelope["width"] and width_key and params[width_key] > envelope["width"][1]:
            continue
        if family.admissible(params):
            return params
    raise RuntimeError(f"{family.name}: no admissible candidate inside the "
                       f"problem envelope in 200 draws")


class UnsearchableFamily(ValueError):
    """A family that cannot be a cantilever of a stated length."""


def primary_of(record, case: LoadCase) -> float:
    """The load case's primary response, as a magnitude."""
    return abs(float(record.labels[PRIMARY_LABEL[case.kind][0]]["value"]))


def dfm_failures(record, step_path: Path, process) -> tuple[int, int, list[str]]:
    """Rules the part fails, rules that could be measured, and their names.

    A preference, not a verdict: the caller ranks with it and the grade stays
    rule_based_dfm_guideline, which is not an evidence level.
    """
    from core.part_dataset.pointcloud import tessellate
    from geometry.manufacturability import assess
    from nodes import step_analyzer as sa

    contents = sa.read_step(str(step_path))
    mesh = tessellate(contents.shapes[0], contents.unit_to_metres)
    report = assess(process, mesh.vertices, mesh.triangles, record)
    measured = [f for f in report.findings if f.assessed]
    failed = [f.rule.quantity for f in measured if not f.passes]
    return len(failed), len(measured), failed


def run(op, candidates: int = DEFAULT_CANDIDATES, top_k: int = DEFAULT_TOP_K,
        seed: int = 0, families: Sequence[str] = DEFAULT_FAMILIES,
        ranker: Ranker | None = None, step_dir: str | Path | None = None,
        load_kind: LoadKind = LoadKind.BENDING, response_limit: float | None = None,
        materials: Sequence[str] | None = None, process=None,
        torque_nm: float = 5.0, gradient_k_per_m: float = 1000.0,
        **_: object) -> DesignOutcome:
    """Build, rank, verify, and return the lightest solver-verified part.

    `families` may name any of SEARCHABLE_FAMILIES; a disc family is refused
    by name. `load_kind` chooses the load case and `response_limit` its limit
    (for bending it defaults to the problem's max_deflection_m; for the others
    the problem states none, so the caller supplies it or the search returns
    the lightest verified part with no margin). `materials` widens the search
    across isotropic materials by exact scaling, with the winner solved again
    in its own material. `process` turns manufacturability into a preference.
    """
    began = time.monotonic()
    reference_material = get_material(op.problem.material_id)
    material = reference_material
    load = op.problem.loads[0]
    length_m = float(op.problem.geometry.length_m)
    limit = response_limit
    if limit is None and load_kind is LoadKind.BENDING:
        limit = op.max_deflection_m
    rng = np.random.default_rng(seed)
    envelope = _envelope(op.problem)
    candidate_materials: list[MaterialSpec] = [reference_material]
    if materials:
        candidate_materials = [get_material(m) for m in materials]
        if reference_material.id not in {m.id for m in candidate_materials}:
            candidate_materials.insert(0, reference_material)

    context = tempfile.TemporaryDirectory() if step_dir is None else None
    directory = Path(context.name) if context else Path(step_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        # --- stage 1: build every candidate, cheaply, and rank -----------
        built: list[Candidate] = []
        refused: list[tuple[str, str]] = []
        fams = []
        for name in families:
            if name not in LENGTH_PARAMETER:
                refused.append((name, NO_LENGTH_REASON.get(
                    name, f"{name}: no parameter carries the span")))
                continue
            fams.append(FAMILIES[name])
        if not fams:
            raise UnsearchableFamily(
                "no family in the request can take an imposed length: "
                + "; ".join(r for _, r in refused))
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

        case = LoadCase(total_load_n=-float(load.magnitude_n), direction=1,
                        kind=load_kind, torque_nm=torque_nm,
                        gradient_k_per_m=gradient_k_per_m)
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
                                      _case_for(c, case), labelled=True)
            except LabellingFailed as exc:
                refused.append((c.record.part_id, str(exc)))
                continue
            verified.append((c, record, primary_of(record, case)))
        if not verified:
            raise RuntimeError("the solver returned nothing for any shortlisted "
                               "part: " + "; ".join(r for _, r in refused))

        # --- stage 3: the other materials, by exact scaling ---------------
        # A scaled label is derived, good enough to choose with. The winner is
        # solved again below in its own material, and that solve is reported.
        options = []          # (candidate, record, response, material, scaled)
        for c, record, response in verified:
            for target in candidate_materials:
                if target.id == reference_material.id:
                    options.append((c, record, response, target, False))
                    continue
                try:
                    copy = scale_record(record, reference_material, target)
                except UnscalableLabel as exc:
                    refused.append((f"{record.part_id}-{target.id}", str(exc)))
                    continue
                options.append((c, copy, primary_of(copy, case), target, True))

        # --- stage 4: choose, with manufacturability as a preference -------
        feasible = [o for o in options if limit is None or o[2] <= limit]
        dfm: dict[str, tuple[int, int, list[str]]] = {}
        if process is not None:
            for _c, record, _r, _m, _scaled in (feasible or options):
                base = record.part_id.split("-" + record.material_id)[0]
                if base not in dfm:
                    dfm[base] = dfm_failures(record, directory / f"{base}.step", process)

        def order_key(option):
            c, record, _response, _material, _scaled = option
            base = record.part_id.split("-" + record.material_id)[0]
            failures = dfm.get(base, (0, 0, []))[0]
            return (failures, float(record.labels["mass_kg"]["value"]))

        if feasible:
            chosen, record, response, chosen_material, was_scaled = min(
                feasible, key=order_key)
            ok = True
        else:
            # Nothing passed. Report the closest, and say it failed.
            chosen, record, response, chosen_material, was_scaled = min(
                options, key=lambda o: o[2])
            ok = False

        # --- stage 5: the winner is solved in its own material -------------
        if was_scaled:
            fam = FAMILIES[chosen.family]
            record, _ = make_part(fam, chosen.parameters, directory,
                                  chosen_material, _case_for(chosen, case),
                                  labelled=True)
            scaled_response, response = response, primary_of(record, case)
        else:
            scaled_response = None
        margin = None if limit is None else float(limit - response)
        mass = float(record.labels["mass_kg"]["value"])
        step_path = directory / f"{record.part_id}.step"
        base_id = record.part_id.split("-" + record.material_id)[0]
        dfm_failed, dfm_measured, dfm_names = dfm.get(base_id, (0, 0, []))
        primary_name = PRIMARY_LABEL[case.kind][0]
        detail = {
            "family": chosen.family, "parameters": dict(chosen.parameters),
            "part_id": record.part_id,
            "step_path": str(step_path) if context is None else None,
            "load_kind": case.kind.value,
            "primary_response_name": primary_name,
            "primary_response": response,
            "response_limit": limit,
            "material_id": chosen_material.id,
            "materials_searched": [m.id for m in candidate_materials],
            "chosen_by_scaling_then_solved": was_scaled,
            "scaled_response_before_solve": scaled_response,
            "predicted_deflection_m": chosen.predicted_deflection_m,
            "ranker": "shape_surrogate" if ranker is not None else "beam_proxy",
            "n_screened": len(built), "n_verified": len(verified),
            "n_options": len(options),
            "families_searched": [f.name for f in fams],
            "shortlist": [built[i].record.part_id for i in shortlist],
            "refused": refused,
            "max_von_mises_pa": float(record.labels["max_von_mises_pa"]["value"]),
            "stress_note": ("not assessed as a constraint: the peak sits at "
                            "a clamped-edge singularity and does not "
                            "converge under refinement"),
            "evidence": record.labels[primary_name]["evidence"],
        }
        if case.kind in (LoadKind.BENDING, LoadKind.COMBINED):
            detail["tip_deflection_m"] = response
        if process is not None:
            detail["process"] = process.value
            detail["dfm_rules_failed"] = dfm_failed
            detail["dfm_rules_measured"] = dfm_measured
            detail["dfm_failed_rules"] = dfm_names
            detail["dfm_note"] = ("rule_based_dfm_guideline, a preference among "
                                  "parts that already passed the physical "
                                  "constraint; not an evidence level")
        return DesignOutcome(
            method=METHOD, mass_kg=mass, feasible=ok,
            # No limit means no margin. Reporting 0.0 would read as a part
            # sitting exactly on a limit that was never stated.
            constraints=({} if margin is None else
                         {"deflection" if case.kind is LoadKind.BENDING
                          else primary_name: margin}),
            evaluations=len(verified), seconds=time.monotonic() - began,
            converged=True, cad_record=record, detail=detail)
    finally:
        if context is not None:
            context.cleanup()
