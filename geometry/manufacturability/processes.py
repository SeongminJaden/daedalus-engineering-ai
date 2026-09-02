"""Design-for-manufacturing rules per process, each with the guide it came from.

A rule here is a number somebody published as a design guideline, applied to
a measurement of the part. It is NOT physics and it is NOT on the evidence
ladder: a part that passes every rule for CNC milling has been checked
against a machining shop's rules of thumb, not shown to be machinable, and a
part that fails one may still be made by a shop that disagrees with the
guide. So every report carries RULE_BASED as its grade, every rule names its
source, and the sources are the pages that were actually read on the
retrieval date, quoted with the number as printed.

What is checked and what is not, per process:

    cnc_milling        minimum wall (metal 0.8 mm, plastic 1.5 mm), hole
                       diameter 2.5 mm, six-axis-direction tool access
    cnc_turning        NOT ASSESSED for axisymmetry; only the shared wall
                       and hole rules, and the report says so
    sheet_metal        uniform thickness (spread of the wall samples), hole
                       diameter at least the thickness; bend radius and
                       hole-to-bend distances are not measured
    fdm                wall 0.8 mm, overhang 45 degrees; bridging, pins and
                       minimum hole not measured
    sls                wall 0.8 mm (PA12), hole 1.5 mm; escape holes only
                       matter for enclosed cavities, which the mesh measure
                       reports as inaccessible area
    slm                wall 0.4 mm, hole 1.5 mm, overhang 50 degrees
    die_casting        wall 1.0 mm minimum and 2.0 to 3.5 recommended
                       (aluminium), draft 1 degree external, internal fillet
                       0.5 mm minimum; core draft and rib rules not measured
    injection_moulding draft 2 degrees, internal radius 0.5 t; wall range,
                       ribs and bosses not measured

Draft is measured with the existing `geometry.surfacing.manufacturability.
draft`, which reports the area fraction of faces lying within a required
angle of the pull direction. Undercuts for a two part mould are the faces
that face neither pull direction; on the axis-aligned families every face
faces some axis, so an undercut measure would read zero on all of them and
prove nothing, which is why it is listed as not assessed rather than
reported as passing.

Rules the guide states only qualitatively, or that need a quantity this
module cannot measure from a mesh (bend radius, hole-to-edge distance, core
depth, rib ratios), produce a finding with passes=None: unassessed, with the
reason. An unassessed rule is not a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .measures import MeshMeasures, measure_mesh

#: The grade of everything this module produces. Deliberately not an
#: EvidenceLevel: a rule of thumb is neither a simulation nor a measurement.
RULE_BASED = "rule_based_dfm_guideline"


@dataclass(frozen=True)
class RuleSource:
    id: str
    title: str
    publisher: str
    url: str
    retrieved: str


RULE_SOURCES: dict[str, RuleSource] = {s.id: s for s in (
    RuleSource("hubs_cnc", "How to design parts for CNC machining", "Protolabs Network (Hubs)",
               "https://www.hubs.com/knowledge-base/how-design-parts-cnc-machining/", "2026-09-03"),
    RuleSource("hubs_fdm", "How to design parts for FDM 3D printing", "Protolabs Network (Hubs)",
               "https://www.hubs.com/knowledge-base/how-design-parts-fdm-3d-printing/", "2026-09-03"),
    RuleSource("hubs_3dp", "What are the key design elements for 3D printing?", "Protolabs Network (Hubs)",
               "https://www.hubs.com/knowledge-base/key-design-considerations-3d-printing/", "2026-09-03"),
    RuleSource("hubs_sls", "How to design parts for SLS 3D printing", "Protolabs Network (Hubs)",
               "https://www.hubs.com/knowledge-base/how-design-parts-sls-3d-printing/", "2026-09-03"),
    RuleSource("hubs_metal", "Metal 3D printing design guide", "Protolabs Network (Hubs)",
               "https://www.hubs.com/knowledge-base/how-design-parts-metal-3d-printing/", "2026-09-03"),
    RuleSource("hubs_im", "Injection molding design guide", "Protolabs Network (Hubs)",
               "https://www.hubs.com/guides/injection-molding/", "2026-09-03"),
    RuleSource("sheetmetal_me", "Design Guidelines", "SheetMetal.Me",
               "https://sheetmetal.me/design-guidelines/", "2026-09-03"),
    RuleSource("amn_diecast", "Designing Parts for Die Casting: Draft Angles and Tolerances",
               "AMN Engineering", "https://amnengineering.com/blog/designing-parts-for-die-casting",
               "2026-09-03"),
)}


class Process(str, Enum):
    CNC_MILLING = "cnc_milling"
    CNC_TURNING = "cnc_turning"
    SHEET_METAL = "sheet_metal"
    FDM = "fdm"
    SLS = "sls"
    SLM = "slm"
    DIE_CASTING = "die_casting"
    INJECTION_MOULDING = "injection_moulding"


@dataclass(frozen=True)
class Rule:
    id: str
    process: Process
    quantity: str            # which measure it reads
    comparison: str          # "min" (measured >= threshold) or "max"
    threshold: float
    unit: str
    source: str              # RuleSource id
    as_printed: str          # the guide's own words and number
    note: str = ""


@dataclass(frozen=True)
class Finding:
    rule: Rule
    measured: float | None
    passes: bool | None      # None: unassessed
    detail: str

    @property
    def assessed(self) -> bool:
        return self.passes is not None


@dataclass
class ProcessReport:
    process: Process
    findings: list[Finding] = field(default_factory=list)
    grade: str = RULE_BASED
    not_measured: tuple[str, ...] = ()

    @property
    def passes(self) -> bool:
        """Every ASSESSED rule passes. Unassessed rules do not count either
        way, and `unassessed` lists them so nobody reads silence as a pass."""
        assessed = [f for f in self.findings if f.assessed]
        return bool(assessed) and all(f.passes for f in assessed)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.passes is False]

    @property
    def unassessed(self) -> list[Finding]:
        return [f for f in self.findings if not f.assessed]

    def summary(self) -> str:
        verdict = "passes the assessed rules" if self.passes else "fails a rule"
        if not any(f.assessed for f in self.findings):
            verdict = "nothing assessed"
        return (f"{self.process.value}: {verdict}; {len(self.failures)} failed, "
                f"{len(self.unassessed)} unassessed; {self.grade}")


def _mm(x: float) -> float:
    return x * 1e-3


RULES: tuple[Rule, ...] = (
    # CNC milling and turning share wall and hole rules
    Rule("cnc_wall_metal", Process.CNC_MILLING, "min_wall_m", "min", _mm(0.8), "m", "hubs_cnc",
         "Wall thickness, metals: recommended 0.8 mm minimum; feasible 0.5 mm"),
    Rule("cnc_wall_plastic", Process.CNC_MILLING, "min_wall_m", "min", _mm(1.5), "m", "hubs_cnc",
         "Wall thickness, plastics: recommended 1.5 mm minimum; feasible 1.0 mm",
         note="applied only when the material is a polymer"),
    Rule("cnc_hole", Process.CNC_MILLING, "min_hole_diameter_m", "min", _mm(2.5), "m", "hubs_cnc",
         "Holes: minimum diameter recommended 2.5 mm; feasible 0.05 mm"),
    Rule("cnc_access", Process.CNC_MILLING, "inaccessible_fraction_6_axis", "max", 0.0, "fraction",
         "hubs_cnc", "Undercuts and tool access: a face the tool cannot reach from any "
         "axis direction cannot be machined in six three-axis setups",
         note="the guide's rule is qualitative; the measure is this module's"),
    Rule("turn_wall_metal", Process.CNC_TURNING, "min_wall_m", "min", _mm(0.8), "m", "hubs_cnc",
         "Wall thickness, metals: recommended 0.8 mm minimum"),
    Rule("turn_hole", Process.CNC_TURNING, "min_hole_diameter_m", "min", _mm(2.5), "m", "hubs_cnc",
         "Holes: minimum diameter recommended 2.5 mm"),
    Rule("turn_axisymmetric", Process.CNC_TURNING, "axisymmetric", "min", 1.0, "flag", "hubs_cnc",
         "Turning needs a part that is a solid of revolution about the spindle axis",
         note="NOT MEASURED: this module has no axisymmetry test, so the finding is unassessed"),
    # sheet metal
    Rule("sheet_uniform", Process.SHEET_METAL, "wall_spread", "max", 0.15, "fraction", "sheetmetal_me",
         "Sheet metal is one thickness everywhere; the 5th to 95th percentile spread of the "
         "measured wall over its median is the check",
         note="the threshold 0.15 is this module's, the uniformity rule is the guide's"),
    Rule("sheet_hole", Process.SHEET_METAL, "min_hole_diameter_m", "min_of_wall", 1.0, "multiple of thickness",
         "sheetmetal_me", "Holes: rule of thumb, never smaller than the material thickness"),
    Rule("sheet_bend_radius", Process.SHEET_METAL, "bend_radius_m", "min_of_wall", 1.0, "multiple of thickness",
         "sheetmetal_me", "Minimum bend radius equal to the material thickness",
         note="NOT MEASURED: bends are not identified on a mesh; unassessed"),
    Rule("sheet_hole_to_bend", Process.SHEET_METAL, "hole_to_bend_m", "min", 0.0, "m", "sheetmetal_me",
         "Holes at least 2.5 times material thickness plus bend radius from a bend",
         note="NOT MEASURED; unassessed"),
    # FDM
    Rule("fdm_wall", Process.FDM, "min_wall_m", "min", _mm(0.8), "m", "hubs_3dp",
         "All 3D printers can successfully print components with wall thicknesses greater than 0.8 mm"),
    Rule("fdm_overhang", Process.FDM, "overhang_fraction_45", "max", 0.0, "fraction", "hubs_fdm",
         "An overhang can usually be printed up to 45 degrees without compromising quality",
         note="area fraction of downward faces steeper than 45 degrees from the vertical, "
              "excluding the build plate; anything above zero needs support"),
    Rule("fdm_bridge", Process.FDM, "bridge_length_m", "max", _mm(5.0), "m", "hubs_fdm",
         "Bridges under 5 mm print without sagging or support marks",
         note="NOT MEASURED; unassessed"),
    # SLS
    Rule("sls_wall", Process.SLS, "min_wall_m", "min", _mm(0.8), "m", "hubs_sls",
         "Minimum thickness between 0.8 mm (PA12) and 2.0 mm (carbon filled polyamide)"),
    Rule("sls_hole", Process.SLS, "min_hole_diameter_m", "min", _mm(1.5), "m", "hubs_sls",
         "All holes should be larger than 1.5 mm in diameter"),
    Rule("sls_escape", Process.SLS, "inaccessible_fraction_6_axis", "max", 0.0, "fraction", "hubs_sls",
         "Escape holes of at least 3.5 mm are needed to remove unsintered powder from hollow parts",
         note="an enclosed cavity shows as inaccessible area; the measure cannot see an escape "
              "hole smaller than the ray model, so a nonzero fraction is a warning, not a fail",
         ),
    # SLM
    Rule("slm_wall", Process.SLM, "min_wall_m", "min", _mm(0.4), "m", "hubs_metal",
         "Minimum wall thickness 0.4 mm"),
    Rule("slm_hole", Process.SLM, "min_hole_diameter_m", "min", _mm(1.5), "m", "hubs_metal",
         "Minimum hole diameter 1.5 mm"),
    Rule("slm_overhang", Process.SLM, "overhang_fraction_50", "max", 0.0, "fraction", "hubs_metal",
         "Maximum overhang angle 50 degrees"),
    # die casting (aluminium)
    Rule("cast_wall_min", Process.DIE_CASTING, "min_wall_m", "min", _mm(1.0), "m", "amn_diecast",
         "Aluminium: minimum wall 1.0 mm, recommended 2.0 to 3.5 mm, maximum 5.0 mm"),
    Rule("cast_wall_rec", Process.DIE_CASTING, "min_wall_m", "min", _mm(2.0), "m", "amn_diecast",
         "Aluminium: recommended 2.0 to 3.5 mm", note="the recommended floor, reported separately"),
    Rule("cast_draft", Process.DIE_CASTING, "dragging_area_fraction_1deg", "max", 0.0, "fraction",
         "amn_diecast", "External walls: 1 to 2 degrees of draft",
         note="area fraction of faces within 1 degree of parallel to the pull, measured by "
              "geometry.surfacing.manufacturability.draft; internal cores need 2 to 3 degrees "
              "and are not told apart from external walls here"),
    Rule("cast_fillet", Process.DIE_CASTING, "min_fillet_radius_m", "min", _mm(0.5), "m", "amn_diecast",
         "Internal corners: minimum 0.5 mm radius, 1 mm or more preferred",
         note="assessed only when the recogniser found fillets; sharp internal corners on the "
              "families are not detected as such, so this is unassessed there"),
    # injection moulding
    Rule("im_draft", Process.INJECTION_MOULDING, "dragging_area_fraction_2deg", "max", 0.0, "fraction",
         "hubs_im", "Draft angle minimum 2 degrees for all vertical walls; add 1 to 2 degrees for "
         "textured surfaces"),
    Rule("im_internal_radius", Process.INJECTION_MOULDING, "min_fillet_over_wall", "min", 0.5,
         "multiple of wall", "hubs_im", "Internal corners: radius 0.5 times the wall thickness",
         note="assessed only when fillets were recognised"),
    Rule("im_rib", Process.INJECTION_MOULDING, "rib_thickness_over_wall", "max", 0.5, "multiple of wall",
         "hubs_im", "Rib thickness equal to 0.5 times the main wall thickness",
         note="NOT MEASURED: ribs are not identified on a mesh; unassessed"),
)

PROCESSES: tuple[Process, ...] = tuple(Process)


def _quantities(measures: MeshMeasures, record, mesh_vertices, mesh_triangles,
                pull_axis: int, is_polymer: bool) -> dict[str, float | None]:
    from geometry.surfacing.manufacturability import draft

    holes = [f["diameter_m"] for f in getattr(record, "features", [])
             if f.get("kind") == "hole"] if record is not None else []
    fillets = [f["radius_m"] for f in getattr(record, "features", [])
               if f.get("kind") == "fillet"] if record is not None else []
    q: dict[str, float | None] = {
        "min_wall_m": measures.min_wall_m,
        "wall_spread": measures.wall_spread,
        "min_hole_diameter_m": min(holes) if holes else None,
        "min_fillet_radius_m": min(fillets) if fillets else None,
        "min_fillet_over_wall": (min(fillets) / measures.median_wall_m
                                 if fillets and measures.median_wall_m else None),
        "inaccessible_fraction_6_axis": measures.inaccessible_fraction_6_axis,
        "overhang_fraction_45": measures.overhang_fraction_45,
        "overhang_fraction_50": measures.overhang_fraction_50,
        "axisymmetric": None, "bend_radius_m": None, "hole_to_bend_m": None,
        "bridge_length_m": None, "rib_thickness_over_wall": None,
        "is_polymer": 1.0 if is_polymer else 0.0,
    }
    report = draft(np.asarray(mesh_vertices), np.asarray(mesh_triangles), pull_axis)
    q["dragging_area_fraction_1deg"] = report.area_fraction_below(1.0)
    q["dragging_area_fraction_2deg"] = report.area_fraction_below(2.0)
    return q


def _judge(rule: Rule, q: dict[str, float | None]) -> Finding:
    if rule.id == "cnc_wall_plastic" and not q.get("is_polymer"):
        return Finding(rule, None, None, "not a polymer; the metal wall rule applies")
    if rule.id in ("cnc_wall_metal", "turn_wall_metal") and q.get("is_polymer"):
        return Finding(rule, None, None, "a polymer; the plastic wall rule applies")
    measured = q.get(rule.quantity)
    if measured is None:
        return Finding(rule, None, None, f"{rule.quantity} not measured: {rule.note or 'no measure'}")
    if rule.comparison == "min":
        ok = measured >= rule.threshold
        detail = f"{measured:.4g} {rule.unit} against a minimum of {rule.threshold:.4g}"
    elif rule.comparison == "max":
        ok = measured <= rule.threshold
        detail = f"{measured:.4g} {rule.unit} against a maximum of {rule.threshold:.4g}"
    elif rule.comparison == "min_of_wall":
        wall = q.get("min_wall_m")
        if wall is None:
            return Finding(rule, None, None, "wall thickness not measured")
        ok = measured >= rule.threshold * wall
        detail = (f"{measured:.4g} m against {rule.threshold:g} times the wall "
                  f"{wall:.4g} m")
    else:
        return Finding(rule, None, None, f"unknown comparison {rule.comparison}")
    return Finding(rule, float(measured), bool(ok), detail)


def assess(process: Process, mesh_vertices, mesh_triangles, record=None,
           build_axis: int = 1, pull_axis: int = 1, is_polymer: bool = False,
           measures: MeshMeasures | None = None) -> ProcessReport:
    """Every rule of one process against the part. Rule-based, not evidence."""
    measures = measures or measure_mesh(mesh_vertices, mesh_triangles, build_axis)
    q = _quantities(measures, record, mesh_vertices, mesh_triangles, pull_axis, is_polymer)
    findings = [_judge(r, q) for r in RULES if r.process is process]
    not_measured = tuple(sorted({f.rule.quantity for f in findings if not f.assessed}))
    return ProcessReport(process=process, findings=findings, not_measured=not_measured)


def assess_all(mesh_vertices, mesh_triangles, record=None, build_axis: int = 1,
               pull_axis: int = 1, is_polymer: bool = False) -> dict[Process, ProcessReport]:
    measures = measure_mesh(mesh_vertices, mesh_triangles, build_axis)
    return {p: assess(p, mesh_vertices, mesh_triangles, record, build_axis, pull_axis,
                      is_polymer, measures) for p in PROCESSES}
