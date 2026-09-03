"""One requirement, two design paths, the same checks on both.

    .venv/bin/python scripts/demo_end_to_end.py --out data/generated/demo_v1

The requirement is a robot forearm link: a cantilever of stated length under a
tip load, with a deflection limit, to be milled from aluminium and bolted into
a two link arm. Two paths produce a design for it.

    family search   the generative CAD executor picks a family, a size and a
                    material, every candidate built as a B-rep and the winner
                    labelled by CalculiX, and writes STEP.
    topology        SIMP with passive load and support patches and the three
                    field projection, the field thresholded, the extracted
                    part re-solved in CalculiX, smoothed to a watertight STL.

Both are then measured the same way: manufacturability rules for the named
process, catalogue fasteners sized for the joint, the part placed in a two
link assembly, this project's statics compared with Gazebo through preloaded
joint springs, and envelope interference from our own kinematics.

EVERY NUMBER HERE IS SIMULATED OR BELOW. Nothing in this demo has been
measured on hardware, and the report says so on every line that carries a
grade.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.execution.cad import run as run_generative  # noqa: E402
from core.assembly import Assembly, Joint, JointType, Link  # noqa: E402
from core.design_genome import DesignGenome, HollowRectangleSection  # noqa: E402
from core.materials import get_material  # noqa: E402
from core.part_dataset.labeller import LoadKind  # noqa: E402
from core.part_dataset.pointcloud import tessellate  # noqa: E402
from geometry.cad_export.standard_parts import (ISO_4762, hex_nut,  # noqa: E402
                                                material_for, socket_head_screw)
from geometry.manufacturability import Process, assess  # noqa: E402
from integration.simulation import gazebo  # noqa: E402
from nodes import step_analyzer as sa  # noqa: E402
from optimization.constraints import build_optimization_problem  # noqa: E402
from optimization.topology import SimpProblem  # noqa: E402
from optimization.topology.smooth import marching_surface, write_stl  # noqa: E402
from optimization.topology.threefield import optimize_projected  # noqa: E402
from optimization.topology.verify import (DisconnectedAtThreshold,  # noqa: E402
                                          elements_touching, verify_extracted)
from physics.fem.mesh import solid_box_mesh  # noqa: E402
from projects.robotic_link.problem import build_mvp_problem  # noqa: E402

PROCESS = Process.CNC_MILLING
MATERIALS = ["al_7075_t6", "al_6061_t6", "ss_304", "ti_6al_4v"]


def dfm_report(vertices, triangles, record=None) -> dict:
    report = assess(PROCESS, vertices, triangles, record)
    measured = [f for f in report.findings if f.assessed]
    failed = [f.rule.quantity for f in measured if not f.passes]
    return {"process": PROCESS.value, "grade": report.grade,
            "rules_measured": len(measured), "rules_failed": len(failed),
            "failed": failed, "not_measured": list(report.not_measured)}


def family_path(op, out: Path) -> dict:
    """The generative CAD executor, with materials and the process."""
    started = time.perf_counter()
    outcome = run_generative(op, candidates=12, top_k=3, seed=3,
                             materials=MATERIALS, process=PROCESS,
                             load_kind=LoadKind.BENDING,
                             step_dir=out / "family_step")
    step = out / "family_step" / f"{outcome.cad_record.part_id}.step"
    contents = sa.read_step(str(step))
    mesh = tessellate(contents.shapes[0], contents.unit_to_metres)
    return {"path": "family search", "seconds": time.perf_counter() - started,
            "feasible": outcome.feasible, "mass_kg": outcome.mass_kg,
            "family": outcome.detail["family"],
            "material_id": outcome.detail["material_id"],
            "tip_deflection_m": outcome.detail["primary_response"],
            "limit_m": outcome.detail["response_limit"],
            "evidence": outcome.detail["evidence"],
            "step": str(step),
            "dfm": dfm_report(mesh.vertices, mesh.triangles, outcome.cad_record),
            "n_screened": outcome.detail["n_screened"],
            "n_verified": outcome.detail["n_verified"]}


def topology_path(op, out: Path, divisions=(32, 12, 4), iterations=100) -> dict:
    """SIMP in the same envelope, extracted, re-solved, smoothed."""
    started = time.perf_counter()
    problem_ir = op.problem
    material = get_material(problem_ir.material_id)
    length = float(problem_ir.geometry.length_m)
    mesh = solid_box_mesh(length, float(problem_ir.geometry.max_height_m),
                          float(problem_ir.geometry.max_width_m), *divisions)
    fixed, tip = mesh.nodes_at_x(0.0), mesh.nodes_at_x(length)
    simp = SimpProblem(mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
                       poisson_ratio=material.poisson_ratio, fixed_nodes=fixed,
                       load_nodes=tip,
                       total_load_n=-float(problem_ir.loads[0].magnitude_n),
                       load_direction=1, volume_fraction=0.35,
                       filter_radius_elements=2.5,
                       passive_solid=(elements_touching(mesh, tip)
                                      | elements_touching(mesh, fixed)))
    result = optimize_projected(simp, max_iterations=iterations)

    checks = []
    for threshold in (0.3, 0.5, 0.7):
        try:
            check = verify_extracted(simp, result.density, threshold,
                                     result.final_compliance,
                                     material.density_kg_m3)
        except DisconnectedAtThreshold as exc:
            checks.append({"threshold": threshold, "refused": str(exc)})
            continue
        checks.append(check.row())
    solved = [c for c in checks if "refused" not in c]
    surface = marching_surface(mesh, result.density, 0.5, smoothing_iterations=10)
    stl = write_stl(surface, out / "topology.stl")
    return {"path": "topology", "seconds": time.perf_counter() - started,
            "elements": mesh.n_elements,
            "field_compliance_j": float(result.final_compliance),
            "grey_fraction": float(np.mean((result.density > 0.1)
                                           & (result.density < 0.9))),
            "thresholds": checks,
            "mass_kg": solved[0]["mass_kg"] if solved else None,
            "surface": surface.row(), "stl": str(stl),
            "dfm": dfm_report(surface.vertices, surface.triangles),
            "evidence": "simulated"}


def fasteners(bolt_size: str = "M6") -> dict:
    """Catalogue parts sized for the joint, with their sources."""
    length_m = 0.030
    screw = socket_head_screw(bolt_size, length_m)
    nut = hex_nut(bolt_size)
    pitch_mm, head_diameter_mm, head_height_mm, socket_mm = ISO_4762[bolt_size]
    return {"screw": {"size": bolt_size, "length_m": length_m,
                      "pitch_mm": pitch_mm,
                      "head_diameter_mm": head_diameter_mm,
                      "head_height_mm": head_height_mm,
                      "hex_key_mm": socket_mm,
                      "volume_m3": screw.volume_m3,
                      "material": material_for(screw)},
            "nut": {"size": bolt_size, "volume_m3": nut.volume_m3,
                    "material": material_for(nut)},
            "note": "envelopes without threads; dimensions from the standard"}


def build_assembly(length_m: float, material_id: str) -> Assembly:
    def link(name, length):
        return Link(name=name, length_m=length,
                    genome=DesignGenome(
                        section=HollowRectangleSection(outer_width_m=0.02,
                                                       outer_height_m=0.04,
                                                       wall_thickness_m=0.002),
                        material_id=material_id))

    origin = np.eye(4)
    origin[0, 3] = length_m
    limits = dict(lower_limit=-3.0, upper_limit=3.0)
    return Assembly(name="forearm_demo", material_id=material_id,
                    links=[link("upper", length_m), link("forearm", length_m * 0.8)],
                    joints=[Joint(name="shoulder", type=JointType.REVOLUTE,
                                  parent=None, child="upper", axis=[0, 0, 1], **limits),
                            Joint(name="elbow", type=JointType.REVOLUTE,
                                  parent="upper", child="forearm", axis=[0, 0, 1],
                                  origin=origin.tolist(), **limits)])


def assembly_checks(assembly: Assembly, material_id: str, out: Path) -> dict:
    material = get_material(material_id)
    q = np.array([0.4, -0.8])
    result: dict = {"pose_rad": q.tolist(), "gazebo_version": gazebo.gazebo_version()}
    interference = gazebo.envelope_interference(assembly, q)
    result["interference"] = {"clashes": len(interference.clashes),
                              "note": interference.summary()}
    if gazebo.gazebo_available():
        hold = gazebo.statics_cross_check(assembly, material.density_kg_m3, q,
                                          directory=out / "gazebo", seconds=4.0)
        result["spring_hold"] = {
            "joints": hold.joint_names,
            "applied_nm": [float(v) for v in hold.applied_nm],
            "settled_rad": [float(v) for v in hold.settled_rad],
            "spring_torque_at_settled_nm":
                [float(v) for v in hold.spring_torque_at_settled_nm],
            "statics_at_settled_nm":
                [float(v) for v in hold.statics_at_settled_nm],
            "relative_errors": [float(v) for v in hold.relative_errors],
            "max_drift_rad": hold.max_drift_rad,
            "summary": hold.summary(),
            "evidence": "simulated, two simulations agreeing"}
    else:
        result["spring_hold"] = {"skipped": "ign is not on PATH"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/generated/demo_v1")
    parser.add_argument("--topology-iterations", type=int, default=100)
    parser.add_argument("--skip-topology", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    op = build_optimization_problem(build_mvp_problem())
    requirement = {
        "name": op.problem.name,
        "length_m": op.problem.geometry.length_m,
        "envelope_m": [op.problem.geometry.max_height_m,
                       op.problem.geometry.max_width_m],
        "tip_load_n": op.problem.loads[0].magnitude_n,
        "max_deflection_m": op.max_deflection_m,
        "material_id": op.problem.material_id,
        "process": PROCESS.value}
    print(json.dumps(requirement, indent=2), flush=True)

    report = {"requirement": requirement, "paths": []}
    family = family_path(op, out)
    print(json.dumps({k: v for k, v in family.items() if k != "dfm"}, indent=2,
                     default=str), flush=True)
    report["paths"].append(family)

    if not args.skip_topology:
        topology = topology_path(op, out, iterations=args.topology_iterations)
        print(json.dumps({k: v for k, v in topology.items()
                          if k not in ("thresholds", "surface", "dfm")},
                         indent=2, default=str), flush=True)
        report["paths"].append(topology)

    report["fasteners"] = fasteners()
    assembly = build_assembly(float(op.problem.geometry.length_m),
                              family["material_id"])
    report["assembly"] = assembly_checks(assembly, family["material_id"], out)
    print(json.dumps(report["assembly"], indent=2, default=str), flush=True)

    report["evidence_note"] = (
        "Every result here is SIMULATED or below. No part in this demo has "
        "been made or measured, and the manufacturability numbers are a rule "
        "set, not evidence.")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
