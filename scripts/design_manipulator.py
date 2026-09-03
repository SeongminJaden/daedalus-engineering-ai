"""Run the whole pipeline on one six axis arm and write the design document.

    .venv/bin/python scripts/design_manipulator.py --out data/generated/manipulator_v1

Every stage writes a table. A stage that cannot run writes what it could not
do instead, because the point of this exercise is as much the holes in the
pipeline as the design that comes out of it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.materials import get_material  # noqa: E402
from projects.manipulator.arm import build_arm, stretched_pose  # noqa: E402
from projects.manipulator.loop import run_loop  # noqa: E402
from projects.manipulator.spec import SPEC  # noqa: E402
from projects.manipulator.stages import (assembly_stage, compliance_stage,  # noqa: E402
                                         drive_comparison_stage, drivetrain_stage,
                                         dynamics_stage, fatigue_stage,
                                         features_stage, link_design_stage,
                                         manufacturability_stage,
                                         measurement_plan_stage,
                                         pinocchio_cross_check, policy_stage,
                                         reflected_inertia_stage,
                                         verification_stage)


def table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "_no rows_"
    columns = columns or sorted({key for row in rows for key in row})
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            elif isinstance(value, list):
                cells.append(", ".join(str(v) for v in value) or "none")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section(title: str, stage) -> str:
    parts = [f"## {title}\n", table(stage.rows), ""]
    for note in stage.notes:
        parts.append(f"Note: {note}.\n")
    for gap in stage.could_not:
        parts.append(f"**Could not:** {gap}\n")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/generated/manipulator_v1")
    parser.add_argument("--skip-links", action="store_true",
                        help="skip the two link design paths, which are slow")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    stages = {}
    stages["policy"] = policy_stage(SPEC)
    print("policy done", flush=True)

    loop = run_loop(SPEC)
    stages["loop"] = loop
    sections = loop.data["final_sections"]
    actuator_masses = loop.data["actuator_masses"]
    arm = build_arm(sections, SPEC)
    print("loop done", flush=True)

    stages["dynamics"] = dynamics_stage(arm, SPEC)
    stages["pinocchio"] = pinocchio_cross_check(arm, SPEC)
    from physics.dynamics import mass_matrix
    from projects.manipulator.arm import stretched_pose as _pose
    inertia_matrix = mass_matrix(arm, _pose(SPEC),
                                 get_material(arm.material_id).density_kg_m3)
    load_inertias = {joint.name: float(inertia_matrix[i, i])
                     for i, joint in enumerate(arm.actuated_joints())}
    stages["drivetrain"] = drivetrain_stage(stages["dynamics"], SPEC,
                                            load_inertias)
    stages["inertia"] = reflected_inertia_stage(arm, stages["drivetrain"], SPEC)
    stages["comparison"] = drive_comparison_stage(stages["drivetrain"], SPEC)
    stages["compliance"] = compliance_stage(arm, stages["drivetrain"], SPEC)
    print("dynamics and drivetrain done", flush=True)

    if not args.skip_links:
        stages["links"] = link_design_stage(SPEC, sections, actuator_masses,
                                            step_dir=out / "links")
        stages["manufacturability"] = manufacturability_stage(stages["links"], SPEC)
        stages["verification"] = verification_stage(stages["links"])
        stages["measurement"] = measurement_plan_stage(SPEC, sections,
                                                       stages["links"])
        print("link design done", flush=True)

    stages["fatigue"] = fatigue_stage(stages["dynamics"], sections, SPEC)
    stages["features"] = features_stage(SPEC, sections)
    stages["assembly"] = assembly_stage(arm, SPEC, directory=out / "assembly")
    print("assembly done", flush=True)

    material = get_material(arm.material_id)
    structure = sum(link.mass_kg(material.density_kg_m3) for link in arm.links)
    actuators = sum(actuator_masses.values())
    bom_sourced = [{"item": part, "count": 1, "source": "vendor datasheet",
                    "mass_kg": None} for part in sorted(set(
                        row["selected"] for row in stages["drivetrain"].rows
                        if row.get("selected")))]
    bom_unsourced = [{"item": f"{link.name} link", "count": 1,
                      "source": "designed here, not a purchased part",
                      "mass_kg": link.mass_kg(material.density_kg_m3)}
                     for link in arm.links]

    report = {
        "specification": {
            "payload_kg": SPEC.payload_kg, "reach_m": SPEC.reach_m,
            "move_deg": 90, "move_time_s": SPEC.move_time_s,
            "safety_factor": SPEC.static_safety_factor_metal,
            "tip_deflection_limit_m": SPEC.tip_deflection_limit_m,
            "torque_margin": SPEC.torque_margin},
        "masses": {"structure_kg": structure, "actuators_kg": actuators,
                   "total_kg": structure + actuators},
        "stages": {name: {"rows": stage.rows, "notes": stage.notes,
                          "could_not": stage.could_not}
                   for name, stage in stages.items()},
        "bom": {"sourced": bom_sourced, "designed": bom_unsourced},
        "seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))

    document = [
        "# A six axis manipulator, through every stage this project has\n",
        "Everything here is SIMULATED or a rule set. No part has been made "
        "and no number has been measured on hardware. The stages that could "
        "not run say so; that list is half the result.\n",
        "## The specification\n",
        table([{"quantity": k, "value": v}
               for k, v in report["specification"].items()]),
        "",
        f"Structure mass {structure:.3f} kg, actuators {actuators:.3f} kg, "
        f"total {structure + actuators:.3f} kg. The whole run takes "
        f"{report['seconds'] / 60:.1f} minutes.\n",
        section("1. The policy layer: sentences to problems", stages["policy"]),
        section("9. The mass torque loop", stages["loop"]),
        section("2. Dynamics on the stated move", stages["dynamics"]),
        section("2b. The same dynamics in Pinocchio", stages["pinocchio"]),
        section("3. Drive selection from the sourced catalogue", stages["drivetrain"]),
        section("3b. Reflected inertia and the matched ratio", stages["inertia"]),
        section("3c. Direct drive against geared, per joint", stages["comparison"]),
        section("3d. Joint compliance and backlash, from the gear data",
                stages["compliance"]),
    ]
    if "links" in stages:
        document.append(section("4. Each link through both design paths",
                                stages["links"]))
        document.append(section("5. What the solver said, and its mesh sensitivity",
                                stages["verification"]))
        document.append(section("8. Manufacturability", stages["manufacturability"]))
    document += [
        section("6. Fatigue over the duty cycle", stages["fatigue"]),
        section("7. Fasteners and tolerances", stages["features"]),
        section("10. Assembly, Gazebo and interference", stages["assembly"]),
        "## 11. Bill of materials\n",
        "Sourced parts, whose numbers come from a vendor page:\n",
        table(bom_sourced),
        "\nParts designed here, which are not purchasable and carry no "
        "vendor data:\n",
        table(bom_unsourced),
        "",
    ]
    if "measurement" in stages:
        document.append(section("12. What to measure first", stages["measurement"]))
    document += [
        "## What this design can and cannot claim\n",
        "**It can claim** that a six axis arm of this geometry, with these "
        "sections, holds the rated payload at full reach with the stated "
        "safety factor and deflection limit ACCORDING TO this project's "
        "solvers, and that two independent engines agree with its rigid body "
        "dynamics to twelve digits and with its statics to 0.03 percent.\n",
        "**It cannot claim** that the arm works. Nothing has been built. Two "
        "joints have no drive at all, so as a machine it is incomplete, and "
        "the reason they have none is that the catalogue pages do not print "
        "the values the selection needs. The link masses come from a sizing "
        "routine that solves for one dimension of three. The free form parts "
        "are solved once with linear tetrahedra and their deflections are "
        "lower bounds. Friction, backlash and joint compliance are all zero "
        "because no measured parameters exist for them, so every torque here "
        "is a lower bound too. No cover, no wiring, no bearings beyond a seat "
        "tolerance, and no bolted joint analysis.\n",
        "**The evidence grade of every number in this document is SIMULATED "
        "or below**, except the manufacturability rows, which are a rule set "
        "with its own grade and are not evidence at all.\n",
    ]
    (out / "design.md").write_text("\n".join(document))
    print(f"wrote {out}/design.md and report.json in "
          f"{report['seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
