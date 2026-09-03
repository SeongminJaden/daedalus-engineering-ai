"""Measure what a topology run hands over: the field against the extracted part.

    .venv/bin/python scripts/topology_design_study.py --out data/generated/topology_v1

Runs the cantilever with and without passive load and support patches, with
plain SIMP and with the three-field projection, re-solves the thresholded part
in CalculiX, and writes the tables that go into docs/topology_design.md.
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

from core.materials import get_material  # noqa: E402
from optimization.topology import SimpProblem, optimize  # noqa: E402
from optimization.topology.threefield import optimize_projected  # noqa: E402
from optimization.topology.verify import (DisconnectedAtThreshold,  # noqa: E402
                                          elements_touching, format_table,
                                          threshold_table, verify_extracted)
from physics.fem.mesh import solid_box_mesh  # noqa: E402

MATERIAL = "al_7075_t6"


def build(divisions, radius, length=0.5, height=0.1, width=0.05,
          volume_fraction=0.35, load_n=-1000.0, passive=True) -> SimpProblem:
    material = get_material(MATERIAL)
    mesh = solid_box_mesh(length, height, width, *divisions)
    fixed, load = mesh.nodes_at_x(0.0), mesh.nodes_at_x(length)
    patches = None
    if passive:
        patches = elements_touching(mesh, load) | elements_touching(mesh, fixed)
    return SimpProblem(mesh=mesh, youngs_modulus_pa=material.youngs_modulus_pa,
                       poisson_ratio=material.poisson_ratio, fixed_nodes=fixed,
                       load_nodes=load, total_load_n=load_n, load_direction=1,
                       volume_fraction=volume_fraction,
                       filter_radius_elements=radius, passive_solid=patches)


def grey_fraction(density: np.ndarray) -> float:
    return float(np.mean((density > 0.1) & (density < 0.9)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/generated/topology_v1")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    density_kg_m3 = get_material(MATERIAL).density_kg_m3
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"material": MATERIAL, "cases": []}

    cases = [
        ("simp, no passive patches", (24, 8, 4), 1.5, False, optimize),
        ("simp, passive patches", (24, 8, 4), 1.5, True, optimize),
        ("three field, passive patches", (24, 8, 4), 1.5, True, optimize_projected),
        ("three field, finer", (32, 12, 4), 2.5, True, optimize_projected),
        ("three field, finest", (40, 16, 6), 2.0, True, optimize_projected),
    ]
    for name, divisions, radius, passive, runner in cases:
        problem = build(divisions, radius, passive=passive)
        started = time.perf_counter()
        result = runner(problem, max_iterations=args.iterations)
        seconds = time.perf_counter() - started
        rows = threshold_table(problem, result.density, result.final_compliance,
                               density_kg_m3)
        entry = {"name": name, "divisions": list(divisions),
                 "filter_radius_elements": radius, "passive": passive,
                 "elements": problem.mesh.n_elements,
                 "field_compliance_j": float(result.final_compliance),
                 "volume_fraction": float(np.mean(result.density)),
                 "grey_fraction": grey_fraction(result.density),
                 "seconds": seconds, "rows": rows}
        report["cases"].append(entry)
        print(f"\n== {name}  {problem.mesh.n_elements} elements  "
              f"grey {entry['grey_fraction']:.2f}  "
              f"field compliance {entry['field_compliance_j']:.4e}  "
              f"{seconds:.0f} s", flush=True)
        print(format_table(rows), flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = ["# Topology extraction study\n"]
    for entry in report["cases"]:
        lines.append(f"## {entry['name']}\n")
        lines.append(f"{entry['elements']} elements, grey fraction "
                     f"{entry['grey_fraction']:.2f}, field compliance "
                     f"{entry['field_compliance_j']:.4e} J, {entry['seconds']:.0f} s\n")
        lines.append(format_table(entry["rows"]) + "\n")
    (out / "report.md").write_text("\n".join(lines))
    print(f"\nwrote {out}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
