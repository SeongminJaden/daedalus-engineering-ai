"""Generate the industrial part dataset described in docs/dataset_spec.md.

Runs every cell (family by load case) through a process pool, each cell
resuming from its own files, then writes the scaled copies for every isotropic
material with a measured Poisson residual and a manifest with counts, timing,
md5 sums and the hash of the specification that produced the set. The data
directory is not committed; the manifest's numbers go into the docs.

    .venv/bin/python scripts/generate_industrial_dataset.py --workers 8 \
        --samples 100 --root data/generated/industrial_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.materials import get_material  # noqa: E402
from core.part_dataset.batch import (cells_for, default_cases, expand_materials,  # noqa: E402
                                     plan, read_cell, run_cell)
from core.part_dataset.families import FAMILIES  # noqa: E402
from core.part_dataset.store import write_jsonl  # noqa: E402

REFERENCE = "al_7075_t6"


def _run(args):
    cell, samples, root, seed = args
    progress = run_cell(cell, samples, root, seed)
    return (cell.name, progress.labelled, progress.refused, progress.seconds)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="data/generated/industrial_v1")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--families", nargs="*", default=list(FAMILIES))
    parser.add_argument("--skip-scaling", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cases = default_cases()
    cells = cells_for(args.families, cases)
    spec = {"families": args.families, "cases": [c.as_dict() for c in cases],
            "samples_per_cell": args.samples, "seed": args.seed,
            "reference_material": REFERENCE,
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip()}
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    print(json.dumps(plan(cells, args.samples, workers=args.workers)), flush=True)
    (root / "spec.json").write_text(json.dumps(spec, indent=2))

    started = time.perf_counter()
    progress = []
    with get_context("spawn").Pool(args.workers) as pool:
        for name, labelled, refused, seconds in pool.imap_unordered(
                _run, [(c, args.samples, root, args.seed) for c in cells]):
            progress.append({"cell": name, "labelled": labelled,
                             "refused": refused, "seconds": seconds})
            print(f"{time.perf_counter() - started:8.0f} s  {name:32s} "
                  f"{labelled} labelled {refused} refused {seconds:.0f} s", flush=True)
    solve_seconds = time.perf_counter() - started

    reference = get_material(REFERENCE)
    records = [r for c in cells for r in read_cell(root, c)]
    scaled_count = 0
    skipped: list[tuple[str, str]] = []
    if not args.skip_scaling:
        scaled_dir = root / "scaled"
        scaled_dir.mkdir(exist_ok=True)
        for cell in cells:
            cell_records = read_cell(root, cell)
            scaled, skip = expand_materials(cell_records, reference)
            skipped.extend(skip)
            scaled_count += len(scaled)
            write_jsonl(scaled_dir / f"{cell.name}.jsonl", scaled)

    files = sorted(p for p in root.rglob("*.jsonl"))
    manifest = {
        "spec_hash_sha256": spec_hash,
        "spec": spec,
        "cells": progress,
        "solved_records": len(records),
        "scaled_records": scaled_count,
        "total_records": len(records) + scaled_count,
        "scaling_skipped": len(skipped),
        "solve_seconds": solve_seconds,
        "total_seconds": time.perf_counter() - started,
        "workers": args.workers,
        "files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                   "md5": _md5(p)} for p in files],
        "bytes": sum(p.stat().st_size for p in files),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("solved_records", "scaled_records",
                                               "total_records", "scaling_skipped",
                                               "solve_seconds", "bytes")}), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    raise SystemExit(main())
