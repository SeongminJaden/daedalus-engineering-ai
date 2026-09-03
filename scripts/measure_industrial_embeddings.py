"""Re-measure the CAD embedding on a generated run: thirteen families instead
of five, and the same comparison against the descriptors and the D2 histogram.

    .venv/bin/python scripts/measure_industrial_embeddings.py \
        --root data/generated/industrial_v1 --per-family 120

Point clouds come from the STEP files of the solved parts (a scaled copy is
the same geometry, so materials add nothing here). The split is the cell draw
order again, so no part is on both sides.
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

from core.part_dataset.descriptors import describe_step  # noqa: E402
from core.part_dataset.embedding import (POINTS_PER_PART, nearest_neighbour_precision,  # noqa: E402
                                         train_embedding)
from core.part_dataset.industrial_surrogate import draw_order  # noqa: E402
from core.part_dataset.pointcloud import d2_signature, point_cloud_of  # noqa: E402
from nodes import step_analyzer as sa  # noqa: E402


def standardise(x, ref):
    mu, sd = ref.mean(axis=0), ref.std(axis=0)
    sd[sd == 0.0] = 1.0
    return (x - mu) / sd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="data/generated/industrial_v1")
    parser.add_argument("--out", default="data/generated/embedding_v1")
    parser.add_argument("--per-family", type=int, default=120)
    parser.add_argument("--samples-per-cell", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    order = draw_order(root)
    cut = int(round(args.samples_per_cell * 0.8))
    rng = np.random.default_rng(0)
    started = time.perf_counter()

    per_family: dict[str, list[tuple[Path, bool]]] = {}
    for cell in sorted(p for p in root.iterdir() if (p / "step").is_dir()):
        family = cell.name.split("__")[0]
        bucket = per_family.setdefault(family, [])
        for step in sorted((cell / "step").glob("*.step")):
            if len(bucket) < args.per_family:
                bucket.append((step, order.get(step.stem, 0) >= cut))

    data = {"train": {"clouds": [], "d2": [], "desc": [], "fams": []},
            "test": {"clouds": [], "d2": [], "desc": [], "fams": []}}
    for family, items in sorted(per_family.items()):
        for step, held_out in items:
            contents = sa.read_step(str(step))
            cloud = point_cloud_of(contents.shapes[0], contents.unit_to_metres,
                                   POINTS_PER_PART, rng)
            side = data["test" if held_out else "train"]
            side["clouds"].append(cloud)
            side["d2"].append(d2_signature(cloud, rng=rng))
            side["desc"].append(describe_step(str(step))[0].vector())
            side["fams"].append(family)
        print(f"{family:16s} {len(items)} parts  "
              f"{time.perf_counter() - started:.0f} s", flush=True)
    for side in data.values():
        for key in ("clouds", "d2", "desc"):
            side[key] = np.array(side[key])

    bundle = train_embedding(data["train"]["clouds"], data["train"]["fams"],
                             epochs=args.epochs, device=args.device)
    e_tr = bundle.embed(data["train"]["clouds"])
    e_te = bundle.embed(data["test"]["clouds"])
    result = {
        "families": len(per_family),
        "train": len(data["train"]["fams"]), "test": len(data["test"]["fams"]),
        "precision_descriptors": nearest_neighbour_precision(
            standardise(data["test"]["desc"], data["train"]["desc"]),
            data["test"]["fams"],
            standardise(data["train"]["desc"], data["train"]["desc"]),
            data["train"]["fams"]),
        "precision_pointnet": nearest_neighbour_precision(
            e_te, data["test"]["fams"], e_tr, data["train"]["fams"]),
        "precision_d2": nearest_neighbour_precision(
            data["test"]["d2"], data["test"]["fams"], data["train"]["d2"],
            data["train"]["fams"]),
        "training": bundle.train_metrics,
        "seconds": time.perf_counter() - started,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "training"}, indent=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle.save(out)
    (out / "report.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
