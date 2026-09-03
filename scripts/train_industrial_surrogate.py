"""Train the shape surrogate on a generated run and measure it per family and
per load case.

    .venv/bin/python scripts/train_industrial_surrogate.py \
        --root data/generated/industrial_v1 --out data/generated/surrogate_v1

Writes the model, a metrics JSON and a markdown report. The held-out set is
the last fifth of every cell's draw order, so every family and every load
case is in it, and a scaled copy is always on the same side as the part it
was scaled from: a copy of a training part in the test set would be a leak,
since the two differ only by an exact material factor.
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

from core.part_dataset.industrial_surrogate import (  # noqa: E402
    TARGET_NAMES, baseline_metrics, cache_descriptors, draw_order, format_table,
    holdout_mask, load_run, metrics_by_group, train_industrial_surrogate,
    without_proxies)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="data/generated/industrial_v1")
    parser.add_argument("--out", default="data/generated/surrogate_v1")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--hidden", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--samples-per-cell", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--solved-only", action="store_true",
                        help="ignore the scaled copies, to isolate their effect")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    started = time.perf_counter()
    descriptors = cache_descriptors(root, workers=args.workers)
    print(f"{len(descriptors)} descriptors in {time.perf_counter() - started:.0f} s",
          flush=True)

    data = load_run(root, descriptors, include_scaled=not args.solved_only)
    order = draw_order(root)
    test_mask = holdout_mask(data, order, args.samples_per_cell)
    train, test = data.subset(~test_mask), data.subset(test_mask)
    assert not (set(train.base_ids) & set(test.base_ids)), "a part is on both sides"
    print(f"{len(data)} rows, {len(train)} train, {len(test)} held out, "
          f"{len(set(data.base_ids))} solved parts, {len(set(data.materials))} materials",
          flush=True)

    surrogate = train_industrial_surrogate(
        train, test, epochs=args.epochs, hidden=tuple(args.hidden),
        batch=args.batch, device=args.device)
    print(json.dumps(surrogate.training), flush=True)
    for name in TARGET_NAMES:
        m = surrogate.test_metrics[name]
        print(f"{name:22s} spearman {m['spearman']:.3f} log_r2 {m['r2_log']:.3f} "
              f"median {m['median_rel_err']:.3f} p95 {m['p95_rel_err']:.3f}", flush=True)

    proxy_by_kind = baseline_metrics(test, ("kinds",))
    proxy_by_family = baseline_metrics(test, ("families",))
    everything = test.subset(np.ones(len(test), dtype=bool))
    everything.families = ["all"] * len(everything)
    proxy_overall = baseline_metrics(everything, ("families",))[0]

    by_family = metrics_by_group(surrogate, test, ("families",))
    by_kind = metrics_by_group(surrogate, test, ("kinds",))
    by_material = metrics_by_group(surrogate, test, ("materials",))
    print(format_table(by_family, ("families",)), flush=True)
    print(format_table(by_kind, ("kinds",)), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    surrogate.save(out)
    ablated = train_industrial_surrogate(
        without_proxies(train), without_proxies(test), epochs=args.epochs,
        hidden=tuple(args.hidden), batch=args.batch, device=args.device)
    print("no proxy features: " + json.dumps(
        {k: round(v, 3) for k, v in ablated.test_metrics["primary_response"].items()
         if isinstance(v, float)}), flush=True)
    print("proxy alone, by load case:", flush=True)
    print(format_table(proxy_by_kind, ("kinds",)), flush=True)

    report = {"root": str(root), "rows": len(data), "train": len(train),
              "test": len(test), "solved_parts": len(set(data.base_ids)),
              "materials": sorted(set(data.materials)),
              "solved_only": args.solved_only,
              "training": surrogate.training, "overall": surrogate.test_metrics,
              "by_family": by_family, "by_kind": by_kind,
              "by_material": by_material,
              "proxy_alone_overall": proxy_overall,
              "proxy_alone_by_kind": proxy_by_kind,
              "proxy_alone_by_family": proxy_by_family,
              "no_proxy_features": ablated.test_metrics,
              "seconds": time.perf_counter() - started}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(
        "# Surrogate on " + str(root) + "\n\n"
        + f"{len(train)} training rows, {len(test)} held out, "
        + f"{len(set(data.base_ids))} solved parts.\n\n## By family\n\n"
        + format_table(by_family, ("families",))
        + "\n\n## By load case\n\n" + format_table(by_kind, ("kinds",))
        + "\n\n## By material\n\n" + format_table(by_material, ("materials",))
        + "\n\n## The proxy alone, by load case\n\n"
        + format_table(proxy_by_kind, ("kinds",))
        + "\n\n## The proxy alone, by family\n\n"
        + format_table(proxy_by_family, ("families",)) + "\n")
    print(f"wrote {out}/report.md in {time.perf_counter() - started:.0f} s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
