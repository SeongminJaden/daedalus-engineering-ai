"""scripts/gpu_sanity.py - verify the GPU stack actually works.

Checks, in order:
  1. NVIDIA Warp initialises and sees a CUDA device
  2. a trivial Warp kernel compiles and runs on the GPU, with a correct result
  3. torch sees CUDA and can do a device round-trip
  4. the profile system resolves a profile for this machine

Run:  env -u PYTHONPATH .venv/bin/python scripts/gpu_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

failures: list[str] = []


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --- 1 & 2: Warp -------------------------------------------------------------
section("NVIDIA Warp")
try:
    import warp as wp

    wp.init()
    print(f"warp version : {wp.config.version}")
    devices = wp.get_devices()
    print(f"devices      : {[str(d) for d in devices]}")
    cuda_devices = [d for d in devices if d.is_cuda]
    if not cuda_devices:
        failures.append("warp: no CUDA device visible")
        print("CUDA device  : NONE")
    else:
        dev = cuda_devices[0]
        print(f"cuda device  : {dev} ({dev.name})")

        @wp.kernel
        def _scale(src: wp.array(dtype=wp.float32),
                   dst: wp.array(dtype=wp.float32),
                   k: float):
            i = wp.tid()
            dst[i] = src[i] * k

        n = 1024
        src = wp.array([float(i) for i in range(n)], dtype=wp.float32, device=dev)
        dst = wp.zeros(n, dtype=wp.float32, device=dev)
        wp.launch(_scale, dim=n, inputs=[src, dst, 3.0], device=dev)
        wp.synchronize_device(dev)
        got = dst.numpy()
        expected_ok = abs(got[10] - 30.0) < 1e-5 and abs(got[-1] - 3069.0) < 1e-3
        print(f"kernel result: dst[10]={got[10]} dst[-1]={got[-1]} -> "
              f"{'OK' if expected_ok else 'WRONG'}")
        if not expected_ok:
            failures.append("warp: kernel produced wrong values")
except Exception as exc:  # noqa: BLE001
    failures.append(f"warp: {type(exc).__name__}: {exc}")
    print(f"FAILED: {type(exc).__name__}: {exc}")


# --- 3: torch ----------------------------------------------------------------
section("torch")
try:
    import torch

    print(f"torch version: {torch.__version__}")
    print(f"cuda build   : {torch.version.cuda}")
    avail = torch.cuda.is_available()
    print(f"is_available : {avail}")
    if not avail:
        failures.append("torch: cuda not available")
    else:
        print(f"device       : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"total vram   : {total:.2f} GB")
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).sum().item()
        print(f"matmul       : sum={y:.4f} -> OK")
except Exception as exc:  # noqa: BLE001
    failures.append(f"torch: {type(exc).__name__}: {exc}")
    print(f"FAILED: {type(exc).__name__}: {exc}")


# --- 4: profile --------------------------------------------------------------
section("profile")
try:
    from core import profile as profile_mod

    vram = profile_mod.detect_vram_gb()
    name = profile_mod.select_profile_name()
    cfg = profile_mod.load_profile(name)
    print(f"detected vram: {'-' if vram is None else f'{vram:.2f} GB'}")
    print(f"available    : {profile_mod.available_profiles()}")
    print(f"selected     : {name}")
    print(f"  vram_gb          : {cfg['gpu']['vram_gb']}")
    print(f"  fidelity         : {cfg['simulation']['fidelity']}")
    print(f"  fem_max_dofs     : {cfg['simulation']['fem_max_dofs']}")
    print(f"  candidate_pool   : {cfg['optimization']['candidate_pool']}")
    print(f"  max_batch        : {cfg['compute']['max_batch']}")
except Exception as exc:  # noqa: BLE001
    failures.append(f"profile: {type(exc).__name__}: {exc}")
    print(f"FAILED: {type(exc).__name__}: {exc}")


# --- verdict -----------------------------------------------------------------
section("verdict")
if failures:
    print("FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
