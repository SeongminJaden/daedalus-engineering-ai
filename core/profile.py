"""core.profile - GPU tier profile resolution and loading.

The same pipeline has to run on a 4 GB laptop GPU and on an 80 GB datacenter
card. Rather than scattering `if vram < x` checks through the codebase, every
size-dependent knob lives in a profile YAML under ``configs/profiles/``; the
rest of the code just reads the resolved dict.

Resolution order (highest priority first):
    1. explicit argument (CLI ``--profile``)
    2. ``ENG_PROFILE`` environment variable
    3. auto-detection from available VRAM
    4. ``profile.fallback`` in ``configs/default.yaml``
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
PROFILE_DIR = CONFIG_DIR / "profiles"

ENV_VAR = "ENG_PROFILE"


def available_profiles() -> list[str]:
    """Names of every profile YAML shipped in configs/profiles/."""
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def detect_vram_gb() -> float | None:
    """Total VRAM of the first visible GPU, or None if there isn't one.

    Tries torch first (already a dependency), falls back to nvidia-smi so this
    still works before torch is importable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.split("\n")[0].strip()
            return float(out) / 1024
        except Exception:
            pass
    return None


def select_profile_name(explicit: str | None = None) -> str:
    """Resolve which profile to use, following the documented priority order."""
    if explicit:
        return explicit
    if os.environ.get(ENV_VAR):
        return os.environ[ENV_VAR]

    defaults = _read_yaml(CONFIG_DIR / "default.yaml").get("profile", {})
    if defaults.get("auto_detect", True):
        vram = detect_vram_gb()
        if vram is not None:
            best = None
            for name in available_profiles():
                tier = _read_yaml(PROFILE_DIR / f"{name}.yaml")
                budget = tier.get("gpu", {}).get("vram_gb")
                # largest tier that still fits in the VRAM we actually have
                if budget is not None and budget <= vram + 0.5:
                    if best is None or budget > best[1]:
                        best = (name, budget)
            if best:
                return best[0]
    return defaults.get("fallback", "laptop_4gb")


def load_profile(name: str | None = None) -> dict[str, Any]:
    """Load a single profile YAML. ``name=None`` resolves it automatically."""
    resolved = select_profile_name(name)
    path = PROFILE_DIR / f"{resolved}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"unknown profile {resolved!r}; available: {available_profiles()}"
        )
    return _read_yaml(path)


def load_config(profile: str | None = None) -> dict[str, Any]:
    """default.yaml with the resolved profile attached under 'active_profile'."""
    cfg = _read_yaml(CONFIG_DIR / "default.yaml")
    cfg["active_profile"] = load_profile(profile)
    return cfg


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as fh:
        return yaml.safe_load(fh) or {}
