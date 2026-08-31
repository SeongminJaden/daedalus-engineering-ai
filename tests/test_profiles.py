"""Profile system tests - schema consistency and resolution priority.

Profiles are discovered from configs/profiles/ at collection time rather than
listed here, so a newly added tier is schema-checked automatically.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import profile as profile_mod  # noqa: E402

ALL_PROFILES = profile_mod.available_profiles()

# The three tiers the project shipped with. Used only to assert the discovery
# floor - the schema checks below cover whatever is actually on disk.
MINIMUM_PROFILE_COUNT = 3


def leaf_keys(d: dict, prefix: str = "") -> set[str]:
    """Every key path in a nested dict, e.g. {'gpu', 'gpu.vram_gb', ...}."""
    out: set[str] = set()
    for k, v in d.items():
        out.add(f"{prefix}{k}")
        if isinstance(v, dict):
            out |= leaf_keys(v, f"{prefix}{k}.")
    return out


def test_profiles_are_discovered():
    assert len(ALL_PROFILES) >= MINIMUM_PROFILE_COUNT, (
        f"expected at least {MINIMUM_PROFILE_COUNT} profiles, found {ALL_PROFILES}"
    )


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_profile_loads(name):
    cfg = profile_mod.load_profile(name)
    assert cfg["name"] == name, f"{name}.yaml declares name={cfg['name']!r}"
    assert cfg["gpu"]["vram_gb"] > 0


@pytest.mark.parametrize("name", ALL_PROFILES)
def test_profile_has_required_sections(name):
    cfg = profile_mod.load_profile(name)
    for section in ("gpu", "compute", "simulation", "optimization",
                    "surrogate", "monitoring"):
        assert section in cfg, f"{name} is missing section {section!r}"


def test_all_profiles_share_one_schema():
    """Every profile on disk must expose the identical key set.

    Discovery-based on purpose: adding a tier with a stray or missing key
    fails here without anyone remembering to update this test.

    The reference is the *majority* key set rather than the first profile
    alphabetically, so when one file drifts the failure names that file
    instead of blaming every correct one against it.
    """
    schemas = {n: leaf_keys(profile_mod.load_profile(n)) for n in ALL_PROFILES}
    counts = Counter(frozenset(s) for s in schemas.values())
    reference = set(counts.most_common(1)[0][0])

    mismatches = {
        n: {"extra": sorted(s - reference), "missing": sorted(reference - s)}
        for n, s in schemas.items()
        if s != reference
    }
    assert not mismatches, (
        f"profiles diverge from the majority schema ({len(reference)} keys): "
        f"{mismatches}"
    )


def test_explicit_argument_wins_over_env(monkeypatch):
    monkeypatch.setenv(profile_mod.ENV_VAR, "cloud_a100")
    assert profile_mod.select_profile_name("laptop_4gb") == "laptop_4gb"


def test_env_var_is_used_when_no_explicit(monkeypatch):
    monkeypatch.setenv(profile_mod.ENV_VAR, "desktop_16gb")
    assert profile_mod.select_profile_name() == "desktop_16gb"


def test_unknown_profile_raises():
    with pytest.raises(FileNotFoundError):
        profile_mod.load_profile("no_such_tier")
