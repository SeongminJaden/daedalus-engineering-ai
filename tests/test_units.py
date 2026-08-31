"""SI unit constants and display-boundary helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import units  # noqa: E402


def test_constants():
    assert units.MM == 1e-3
    assert units.CM == 1e-2
    assert units.MPA == 1e6
    assert units.GPA == 1e9
    assert units.KN == 1e3
    assert units.DEG == pytest.approx(math.pi / 180.0)
    assert units.KGF == pytest.approx(9.80665)


@pytest.mark.parametrize("value,const,to_fn", [
    (50.0, units.MM, units.to_mm),
    (120.0, units.MPA, units.to_mpa),
    (71.7, units.GPA, units.to_gpa),
    (2.5, units.KN, units.to_kn),
    (45.0, units.DEG, units.to_deg),
])
def test_round_trip_through_si(value, const, to_fn):
    """human value -> SI -> human value must be identity."""
    assert to_fn(value * const) == pytest.approx(value)


def test_kgf_to_n():
    assert units.kgf_to_n(20.0) == pytest.approx(20.0 * 9.80665)
