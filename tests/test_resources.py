"""Resource readings must be measured, or say they were not.

The failure this file guards against is a plausible zero. An unread field and
a genuinely idle device look identical once both are 0, and the second is a
lie that no later reader can detect.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from monitoring.resources import (GpuReading, ResourceSnapshot, _bar, _number,
                                  read_cpu, read_gpus, read_memory,
                                  render_panel, snapshot)

has_nvidia_smi = shutil.which("nvidia-smi") is not None
requires_gpu = pytest.mark.skipif(not has_nvidia_smi,
                                  reason="nvidia-smi is not installed")


# --------------------------------------------- unreadable is not zero

@pytest.mark.parametrize("text", ["[N/A]", "N/A", "[Not Supported]",
                                  "not supported", "", "   ", "garbage"])
def test_an_unreadable_field_is_none_and_never_zero(text):
    assert _number(text) is None


@pytest.mark.parametrize("text,expected", [("6.62", 6.62), ("0", 0.0),
                                           (" 43 ", 43.0)])
def test_a_real_reading_is_kept_including_a_real_zero(text, expected):
    """A measured zero is a fact and must survive. Only gaps become None."""
    assert _number(text) == expected


def test_the_stored_form_says_unavailable_rather_than_zero():
    reading = GpuReading(name="test", utilization_percent=None,
                         memory_used_mib=None, memory_total_mib=None,
                         temperature_c=None, power_w=12.5,
                         power_limit_w=None)
    stored = ResourceSnapshot(timestamp=0.0, cpu=read_cpu(sample_s=0.01),
                              memory=read_memory(),
                              gpus=(reading,)).as_dict()
    gpu = stored["gpus"][0]
    assert gpu["power_w"] == 12.5
    assert gpu["power_limit_w"] == "unavailable"
    assert gpu["utilization_percent"] == "unavailable"
    assert gpu["memory_percent"] == "unavailable"


def test_an_unreadable_bar_does_not_draw_an_empty_one():
    """An empty bar and a missing reading look the same at a glance."""
    assert "unavailable" in _bar(None)
    assert set(_bar(0.0)) == {"."}
    assert set(_bar(1.0)) == {"#"}


def test_memory_percent_needs_both_numbers():
    partial = GpuReading(name="x", utilization_percent=None,
                         memory_used_mib=100.0, memory_total_mib=None,
                         temperature_c=None, power_w=None, power_limit_w=None)
    assert partial.memory_percent is None


# ------------------------------------------------ the readings are real

def test_cpu_readings_are_plausible_and_per_core():
    reading = read_cpu(sample_s=0.05)
    assert reading.cores >= 1
    assert len(reading.percent_per_core) == reading.cores
    assert all(0.0 <= core <= 100.0 for core in reading.percent_per_core)
    assert 0.0 <= reading.percent_total <= 100.0
    assert reading.busiest_core_percent >= reading.percent_total - 1e-9


def test_memory_matches_psutil():
    """Checked against the source rather than against itself."""
    import psutil

    reading = read_memory()
    virtual = psutil.virtual_memory()
    assert reading.total_gb == pytest.approx(virtual.total / 1024 ** 3,
                                             rel=1e-9)
    assert reading.available_gb <= reading.total_gb
    assert 0.0 <= reading.percent <= 100.0


@requires_gpu
def test_gpu_readings_match_nvidia_smi():
    """The independent check: our parse against a fresh nvidia-smi call."""
    gpus, reason = read_gpus()
    assert reason == ""
    assert gpus
    raw = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    assert len(gpus) == len(raw)
    for gpu, line in zip(gpus, raw):
        name, total = [p.strip() for p in line.split(",")]
        assert gpu.name == name
        assert gpu.memory_total_mib == pytest.approx(float(total))


@requires_gpu
def test_vram_used_is_within_the_total():
    for gpu in read_gpus()[0]:
        if gpu.memory_used_mib is None or gpu.memory_total_mib is None:
            continue
        assert 0.0 <= gpu.memory_used_mib <= gpu.memory_total_mib


def test_a_missing_nvidia_smi_reports_a_reason_rather_than_raising(monkeypatch):
    """A monitor must never take down the run it is watching."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    gpus, reason = read_gpus()
    assert gpus == ()
    assert "unavailable" in reason


def test_a_failing_nvidia_smi_reports_a_reason_rather_than_raising(monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("no such device")

    monkeypatch.setattr(subprocess, "run", explode)
    gpus, reason = read_gpus()
    assert gpus == ()
    assert "unavailable" in reason


# ----------------------------------------------------- it stays out of the way

def test_a_snapshot_renders_without_touching_anything():
    reading = snapshot(sample_s=0.05)
    assert render_panel(reading) is not None
    assert reading.timestamp > 0.0


def test_the_snapshot_is_json_serialisable():
    """The Brain and a dashboard artifact both need this to round trip."""
    import json

    restored = json.loads(json.dumps(snapshot(sample_s=0.05).as_dict()))
    assert restored["cpu"]["cores"] >= 1
    assert "gpus" in restored
