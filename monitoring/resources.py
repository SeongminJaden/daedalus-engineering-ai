"""Live CPU, memory and GPU readings, with unreadable values left unread.

The one rule here is that a number must have been measured. A field this
module cannot read comes back as None and renders as "unavailable"; it never
comes back as zero. On this machine that is not hypothetical: an RTX 3050
Laptop reports its power LIMIT as "[N/A]" while reporting draw perfectly well,
so a monitor that defaulted to 0 W would draw a power bar against a limit of
zero and look like a bug in the GPU rather than a gap in the driver.

VALIDITY DOMAIN
===============
Stated before implementing, per the standing discipline.

What these numbers mean
    CPU percent is sampled over a real interval, because psutil's first
    non-blocking call returns whatever has accumulated since the process
    started, which for a long lived process is a meaningless average. The
    default here is a short blocking sample.

    Memory "available" is the kernel's estimate of what can be allocated
    without swapping, which is the number worth watching. It is not
    total minus used: buffers and cache are reclaimable.

    GPU utilisation from nvidia-smi is the percentage of the last sampling
    period during which any kernel was running. It says nothing about how
    much of the device that kernel used, so a single tiny kernel looping can
    read 100 percent while the GPU is nearly idle in any useful sense.

What this module is not
    It observes and never influences. Nothing here is on a compute path,
    nothing it reads changes a result, and a failure to read must degrade to
    "unavailable" rather than raise into whatever is being monitored.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field

#: Long enough for a meaningful CPU sample, short enough to feel live.
DEFAULT_SAMPLE_S = 0.2

#: nvidia-smi writes this when a field exists but the driver will not report
#: it. It is a gap, not a value.
_NOT_AVAILABLE = {"[n/a]", "n/a", "[not supported]", "not supported", ""}

_GPU_FIELDS = ("name", "utilization.gpu", "memory.used", "memory.total",
               "temperature.gpu", "power.draw", "power.limit")


def _number(text: str) -> float | None:
    """A reading, or None when the driver declined to give one."""
    cleaned = text.strip()
    if cleaned.lower() in _NOT_AVAILABLE:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass(frozen=True)
class CpuReading:
    cores: int
    percent_total: float
    percent_per_core: tuple[float, ...]
    load_1m: float | None
    load_5m: float | None
    load_15m: float | None

    @property
    def busiest_core_percent(self) -> float:
        return max(self.percent_per_core, default=0.0)


@dataclass(frozen=True)
class MemoryReading:
    total_gb: float
    used_gb: float
    available_gb: float
    percent: float


@dataclass(frozen=True)
class GpuReading:
    """A GPU as the driver describes it. Any field may be unreadable."""

    name: str
    utilization_percent: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None
    temperature_c: float | None
    power_w: float | None
    power_limit_w: float | None

    @property
    def memory_percent(self) -> float | None:
        if not self.memory_used_mib or not self.memory_total_mib:
            return None
        return 100.0 * self.memory_used_mib / self.memory_total_mib


@dataclass(frozen=True)
class ResourceSnapshot:
    """One moment, with its timestamp, for a panel or a stored record."""

    timestamp: float
    cpu: CpuReading
    memory: MemoryReading
    gpus: tuple[GpuReading, ...] = field(default_factory=tuple)
    gpu_unavailable_reason: str = ""

    def as_dict(self) -> dict:
        """The stored form. Unreadable fields say so rather than reading 0."""

        def show(value):
            return "unavailable" if value is None else value

        return {
            "timestamp": self.timestamp,
            "cpu": {
                "cores": self.cpu.cores,
                "percent_total": self.cpu.percent_total,
                "percent_per_core": list(self.cpu.percent_per_core),
                "busiest_core_percent": self.cpu.busiest_core_percent,
                "load_1m": show(self.cpu.load_1m),
                "load_5m": show(self.cpu.load_5m),
                "load_15m": show(self.cpu.load_15m),
            },
            "memory_gb": {
                "total": self.memory.total_gb,
                "used": self.memory.used_gb,
                "available": self.memory.available_gb,
                "percent": self.memory.percent,
            },
            "gpus": [
                {
                    "name": gpu.name,
                    "utilization_percent": show(gpu.utilization_percent),
                    "memory_used_mib": show(gpu.memory_used_mib),
                    "memory_total_mib": show(gpu.memory_total_mib),
                    "memory_percent": show(gpu.memory_percent),
                    "temperature_c": show(gpu.temperature_c),
                    "power_w": show(gpu.power_w),
                    "power_limit_w": show(gpu.power_limit_w),
                }
                for gpu in self.gpus
            ],
            "gpu_unavailable_reason": self.gpu_unavailable_reason,
        }


def read_cpu(sample_s: float = DEFAULT_SAMPLE_S) -> CpuReading:
    """CPU load, sampled over a real interval rather than since boot."""
    import os

    import psutil

    per_core = tuple(psutil.cpu_percent(interval=sample_s, percpu=True))
    total = sum(per_core) / len(per_core) if per_core else 0.0
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        one = five = fifteen = None
    return CpuReading(cores=psutil.cpu_count(logical=True) or len(per_core),
                      percent_total=total, percent_per_core=per_core,
                      load_1m=one, load_5m=five, load_15m=fifteen)


def read_memory() -> MemoryReading:
    import psutil

    virtual = psutil.virtual_memory()
    gib = 1024 ** 3
    return MemoryReading(total_gb=virtual.total / gib,
                         used_gb=virtual.used / gib,
                         available_gb=virtual.available / gib,
                         percent=virtual.percent)


def read_gpus(timeout_s: float = 5.0) -> tuple[tuple[GpuReading, ...], str]:
    """Every visible GPU, plus a reason when there are none.

    Returns the reason rather than raising: a monitor that crashes the run it
    was watching is worse than a monitor that says it cannot see the GPU.
    """
    if shutil.which("nvidia-smi") is None:
        return (), "unavailable: nvidia-smi is not on PATH"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_GPU_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as error:
        return (), f"unavailable: nvidia-smi failed ({error})"
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        return (), (f"unavailable: nvidia-smi exited {completed.returncode}"
                    + (f" ({detail[0]})" if detail else ""))

    readings = []
    for line in completed.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_GPU_FIELDS):
            continue
        readings.append(GpuReading(
            name=parts[0], utilization_percent=_number(parts[1]),
            memory_used_mib=_number(parts[2]),
            memory_total_mib=_number(parts[3]),
            temperature_c=_number(parts[4]), power_w=_number(parts[5]),
            power_limit_w=_number(parts[6])))
    if not readings:
        return (), "unavailable: nvidia-smi reported no GPUs"
    return tuple(readings), ""


def snapshot(sample_s: float = DEFAULT_SAMPLE_S) -> ResourceSnapshot:
    """One reading of everything, for a panel or a stored record."""
    gpus, reason = read_gpus()
    return ResourceSnapshot(timestamp=time.time(), cpu=read_cpu(sample_s),
                            memory=read_memory(), gpus=gpus,
                            gpu_unavailable_reason=reason)


def _bar(fraction: float | None, width: int = 18) -> str:
    """A meter. An unreadable value draws nothing rather than an empty bar.

    An empty bar and a missing reading look identical at a glance, which is
    the confusion this whole module exists to avoid.
    """
    if fraction is None:
        return "unavailable".ljust(width)
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "#" * filled + "." * (width - filled)


def render_panel(reading: ResourceSnapshot | None = None):
    """A rich renderable for one snapshot."""
    from rich.panel import Panel
    from rich.table import Table

    reading = reading or snapshot()
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()

    cpu = reading.cpu
    table.add_row(f"cpu ({cpu.cores} threads)",
                  f"{_bar(cpu.percent_total / 100.0)} {cpu.percent_total:5.1f} %")
    table.add_row("busiest core", f"{cpu.busiest_core_percent:5.1f} %")
    if cpu.load_1m is None:
        table.add_row("load average", "unavailable")
    else:
        table.add_row("load average",
                      f"{cpu.load_1m:.2f} / {cpu.load_5m:.2f} / {cpu.load_15m:.2f}")

    memory = reading.memory
    table.add_row("memory",
                  f"{_bar(memory.percent / 100.0)} "
                  f"{memory.used_gb:5.2f} / {memory.total_gb:5.2f} GiB")
    table.add_row("available", f"{memory.available_gb:5.2f} GiB")

    if not reading.gpus:
        table.add_row("gpu", reading.gpu_unavailable_reason or "unavailable")
    for gpu in reading.gpus:
        table.add_row("", "")
        table.add_row("gpu", gpu.name)
        share = None if gpu.utilization_percent is None \
            else gpu.utilization_percent / 100.0
        table.add_row("utilisation",
                      f"{_bar(share)} " + ("unavailable"
                                           if gpu.utilization_percent is None
                                           else f"{gpu.utilization_percent:5.1f} %"))
        if gpu.memory_total_mib is None or gpu.memory_used_mib is None:
            table.add_row("vram", "unavailable")
        else:
            fraction = gpu.memory_used_mib / gpu.memory_total_mib
            table.add_row("vram",
                          f"{_bar(fraction)} {gpu.memory_used_mib:6.0f} / "
                          f"{gpu.memory_total_mib:6.0f} MiB")
        table.add_row("temperature",
                      "unavailable" if gpu.temperature_c is None
                      else f"{gpu.temperature_c:.0f} C")
        # Draw and limit are separate readings and either can be missing. A
        # driver that reports draw but not limit is normal on a laptop GPU.
        if gpu.power_w is None:
            table.add_row("power", "unavailable")
        elif gpu.power_limit_w is None:
            table.add_row("power", f"{gpu.power_w:.2f} W (limit unavailable)")
        else:
            table.add_row("power",
                          f"{gpu.power_w:.2f} / {gpu.power_limit_w:.0f} W")

    return Panel(table, title="resources", border_style="cyan")


def live(refresh_hz: int = 2, duration_s: float | None = None) -> None:
    """Redraw the panel until interrupted, or until `duration_s` elapses."""
    from rich.console import Console
    from rich.live import Live

    if refresh_hz <= 0:
        raise ValueError("refresh_hz must be positive")
    period = 1.0 / refresh_hz
    started = time.time()
    console = Console()
    try:
        with Live(render_panel(), console=console, refresh_per_second=refresh_hz,
                  screen=False) as display:
            while duration_s is None or time.time() - started < duration_s:
                # The CPU sample IS the wait, so the loop does not spin.
                display.update(render_panel(snapshot(sample_s=period)))
    except KeyboardInterrupt:
        pass
