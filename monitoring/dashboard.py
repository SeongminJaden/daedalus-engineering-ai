"""monitoring.dashboard - live terminal dashboard for a running design loop.

CLI/TUI only, by design: no GUI anywhere in this project. Built on rich.Live
so it renders over SSH and inside tmux without a display server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunState:
    """Whatever the dashboard needs to draw one frame."""

    profile: str = "unknown"
    generation: int = 0
    generations_total: int = 0
    best_objective: float | None = None
    evaluated: int = 0
    # None, not 0.0. An unread GPU and an idle GPU are different states, and
    # a default of zero makes them indistinguishable on the panel.
    gpu_mem_used_gb: float | None = None
    gpu_mem_total_gb: float | None = None
    status: str = "idle"
    extra: dict[str, Any] = field(default_factory=dict)


class Dashboard:
    """Live TUI over a RunState. Stub: layout is real, wiring is not yet."""

    def __init__(self, state: RunState | None = None, refresh_hz: int = 4):
        self.state = state or RunState()
        self.refresh_hz = refresh_hz
        self._live = None

    def _render(self):
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", justify="right")
        table.add_column()

        s = self.state
        table.add_row("profile", s.profile)
        table.add_row("status", s.status)
        table.add_row("generation", f"{s.generation}/{s.generations_total}")
        table.add_row("evaluated", str(s.evaluated))
        table.add_row(
            "best objective",
            "-" if s.best_objective is None else f"{s.best_objective:.6g}",
        )
        # Silently dropping the row would make an unread GPU look like a
        # machine that has none, so say which it is.
        if s.gpu_mem_total_gb is None or s.gpu_mem_used_gb is None:
            table.add_row("gpu memory", "unavailable")
        else:
            table.add_row(
                "gpu memory",
                f"{s.gpu_mem_used_gb:.2f} / {s.gpu_mem_total_gb:.2f} GB",
            )
        for key, value in s.extra.items():
            table.add_row(key, str(value))

        return Panel(table, title="Daedalus", border_style="cyan")

    def __enter__(self):
        from rich.live import Live

        self._live = Live(self._render(), refresh_per_second=self.refresh_hz)
        self._live.__enter__()
        return self

    def update(self, **fields) -> None:
        for key, value in fields.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
            else:
                self.state.extra[key] = value
        if self._live is not None:
            self._live.update(self._render())

    def __exit__(self, *exc):
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None
        return False
