"""monitoring.logging_setup - structured (JSONL) logging + rich console.

Two sinks, deliberately separate:
  * a JSONL file, one object per line, for later analysis of a run
  * a rich-formatted console handler for a human watching the terminal
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class JsonlFormatter(logging.Formatter):
    """One JSON object per line. Extra fields ride along via `extra=`."""

    RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "asctime", "message"
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self.RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_dir: str | Path = "logs", level: int = logging.INFO,
                  console: bool = True) -> logging.Logger:
    """Configure the root logger. Returns the project logger."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"run-{stamp}.jsonl")
    file_handler.setFormatter(JsonlFormatter())
    root.addHandler(file_handler)

    if console:
        try:
            from rich.logging import RichHandler

            root.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
        except ImportError:
            root.addHandler(logging.StreamHandler())

    return logging.getLogger("engineering_ai")
