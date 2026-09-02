"""Records on disk, one JSON object per line, validated on the way in and out.

Every read goes through `validate_record`, so a file edited by hand or
written by an older version is refused at the first bad line rather than
half-loaded. Every write to a public file checks `is_publishable`, so a
proprietary part cannot reach the repository by way of a dataset file, which
is the route it would otherwise take.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import PartRecord, validate_record


class NotPublishable(ValueError):
    """A record that may not be written to a public dataset file."""


def write_jsonl(path: str | Path, records: Iterable[PartRecord],
                public: bool = True) -> int:
    """Write records; returns how many. Refuses the first unpublishable one
    BEFORE writing anything when `public` is true, so a failed write leaves
    no partial file behind."""
    records = list(records)
    if public:
        for record in records:
            if not record.is_publishable:
                raise NotPublishable(
                    f"{record.part_id} is {record.provenance.kind.value} under "
                    f"licence {record.provenance.licence.identifier!r} and may "
                    f"not be written to a public dataset file")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"),
                                    sort_keys=True))
            handle.write("\n")
    return len(records)


def read_jsonl(path: str | Path) -> list[PartRecord]:
    """Read and validate every line. A bad line names itself."""
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(validate_record(json.loads(line)))
            except Exception as exc:      # noqa: BLE001 - re-raised with location
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return records
