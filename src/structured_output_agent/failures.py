from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .errors import StructuredOutputError

logger = logging.getLogger("structured_output_agent")
_MAX_RAW = 2000


@dataclass
class FailureRecord:
    run_id: str
    schema: str
    attempt: int
    kind: str
    message: str
    details: list[str]
    raw_output: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ValidationFailureLog:
    """Keeps every validation failure in memory and, if a path is given, appends it to a JSONL file."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: list[FailureRecord] = []

    def record(
        self, *, run_id: str, schema: str, attempt: int, error: StructuredOutputError, raw_output: str
    ) -> FailureRecord:
        rec = FailureRecord(
            run_id=run_id,
            schema=schema,
            attempt=attempt,
            kind=error.kind,
            message=error.message,
            details=error.details,
            raw_output=raw_output[:_MAX_RAW],
        )
        self.records.append(rec)
        logger.warning("validation failure [%s] schema=%s attempt=%d: %s", rec.kind, schema, attempt, error.message)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec)) + "\n")
        return rec
