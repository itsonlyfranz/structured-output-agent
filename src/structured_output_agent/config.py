from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-5-5"


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines; never overrides variables already set."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str = DEFAULT_MODEL
    max_retries: int = 3
    failure_log: Path = Path("logs/validation_failures.jsonl")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            model=os.environ.get("SOA_MODEL") or DEFAULT_MODEL,
            max_retries=int(os.environ.get("SOA_MAX_RETRIES") or 3),
            failure_log=Path(os.environ.get("SOA_FAILURE_LOG") or "logs/validation_failures.jsonl"),
        )
