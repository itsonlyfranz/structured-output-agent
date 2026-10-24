from __future__ import annotations

from typing import Literal

FailureKind = Literal["parse_error", "schema_error", "tool_args_error", "tool_error", "tool_result_error"]


class StructuredOutputError(Exception):
    """A single model output (or tool result) that failed parsing or validation."""

    def __init__(self, kind: FailureKind, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.details = details or []

    def feedback(self) -> str:
        lines = [self.message, *(f"- {d}" for d in self.details)]
        return "\n".join(lines)


class RetriesExhausted(Exception):
    """The model never produced a valid output within the retry budget."""

    def __init__(self, schema: str, errors: list[StructuredOutputError]):
        super().__init__(f"{schema}: no valid output after {len(errors)} attempts; last error: {errors[-1].message}")
        self.schema = schema
        self.errors = errors


class AgentError(Exception):
    """The tool-using agent could not finish (e.g. step budget exhausted)."""
