from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import RetriesExhausted, StructuredOutputError
from .extract import extract_json
from .failures import ValidationFailureLog
from .llm import LLMClient, Message

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


@dataclass
class StructuredResult(Generic[T]):
    value: T
    attempts: int
    run_id: str


def format_validation_error(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()]


def parse_model(text: str, model: type[M]) -> M:
    """Parse a model reply into `model`, raising StructuredOutputError on any failure."""
    data = extract_json(text)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise StructuredOutputError(
            "schema_error", "Your JSON did not match the required schema.", format_validation_error(exc)
        ) from exc


def schema_prompt(model: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object that conforms to this JSON Schema. "
        "Output only the JSON, no prose.\n" + json.dumps(model.model_json_schema())
    )


class StructuredRunner:
    """Asks the LLM for output validated against a Pydantic model, retrying with error feedback."""

    def __init__(self, client: LLMClient, failure_log: ValidationFailureLog | None = None, max_retries: int = 3):
        self.client = client
        self.failure_log = failure_log or ValidationFailureLog()
        self.max_retries = max_retries

    def run(self, prompt: str, model: type[M], *, system: str = "") -> StructuredResult[M]:
        full_system = "\n\n".join(s for s in (system, schema_prompt(model)) if s)
        return self.ask(
            [Message(role="user", content=prompt)],
            lambda text: parse_model(text, model),
            system=full_system,
            schema=model.__name__,
        )

    def ask(
        self,
        messages: list[Message],
        validate: Callable[[str], T],
        *,
        system: str,
        schema: str,
        run_id: str | None = None,
    ) -> StructuredResult[T]:
        """Core retry loop. `validate` raises StructuredOutputError to trigger a corrective retry."""
        run_id = run_id or uuid.uuid4().hex[:8]
        convo = list(messages)
        errors: list[StructuredOutputError] = []
        for attempt in range(1, self.max_retries + 2):
            raw = self.client.complete(system=system, messages=convo)
            try:
                return StructuredResult(validate(raw), attempt, run_id)
            except StructuredOutputError as err:
                errors.append(err)
                self.failure_log.record(run_id=run_id, schema=schema, attempt=attempt, error=err, raw_output=raw)
                convo += [
                    Message(role="assistant", content=raw),
                    Message(
                        role="user",
                        content=f"That reply was rejected:\n{err.feedback()}\nReply again with corrected JSON only.",
                    ),
                ]
        raise RetriesExhausted(schema, errors)
