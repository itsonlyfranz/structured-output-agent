from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ValidationError

from .errors import RetriesExhausted, StructuredOutputError
from .extract import extract_json
from .failures import ValidationFailureLog
from .graph import build_graph, run_graph
from .llm import LLMClient, Message

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


@dataclass
class StructuredResult(Generic[T]):
    value: T
    attempts: int
    run_id: str
    trace: list[str] = field(default_factory=list)  # LangGraph nodes visited, in order


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


def errors_from_state(errors: list[dict]) -> list[StructuredOutputError]:
    return [StructuredOutputError(e["kind"], e["message"], e["details"]) for e in errors]


class StructuredRunner:
    """Asks the LLM for output validated against a Pydantic model, retrying with error feedback.

    Each run is a LangGraph StateGraph (generate -> validate -> retry | done). Pass a `checkpointer`
    to persist each run's state under its run id; by default nothing is retained between runs.
    """

    def __init__(
        self,
        client: LLMClient,
        failure_log: ValidationFailureLog | None = None,
        max_retries: int = 3,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.client = client
        self.failure_log = failure_log or ValidationFailureLog()
        self.max_retries = max_retries
        self.checkpointer = checkpointer

    def run(self, prompt: str, model: type[M], *, system: str = "") -> StructuredResult[M]:
        run_id = uuid.uuid4().hex[:8]
        graph = build_graph(
            client=self.client,
            system="\n\n".join(s for s in (system, schema_prompt(model)) if s),
            schema=model.__name__,
            run_id=run_id,
            validate=lambda text: (True, parse_model(text, model), text),
            failure_log=self.failure_log,
            max_retries=self.max_retries,
            checkpointer=self.checkpointer,
        )
        state, trace = run_graph(graph, [Message(role="user", content=prompt)], run_id, 3 * (self.max_retries + 2))
        if state["status"] != "done":
            raise RetriesExhausted(model.__name__, errors_from_state(state["errors"]))
        return StructuredResult(state["value"], state["attempt"], run_id, trace)
