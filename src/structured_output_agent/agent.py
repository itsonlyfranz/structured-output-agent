from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .errors import AgentError, RetriesExhausted, StructuredOutputError
from .extract import extract_json
from .graph import ActResult, build_graph, run_graph
from .llm import Message
from .runner import StructuredRunner, errors_from_state, format_validation_error
from .tools import ToolRegistry

M = TypeVar("M", bound=BaseModel)


class CallTool(BaseModel):
    action: Literal["call_tool"]
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    action: Literal["final"]
    output: dict[str, Any]


Step = Annotated[CallTool | FinalAnswer, Field(discriminator="action")]
_step_adapter: TypeAdapter[CallTool | FinalAnswer] = TypeAdapter(Step)


@dataclass
class AgentResult(Generic[M]):
    output: M
    steps: int
    run_id: str
    tool_calls: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)  # LangGraph nodes visited, in order


class ToolAgent:
    """LangGraph agent: generate -> validate -> (act -> generate)* -> final. Every model step and tool result is validated."""

    def __init__(self, runner: StructuredRunner, tools: ToolRegistry, max_steps: int = 8):
        self.runner = runner
        self.tools = tools
        self.max_steps = max_steps

    def _system(self, output_model: type[BaseModel]) -> str:
        return (
            "You are an agent that answers by emitting exactly one JSON object per turn, nothing else.\n"
            'To use a tool: {"action": "call_tool", "tool": "<name>", "arguments": {...}}\n'
            'To finish: {"action": "final", "output": {...}} where output matches the final schema.\n'
            f"Tools:\n{self.tools.describe()}\n"
            f"Final output JSON Schema: {json.dumps(output_model.model_json_schema())}"
        )

    def _validator(self, output_model: type[M]):
        def validate(text: str) -> tuple[bool, Any, str]:
            data = extract_json(text)
            try:
                step = _step_adapter.validate_python(data)
            except ValidationError as exc:
                raise StructuredOutputError(
                    "schema_error", "Your JSON was not a valid agent step.", format_validation_error(exc)
                ) from exc
            if isinstance(step, FinalAnswer):
                try:
                    return True, output_model.model_validate(step.output), text
                except ValidationError as exc:
                    raise StructuredOutputError(
                        "schema_error",
                        "Your final output did not match the required schema.",
                        format_validation_error(exc),
                    ) from exc
            args = self.tools.get(step.tool).parse_args(step.arguments)
            return False, (step.tool, args), step.model_dump_json()

        return validate

    def _act(self, value: tuple[str, BaseModel]) -> ActResult:
        name, args = value
        try:
            return ActResult(name, f"Tool result: {self.tools.get(name).call(args).model_dump_json()}")
        except StructuredOutputError as err:
            return ActResult(name, f"Tool failed: {err.feedback()}\nFix your arguments or try another approach.", err)

    def run(self, task: str, output_model: type[M]) -> AgentResult[M]:
        run_id = uuid.uuid4().hex[:8]
        graph = build_graph(
            client=self.runner.client,
            system=self._system(output_model),
            schema="AgentStep",
            run_id=run_id,
            validate=self._validator(output_model),
            failure_log=self.runner.failure_log,
            max_retries=self.runner.max_retries,
            act=self._act,
            max_steps=self.max_steps,
            checkpointer=self.runner.checkpointer,
        )
        # Generous cap: LangGraph's recursion limit is a backstop; the explicit budgets above fire first.
        limit = 3 * (self.max_steps + 1) * (self.runner.max_retries + 2)
        state, trace = run_graph(graph, [Message(role="user", content=task)], run_id, limit)
        if state["status"] == "retries_exhausted":
            raise RetriesExhausted("AgentStep", errors_from_state(state["errors"]))
        if state["status"] != "done":
            raise AgentError(f"agent did not finish within {self.max_steps} steps")
        return AgentResult(state["value"], state["steps"] + 1, run_id, state["tool_calls"], trace)
