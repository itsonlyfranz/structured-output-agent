from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .errors import AgentError, StructuredOutputError
from .extract import extract_json
from .llm import LLMClient, Message
from .runner import StructuredRunner, format_validation_error
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


class ToolAgent:
    """Loop: model picks a tool or a final answer as schema-validated JSON; every tool result is validated too."""

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
        def validate(text: str) -> tuple[CallTool, Any] | tuple[FinalAnswer, M]:
            data = extract_json(text)
            try:
                step = _step_adapter.validate_python(data)
            except ValidationError as exc:
                raise StructuredOutputError(
                    "schema_error", "Your JSON was not a valid agent step.", format_validation_error(exc)
                ) from exc
            if isinstance(step, FinalAnswer):
                try:
                    return step, output_model.model_validate(step.output)
                except ValidationError as exc:
                    raise StructuredOutputError(
                        "schema_error",
                        "Your final output did not match the required schema.",
                        format_validation_error(exc),
                    ) from exc
            return step, self.tools.get(step.tool).parse_args(step.arguments)

        return validate

    def run(self, task: str, output_model: type[M]) -> AgentResult[M]:
        run_id = uuid.uuid4().hex[:8]
        convo: list[Message] = [Message(role="user", content=task)]
        validate = self._validator(output_model)
        called: list[str] = []
        for step_no in range(1, self.max_steps + 1):
            res = self.runner.ask(
                convo, validate, system=self._system(output_model), schema="AgentStep", run_id=run_id
            )
            step, parsed = res.value
            convo.append(Message(role="assistant", content=step.model_dump_json()))
            if isinstance(step, FinalAnswer):
                return AgentResult(parsed, step_no, run_id, called)
            tool = self.tools.get(step.tool)
            called.append(step.tool)
            try:
                observation = f"Tool result: {tool.call(parsed).model_dump_json()}"
            except StructuredOutputError as err:
                self.runner.failure_log.record(
                    run_id=run_id, schema=f"tool:{step.tool}", attempt=step_no, error=err, raw_output=step.model_dump_json()
                )
                observation = f"Tool failed: {err.feedback()}\nFix your arguments or try another approach."
            convo.append(Message(role="user", content=observation))
        raise AgentError(f"agent did not finish within {self.max_steps} steps")
