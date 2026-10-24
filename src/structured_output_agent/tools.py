from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .errors import StructuredOutputError
from .runner import format_validation_error


@dataclass
class Tool:
    """A callable with Pydantic-typed input and output; both sides are validated."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: Callable[..., Any]

    def parse_args(self, arguments: dict[str, Any]) -> BaseModel:
        try:
            return self.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise StructuredOutputError(
                "tool_args_error",
                f"Arguments for tool '{self.name}' did not match its input schema.",
                format_validation_error(exc),
            ) from exc

    def call(self, args: BaseModel) -> BaseModel:
        try:
            raw = self.fn(**args.model_dump())
        except Exception as exc:  # tool bugs must not crash the agent loop
            raise StructuredOutputError(
                "tool_error", f"Tool '{self.name}' raised {type(exc).__name__}: {exc}"
            ) from exc
        try:
            return self.output_model.model_validate(raw)
        except ValidationError as exc:
            raise StructuredOutputError(
                "tool_result_error",
                f"Tool '{self.name}' returned data that did not match its output schema.",
                format_validation_error(exc),
            ) from exc


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise StructuredOutputError(
                "tool_args_error", f"Unknown tool '{name}'.", [f"available tools: {', '.join(self._tools)}"]
            )
        return self._tools[name]

    def describe(self) -> str:
        import json

        return "\n".join(
            f"- {t.name}: {t.description}\n  input schema: {json.dumps(t.input_model.model_json_schema())}"
            for t in self._tools.values()
        )
