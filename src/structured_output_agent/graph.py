"""The agent flow as a LangGraph StateGraph.

    START -> generate -> validate --valid, final--> END
                ^           |  \\--valid, action--> act --budget left--> generate
                |           |                          \\--budget spent--> fail
                +--invalid--+ (retries left)
                            \\--invalid, retries spent--> fail
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .errors import StructuredOutputError
from .failures import ValidationFailureLog
from .llm import LLMClient, Message


@dataclass
class ActResult:
    """Outcome of executing one tool call. `error` is set when args/result/execution was invalid."""

    tool: str
    observation: str
    error: StructuredOutputError | None = None


# validate(raw) -> (is_final, value, echo); raises StructuredOutputError when the reply is unusable.
Validate = Callable[[str], tuple[bool, Any, str]]
Act = Callable[[Any], ActResult]


class State(TypedDict):
    messages: list[Message]
    raw: str
    value: Any
    is_final: bool
    attempt: int  # attempts spent on the current turn
    steps: int  # tool calls executed so far
    errors: list[dict[str, Any]]  # failures on the current turn, as plain dicts (checkpoint-safe)
    tool_calls: list[str]
    status: Literal["running", "done", "retries_exhausted", "step_budget"]


def initial_state(messages: list[Message]) -> State:
    return State(
        messages=messages, raw="", value=None, is_final=False, attempt=0, steps=0,
        errors=[], tool_calls=[], status="running",
    )  # fmt: skip


def build_graph(
    *,
    client: LLMClient,
    system: str,
    schema: str,
    run_id: str,
    validate: Validate,
    failure_log: ValidationFailureLog,
    max_retries: int,
    act: Act | None = None,
    max_steps: int = 1,
    checkpointer: BaseCheckpointSaver | None = None,
):
    def generate(state: State) -> dict[str, Any]:
        return {"raw": client.complete(system=system, messages=state["messages"]), "attempt": state["attempt"] + 1}

    def validate_node(state: State) -> dict[str, Any]:
        raw = state["raw"]
        try:
            is_final, value, echo = validate(raw)
        except StructuredOutputError as err:
            failure_log.record(run_id=run_id, schema=schema, attempt=state["attempt"], error=err, raw_output=raw)
            feedback = f"That reply was rejected:\n{err.feedback()}\nReply again with corrected JSON only."
            return {
                "errors": [*state["errors"], {"kind": err.kind, "message": err.message, "details": err.details}],
                "messages": [*state["messages"], Message(role="assistant", content=raw), Message(role="user", content=feedback)],
                "value": None,
            }  # fmt: skip
        update: dict[str, Any] = {"value": value, "is_final": is_final, "status": "done" if is_final else "running"}
        if not is_final:
            update["messages"] = [*state["messages"], Message(role="assistant", content=echo)]
        return update

    def act_node(state: State) -> dict[str, Any]:
        assert act is not None
        res = act(state["value"])
        if res.error:
            failure_log.record(
                run_id=run_id, schema=f"tool:{res.tool}", attempt=state["steps"] + 1, error=res.error, raw_output=state["raw"]
            )  # fmt: skip
        return {
            "messages": [*state["messages"], Message(role="user", content=res.observation)],
            "steps": state["steps"] + 1,
            "tool_calls": [*state["tool_calls"], res.tool],
            "attempt": 0,
            "errors": [],
        }

    def fail(state: State) -> dict[str, Any]:
        return {"status": "retries_exhausted" if state["errors"] else "step_budget"}

    def after_validate(state: State) -> str:
        if state["value"] is None:  # validation failed
            return "generate" if state["attempt"] <= max_retries else "fail"
        if state["is_final"]:
            return END
        return "act"

    def after_act(state: State) -> str:
        return "generate" if state["steps"] < max_steps else "fail"

    g = StateGraph(State)
    g.add_node("generate", generate)
    g.add_node("validate", validate_node)
    g.add_node("fail", fail)
    g.add_edge(START, "generate")
    g.add_edge("generate", "validate")
    g.add_edge("fail", END)
    if act is not None:
        g.add_node("act", act_node)
        g.add_conditional_edges("validate", after_validate, ["generate", "fail", "act", END])
        g.add_conditional_edges("act", after_act, ["generate", "fail"])
    else:
        g.add_conditional_edges("validate", after_validate, ["generate", "fail", END])
    return g.compile(checkpointer=checkpointer)


def run_graph(graph, messages: list[Message], run_id: str, recursion_limit: int) -> tuple[State, list[str]]:
    """Execute the graph, returning the final state (read back from the checkpointer) and the node trace."""
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": recursion_limit}
    trace: list[str] = []
    for update in graph.stream(initial_state(messages), config, stream_mode="updates"):
        trace.extend(update)
    return graph.get_state(config).values, trace
