import json

import pytest
from pydantic import BaseModel, Field

from structured_output_agent import AgentError, RetriesExhausted, StructuredRunner, Tool, ToolAgent, ToolRegistry
from structured_output_agent import ValidationFailureLog
from structured_output_agent.testing import FakeClient


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int = Field(ge=0)


class Final(BaseModel):
    answer: int


def call(**args):
    return json.dumps({"action": "call_tool", "tool": "double", "arguments": args})


FINAL = json.dumps({"action": "final", "output": {"answer": 4}})


def make(replies, fn, **kw):
    log = ValidationFailureLog()
    tools = ToolRegistry([Tool("double", "doubles x", In, Out, fn)])
    return ToolAgent(StructuredRunner(FakeClient(replies), log), tools, **kw), log


def test_tool_call_then_final():
    agent, log = make([call(x=2), FINAL], lambda x: {"y": x * 2})
    res = agent.run("double 2", Final)
    assert res.output == Final(answer=4) and res.tool_calls == ["double"] and res.steps == 2
    assert log.records == []


def test_bad_tool_args_trigger_retry_not_tool_execution():
    ran = []
    agent, log = make([call(x="abc"), call(x=2), FINAL], lambda x: ran.append(x) or {"y": x * 2})
    res = agent.run("go", Final)
    assert ran == [2] and res.output.answer == 4
    assert [r.kind for r in log.records] == ["tool_args_error"]


def test_unknown_tool_is_rejected_and_retried():
    bad = json.dumps({"action": "call_tool", "tool": "nope", "arguments": {}})
    agent, log = make([bad, FINAL], lambda x: {"y": x})
    assert agent.run("go", Final).output.answer == 4
    assert log.records[0].kind == "tool_args_error"


def test_invalid_tool_result_is_logged_and_reported_to_model():
    agent, log = make([call(x=1), call(x=2), FINAL], lambda x: {"y": -1 if x == 1 else 4})
    client = agent.runner.client
    agent.run("go", Final)
    assert [r.kind for r in log.records] == ["tool_result_error"]
    assert "Tool failed" in client.calls[1][1][-1]["content"]


def test_tool_exception_does_not_crash_agent():
    def boom(x):
        raise ValueError("kaput")

    agent, log = make([call(x=1), FINAL], boom)
    agent.run("go", Final)
    assert log.records[0].kind == "tool_error" and "kaput" in log.records[0].message


def test_final_output_validated_against_schema():
    wrong = json.dumps({"action": "final", "output": {"answer": "four"}})
    agent, log = make([wrong, FINAL], lambda x: {"y": x})
    assert agent.run("go", Final).output.answer == 4
    assert log.records[0].kind == "schema_error"


def test_step_budget_enforced():
    agent, _ = make([call(x=1)] * 2, lambda x: {"y": x}, max_steps=2)
    with pytest.raises(AgentError):
        agent.run("go", Final)


def test_persistent_garbage_raises_retries_exhausted():
    agent, _ = make(["junk"] * 4, lambda x: {"y": x})
    with pytest.raises(RetriesExhausted):
        agent.run("go", Final)
