from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from structured_output_agent import RetriesExhausted, StructuredRunner, ValidationFailureLog
from structured_output_agent.llm import AnthropicClient
from structured_output_agent.testing import FakeClient


class P(BaseModel):
    n: int


def test_trace_shows_conditional_retry_edges():
    res = StructuredRunner(FakeClient(["bad", '{"n": 1}'])).run("q", P)
    assert res.trace == ["generate", "validate", "generate", "validate"]


def test_clean_run_has_no_retry_edge():
    assert StructuredRunner(FakeClient(['{"n": 1}'])).run("q", P).trace == ["generate", "validate"]


def test_exhaustion_ends_in_fail_node_with_checkpointed_status():
    log = ValidationFailureLog()
    runner = StructuredRunner(FakeClient(["x"] * 2), log, max_retries=1, checkpointer=MemorySaver())
    with pytest.raises(RetriesExhausted):
        runner.run("q", P)
    thread = log.records[0].run_id
    values = runner.checkpointer.get_tuple({"configurable": {"thread_id": thread}}).checkpoint["channel_values"]
    assert values["status"] == "retries_exhausted" and len(values["errors"]) == 2


def test_run_state_is_checkpointed_per_run_when_opted_in():
    runner = StructuredRunner(FakeClient(["bad", '{"n": 1}']), checkpointer=MemorySaver())
    res = runner.run("q", P)
    tup = runner.checkpointer.get_tuple({"configurable": {"thread_id": res.run_id}})
    values = tup.checkpoint["channel_values"]
    assert values["status"] == "done" and values["attempt"] == 2 and values["value"] == P(n=1)


def test_runner_retains_no_checkpoints_by_default():
    runner = StructuredRunner(FakeClient(['{"n": 1}']))
    res = runner.run("q", P)
    assert runner.checkpointer is None and res.value == P(n=1) and res.attempts == 1


def test_checkpointed_run_returns_in_process_validated_value():
    class Local(BaseModel):
        total: int = Field(alias="Total")

    res = StructuredRunner(FakeClient(['{"Total": 3}']), checkpointer=MemorySaver()).run("q", Local)
    assert type(res.value) is Local and res.value.total == 3


def test_anthropic_client_flattens_langchain_content_blocks():
    seen = []

    class Chat:
        def invoke(self, msgs):
            seen.append(msgs)
            return SimpleNamespace(content=[{"type": "text", "text": '{"n"'}, {"type": "text", "text": ": 1}"}])

    out = AnthropicClient(None, "m", chat_model=Chat()).complete(system="S", messages=[{"role": "user", "content": "hi"}])
    assert out == '{"n": 1}' and seen[0] == [("system", "S"), ("user", "hi")]
