from types import SimpleNamespace

import pytest
from pydantic import BaseModel

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
    runner = StructuredRunner(FakeClient(["x"] * 2), log, max_retries=1)
    with pytest.raises(RetriesExhausted):
        runner.run("q", P)
    thread = log.records[0].run_id
    values = runner.checkpointer.get_tuple({"configurable": {"thread_id": thread}}).checkpoint["channel_values"]
    assert values["status"] == "retries_exhausted" and len(values["errors"]) == 2


def test_run_state_is_checkpointed_per_run():
    runner = StructuredRunner(FakeClient(["bad", '{"n": 1}']))
    res = runner.run("q", P)
    tup = runner.checkpointer.get_tuple({"configurable": {"thread_id": res.run_id}})
    values = tup.checkpoint["channel_values"]
    assert values["status"] == "done" and values["attempt"] == 2 and values["value"] == P(n=1)


def test_anthropic_client_flattens_langchain_content_blocks():
    seen = []

    class Chat:
        def invoke(self, msgs):
            seen.append(msgs)
            return SimpleNamespace(content=[{"type": "text", "text": '{"n"'}, {"type": "text", "text": ": 1}"}])

    out = AnthropicClient(None, "m", chat_model=Chat()).complete(system="S", messages=[{"role": "user", "content": "hi"}])
    assert out == '{"n": 1}' and seen[0] == [("system", "S"), ("user", "hi")]
