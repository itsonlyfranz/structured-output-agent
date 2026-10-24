import json

import pytest
from pydantic import BaseModel, Field

from structured_output_agent import RetriesExhausted, StructuredRunner, ValidationFailureLog
from structured_output_agent.testing import FakeClient


class Person(BaseModel):
    name: str
    age: int = Field(ge=0)


GOOD = '{"name": "Ada", "age": 36}'


def test_valid_first_try_makes_one_call():
    client = FakeClient([GOOD])
    res = StructuredRunner(client).run("who?", Person)
    assert res.value == Person(name="Ada", age=36) and res.attempts == 1
    assert len(client.calls) == 1


def test_schema_is_sent_to_the_model():
    client = FakeClient([GOOD])
    StructuredRunner(client).run("who?", Person)
    assert '"age"' in client.calls[0][0]


def test_retries_on_parse_error_with_feedback():
    client = FakeClient(["{oops", GOOD])
    res = StructuredRunner(client).run("who?", Person)
    assert res.attempts == 2
    retry_messages = client.calls[1][1]
    assert retry_messages[-2] == {"role": "assistant", "content": "{oops"}
    assert "not valid JSON" in retry_messages[-1]["content"]


def test_retries_on_schema_error_and_feedback_names_field():
    client = FakeClient(['{"name": "Ada", "age": -1}', GOOD])
    res = StructuredRunner(client).run("who?", Person)
    assert res.attempts == 2
    assert "age" in client.calls[1][1][-1]["content"]


def test_exhausted_retries_raise_and_log_every_attempt():
    log = ValidationFailureLog()
    client = FakeClient(["nope"] * 3)
    with pytest.raises(RetriesExhausted) as e:
        StructuredRunner(client, log, max_retries=2).run("who?", Person)
    assert len(e.value.errors) == 3 and len(client.calls) == 3
    assert [r.attempt for r in log.records] == [1, 2, 3]


def test_failures_logged_to_jsonl(tmp_path):
    path = tmp_path / "sub" / "fail.jsonl"
    log = ValidationFailureLog(path)
    StructuredRunner(FakeClient(["{bad", '{"name": "A", "age": "x"}', GOOD]), log).run("q", Person)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["parse_error", "schema_error"]
    assert rows[1]["schema"] == "Person" and rows[1]["raw_output"] == '{"name": "A", "age": "x"}'
    assert any("age" in d for d in rows[1]["details"])
    assert len({r["run_id"] for r in rows}) == 1
