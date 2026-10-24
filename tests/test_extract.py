import pytest

from structured_output_agent.errors import StructuredOutputError
from structured_output_agent.extract import extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_code_fence_and_prose():
    assert extract_json('Here you go:\n```json\n{"a": 1}\n```\nthanks') == {"a": 1}
    assert extract_json('Result: {"a": {"b": 2}} done') == {"a": {"b": 2}}


def test_garbage_raises_parse_error():
    with pytest.raises(StructuredOutputError) as e:
        extract_json("not json at all")
    assert e.value.kind == "parse_error"
