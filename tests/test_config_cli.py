import pytest

from structured_output_agent.cli import main
from structured_output_agent.config import DEFAULT_MODEL, Settings, load_dotenv
from structured_output_agent.llm import AnthropicClient


def test_settings_defaults_and_overrides(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "SOA_MODEL", "SOA_MAX_RETRIES"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.model == DEFAULT_MODEL == "claude-sonnet-5-5" and s.api_key is None and s.max_retries == 3
    monkeypatch.setenv("SOA_MODEL", "other")
    monkeypatch.setenv("SOA_MAX_RETRIES", "5")
    assert Settings.from_env().model == "other" and Settings.from_env().max_retries == 5


def test_dotenv_loaded_but_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nFOO_SOA=1\nBAR_SOA='two'\n")
    monkeypatch.setenv("BAR_SOA", "keep")
    monkeypatch.delenv("FOO_SOA", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["FOO_SOA"] == "1" and os.environ["BAR_SOA"] == "keep"
    monkeypatch.delenv("FOO_SOA")


def test_anthropic_client_requires_key():
    with pytest.raises(RuntimeError):
        AnthropicClient(None, "m")


def test_demo_runs_offline_and_writes_log(tmp_path, capsys):
    log = tmp_path / "f.jsonl"
    assert main(["demo", "--log", str(log)]) == 0
    out = capsys.readouterr().out
    assert "valid after 3 attempts" in out and "final output" in out
    kinds = [__import__("json").loads(l)["kind"] for l in log.read_text().splitlines()]
    assert kinds == ["parse_error", "schema_error", "tool_args_error", "tool_result_error"]
