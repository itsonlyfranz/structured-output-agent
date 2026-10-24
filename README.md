# Structured Output Agent

**Making LLMs reliable, not random.** An agent harness that forces model output into Pydantic schemas, validates every tool call and tool result, retries with targeted error feedback when the model gets it wrong, and keeps an audit log of every validation failure.

**Technologies:** Python 3.11+, Pydantic v2, LangGraph, langchain-anthropic (Claude `claude-sonnet-5-5`), pytest

## Highlights

- The agent flow is a LangGraph `StateGraph` with typed state, explicit nodes (`generate`, `validate`, `act`, `fail`) and conditional edges for retry, tool execution and budget exhaustion; each run is checkpointed with LangGraph's `MemorySaver`.
- Every model reply is parsed and validated against a Pydantic schema; invalid output is never returned, only retried or raised as `RetriesExhausted`.
- Corrective retries feed back the rejected reply and exact field errors: in the offline demo, a non-JSON reply and a schema-violating reply (2 field errors) are fixed on attempt 3.
- Tool arguments are validated before execution and tool results after it, so bad arguments never reach a tool and bad data never reaches the model unflagged.
- Five failure kinds (`parse_error`, `schema_error`, `tool_args_error`, `tool_result_error`, `tool_error`) are logged to JSONL; the demo records 4 of them.
- The LLM sits behind a one-method interface, so all 26 tests run offline in about a second with a scripted fake client.

## What it demonstrates

| Capability | Where | Proved by |
|---|---|---|
| Enforce Pydantic JSON schemas | `runner.py` (`StructuredRunner.run`) sends the model's JSON Schema and validates every reply with `model_validate` | `tests/test_runner.py::test_schema_is_sent_to_the_model`, `test_retries_on_schema_error_and_feedback_names_field` |
| Validate tool responses | `tools.py` validates tool **arguments** before execution and tool **results** after; tool exceptions are contained | `tests/test_agent.py::test_bad_tool_args_trigger_retry_not_tool_execution`, `test_invalid_tool_result_is_logged_and_reported_to_model`, `test_tool_exception_does_not_crash_agent` |
| Retry on parse errors | the `validate` → `generate` retry edge in `graph.py` re-prompts with the rejected reply plus precise errors, up to `max_retries` | `test_retries_on_parse_error_with_feedback`, `test_exhausted_retries_raise_and_log_every_attempt`, `tests/test_graph.py::test_trace_shows_conditional_retry_edges` |
| Log validation failures | `failures.py`: in-memory records + append-only JSONL (kind, schema, attempt, errors, raw output, run id) | `test_failures_logged_to_jsonl`, `tests/test_config_cli.py::test_demo_runs_offline_and_writes_log` |

## Architecture

```
 task ──▶ START ─▶ generate ─▶ validate ──valid, final──────────▶ END
                      ▲            │  └──valid, tool call──▶ act ─┐
                      │            │                              │ budget left
                      └──invalid, retries left                    │
                      └─────────────────────────────◀─────────────┘
                                   ├──invalid, retries spent──▶ fail ─▶ END
                                   ▼ every failure
                        ValidationFailureLog → logs/validation_failures.jsonl
```

The graph (`graph.py`) is LangGraph `StateGraph` over a typed `State` (messages, raw reply, attempt/step counters, plain-dict errors so checkpoints serialise). `generate` calls the LLM, `validate` parses and validates, `act` runs a validated tool call, `fail` records why the run ended. Conditional edges choose retry / act / finish / fail. LangGraph's recursion limit is set as a backstop behind the explicit retry and step budgets.

Components and responsibilities:

- `llm.py`: tiny `LLMClient` protocol (`complete(system, messages) -> str`) plus `AnthropicClient` (LangChain `ChatAnthropic`). This is the only provider seam.
- `extract.py`: tolerant JSON extraction (code fences, surrounding prose).
- `graph.py`: the LangGraph `StateGraph` (nodes, conditional edges, `MemorySaver` checkpointing per run id) shared by both entry points.
- `runner.py`: `StructuredRunner`, the no-tools graph; failures become corrective user messages naming the offending fields.
- `agent.py`: `ToolAgent`, the graph with an `act` node, where each turn is a discriminated-union step (`call_tool` | `final`). Bad arguments and a malformed final answer are retried; bad tool results/exceptions are logged and fed back as observations. Step budget prevents runaway loops.
- `tools.py`: `Tool` / `ToolRegistry` with Pydantic input and output models; validates arguments, executes, validates results.
- `failures.py`: `ValidationFailureLog`, the in-memory and JSONL audit trail.
- `config.py` / `cli.py` / `demo.py`: environment settings, the `soa` command, and the offline demo.
- `testing.py`: `FakeClient` replays scripted replies, so the whole suite and the demo are offline.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate   # or: uv venv && uv pip install -e ".[dev]"
pip install -e ".[dev]"
pytest                                          # offline, no API key needed
```

### Demo (offline)

```bash
soa demo
```

Replays a scripted "model" that emits non-JSON, schema-violating JSON, bad tool arguments and a tool that returns invalid data. You see each one rejected, the corrected retry, the final validated result, and the failure log (also written to `logs/validation_failures.jsonl`).

### Live with Claude

```bash
cp .env.example .env     # then set ANTHROPIC_API_KEY (the .env file is gitignored)
soa extract "Invoice from Acme Corp for 120.50 USD"
```

Config via environment: `SOA_MODEL` (default `claude-sonnet-5-5`), `SOA_MAX_RETRIES` (default 3), `SOA_FAILURE_LOG`.

Library use:

```python
from structured_output_agent import StructuredRunner, ValidationFailureLog
from structured_output_agent.llm import AnthropicClient

runner = StructuredRunner(AnthropicClient(api_key, "claude-sonnet-5-5"), ValidationFailureLog("fail.jsonl"))
result = runner.run("Invoice from Acme, 120.50 USD", Invoice)   # result.value is a validated Invoice
```

## Design decisions and trade-offs

- **Prompt-level schema + validate + retry, not provider-specific constrained decoding.** It works with any text-in/text-out model and makes the reliability layer explicit and testable. Trade-off: a retry costs extra tokens/latency; native structured output or forced tool-use could cut first-try failures and could be added behind `LLMClient`.
- **Feedback-driven retries.** The rejected reply and exact field errors go back to the model, which fixes far more than blind resampling. Retries are bounded; exhaustion raises `RetriesExhausted` carrying every error rather than returning unvalidated data.
- **Validate both sides of every tool.** Models hallucinate arguments, and tools return surprising data; both are schema-checked. Invalid arguments never reach the tool.
- **Failures are data.** JSONL records include the raw output (truncated to 2 KB) so failure rates by schema/kind can be analysed with standard tools. Raw outputs may contain user content, so the log path is gitignored.
- **LangGraph where it fits.** The retry/act/fail flow is a real state machine, so a `StateGraph` with conditional edges makes it explicit, inspectable (`result.trace`) and checkpointed (`tests/test_graph.py`). Human-in-the-loop interrupts and parallel branches were not used: nothing in this flow needs a human or fan-out, and adding them would be decoration.
- **Local only.** Checkpoints are in-process (`MemorySaver`); no database or hosted service. A minimal built-in `.env` loader avoids another dependency.
- **Not covered:** streaming, parallel tool calls, async, and semantic correctness beyond what the schema can express.

## Tests

`pytest` runs 26 offline tests against `FakeClient`; the table above maps each capability to its tests.
