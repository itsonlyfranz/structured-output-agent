"""Offline demo: scripted 'LLM' replies that fail in realistic ways, so the safeguards are visible."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .agent import ToolAgent
from .failures import ValidationFailureLog
from .runner import StructuredRunner
from .testing import FakeClient
from .tools import Tool, ToolRegistry


class Invoice(BaseModel):
    vendor: str
    total: float = Field(gt=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")


class LookupIn(BaseModel):
    sku: str


class LookupOut(BaseModel):
    sku: str
    price: float = Field(gt=0)


class Quote(BaseModel):
    sku: str
    total_price: float


def _lookup(sku: str):
    return {"sku": sku, "price": 19.5} if sku == "A1" else {"sku": sku, "price": -1}


def run_demo(log_path: Path | None = None) -> ValidationFailureLog:
    log = ValidationFailureLog(log_path)

    print("== 1. Structured extraction with retries ==")
    client = FakeClient(
        [
            "Sure! Here is the invoice: {vendor: 'Acme'",  # not JSON
            '{"vendor": "Acme", "total": -5, "currency": "usd"}',  # schema violations
            '```json\n{"vendor": "Acme", "total": 120.5, "currency": "USD"}\n```',  # valid
        ]
    )
    result = StructuredRunner(client, log).run("Invoice from Acme, 120.50 USD", Invoice)
    print(f"valid after {result.attempts} attempts -> {result.value}")

    print("\n== 2. Agent with validated tool calls and results ==")
    tools = ToolRegistry([Tool("lookup_price", "Get unit price for a SKU", LookupIn, LookupOut, _lookup)])
    steps = [
        json.dumps({"action": "call_tool", "tool": "lookup_price", "arguments": {"sku": 7}}),  # bad args
        json.dumps({"action": "call_tool", "tool": "lookup_price", "arguments": {"sku": "B2"}}),  # tool returns bad data
        json.dumps({"action": "call_tool", "tool": "lookup_price", "arguments": {"sku": "A1"}}),
        json.dumps({"action": "final", "output": {"sku": "A1", "total_price": 19.5}}),
    ]
    agent_result = ToolAgent(StructuredRunner(FakeClient(steps), log), tools).run("Quote SKU A1", Quote)
    print(f"final output {agent_result.output} after {agent_result.steps} steps, tools: {agent_result.tool_calls}")

    print(f"\n== Validation failures logged: {len(log.records)} ==")
    for r in log.records:
        print(f"[{r.kind}] {r.schema} attempt {r.attempt}: {r.message} {r.details}")
    return log
