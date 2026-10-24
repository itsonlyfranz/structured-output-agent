from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel

from .config import Settings, load_dotenv
from .demo import Invoice, run_demo
from .failures import ValidationFailureLog
from .llm import AnthropicClient
from .runner import StructuredRunner


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(prog="soa", description="Structured Output Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="offline demo with a scripted fake LLM (no API key needed)")
    demo.add_argument("--log", type=Path, default=settings.failure_log, help="JSONL file for validation failures")
    ext = sub.add_parser("extract", help="live: extract an Invoice from text using Claude")
    ext.add_argument("text")
    args = parser.parse_args(argv)

    if args.cmd == "demo":
        run_demo(args.log)
        print(f"\nFailure log written to {args.log}")
        return 0

    client = AnthropicClient(settings.api_key, settings.model)
    runner = StructuredRunner(client, ValidationFailureLog(settings.failure_log), settings.max_retries)
    result = runner.run(args.text, Invoice)
    print(result.value.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
