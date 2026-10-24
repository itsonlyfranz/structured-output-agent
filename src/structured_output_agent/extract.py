from __future__ import annotations

import json
import re
from typing import Any

from .errors import StructuredOutputError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse the JSON value in a model reply, tolerating code fences and surrounding prose."""
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    first_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            first_error = first_error or exc
    assert first_error is not None
    start = text.find("{")
    if start != -1:
        try:
            return json.JSONDecoder().raw_decode(text[start:])[0]
        except json.JSONDecodeError:
            pass
    raise StructuredOutputError(
        "parse_error",
        "Your reply was not valid JSON.",
        [f"{first_error.msg} at line {first_error.lineno} column {first_error.colno}"],
    )
