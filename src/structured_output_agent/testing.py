"""Offline LLMClient doubles used by the test suite and the demo."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .llm import Message


class FakeClient:
    """Replays scripted replies in order and records every call it receives."""

    def __init__(self, replies: Sequence[str | Callable[[Sequence[Message]], str]]):
        self._replies = list(replies)
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, *, system: str, messages: Sequence[Message]) -> str:
        self.calls.append((system, [dict(m) for m in messages]))  # type: ignore[misc]
        if len(self.calls) > len(self._replies):
            raise AssertionError("FakeClient ran out of scripted replies")
        reply = self._replies[len(self.calls) - 1]
        return reply(messages) if callable(reply) else reply
