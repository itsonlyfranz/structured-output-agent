"""The only seam between this project and an LLM provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class LLMClient(Protocol):
    def complete(self, *, system: str, messages: Sequence[Message]) -> str:
        """Return the assistant's text reply for the conversation."""
        ...


class AnthropicClient:
    """LLMClient backed by the official Anthropic SDK."""

    def __init__(self, api_key: str | None, model: str, max_tokens: int = 2048):
        import anthropic

        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (export it or put it in a gitignored .env)")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, *, system: str, messages: Sequence[Message]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=list(messages),
        )
        return "".join(block.text for block in response.content if block.type == "text")
