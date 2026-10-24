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


def _text_of(content: object) -> str:
    """Flatten a LangChain message `content` (str or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    return "".join(
        b if isinstance(b, str) else b.get("text", "") for b in content if isinstance(b, str) or b.get("type") == "text"
    )  # type: ignore[union-attr]


class AnthropicClient:
    """LLMClient backed by Claude through langchain-anthropic's ChatAnthropic.

    `chat_model` may be injected (any LangChain chat model) for tests.
    """

    def __init__(self, api_key: str | None, model: str, max_tokens: int = 2048, chat_model: object | None = None):
        if chat_model is None:
            from langchain_anthropic import ChatAnthropic

            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (export it or put it in a gitignored .env)")
            chat_model = ChatAnthropic(model=model, api_key=api_key, max_tokens=max_tokens)
        self._chat = chat_model
        self.model = model

    def complete(self, *, system: str, messages: Sequence[Message]) -> str:
        reply = self._chat.invoke([("system", system), *((m["role"], m["content"]) for m in messages)])  # type: ignore[attr-defined]
        return _text_of(reply.content)
