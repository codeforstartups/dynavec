"""Chat model abstraction for dynavec."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class Message:
    """A single message in a chat conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class Tool:
    """A tool/function specification provided to the model."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema dict


@dataclass
class ChatResult:
    """The final result of a chat invocation."""

    message: Message
    finish_reason: str | None = None


@dataclass
class ChatChunk:
    """A streamed chunk from a chat invocation."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(ABC):
    """Base class for all chat/LLM backends."""

    @abstractmethod
    def invoke(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> ChatResult:
        """Invoke the chat model synchronously."""

    @abstractmethod
    def stream(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> Iterator[ChatChunk]:
        """Stream the chat model response."""

    async def ainvoke(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> ChatResult:
        """Invoke the chat model asynchronously. Default delegates to thread."""
        return await asyncio.to_thread(self.invoke, messages, tools, **kwargs)

    async def astream(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> AsyncIterator[ChatChunk]:
        """Stream the chat model response asynchronously. Default delegates to thread."""

        def _sync_stream() -> list[ChatChunk]:
            return list(self.stream(messages, tools, **kwargs))

        chunks = await asyncio.to_thread(_sync_stream)
        for chunk in chunks:
            yield chunk
