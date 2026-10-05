"""OpenAI chat model (bring your own OPENAI_API_KEY)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..exceptions import MissingDependencyError
from .base import ChatChunk, ChatModel, ChatResult, Message, Tool, ToolCall


class OpenAIChatModel(ChatModel):
    """Chat with OpenAI's models.

    Parameters
    ----------
    model:
        Model id, e.g. ``"gpt-4o"``.
    api_key:
        Optional; falls back to the ``OPENAI_API_KEY`` environment variable.
    base_url:
        Optional custom endpoint URL.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependencyError("OpenAIChatModel", "openai", "openai") from exc

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role}

            if msg.content is not None:
                d["content"] = msg.content

            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in msg.tool_calls
                ]

            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
                d["name"] = "tool"  # some models/APIs require a name

            out.append(d)
        return out

    def _convert_tools(self, tools: list[Tool] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def invoke(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> ChatResult:
        """Invoke the chat model synchronously."""
        openai_msgs = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools)

        args: dict[str, Any] = {"model": self.model, "messages": openai_msgs, **kwargs}
        if openai_tools:
            args["tools"] = openai_tools

        resp = self._client.chat.completions.create(**args)
        choice = resp.choices[0]

        out_msg = Message(role=choice.message.role or "assistant", content=choice.message.content)
        if choice.message.tool_calls:
            out_msg.tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in choice.message.tool_calls
            ]

        return ChatResult(message=out_msg, finish_reason=choice.finish_reason)

    def stream(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> Iterator[ChatChunk]:
        """Stream the chat model response."""
        openai_msgs = self._convert_messages(messages)
        openai_tools = self._convert_tools(tools)

        args: dict[str, Any] = {
            "model": self.model,
            "messages": openai_msgs,
            "stream": True,
            **kwargs,
        }
        if openai_tools:
            args["tools"] = openai_tools

        resp = self._client.chat.completions.create(**args)

        for chunk in resp:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            tool_calls = []
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id or "",
                            name=tc.function.name if (tc.function and tc.function.name) else "",
                            arguments=tc.function.arguments
                            if (tc.function and tc.function.arguments)
                            else "",
                        )
                    )

            yield ChatChunk(content=delta.content, tool_calls=tool_calls)
