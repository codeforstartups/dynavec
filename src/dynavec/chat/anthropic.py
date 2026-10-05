"""Anthropic chat model (bring your own ANTHROPIC_API_KEY)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..exceptions import MissingDependencyError
from .base import ChatChunk, ChatModel, ChatResult, Message, Tool, ToolCall


class AnthropicChatModel(ChatModel):
    """Chat with Anthropic's Claude models.

    Parameters
    ----------
    model:
        Model id, e.g. ``"claude-3-5-sonnet-20240620"``.
    api_key:
        Optional; falls back to the ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20240620",
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise MissingDependencyError("AnthropicChatModel", "anthropic", "anthropic") from exc

        self.model = model
        self._client = Anthropic(api_key=api_key)

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_prompt = ""
        anthropic_msgs = []

        for msg in messages:
            if msg.role == "system":
                system_prompt += (msg.content or "") + "\n"
                continue

            content_blocks: list[dict[str, Any]] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": json.loads(tc.arguments) if tc.arguments else {},
                        }
                    )

            if msg.role == "tool" and msg.tool_call_id:
                content_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    }
                )
                anthropic_msgs.append({"role": "user", "content": content_blocks})
                continue

            anthropic_msgs.append(
                {
                    "role": "assistant" if msg.role == "assistant" else "user",
                    "content": content_blocks,
                }
            )

        return system_prompt.strip(), anthropic_msgs

    def _convert_tools(self, tools: list[Tool] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def invoke(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> ChatResult:
        system, anthropic_msgs = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        args: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_msgs,
            "max_tokens": 1024,
            **kwargs,
        }
        if system:
            args["system"] = system
        if anthropic_tools:
            args["tools"] = anthropic_tools

        resp = self._client.messages.create(**args)

        text_content = ""
        tool_calls = []

        for block in resp.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=json.dumps(block.input))
                )

        out_msg = Message(
            role="assistant", content=text_content if text_content else None, tool_calls=tool_calls
        )
        return ChatResult(message=out_msg, finish_reason=resp.stop_reason)

    def stream(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> Iterator[ChatChunk]:
        system, anthropic_msgs = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        args: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_msgs,
            "max_tokens": 1024,
            **kwargs,
        }
        if system:
            args["system"] = system
        if anthropic_tools:
            args["tools"] = anthropic_tools

        with self._client.messages.stream(**args) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue

                delta = event.delta

                if delta.type == "text_delta":
                    yield ChatChunk(content=delta.text)
                elif delta.type == "input_json_delta":
                    yield ChatChunk(
                        tool_calls=[
                            ToolCall(
                                id="",
                                name="",
                                arguments=delta.partial_json,
                            )
                        ]
                    )
