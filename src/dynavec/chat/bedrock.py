"""AWS Bedrock chat model."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ..exceptions import MissingDependencyError
from .base import ChatChunk, ChatModel, ChatResult, Message, Tool, ToolCall


class BedrockChatModel(ChatModel):
    """Chat with AWS Bedrock via the unified Converse API.

    Parameters
    ----------
    model:
        Model id, e.g. ``"anthropic.claude-3-haiku-20240307-v1:0"``.
    region_name:
        Optional AWS region name.
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region_name: str | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise MissingDependencyError("BedrockChatModel", "boto3", "boto3") from exc

        self.model = model
        self._client = boto3.client("bedrock-runtime", region_name=region_name)

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system_prompts: list[dict[str, Any]] = []
        bedrock_msgs: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_prompts.append({"text": msg.content})
                continue

            content_blocks: list[dict[str, Any]] = []
            if msg.content:
                content_blocks.append({"text": msg.content})

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tc.id,
                                "name": tc.name,
                                "input": json.loads(tc.arguments) if tc.arguments else {},
                            }
                        }
                    )

            if msg.role == "tool" and msg.tool_call_id:
                content_blocks.append(
                    {
                        "toolResult": {
                            "toolUseId": msg.tool_call_id,
                            "content": [{"text": msg.content or ""}],
                            "status": "success",
                        }
                    }
                )
                bedrock_msgs.append({"role": "user", "content": content_blocks})
                continue

            bedrock_msgs.append(
                {
                    "role": "assistant" if msg.role == "assistant" else "user",
                    "content": content_blocks,
                }
            )

        return system_prompts, bedrock_msgs

    def _convert_tools(self, tools: list[Tool] | None) -> dict[str, Any] | None:
        if not tools:
            return None
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"json": t.parameters},
                    }
                }
                for t in tools
            ]
        }

    def invoke(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> ChatResult:
        system, bedrock_msgs = self._convert_messages(messages)
        bedrock_tools = self._convert_tools(tools)

        args: dict[str, Any] = {"modelId": self.model, "messages": bedrock_msgs, **kwargs}
        if system:
            args["system"] = system
        if bedrock_tools:
            args["toolConfig"] = bedrock_tools

        resp = self._client.converse(**args)

        output_msg = resp["output"]["message"]
        role = output_msg["role"]

        text_content = ""
        tool_calls = []

        for block in output_msg.get("content", []):
            if "text" in block:
                text_content += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    ToolCall(id=tu["toolUseId"], name=tu["name"], arguments=json.dumps(tu["input"]))
                )

        stop_reason = resp.get("stopReason")
        out = Message(
            role=role, content=text_content if text_content else None, tool_calls=tool_calls
        )
        return ChatResult(message=out, finish_reason=stop_reason)

    def stream(
        self, messages: list[Message], tools: list[Tool] | None = None, **kwargs: Any
    ) -> Iterator[ChatChunk]:
        system, bedrock_msgs = self._convert_messages(messages)
        bedrock_tools = self._convert_tools(tools)

        args: dict[str, Any] = {"modelId": self.model, "messages": bedrock_msgs, **kwargs}
        if system:
            args["system"] = system
        if bedrock_tools:
            args["toolConfig"] = bedrock_tools

        resp = self._client.converse_stream(**args)
        stream = resp.get("stream")
        if not stream:
            return

        for event in stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield ChatChunk(content=delta["text"])
                elif "toolUse" in delta:
                    yield ChatChunk(
                        tool_calls=[
                            ToolCall(id="", name="", arguments=delta["toolUse"].get("input", ""))
                        ]
                    )
