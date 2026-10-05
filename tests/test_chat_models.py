"""Comprehensive tests for chat model providers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import dynavec.chat as chat
from dynavec.chat.anthropic import AnthropicChatModel
from dynavec.chat.base import Message, Tool
from dynavec.chat.bedrock import BedrockChatModel
from dynavec.chat.openai import OpenAIChatModel
from dynavec.exceptions import MissingDependencyError


# ---------------------------------------------------------------------------
# OpenAI Fakes
# ---------------------------------------------------------------------------
class FakeOpenAIChatCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content="Streamed ",
                                tool_calls=[
                                    SimpleNamespace(
                                        id="call_1",
                                        function=SimpleNamespace(
                                            name="search", arguments='{"q":"dynavec"}'
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                )
            ]

        if kwargs.get("tools"):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_abc",
                                    function=SimpleNamespace(
                                        name="search",
                                        arguments='{"query": "vector search"}',
                                    ),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="Hello from fake OpenAI",
                        tool_calls=[],
                    ),
                    finish_reason="stop",
                )
            ]
        )


class FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeOpenAIChatCompletions())


# ---------------------------------------------------------------------------
# Anthropic Fakes
# ---------------------------------------------------------------------------
class FakeAnthropicMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("tools"):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="toolu_123",
                        name="get_weather",
                        input={"location": "SF"},
                    )
                ],
                stop_reason="tool_use",
            )
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text="Hello from fake Anthropic",
                )
            ],
            stop_reason="end_turn",
        )

    def stream(self, **kwargs):
        self.last_kwargs = kwargs

        class FakeStreamContext:
            def __enter__(self):
                return [
                    SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="text_delta", text="Streamed "),
                    ),
                    SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="input_json_delta", partial_json='{"loc":"NY"}'),
                    ),
                ]

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return FakeStreamContext()


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeAnthropicMessages()


# ---------------------------------------------------------------------------
# Bedrock Fakes
# ---------------------------------------------------------------------------
class FakeBedrockClient:
    def __init__(self):
        self.last_kwargs = None

    def converse(self, **kwargs):
        self.last_kwargs = kwargs
        if "toolConfig" in kwargs:
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tool_bedrock_1",
                                    "name": "lookup",
                                    "input": {"id": "123"},
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
            }
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello from fake Bedrock"}],
                }
            },
            "stopReason": "end_turn",
        }

    def converse_stream(self, **kwargs):
        self.last_kwargs = kwargs
        return {
            "stream": [
                {
                    "contentBlockDelta": {
                        "delta": {"text": "Streamed "},
                    }
                },
                {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": '{"arg":"val"}'}},
                    }
                },
            ]
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_chat_module_exports():
    assert hasattr(chat, "ChatModel")
    assert hasattr(chat, "Message")
    assert hasattr(chat, "Tool")
    assert hasattr(chat, "ToolCall")
    assert hasattr(chat, "ChatResult")
    assert hasattr(chat, "ChatChunk")
    assert hasattr(chat, "OpenAIChatModel")
    assert hasattr(chat, "AnthropicChatModel")
    assert hasattr(chat, "BedrockChatModel")


def test_openai_chat_model(monkeypatch):
    fake_client = FakeOpenAIClient()

    def fake_init(self, *args, **kwargs):
        self.model = "gpt-4o"
        self._client = fake_client

    monkeypatch.setattr("dynavec.chat.openai.OpenAIChatModel.__init__", fake_init)

    model = OpenAIChatModel()

    # Plain invoke
    res = model.invoke([Message(role="user", content="Hi")])
    assert res.message.content == "Hello from fake OpenAI"
    assert res.finish_reason == "stop"

    # Tool invoke
    tool = Tool(
        name="search",
        description="Search vector store",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    res_tool = model.invoke([Message(role="user", content="Search query")], tools=[tool])
    assert res_tool.finish_reason == "tool_calls"
    assert len(res_tool.message.tool_calls) == 1
    assert res_tool.message.tool_calls[0].name == "search"
    assert json.loads(res_tool.message.tool_calls[0].arguments) == {"query": "vector search"}

    # Stream
    chunks = list(model.stream([Message(role="user", content="Hi")]))
    assert chunks[0].content == "Streamed "
    assert len(chunks[0].tool_calls) == 1


def test_anthropic_chat_model(monkeypatch):
    fake_client = FakeAnthropicClient()

    def fake_init(self, *args, **kwargs):
        self.model = "claude-3-5-sonnet-20240620"
        self._client = fake_client

    monkeypatch.setattr("dynavec.chat.anthropic.AnthropicChatModel.__init__", fake_init)

    model = AnthropicChatModel()

    # Plain invoke with system prompt
    res = model.invoke(
        [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hi"),
        ]
    )
    assert res.message.content == "Hello from fake Anthropic"
    assert res.finish_reason == "end_turn"
    assert fake_client.messages.last_kwargs.get("system") == "You are a helpful assistant."

    # Tool invoke
    tool = Tool(
        name="get_weather",
        description="Get current weather",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}},
    )
    res_tool = model.invoke([Message(role="user", content="What's weather?")], tools=[tool])
    assert res_tool.finish_reason == "tool_use"
    assert len(res_tool.message.tool_calls) == 1
    assert res_tool.message.tool_calls[0].id == "toolu_123"
    assert json.loads(res_tool.message.tool_calls[0].arguments) == {"location": "SF"}

    # Stream
    chunks = list(model.stream([Message(role="user", content="Hi")]))
    assert chunks[0].content == "Streamed "
    assert chunks[1].tool_calls[0].arguments == '{"loc":"NY"}'


def test_bedrock_chat_model(monkeypatch):
    fake_client = FakeBedrockClient()

    def fake_init(self, *args, **kwargs):
        self.model = "anthropic.claude-3-haiku-20240307-v1:0"
        self._client = fake_client

    monkeypatch.setattr("dynavec.chat.bedrock.BedrockChatModel.__init__", fake_init)

    model = BedrockChatModel()

    # Plain invoke
    res = model.invoke([Message(role="user", content="Hi")])
    assert res.message.content == "Hello from fake Bedrock"
    assert res.finish_reason == "end_turn"

    # Tool invoke
    tool = Tool(
        name="lookup",
        description="Lookup record",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
    )
    res_tool = model.invoke([Message(role="user", content="Lookup")], tools=[tool])
    assert res_tool.finish_reason == "tool_use"
    assert len(res_tool.message.tool_calls) == 1
    assert res_tool.message.tool_calls[0].name == "lookup"

    # Stream
    chunks = list(model.stream([Message(role="user", content="Hi")]))
    assert chunks[0].content == "Streamed "
    assert chunks[1].tool_calls[0].arguments == '{"arg":"val"}'


@pytest.mark.asyncio
async def test_async_chat_model_delegation(monkeypatch):
    def fake_init(self, *args, **kwargs):
        self.model = "gpt-4o"
        self._client = FakeOpenAIClient()

    monkeypatch.setattr("dynavec.chat.openai.OpenAIChatModel.__init__", fake_init)

    model = OpenAIChatModel()
    res = await model.ainvoke([Message(role="user", content="Hi")])
    assert res.message.content == "Hello from fake OpenAI"

    streamed_chunks = []
    async for chunk in model.astream([Message(role="user", content="Hi")]):
        streamed_chunks.append(chunk)

    assert len(streamed_chunks) == 1
    assert streamed_chunks[0].content == "Streamed "


def test_missing_dependency_guards(monkeypatch):
    import sys

    # Simulate missing openai module
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(MissingDependencyError) as exc_info:
        OpenAIChatModel()
    assert "OpenAIChatModel" in str(exc_info.value)
