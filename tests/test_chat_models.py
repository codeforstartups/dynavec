"""Comprehensive tests for chat model providers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

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
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
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


class FakeAsyncOpenAIChatCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if kwargs.get("stream"):

            async def _async_gen() -> Any:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content="Async streamed ",
                                tool_calls=[
                                    SimpleNamespace(
                                        id="async_call_1",
                                        function=SimpleNamespace(
                                            name="search", arguments='{"q":"async"}'
                                        ),
                                    )
                                ],
                            )
                        )
                    ]
                )

            return _async_gen()

        if kwargs.get("tools"):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_async_tool",
                                    function=SimpleNamespace(
                                        name="search",
                                        arguments='{"query": "async vector search"}',
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
                        content="Hello from fake Async OpenAI",
                        tool_calls=[],
                    ),
                    finish_reason="stop",
                )
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeOpenAIChatCompletions())


class FakeAsyncOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeAsyncOpenAIChatCompletions())


# ---------------------------------------------------------------------------
# Anthropic Fakes
# ---------------------------------------------------------------------------
class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
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

    def stream(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs

        class FakeStreamContext:
            def __enter__(self) -> list[Any]:
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

            def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        return FakeStreamContext()


class FakeAsyncAnthropicMessages:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if kwargs.get("tools"):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="async_toolu_123",
                        name="get_weather",
                        input={"location": "Async SF"},
                    )
                ],
                stop_reason="tool_use",
            )
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text="Hello from fake Async Anthropic",
                )
            ],
            stop_reason="end_turn",
        )

    def stream(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs

        class FakeAsyncStreamContext:
            async def __aenter__(self) -> Any:
                async def _gen() -> Any:
                    yield SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="text_delta", text="Async Streamed "),
                    )
                    yield SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(
                            type="input_json_delta", partial_json='{"loc":"Async NY"}'
                        ),
                    )

                return _gen()

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                pass

        return FakeAsyncStreamContext()


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeAnthropicMessages()


class FakeAsyncAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeAsyncAnthropicMessages()


# ---------------------------------------------------------------------------
# Bedrock Fakes
# ---------------------------------------------------------------------------
class FakeBedrockClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def converse(self, **kwargs: Any) -> dict[str, Any]:
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

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
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
def test_chat_module_exports() -> None:
    assert hasattr(chat, "ChatModel")
    assert hasattr(chat, "Message")
    assert hasattr(chat, "Tool")
    assert hasattr(chat, "ToolCall")
    assert hasattr(chat, "ChatResult")
    assert hasattr(chat, "ChatChunk")
    assert hasattr(chat, "OpenAIChatModel")
    assert hasattr(chat, "AnthropicChatModel")
    assert hasattr(chat, "BedrockChatModel")


def test_openai_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeOpenAIClient()
    fake_async_client = FakeAsyncOpenAIClient()

    def fake_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self.model = "gpt-4o"
        self._client = fake_client
        self._async_client = fake_async_client

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


@pytest.mark.asyncio
async def test_openai_async_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeOpenAIClient()
    fake_async_client = FakeAsyncOpenAIClient()

    def fake_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self.model = "gpt-4o"
        self._client = fake_client
        self._async_client = fake_async_client

    monkeypatch.setattr("dynavec.chat.openai.OpenAIChatModel.__init__", fake_init)

    model = OpenAIChatModel()

    # Async invoke
    res = await model.ainvoke([Message(role="user", content="Hi")])
    assert res.message.content == "Hello from fake Async OpenAI"
    assert res.finish_reason == "stop"

    # Async stream
    streamed_chunks = []
    async for chunk in model.astream([Message(role="user", content="Hi")]):
        streamed_chunks.append(chunk)

    assert len(streamed_chunks) == 1
    assert streamed_chunks[0].content == "Async streamed "
    assert len(streamed_chunks[0].tool_calls) == 1
    assert streamed_chunks[0].tool_calls[0].arguments == '{"q":"async"}'


def test_anthropic_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeAnthropicClient()
    fake_async_client = FakeAsyncAnthropicClient()

    def fake_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self.model = "claude-3-5-sonnet-20240620"
        self._client = fake_client
        self._async_client = fake_async_client

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
    assert fake_client.messages.last_kwargs is not None
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


@pytest.mark.asyncio
async def test_anthropic_async_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeAnthropicClient()
    fake_async_client = FakeAsyncAnthropicClient()

    def fake_init(self: Any, *args: Any, **kwargs: Any) -> None:
        self.model = "claude-3-5-sonnet-20240620"
        self._client = fake_client
        self._async_client = fake_async_client

    monkeypatch.setattr("dynavec.chat.anthropic.AnthropicChatModel.__init__", fake_init)

    model = AnthropicChatModel()

    # Async invoke
    res = await model.ainvoke([Message(role="user", content="Hi")])
    assert res.message.content == "Hello from fake Async Anthropic"
    assert res.finish_reason == "end_turn"

    # Async stream
    chunks = []
    async for chunk in model.astream([Message(role="user", content="Hi")]):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].content == "Async Streamed "
    assert chunks[1].tool_calls[0].arguments == '{"loc":"Async NY"}'


def test_bedrock_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeBedrockClient()

    def fake_init(self: Any, *args: Any, **kwargs: Any) -> None:
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


def test_missing_dependency_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    # Simulate missing openai module
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(MissingDependencyError) as exc_info:
        OpenAIChatModel()
    assert "OpenAIChatModel" in str(exc_info.value)
