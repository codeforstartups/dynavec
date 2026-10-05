"""Tests for Human-in-the-loop / interrupts functionality in dynavec agents."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from dynavec.agents.base import interrupt, tool
from dynavec.agents.react import ReActAgent
from dynavec.chat.base import ChatChunk, ChatModel, ChatResult, Message, ToolCall


class DummyChatModel(ChatModel):
    """Simple mock chat model for HITL tests."""

    def __init__(self, responses: list[ChatResult]) -> None:
        super().__init__()
        self.responses = responses
        self.call_count = 0

    def invoke(
        self, messages: list[Message], tools: list[Any] | None = None, **kwargs: Any
    ) -> ChatResult:
        res = self.responses[self.call_count]
        self.call_count += 1
        return res

    async def ainvoke(
        self, messages: list[Message], tools: list[Any] | None = None, **kwargs: Any
    ) -> ChatResult:
        return self.invoke(messages, tools, **kwargs)

    def stream(
        self, messages: list[Message], tools: list[Any] | None = None, **kwargs: Any
    ) -> Iterator[ChatChunk]:
        yield ChatChunk(content="")

    async def astream(
        self, messages: list[Message], tools: list[Any] | None = None, **kwargs: Any
    ) -> AsyncIterator[ChatChunk]:
        yield ChatChunk(content="")


@tool
def sensitive_action_tool(action: str) -> str:
    """Action requiring human approval."""
    interrupt("approval_node", f"Action '{action}' requires approval. Confirm?")
    return f"Action '{action}' executed successfully."


def test_interrupt_raised_and_caught() -> None:
    mock_model = DummyChatModel(
        responses=[
            ChatResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            name="sensitive_action_tool",
                            arguments=json.dumps({"action": "delete_database"}),
                        )
                    ],
                )
            )
        ]
    )

    agent = ReActAgent(model=mock_model, tools=[sensitive_action_tool])
    result = agent.run("Please delete the database.")

    assert result.finished is False
    assert result.termination_reason == "interrupted"
    assert result.interrupt_payload is not None


def test_resume_execution() -> None:
    mock_model = DummyChatModel(
        responses=[
            ChatResult(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            name="sensitive_action_tool",
                            arguments=json.dumps({"action": "delete_database"}),
                        )
                    ],
                )
            ),
            ChatResult(
                message=Message(
                    role="assistant",
                    content="Database action completed.",
                )
            ),
        ]
    )

    agent = ReActAgent(model=mock_model, tools=[sensitive_action_tool])
    result = agent.run("Please delete the database.")

    assert result.finished is False
    assert result.interrupt_payload is not None

    payload = result.interrupt_payload
    saved_messages = payload["messages"]
    start_step = payload["start_step"]

    resume_agent = ReActAgent(model=mock_model, tools=[sensitive_action_tool])
    final_result = resume_agent.resume(
        saved_messages,
        start_step=start_step,
        human_decision="approved",
    )

    assert final_result.finished is True
    assert final_result.termination_reason == "completed"
    assert "Database action completed" in final_result.output


def test_async_interrupt_and_resume() -> None:
    async def _run_async() -> None:
        mock_model = DummyChatModel(
            responses=[
                ChatResult(
                    message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="tc_1",
                                name="sensitive_action_tool",
                                arguments=json.dumps({"action": "delete_database"}),
                            )
                        ],
                    )
                ),
                ChatResult(
                    message=Message(
                        role="assistant",
                        content="Deployment completed.",
                    )
                ),
            ]
        )

        agent = ReActAgent(model=mock_model, tools=[sensitive_action_tool])
        result = await agent.arun("Deploy latest release.")

        assert result.finished is False
        assert result.interrupt_payload is not None

        payload = result.interrupt_payload
        assert payload is not None

        resume_agent = ReActAgent(model=mock_model, tools=[sensitive_action_tool])
        final_result = await resume_agent.aresume(
            payload["messages"],
            start_step=payload["start_step"],
            human_decision="approved",
        )

        assert final_result.finished is True
        assert final_result.termination_reason == "completed"

    asyncio.run(_run_async())
