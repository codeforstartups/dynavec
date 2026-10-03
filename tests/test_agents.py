"""Unit tests for agent primitives (ReActAgent, Planner, AgentTool)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from dynavec.agents import (
    AgentTool,
    Plan,
    Planner,
    ReActAgent,
    tool,
)
from dynavec.chat.base import (
    ChatChunk,
    ChatModel,
    ChatResult,
    Message,
    Tool,
    ToolCall,
)


class ScriptedChatModel(ChatModel):
    """A deterministic test fake for ChatModel that replays configured responses."""

    def __init__(self, responses: list[ChatResult | Message | str]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.received_messages: list[list[Message]] = []

    def invoke(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received_messages.append(list(messages))
        if self.call_count >= len(self.responses):
            # Default response if script exhausted
            return ChatResult(
                message=Message(
                    role="assistant", content="Scripted responses exhausted."
                ),
                finish_reason="stop",
            )

        resp = self.responses[self.call_count]
        self.call_count += 1

        if isinstance(resp, ChatResult):
            return resp
        if isinstance(resp, Message):
            return ChatResult(message=resp, finish_reason="stop")
        return ChatResult(
            message=Message(role="assistant", content=resp),
            finish_reason="stop",
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatChunk]:
        res = self.invoke(messages, tools, **kwargs)
        yield ChatChunk(content=res.message.content, tool_calls=res.message.tool_calls)


def test_tool_decorator_and_schema_generation() -> None:
    """Verify @tool extracts docstrings, names, parameter types, and executes correctly."""

    @tool(name="custom_add", description="Add two integers.")
    def add(a: int, b: int = 0) -> int:
        """Add two numbers."""
        return a + b

    assert isinstance(add, AgentTool)
    assert add.name == "custom_add"
    assert add.description == "Add two integers."
    assert add.parameters["type"] == "object"
    assert add.parameters["properties"]["a"]["type"] == "integer"
    assert add.parameters["properties"]["b"]["type"] == "integer"
    assert "a" in add.parameters.get("required", [])
    assert "b" not in add.parameters.get("required", [])

    # Test execution with dict
    assert add.execute({"a": 5, "b": 10}) == "15"
    # Test execution with JSON string
    assert add.execute('{"a": 20, "b": 22}') == "42"
    # Test direct call
    assert add(3, 4) == 7


def test_react_agent_two_tool_task() -> None:
    """Acceptance criteria: Agent solves a 2-tool sequential task via the ReAct loop."""

    # Define tools
    @tool()
    def get_user_id(username: str) -> str:
        """Retrieve user ID by username."""
        if username == "alice":
            return "user_123"
        return "not_found"

    @tool()
    def get_account_balance(user_id: str) -> float:
        """Fetch account balance for a user ID."""
        if user_id == "user_123":
            return 2500.50
        return 0.0

    # Scripted model responses:
    # 1. Call get_user_id("alice")
    # 2. Call get_account_balance("user_123")
    # 3. Final answer
    step1_msg = Message(
        role="assistant",
        content="I will look up Alice's user ID.",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="get_user_id",
                arguments=json.dumps({"username": "alice"}),
            )
        ],
    )
    step2_msg = Message(
        role="assistant",
        content="I found the user ID. Now I will check the balance.",
        tool_calls=[
            ToolCall(
                id="call_2",
                name="get_account_balance",
                arguments=json.dumps({"user_id": "user_123"}),
            )
        ],
    )
    step3_msg = Message(
        role="assistant",
        content="Alice's account balance is $2,500.50.",
        tool_calls=[],
    )

    fake_model = ScriptedChatModel([step1_msg, step2_msg, step3_msg])
    agent = ReActAgent(
        model=fake_model,
        tools=[get_user_id, get_account_balance],
        max_steps=5,
    )

    result = agent.run("What is Alice's balance?")

    assert result.finished is True
    assert result.termination_reason == "completed"
    assert result.total_steps == 3
    assert result.tool_calls_count == 2
    assert "2,500.50" in result.output
    assert len(result.steps) == 3

    # Validate step observations
    assert result.steps[0].observations == ["user_123"]
    assert result.steps[1].observations == ["2500.5"]
    assert result.steps[2].observations == []


def test_react_agent_max_steps_budget_guard() -> None:
    """Acceptance criteria: Agent terminates cleanly when max_steps budget is exhausted."""

    # Infinite loop model that keeps asking for tools
    infinite_tool_msg = Message(
        role="assistant",
        content="Still thinking...",
        tool_calls=[
            ToolCall(
                id="call_loop",
                name="dummy_tool",
                arguments='{"val": "x"}',
            )
        ],
    )
    fake_model = ScriptedChatModel([infinite_tool_msg] * 10)

    @tool()
    def dummy_tool(val: str) -> str:
        """A dummy test tool."""
        return f"result_{val}"

    agent = ReActAgent(
        model=fake_model,
        tools=[dummy_tool],
        max_steps=3,
    )

    result = agent.run("Run indefinitely")

    assert result.finished is False
    assert result.termination_reason == "max_steps_reached"
    assert result.total_steps == 3
    assert result.tool_calls_count == 3


def test_react_agent_tool_error_handling() -> None:
    """Verify tool execution exceptions are caught and passed back as observations."""

    @tool()
    def buggy_tool(x: int) -> float:
        """A tool that raises zero division."""
        return 10 / x

    # Model calls buggy_tool with x=0, gets error observation, then returns answer
    step1_msg = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_err",
                name="buggy_tool",
                arguments='{"x": 0}',
            )
        ],
    )
    step2_msg = Message(
        role="assistant",
        content="Encountered division by zero error.",
        tool_calls=[],
    )

    fake_model = ScriptedChatModel([step1_msg, step2_msg])
    agent = ReActAgent(model=fake_model, tools=[buggy_tool], max_steps=5)

    result = agent.run("Divide 10 by 0")

    assert result.finished is True
    assert "division by zero" in result.steps[0].observations[0].lower()


def test_react_agent_unregistered_tool() -> None:
    """Verify calling an unregistered tool returns an error message observation."""
    step1_msg = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_unknown",
                name="unknown_tool",
                arguments="{}",
            )
        ],
    )
    step2_msg = Message(role="assistant", content="Unknown tool handled.")

    fake_model = ScriptedChatModel([step1_msg, step2_msg])
    agent = ReActAgent(model=fake_model, tools=[], max_steps=5)

    result = agent.run("Call missing tool")

    assert result.finished is True
    assert "not registered" in result.steps[0].observations[0].lower()


@pytest.mark.asyncio
async def test_react_agent_async_arun() -> None:
    """Verify async arun execution works."""

    @tool()
    def greet(name: str) -> str:
        """Greet a person."""
        return f"Hello, {name}!"

    step1_msg = Message(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="call_greet",
                name="greet",
                arguments='{"name": "Dynavec"}',
            )
        ],
    )
    step2_msg = Message(role="assistant", content="Greeting completed.")

    fake_model = ScriptedChatModel([step1_msg, step2_msg])
    agent = ReActAgent(model=fake_model, tools=[greet], max_steps=5)

    result = await agent.arun("Say hello")

    assert result.finished is True
    assert result.total_steps == 2
    assert result.steps[0].observations == ["Hello, Dynavec!"]


def test_planner_structured_decomposition() -> None:
    """Verify Planner decomposes a goal into ordered PlanSteps from JSON."""
    plan_json = json.dumps(
        {
            "steps": [
                {
                    "step_number": 1,
                    "description": "Fetch documentation from knowledge base",
                    "tool_hint": "dynavec_search",
                },
                {
                    "step_number": 2,
                    "description": "Summarize key findings",
                    "tool_hint": "summarizer",
                },
            ]
        }
    )

    fake_model = ScriptedChatModel([f"```json\n{plan_json}\n```"])
    planner = Planner(model=fake_model)

    plan = planner.plan("Research vector database scaling")

    assert isinstance(plan, Plan)
    assert plan.goal == "Research vector database scaling"
    assert len(plan.steps) == 2
    assert plan.steps[0].step_number == 1
    assert "documentation" in plan.steps[0].description
    assert plan.steps[0].tool_hint == "dynavec_search"
    assert plan.steps[1].step_number == 2


@pytest.mark.asyncio
async def test_planner_async_aplan() -> None:
    """Verify async aplan method."""
    plan_json = json.dumps(
        {
            "steps": [
                {"step_number": 1, "description": "Step A"},
                {"step_number": 2, "description": "Step B"},
            ]
        }
    )
    fake_model = ScriptedChatModel([plan_json])
    planner = Planner(model=fake_model)

    plan = await planner.aplan("Async task")
    assert len(plan.steps) == 2
    assert plan.steps[0].description == "Step A"
    assert plan.steps[1].description == "Step B"
