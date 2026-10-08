from __future__ import annotations

from typing import Any

from dynavec.agents import (
    AgentResult,
    AgentState,
    ReActAgent,
    Supervisor,
)
from dynavec.chat.base import ChatChunk, ChatModel, ChatResult, Message, Tool


class ScriptedChatModel(ChatModel):
    """Deterministic chat model for supervisor tests."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    def invoke(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.responses[self.call_count]
        self.call_count += 1

        return ChatResult(
            message=Message(
                role="assistant",
                content=response,
            ),
            finish_reason="stop",
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ):
        result = self.invoke(messages, tools, **kwargs)

        yield ChatChunk(
            content=result.message.content,
            tool_calls=result.message.tool_calls,
        )


class TransactionAgent:
    def run(self, goal: str, *, state: AgentState) -> AgentResult:
        state.data["transaction_id"] = "TX123"
        state.data["decline_code"] = "51"

        return AgentResult(
            output="Transaction declined with code 51.",
            finished=True,
            termination_reason="completed",
            total_steps=0,
            tool_calls_count=0,
        )


class PolicyAgent:
    def run(self, goal: str, *, state: AgentState) -> AgentResult:
        decline_code = state.data["decline_code"]

        return AgentResult(
            output=f"Policy analysis for decline code {decline_code}.",
            finished=True,
            termination_reason="completed",
            total_steps=0,
            tool_calls_count=0,
        )


def test_supervisor_delegates_to_multiple_agents_with_shared_state():
    state = AgentState()

    supervisor = Supervisor(
        agents=[
            TransactionAgent(),
            PolicyAgent(),
        ],
    )

    results = supervisor.run(
        "Analyze the declined transaction.",
        state=state,
    )

    assert len(results) == 2

    assert results[0].output == "Transaction declined with code 51."
    assert results[1].output == "Policy analysis for decline code 51."

    assert state.data["transaction_id"] == "TX123"
    assert state.data["decline_code"] == "51"


def test_supervisor_runs_react_agents():
    state = AgentState()

    first_agent = ReActAgent(model=ScriptedChatModel(["Transaction analysis completed."]))

    second_agent = ReActAgent(model=ScriptedChatModel(["Policy analysis completed."]))

    supervisor = Supervisor(
        agents=[first_agent, second_agent],
    )

    results = supervisor.run(
        "Analyze transaction TX123.",
        state=state,
    )

    assert [result.output for result in results] == [
        "Transaction analysis completed.",
        "Policy analysis completed.",
    ]


def test_supervisor_preserves_state_between_specialists():
    state = AgentState()

    class FirstAgent:
        def run(self, goal: str, *, state: AgentState) -> AgentResult:
            state.data["transaction_id"] = "TX123"
            state.data["decline_code"] = "51"

            return AgentResult(
                output="Transaction analyzed.",
                finished=True,
                termination_reason="completed",
                total_steps=0,
                tool_calls_count=0,
            )

    class SecondAgent:
        def run(self, goal: str, *, state: AgentState) -> AgentResult:
            assert state.data["transaction_id"] == "TX123"
            assert state.data["decline_code"] == "51"

            return AgentResult(
                output="Policy analyzed.",
                finished=True,
                termination_reason="completed",
                total_steps=0,
                tool_calls_count=0,
            )

    supervisor = Supervisor(
        agents=[FirstAgent(), SecondAgent()],
    )

    results = supervisor.run(
        "Analyze the declined transaction.",
        state=state,
    )

    assert [result.output for result in results] == [
        "Transaction analyzed.",
        "Policy analyzed.",
    ]
