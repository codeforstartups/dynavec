from __future__ import annotations

from typing import Any

import pytest

from dynavec.agents import (
    AgentResult,
    AgentState,
    Handoff,
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


def test_supervisor_routes_only_to_selected_specialist():
    calls: list[str] = []

    class NamedAgent:
        def __init__(self, name: str, output: str) -> None:
            self.name = name
            self.output = output

        def run(self, goal: str, *, state: AgentState) -> AgentResult:
            calls.append(self.name)
            return AgentResult(output=self.output)

    agents = {
        "transaction": NamedAgent("transaction", "Declined with code 51."),
        "policy": NamedAgent("policy", "Manual review required."),
    }

    supervisor = Supervisor(
        agents=agents,
        router=lambda goal, state: "transaction",
    )

    results = supervisor.run("Analyze transaction TX123.", state=AgentState())

    assert calls == ["transaction"]
    assert len(results) == 1
    assert results[0].output == "Declined with code 51."


def test_supervisor_rejects_unknown_agent_from_router():
    class NamedAgent:
        def run(self, goal: str, *, state: AgentState) -> AgentResult:
            return AgentResult(output="Transaction analyzed.")

    supervisor = Supervisor(
        agents={"transaction": NamedAgent()},
        router=lambda goal, state: "unknown",
    )

    with pytest.raises(ValueError, match="Router selected unknown agent"):
        supervisor.run("Analyze transaction TX123.", state=AgentState())


def test_supervisor_aggregates_only_routed_specialist():
    calls: list[str] = []

    class NamedAgent:
        def __init__(self, name: str, output: str) -> None:
            self.name = name
            self.output = output

        def run(self, goal: str, *, state: AgentState) -> AgentResult:
            calls.append(self.name)
            return AgentResult(output=self.output)

    supervisor = Supervisor(
        agents={
            "transaction": NamedAgent("transaction", "Declined with code 51."),
            "policy": NamedAgent("policy", "Manual review required."),
        },
        router=lambda goal, state: "policy",
    )

    result = supervisor.run_and_aggregate(
        "Review this transaction.",
        state=AgentState(),
    )

    assert calls == ["policy"]
    assert result.output == "Manual review required."


def test_supervisor_handoff_routes_to_named_specialist_with_shared_state():
    state = AgentState(data={"transaction_id": "TX123"})
    calls = []

    class FakeAgent:
        def __init__(self, output):
            self.output = output

        def run(self, goal, *, state):
            calls.append((goal, state))
            state.data["policy_review"] = "decline_code_51"
            return AgentResult(output=self.output)

    transaction_agent = FakeAgent("Transaction identified")
    policy_agent = FakeAgent("Policy review completed")

    supervisor = Supervisor(
        {
            "transaction": transaction_agent,
            "policy": policy_agent,
        }
    )

    request = Handoff()(
        "policy",
        "Review transaction TX123 using the transaction findings.",
    )

    result = supervisor.handoff(request, state=state)

    assert result.output == "Policy review completed"
    assert len(calls) == 1
    assert calls[0][0] == ("Review transaction TX123 using the transaction findings.")
    assert calls[0][1] is state
    assert state.data["policy_review"] == "decline_code_51"
