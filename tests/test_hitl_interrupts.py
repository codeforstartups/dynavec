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


@tool()
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


# ---------------------------------------------------------------------------
# Resume semantics: the paused tool re-runs and interrupt() returns the decision.
# ---------------------------------------------------------------------------

from dynavec.agents import MemoryCheckpointer, NodeInterrupt  # noqa: E402

DECISIONS: list[Any] = []


@tool()
def transfer_funds(amount: int) -> str:
    """Transfer money; needs a human decision first."""
    decision = interrupt("approve_transfer", {"amount": amount})
    DECISIONS.append(decision)
    if decision != "approved":
        return f"Transfer of {amount} cancelled by reviewer."
    return f"Transferred {amount}."


@tool()
def lookup_balance() -> str:
    """Read-only lookup."""
    return "balance=500"


def _tool_call(call_id: str, name: str, args: dict[str, Any]) -> ChatResult:
    return ChatResult(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args))],
        )
    )


def _final(text: str) -> ChatResult:
    return ChatResult(message=Message(role="assistant", content=text))


def _assert_provider_valid(messages: list[Message]) -> None:
    """Every assistant tool call is answered by a tool message before the next turn."""
    for i, m in enumerate(messages):
        if m.role == "assistant" and m.tool_calls:
            replies = []
            for nxt in messages[i + 1 :]:
                if nxt.role != "tool":
                    break
                replies.append(nxt.tool_call_id)
            assert sorted(replies) == sorted(tc.id for tc in m.tool_calls)


def test_interrupt_payload_carries_node_and_payload() -> None:
    model = DummyChatModel([_tool_call("t1", "transfer_funds", {"amount": 120})])
    result = ReActAgent(model=model, tools=[transfer_funds]).run("pay rent", thread_id="th-1")

    assert result.termination_reason == "interrupted"
    payload = result.interrupt_payload
    assert payload is not None
    assert payload["node_id"] == "approve_transfer"
    assert payload["payload"] == {"amount": 120}
    assert payload["thread_id"] == "th-1"
    assert payload["start_step"] == 1


def test_resume_reruns_tool_with_decision_and_records_its_result() -> None:
    DECISIONS.clear()
    model = DummyChatModel(
        [_tool_call("t1", "transfer_funds", {"amount": 120}), _final("Done.")]
    )
    agent = ReActAgent(model=model, tools=[transfer_funds])
    paused = agent.run("pay rent")
    final = agent.resume(
        paused.interrupt_payload["messages"],  # type: ignore[index]
        human_decision="approved",
        start_step=paused.interrupt_payload["start_step"],  # type: ignore[index]
    )

    assert DECISIONS == ["approved"]
    assert final.termination_reason == "completed"
    assert final.steps[0].observations == ["Transferred 120."]
    assert final.total_steps == 2


def test_rejected_decision_reaches_the_tool() -> None:
    DECISIONS.clear()
    model = DummyChatModel(
        [_tool_call("t1", "transfer_funds", {"amount": 99}), _final("Cancelled.")]
    )
    agent = ReActAgent(model=model, tools=[transfer_funds])
    paused = agent.run("pay")
    final = agent.resume(
        paused.interrupt_payload["messages"],  # type: ignore[index]
        human_decision="rejected",
    )

    assert DECISIONS == ["rejected"]
    assert final.steps[0].observations == ["Transfer of 99 cancelled by reviewer."]


def test_resume_from_checkpoint_in_a_fresh_agent() -> None:
    """A paused run survives a process restart when a checkpointer is configured."""
    DECISIONS.clear()
    checkpointer = MemoryCheckpointer()
    first = ReActAgent(
        model=DummyChatModel([_tool_call("t1", "transfer_funds", {"amount": 7})]),
        tools=[transfer_funds],
        checkpointer=checkpointer,
    )
    assert first.run("pay", thread_id="th-x").termination_reason == "interrupted"

    # New agent instance, nothing carried over except the checkpointer.
    second = ReActAgent(
        model=DummyChatModel([_final("Paid.")]),
        tools=[transfer_funds],
        checkpointer=checkpointer,
    )
    final = second.resume(thread_id="th-x", human_decision="approved")

    assert final.termination_reason == "completed"
    assert final.output == "Paid."
    assert DECISIONS == ["approved"]
    latest = checkpointer.get("th-x")
    assert latest is not None and latest.state["finished"] is True
    assert "interrupted" not in latest.state


def test_run_refuses_a_paused_thread() -> None:
    checkpointer = MemoryCheckpointer()
    agent = ReActAgent(
        model=DummyChatModel([_tool_call("t1", "transfer_funds", {"amount": 1})]),
        tools=[transfer_funds],
        checkpointer=checkpointer,
    )
    agent.run("pay", thread_id="th-p")
    try:
        agent.run("pay", thread_id="th-p")
    except ValueError as exc:
        assert "resume" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("run() on a paused thread should raise")


def test_resume_without_paused_tool_call_raises() -> None:
    agent = ReActAgent(model=DummyChatModel([]), tools=[transfer_funds])
    try:
        agent.resume([Message(role="user", content="hi")], human_decision="approved")
    except ValueError as exc:
        assert "Nothing to resume" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_second_interrupt_in_same_batch_pauses_again() -> None:
    DECISIONS.clear()
    batch = ChatResult(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="a", name="lookup_balance", arguments="{}"),
                ToolCall(id="b", name="transfer_funds", arguments=json.dumps({"amount": 1})),
                ToolCall(id="c", name="transfer_funds", arguments=json.dumps({"amount": 2})),
            ],
        )
    )
    model = DummyChatModel([batch, _final("All done.")])
    agent = ReActAgent(model=model, tools=[lookup_balance, transfer_funds])

    first = agent.run("move money")
    assert first.termination_reason == "interrupted"
    second = agent.resume(first.interrupt_payload["messages"], human_decision="approved")  # type: ignore[index]
    assert second.termination_reason == "interrupted"
    assert second.interrupt_payload["payload"] == {"amount": 2}  # type: ignore[index]
    final = agent.resume(second.interrupt_payload["messages"], human_decision="rejected")  # type: ignore[index]

    assert final.termination_reason == "completed"
    assert DECISIONS == ["approved", "rejected"]
    assert final.steps[0].observations == [
        "balance=500",
        "Transferred 1.",
        "Transfer of 2 cancelled by reviewer.",
    ]


def test_message_history_stays_provider_valid_after_resume() -> None:
    seen: list[list[Message]] = []

    class RecordingModel(DummyChatModel):
        def invoke(self, messages, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(list(messages))
            return super().invoke(messages, tools, **kwargs)

    model = RecordingModel([_tool_call("t1", "transfer_funds", {"amount": 5}), _final("ok")])
    agent = ReActAgent(model=model, tools=[transfer_funds])
    paused = agent.run("pay")
    agent.resume(paused.interrupt_payload["messages"], human_decision="approved")  # type: ignore[index]

    _assert_provider_valid(seen[-1])
    assert seen[-1][-1].role == "tool"


def test_tool_errors_still_become_observations() -> None:
    @tool()
    def broken() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    model = DummyChatModel([_tool_call("t1", "broken", {}), _final("handled")])
    result = ReActAgent(model=model, tools=[broken]).run("go")
    assert result.termination_reason == "completed"
    assert "boom" in result.steps[0].observations[0]
    assert issubclass(NodeInterrupt, Exception)
