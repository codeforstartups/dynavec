"""Unit tests for agent & workflow state checkpointers (Issue #280)."""

from __future__ import annotations

import boto3
from moto import mock_aws

from dynavec.agents import ReActAgent
from dynavec.chat.base import ChatModel, ChatResult, Message, Tool, ToolCall
from dynavec.checkpoint import (
    BaseCheckpointer,
    Checkpoint,
    DynamoDBCheckpointer,
    MemoryCheckpointer,
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
        **kwargs: object,
    ) -> ChatResult:
        self.received_messages.append(list(messages))
        if self.call_count >= len(self.responses):
            return ChatResult(
                message=Message(role="assistant", content="Responses exhausted."),
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

    async def ainvoke(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs: object,
    ) -> ChatResult:
        return self.invoke(messages, tools, **kwargs)

    def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs: object,
    ):
        raise NotImplementedError

    async def astream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        **kwargs: object,
    ):
        raise NotImplementedError


def test_checkpoint_dataclass_round_trip():
    """Verify Checkpoint serialization and deserialization round-trip."""
    cp = Checkpoint(
        thread_id="th-001",
        checkpoint_id="cp-001",
        parent_checkpoint_id=None,
        node_id="start_node",
        step=1,
        state={"query": "test", "count": 42, "scores": [0.9, 0.8]},
        metadata={"user": "alice"},
    )

    d = cp.to_dict()
    assert d["thread_id"] == "th-001"
    assert d["checkpoint_id"] == "cp-001"
    assert d["state"]["count"] == 42

    restored = Checkpoint.from_dict(d)
    assert restored.thread_id == cp.thread_id
    assert restored.checkpoint_id == cp.checkpoint_id
    assert restored.state == cp.state
    assert restored.metadata == cp.metadata


def test_memory_checkpointer_crud():
    """Verify MemoryCheckpointer save, get, list, put, and clear operations."""
    chk = MemoryCheckpointer()
    assert isinstance(chk, BaseCheckpointer)

    # Empty thread returns None
    assert chk.get("thread-1") is None
    assert chk.list("thread-1") == []

    # Put initial checkpoint
    cp1 = chk.put(
        "thread-1",
        state={"val": 10},
        node_id="node_a",
        step=0,
    )
    assert cp1.step == 0
    assert cp1.state["val"] == 10

    # Put second checkpoint (auto-chains parent and step)
    cp2 = chk.put(
        "thread-1",
        state={"val": 20},
        node_id="node_b",
    )
    assert cp2.step == 1
    assert cp2.parent_checkpoint_id == cp1.checkpoint_id

    # Get latest
    latest = chk.get("thread-1")
    assert latest is not None
    assert latest.checkpoint_id == cp2.checkpoint_id

    # Get specific historical checkpoint (time travel)
    first = chk.get("thread-1", cp1.checkpoint_id)
    assert first is not None
    assert first.checkpoint_id == cp1.checkpoint_id
    assert first.state["val"] == 10

    # List
    all_cps = chk.list("thread-1")
    assert len(all_cps) == 2
    assert [c.step for c in all_cps] == [0, 1]

    # List with limit
    limited = chk.list("thread-1", limit=1)
    assert len(limited) == 1
    assert limited[0].checkpoint_id == cp2.checkpoint_id

    # Clear
    chk.clear("thread-1")
    assert chk.get("thread-1") is None


def test_dynamodb_checkpointer_moto():
    """Verify DynamoDBCheckpointer save, get, list with mocked AWS DynamoDB table."""
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        table = ddb.create_table(
            TableName="test_checkpoints",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        chk = DynamoDBCheckpointer(table=table, namespace="tenant_1")

        # Initial lookup is empty
        assert chk.get("th-ddb-1") is None
        assert chk.list("th-ddb-1") == []

        # Save checkpoint 1
        cp1 = Checkpoint(
            thread_id="th-ddb-1",
            checkpoint_id="cp-1",
            node_id="node_1",
            step=1,
            state={"status": "in_progress", "ratio": 0.75, "tags": ["a", "b"]},
            metadata={"source": "api"},
        )
        chk.save(cp1)

        # Get latest
        got_latest = chk.get("th-ddb-1")
        assert got_latest is not None
        assert got_latest.checkpoint_id == "cp-1"
        assert got_latest.step == 1
        assert got_latest.state["status"] == "in_progress"
        assert got_latest.state["ratio"] == 0.75
        assert got_latest.state["tags"] == ["a", "b"]

        # Save checkpoint 2
        cp2 = Checkpoint(
            thread_id="th-ddb-1",
            checkpoint_id="cp-2",
            parent_checkpoint_id="cp-1",
            node_id="node_2",
            step=2,
            state={"status": "completed", "ratio": 1.0},
        )
        chk.save(cp2)

        # Get latest now points to cp-2
        latest2 = chk.get("th-ddb-1")
        assert latest2 is not None
        assert latest2.checkpoint_id == "cp-2"
        assert latest2.step == 2
        assert latest2.state["status"] == "completed"

        # Time-travel get returns cp-1
        hist1 = chk.get("th-ddb-1", "cp-1")
        assert hist1 is not None
        assert hist1.checkpoint_id == "cp-1"
        assert hist1.step == 1

        # List checkpoints
        all_cps = chk.list("th-ddb-1")
        assert len(all_cps) == 2
        assert [c.checkpoint_id for c in all_cps] == ["cp-1", "cp-2"]

        # List with limit
        lim = chk.list("th-ddb-1", limit=1)
        assert len(lim) == 1
        assert lim[0].checkpoint_id == "cp-2"


def test_dynamodb_checkpointer_thread_isolation():
    """Verify separate threads do not interfere with each other's checkpoints."""
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        table = ddb.create_table(
            TableName="test_checkpoints_iso",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        chk = DynamoDBCheckpointer(table=table, namespace="tenant_iso")

        chk.put("thread_A", state={"agent": "A1"}, node_id="init", step=0)
        chk.put("thread_B", state={"agent": "B1"}, node_id="init", step=0)
        chk.put("thread_A", state={"agent": "A2"}, node_id="step_1", step=1)

        list_a = chk.list("thread_A")
        list_b = chk.list("thread_B")

        assert len(list_a) == 2
        assert len(list_b) == 1
        assert list_a[-1].state["agent"] == "A2"
        assert list_b[0].state["agent"] == "B1"


def test_react_agent_checkpointing():
    """Verify ReActAgent records checkpoints after each step when checkpointer is configured."""
    model = ScriptedChatModel(
        [
            # Step 1: Model requests tool call
            ChatResult(
                message=Message(
                    role="assistant",
                    content="Let me check the weather.",
                    tool_calls=[
                        ToolCall(
                            id="tc-1",
                            name="get_weather",
                            arguments='{"location": "Seattle"}',
                        )
                    ],
                )
            ),
            # Step 2: Model gives final answer
            ChatResult(
                message=Message(
                    role="assistant",
                    content="The weather in Seattle is 65F and sunny.",
                )
            ),
        ]
    )

    def get_weather(location: str) -> str:
        return f"{location}: 65F and sunny"

    checkpointer = MemoryCheckpointer()
    agent = ReActAgent(
        model=model,
        tools=[get_weather],
        checkpointer=checkpointer,
    )

    result = agent.run("What's the weather in Seattle?", thread_id="thread-weather")

    assert result.finished is True
    assert "65F and sunny" in result.output

    # Check that checkpoints were written: step 0 (init), step 1 (tool), step 2 (completion)
    cps = checkpointer.list("thread-weather")
    assert len(cps) == 3
    assert [c.step for c in cps] == [0, 1, 2]
    assert cps[0].node_id == "init"
    assert cps[1].node_id == "step_1"
    assert cps[2].node_id == "step_2"
    assert cps[2].state["finished"] is True


def test_react_agent_resumption():
    """Verify resuming an agent from a previous thread checkpoint reproduces state and finishes."""
    # First agent execution stops after step 1
    model_part1 = ScriptedChatModel(
        [
            ChatResult(
                message=Message(
                    role="assistant",
                    content="Looking up data.",
                    tool_calls=[
                        ToolCall(
                            id="tc-1",
                            name="fetch_data",
                            arguments='{"item": "widget"}',
                        )
                    ],
                )
            ),
        ]
    )

    def fetch_data(item: str) -> str:
        return f"Item {item} inventory: 42 units"

    checkpointer = MemoryCheckpointer()
    agent1 = ReActAgent(
        model=model_part1,
        tools=[fetch_data],
        max_steps=1,  # Intentional stop after step 1
        checkpointer=checkpointer,
    )

    res1 = agent1.run("Check inventory for widget", thread_id="thread-resume")
    assert res1.finished is False
    assert res1.termination_reason == "max_steps_reached"

    # Verify checkpointer recorded step 1
    cp_step1 = checkpointer.get("thread-resume")
    assert cp_step1 is not None
    assert cp_step1.step == 1

    # Second agent picks up from checkpoint on the same thread
    model_part2 = ScriptedChatModel(
        [
            ChatResult(
                message=Message(
                    role="assistant",
                    content="We have 42 widgets available in stock.",
                )
            ),
        ]
    )

    agent2 = ReActAgent(
        model=model_part2,
        tools=[fetch_data],
        max_steps=5,
        checkpointer=checkpointer,
    )

    res2 = agent2.run("Check inventory for widget", thread_id="thread-resume")

    assert res2.finished is True
    assert res2.output == "We have 42 widgets available in stock."

    # Final checkpointer state
    cps = checkpointer.list("thread-resume")
    assert len(cps) >= 3
    assert cps[-1].step == 2
    assert cps[-1].state["finished"] is True
