"""Unit and integration tests for dynaflow snapshot, time-travel, and fork/replay (Issue #281)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import boto3
import pytest
from moto import mock_aws

from dynavec.agents import ReActAgent
from dynavec.chat.base import ChatModel, ChatResult, Message, Tool, ToolCall
from dynavec.checkpoint import (
    BaseCheckpointer,
    DynamoDBCheckpointer,
    MemoryCheckpointer,
    Snapshot,
    SnapshotImmutableError,
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


def _create_mock_ddb_table(table_name: str = "dynaflow_snapshots_test"):
    """Helper to create a single-table DynamoDB resource for tests."""
    ddb = boto3.resource("dynamodb", region_name="us-east-1")
    return ddb.create_table(
        TableName=table_name,
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


# ---------------------------------------------------------------------------
# 1. Snapshot Data Model Tests
# ---------------------------------------------------------------------------


def test_snapshot_model_fields_and_roundtrip():
    """Verify Snapshot dataclass fields, defaults, to_dict, and from_dict."""
    snap = Snapshot(
        snapshot_id="snap-001",
        thread_id="th-main",
        checkpoint_id="cp-001",
        step=2,
        state={"counter": 42, "results": ["a", "b"]},
        inputs={"prompt": "hello world", "temperature": 0.7},
        parent_snapshot_id="snap-000",
        metadata={"author": "alice", "env": "test"},
    )

    assert snap.snapshot_id == "snap-001"
    assert snap.thread_id == "th-main"
    assert snap.checkpoint_id == "cp-001"
    assert snap.step == 2
    assert snap.state["counter"] == 42
    assert snap.inputs["temperature"] == 0.7
    assert snap.parent_snapshot_id == "snap-000"
    assert snap.metadata["author"] == "alice"
    assert snap.timestamp is not None

    # Serialization roundtrip
    data = snap.to_dict()
    assert data["snapshot_id"] == "snap-001"
    assert data["thread_id"] == "th-main"
    assert data["checkpoint_id"] == "cp-001"
    assert data["step"] == 2
    assert data["state"]["counter"] == 42
    assert data["inputs"]["prompt"] == "hello world"
    assert data["parent_snapshot_id"] == "snap-000"
    assert data["metadata"]["env"] == "test"

    restored = Snapshot.from_dict(data)
    assert restored.snapshot_id == snap.snapshot_id
    assert restored.thread_id == snap.thread_id
    assert restored.checkpoint_id == snap.checkpoint_id
    assert restored.step == snap.step
    assert restored.state == snap.state
    assert restored.inputs == snap.inputs
    assert restored.parent_snapshot_id == snap.parent_snapshot_id
    assert restored.timestamp == snap.timestamp
    assert restored.metadata == snap.metadata


def test_snapshot_model_immutability():
    """Verify Snapshot is frozen and attributes cannot be mutated in place."""
    snap = Snapshot(
        snapshot_id="snap-frozen",
        thread_id="th-1",
        checkpoint_id="cp-1",
        step=1,
        state={"x": 10},
        inputs={"arg": "val"},
    )

    with pytest.raises((FrozenInstanceError, AttributeError)):
        # Attempt direct attribute assignment
        snap.step = 2  # type: ignore[misc]

    with pytest.raises((FrozenInstanceError, AttributeError)):
        snap.state = {"x": 99}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. In-Memory Checkpointer Snapshot Tests
# ---------------------------------------------------------------------------


def test_memory_checkpointer_snapshot_steps_and_fetch():
    """Verify snapshot creation at steps and retrieval by id on MemoryCheckpointer."""
    chk = MemoryCheckpointer()
    assert isinstance(chk, BaseCheckpointer)

    # Initial lookups return None / empty
    assert chk.get_snapshot("th-1", "snap-nonexistent") is None
    assert chk.list_snapshots("th-1") == []

    # Step 0 snapshot
    snap0 = Snapshot(
        snapshot_id="snap-step-0",
        thread_id="th-1",
        checkpoint_id="cp-0",
        step=0,
        state={"status": "init", "value": 100},
        inputs={"task": "process_order", "user_id": 42},
    )
    sid0 = chk.save_snapshot(snap0)
    assert sid0 == "snap-step-0"

    # Step 1 snapshot
    snap1 = Snapshot(
        snapshot_id="snap-step-1",
        thread_id="th-1",
        checkpoint_id="cp-1",
        step=1,
        state={"status": "validated", "value": 100, "validated_at": "now"},
        inputs={"task": "process_order", "user_id": 42},
        parent_snapshot_id=sid0,
    )
    sid1 = chk.save_snapshot(snap1)
    assert sid1 == "snap-step-1"

    # Fetch by id
    fetched0 = chk.get_snapshot("th-1", "snap-step-0")
    assert fetched0 is not None
    assert fetched0.step == 0
    assert fetched0.state["status"] == "init"
    assert fetched0.inputs["user_id"] == 42
    assert fetched0.parent_snapshot_id is None

    fetched1 = chk.get_snapshot("th-1", "snap-step-1")
    assert fetched1 is not None
    assert fetched1.step == 1
    assert fetched1.state["status"] == "validated"
    assert fetched1.parent_snapshot_id == "snap-step-0"


def test_memory_checkpointer_list_snapshots_chronological():
    """Verify list_snapshots returns snapshots in chronological order and isolates threads."""
    chk = MemoryCheckpointer()

    for step_i in range(4):
        chk.save_snapshot(
            Snapshot(
                snapshot_id=f"snap-thA-{step_i}",
                thread_id="thread_A",
                checkpoint_id=f"cp-thA-{step_i}",
                step=step_i,
                state={"step_count": step_i},
                inputs={"job": "A"},
            )
        )

    # Separate thread
    chk.save_snapshot(
        Snapshot(
            snapshot_id="snap-thB-0",
            thread_id="thread_B",
            checkpoint_id="cp-thB-0",
            step=0,
            state={"step_count": 0},
            inputs={"job": "B"},
        )
    )

    list_a = chk.list_snapshots("thread_A")
    assert len(list_a) == 4
    assert [s.step for s in list_a] == [0, 1, 2, 3]
    assert [s.snapshot_id for s in list_a] == [
        "snap-thA-0",
        "snap-thA-1",
        "snap-thA-2",
        "snap-thA-3",
    ]

    # Verify limit parameter
    list_a_limit = chk.list_snapshots("thread_A", limit=2)
    assert len(list_a_limit) == 2
    assert [s.snapshot_id for s in list_a_limit] == ["snap-thA-2", "snap-thA-3"]

    list_b = chk.list_snapshots("thread_B")
    assert len(list_b) == 1
    assert list_b[0].snapshot_id == "snap-thB-0"

    # Non-existent thread
    assert chk.list_snapshots("thread_unknown") == []


def test_memory_checkpointer_immutability():
    """Verify historical snapshots in MemoryCheckpointer are strictly immutable."""
    chk = MemoryCheckpointer()

    snap = Snapshot(
        snapshot_id="snap-immutable-1",
        thread_id="th-immut",
        checkpoint_id="cp-1",
        step=1,
        state={"score": 0.85, "tags": ["prod", "v1"]},
        inputs={"query": "dynaflow memory"},
    )
    chk.save_snapshot(snap)

    # 1. Attempting to overwrite an existing snapshot raises SnapshotImmutableError (ValueError)
    overwrite_attempt = Snapshot(
        snapshot_id="snap-immutable-1",
        thread_id="th-immut",
        checkpoint_id="cp-2",
        step=2,
        state={"score": 0.99},
        inputs={"query": "altered"},
    )
    with pytest.raises((SnapshotImmutableError, ValueError)):
        chk.save_snapshot(overwrite_attempt)

    # 2. Mutating the local state dict after saving does not alter the stored snapshot
    snap.state["score"] = 0.00
    snap.inputs["query"] = "mutated_locally"
    fetched = chk.get_snapshot("th-immut", "snap-immutable-1")
    assert fetched is not None
    assert fetched.state["score"] == 0.85
    assert fetched.inputs["query"] == "dynaflow memory"

    # 3. Mutating the dictionary returned from get_snapshot does not alter the stored snapshot
    fetched.state["tags"].append("mutated_after_get")
    fetched_again = chk.get_snapshot("th-immut", "snap-immutable-1")
    assert fetched_again is not None
    assert "mutated_after_get" not in fetched_again.state["tags"]


def test_memory_checkpointer_time_travel_rewind():
    """Verify time-travel rewind to historical snapshots in MemoryCheckpointer."""
    chk = MemoryCheckpointer()
    thread_id = "th-time-travel"

    # Simulate 3 execution steps
    states = [
        {"stage": "init", "data": "initial"},
        {"stage": "processing", "data": "intermediate"},
        {"stage": "completed", "data": "final_output"},
    ]

    snap_ids = []
    for step, st in enumerate(states):
        snap = Snapshot(
            snapshot_id=f"snap-tt-{step}",
            thread_id=thread_id,
            checkpoint_id=f"cp-tt-{step}",
            step=step,
            state=st,
            inputs={"pipeline": "data_sync"},
            parent_snapshot_id=snap_ids[-1] if snap_ids else None,
        )
        chk.save_snapshot(snap)
        snap_ids.append(snap.snapshot_id)

    # Verify current state is step 2 (completed)
    all_snaps = chk.list_snapshots(thread_id)
    assert len(all_snaps) == 3
    assert all_snaps[-1].state["stage"] == "completed"

    # Time-travel rewind to step 1
    rewound_step1 = chk.get_snapshot(thread_id, "snap-tt-1")
    assert rewound_step1 is not None
    assert rewound_step1.step == 1
    assert rewound_step1.state["stage"] == "processing"
    assert rewound_step1.state["data"] == "intermediate"

    # Time-travel rewind to step 0
    rewound_step0 = chk.get_snapshot(thread_id, "snap-tt-0")
    assert rewound_step0 is not None
    assert rewound_step0.step == 0
    assert rewound_step0.state["stage"] == "init"


def test_memory_checkpointer_fork_from_snapshot():
    """Verify forking a new thread from a snapshot with overridden inputs."""
    chk = MemoryCheckpointer()
    parent_thread = "th-parent"
    child_thread = "th-child-forked"

    # Setup parent thread snapshots
    snap0 = Snapshot(
        snapshot_id="snap-p0",
        thread_id=parent_thread,
        checkpoint_id="cp-p0",
        step=0,
        state={"balance": 100},
        inputs={"currency": "USD", "rate": 1.0},
    )
    chk.save_snapshot(snap0)

    snap1 = Snapshot(
        snapshot_id="snap-p1",
        thread_id=parent_thread,
        checkpoint_id="cp-p1",
        step=1,
        state={"balance": 150, "transactions": [50]},
        inputs={"currency": "USD", "rate": 1.0},
        parent_snapshot_id="snap-p0",
    )
    chk.save_snapshot(snap1)

    # Fork from step 1 into child thread with input overrides
    forked = chk.fork_from_snapshot(
        thread_id=parent_thread,
        snapshot_id="snap-p1",
        new_thread_id=child_thread,
        override_inputs={"rate": 1.25, "fee": 5},
    )

    assert isinstance(forked, Snapshot)
    assert forked.thread_id == child_thread
    assert forked.parent_snapshot_id == "snap-p1"
    assert forked.step == 1
    assert forked.state["balance"] == 150
    assert forked.state["transactions"] == [50]
    # Inputs merged: original currency + overridden rate + new fee
    assert forked.inputs["currency"] == "USD"
    assert forked.inputs["rate"] == 1.25
    assert forked.inputs["fee"] == 5
    assert forked.metadata["forked_from_thread"] == parent_thread
    assert forked.metadata["forked_from_snapshot"] == "snap-p1"

    # Check forked snapshot is persisted in checkpointer under child_thread
    child_snaps = chk.list_snapshots(child_thread)
    assert len(child_snaps) == 1
    assert child_snaps[0].snapshot_id == forked.snapshot_id

    # Check parent thread snapshots remain unchanged
    parent_snaps = chk.list_snapshots(parent_thread)
    assert len(parent_snaps) == 2

    # Forking from non-existent snapshot raises ValueError
    with pytest.raises(ValueError, match="not found"):
        chk.fork_from_snapshot(parent_thread, "non_existent_snap", "th-fail")


# ---------------------------------------------------------------------------
# 3. DynamoDB Checkpointer Snapshot Tests (with Moto)
# ---------------------------------------------------------------------------


def test_dynamodb_checkpointer_snapshot_steps_and_schema():
    """Verify DynamoDBCheckpointer stores snapshots matching single-table pk pattern."""
    with mock_aws():
        table = _create_mock_ddb_table("test_ddb_snapshots_schema")
        chk = DynamoDBCheckpointer(table=table, namespace="prod")

        # Initial checks
        assert chk.get_snapshot("th-ddb-1", "snap-none") is None
        assert chk.list_snapshots("th-ddb-1") == []

        # Create step 0 snapshot
        snap0 = Snapshot(
            snapshot_id="snap-ddb-0",
            thread_id="th-ddb-1",
            checkpoint_id="cp-ddb-0",
            step=0,
            state={"node": "start", "values": [1.5, 2.5]},
            inputs={"param": "value_a"},
        )
        sid0 = chk.save_snapshot(snap0)
        assert sid0 == "snap-ddb-0"

        # Verify exact DynamoDB single-table primary key pattern
        expected_snap_pk = "prod#snapshot#th-ddb-1#snap-ddb-0"
        raw_snap = table.get_item(Key={"pk": expected_snap_pk}).get("Item")
        assert raw_snap is not None
        assert raw_snap["type"] == "snapshot"
        assert raw_snap["snapshot_id"] == "snap-ddb-0"
        assert raw_snap["step"] == 0

        # Verify thread snapshots index item
        expected_index_pk = "prod#snapshots#th-ddb-1"
        raw_index = table.get_item(Key={"pk": expected_index_pk}).get("Item")
        assert raw_index is not None
        assert raw_index["type"] == "thread_snapshots"
        assert raw_index["latest_snapshot_id"] == "snap-ddb-0"

        # Fetch using checkpointer
        fetched0 = chk.get_snapshot("th-ddb-1", "snap-ddb-0")
        assert fetched0 is not None
        assert fetched0.snapshot_id == "snap-ddb-0"
        assert fetched0.state["node"] == "start"
        assert fetched0.state["values"] == [1.5, 2.5]
        assert fetched0.inputs["param"] == "value_a"


def test_dynamodb_checkpointer_list_chronological():
    """Verify DynamoDBCheckpointer lists snapshots chronologically with thread isolation."""
    with mock_aws():
        table = _create_mock_ddb_table("test_ddb_snapshots_list")
        chk = DynamoDBCheckpointer(table=table, namespace="tenant_x")

        # Save 3 steps on thread_1
        for s in range(3):
            chk.save_snapshot(
                Snapshot(
                    snapshot_id=f"snap-ddb-t1-{s}",
                    thread_id="thread_1",
                    checkpoint_id=f"cp-t1-{s}",
                    step=s,
                    state={"step": s},
                    inputs={"env": "prod"},
                )
            )

        # Save 1 step on thread_2
        chk.save_snapshot(
            Snapshot(
                snapshot_id="snap-ddb-t2-0",
                thread_id="thread_2",
                checkpoint_id="cp-t2-0",
                step=0,
                state={"step": 0},
                inputs={"env": "dev"},
            )
        )

        list_t1 = chk.list_snapshots("thread_1")
        assert len(list_t1) == 3
        assert [s.step for s in list_t1] == [0, 1, 2]
        assert [s.snapshot_id for s in list_t1] == [
            "snap-ddb-t1-0",
            "snap-ddb-t1-1",
            "snap-ddb-t1-2",
        ]

        # Verify limit parameter on DynamoDB
        list_t1_limit = chk.list_snapshots("thread_1", limit=2)
        assert len(list_t1_limit) == 2
        assert [s.snapshot_id for s in list_t1_limit] == [
            "snap-ddb-t1-1",
            "snap-ddb-t1-2",
        ]

        list_t2 = chk.list_snapshots("thread_2")
        assert len(list_t2) == 1
        assert list_t2[0].snapshot_id == "snap-ddb-t2-0"


def test_dynamodb_checkpointer_immutability():
    """Verify DynamoDBCheckpointer rejects snapshot overwrites and preserves immutability."""
    with mock_aws():
        table = _create_mock_ddb_table("test_ddb_snapshots_immut")
        chk = DynamoDBCheckpointer(table=table, namespace="ns_sec")

        snap = Snapshot(
            snapshot_id="snap-locked-1",
            thread_id="th-ddb-sec",
            checkpoint_id="cp-1",
            step=1,
            state={"payload": "original_state"},
            inputs={"auth": "bearer-token"},
        )
        chk.save_snapshot(snap)

        # 1. Attempting to save another snapshot with same ID must fail
        attempt = Snapshot(
            snapshot_id="snap-locked-1",
            thread_id="th-ddb-sec",
            checkpoint_id="cp-modified",
            step=2,
            state={"payload": "tampered_state"},
            inputs={"auth": "hacked"},
        )
        with pytest.raises((SnapshotImmutableError, ValueError)):
            chk.save_snapshot(attempt)

        # 2. Stored snapshot remains untouched
        fetched = chk.get_snapshot("th-ddb-sec", "snap-locked-1")
        assert fetched is not None
        assert fetched.state["payload"] == "original_state"
        assert fetched.inputs["auth"] == "bearer-token"


def test_dynamodb_checkpointer_time_travel_rewind():
    """Verify time-travel rewind in DynamoDB-backed checkpointer."""
    with mock_aws():
        table = _create_mock_ddb_table("test_ddb_snapshots_tt")
        chk = DynamoDBCheckpointer(table=table, namespace="tenant_tt")
        thread_id = "th-flow-tt"

        # Save 3 stages
        chk.save_snapshot(
            Snapshot(
                snapshot_id="snap-s0",
                thread_id=thread_id,
                checkpoint_id="cp-0",
                step=0,
                state={"iteration": 0, "status": "pending"},
                inputs={"max_retries": 3},
            )
        )
        chk.save_snapshot(
            Snapshot(
                snapshot_id="snap-s1",
                thread_id=thread_id,
                checkpoint_id="cp-1",
                step=1,
                state={"iteration": 1, "status": "running"},
                inputs={"max_retries": 3},
                parent_snapshot_id="snap-s0",
            )
        )
        chk.save_snapshot(
            Snapshot(
                snapshot_id="snap-s2",
                thread_id=thread_id,
                checkpoint_id="cp-2",
                step=2,
                state={"iteration": 2, "status": "done"},
                inputs={"max_retries": 3},
                parent_snapshot_id="snap-s1",
            )
        )

        # Rewind to step 1
        rewound = chk.get_snapshot(thread_id, "snap-s1")
        assert rewound is not None
        assert rewound.step == 1
        assert rewound.state["status"] == "running"
        assert rewound.parent_snapshot_id == "snap-s0"


def test_dynamodb_checkpointer_fork_from_snapshot():
    """Verify DynamoDBCheckpointer forks from a historical snapshot with input overrides."""
    with mock_aws():
        table = _create_mock_ddb_table("test_ddb_snapshots_fork")
        chk = DynamoDBCheckpointer(table=table, namespace="tenant_fork")

        # Initial thread with 2 steps
        chk.save_snapshot(
            Snapshot(
                snapshot_id="snap-orig-0",
                thread_id="th-orig",
                checkpoint_id="cp-orig-0",
                step=0,
                state={"items_processed": 5},
                inputs={"batch_size": 10, "strategy": "fifo"},
            )
        )
        chk.save_snapshot(
            Snapshot(
                snapshot_id="snap-orig-1",
                thread_id="th-orig",
                checkpoint_id="cp-orig-1",
                step=1,
                state={"items_processed": 15},
                inputs={"batch_size": 10, "strategy": "fifo"},
                parent_snapshot_id="snap-orig-0",
            )
        )

        # Fork from step 1 into a replay branch with different strategy & batch size
        forked = chk.fork_from_snapshot(
            thread_id="th-orig",
            snapshot_id="snap-orig-1",
            new_thread_id="th-forked-branch",
            override_inputs={"strategy": "priority", "batch_size": 25, "notify": True},
        )

        assert forked.thread_id == "th-forked-branch"
        assert forked.parent_snapshot_id == "snap-orig-1"
        assert forked.step == 1
        assert forked.state["items_processed"] == 15
        assert forked.inputs["strategy"] == "priority"
        assert forked.inputs["batch_size"] == 25
        assert forked.inputs["notify"] is True
        assert forked.metadata["forked_from_thread"] == "th-orig"
        assert forked.metadata["forked_from_snapshot"] == "snap-orig-1"

        # Check DynamoDB storage for the forked snapshot
        forked_fetch = chk.get_snapshot("th-forked-branch", forked.snapshot_id)
        assert forked_fetch is not None
        assert forked_fetch.snapshot_id == forked.snapshot_id
        assert forked_fetch.inputs["strategy"] == "priority"

        # Forking from non-existent snapshot raises ValueError
        with pytest.raises(ValueError):
            chk.fork_from_snapshot("th-orig", "snap-missing", "th-forked-fail")


# ---------------------------------------------------------------------------
# 4. ReActAgent Snapshot Integration & Replay
# ---------------------------------------------------------------------------


def test_react_agent_snapshot_capture_and_replay():
    """Verify taking snapshots during ReActAgent run and rewinding/resuming."""
    model = ScriptedChatModel(
        [
            # Step 1
            ChatResult(
                message=Message(
                    role="assistant",
                    content="Computing step 1.",
                    tool_calls=[
                        ToolCall(
                            id="tc-1",
                            name="multiply",
                            arguments='{"a": 6, "b": 7}',
                        )
                    ],
                )
            ),
            # Step 2
            ChatResult(
                message=Message(
                    role="assistant",
                    content="The result is 42.",
                )
            ),
        ]
    )

    def multiply(a: int, b: int) -> int:
        return a * b

    checkpointer = MemoryCheckpointer()
    agent = ReActAgent(
        model=model,
        tools=[multiply],
        checkpointer=checkpointer,
    )

    thread_id = "th-agent-run"
    result = agent.run("Multiply 6 by 7", thread_id=thread_id)
    assert result.finished is True
    assert "42" in result.output

    # Check that checkpointer has recorded checkpoints
    cps = checkpointer.list(thread_id)
    assert len(cps) >= 2

    # Save snapshots corresponding to checkpoints
    snap_step1 = Snapshot(
        snapshot_id="snap-agent-step1",
        thread_id=thread_id,
        checkpoint_id=cps[1].checkpoint_id,
        step=cps[1].step,
        state=cps[1].state,
        inputs={"prompt": "Multiply 6 by 7"},
    )
    checkpointer.save_snapshot(snap_step1)

    # Fork to new thread with modified input parameter
    forked_agent_snap = checkpointer.fork_from_snapshot(
        thread_id=thread_id,
        snapshot_id="snap-agent-step1",
        new_thread_id="th-agent-forked",
        override_inputs={"prompt": "Multiply 6 by 7 with high precision"},
    )

    assert forked_agent_snap.thread_id == "th-agent-forked"
    assert forked_agent_snap.parent_snapshot_id == "snap-agent-step1"
    assert forked_agent_snap.inputs["prompt"] == "Multiply 6 by 7 with high precision"

    # Verify checkpoint forked with a distinct checkpoint ID and points to source checkpoint as parent
    forked_cp = checkpointer.get("th-agent-forked")
    assert forked_cp is not None
    assert forked_cp.checkpoint_id != cps[1].checkpoint_id
    assert forked_cp.parent_checkpoint_id == cps[1].checkpoint_id
    assert forked_cp.state["goal"] == "Multiply 6 by 7 with high precision"

    # Model response for replaying on the forked thread
    model.responses.append(
        ChatResult(
            message=Message(
                role="assistant",
                content="The result is 42.0000 with high precision.",
            )
        )
    )

    # Replay/resume the agent on the forked thread with the new input
    replay_result = agent.run("Multiply 6 by 7 with high precision", thread_id="th-agent-forked")
    assert replay_result.finished is True
    assert "42.0000" in replay_result.output
