"""Run and thread state checkpointer for agent orchestration (dynaflow).

Provides persistent snapshots and time-travel resumability for agent runs
keyed by thread_id and checkpoint_id. Supports both an in-memory checkpointer
for local execution/testing and a serverless DynamoDB-backed checkpointer
for durable, in-account agent execution.
"""

from __future__ import annotations

import copy
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from botocore.exceptions import ClientError

from .stores.dynamodb import _from_dynamo, _to_dynamo


class SnapshotImmutableError(ValueError):
    """Raised when attempting to modify or overwrite an immutable historical snapshot."""


@dataclass(frozen=True)
class Snapshot:
    """An immutable point-in-time snapshot of thread state and inputs for replay and time travel."""

    snapshot_id: str = field(
        default_factory=lambda: (
            f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
    )
    thread_id: str = ""
    checkpoint_id: str = ""
    step: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    parent_snapshot_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            object.__setattr__(
                self,
                "snapshot_id",
                f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}",
            )
        if not self.timestamp:
            object.__setattr__(
                self,
                "timestamp",
                datetime.now(timezone.utc).isoformat(),
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to a dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "thread_id": self.thread_id,
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "state": copy.deepcopy(self.state),
            "inputs": copy.deepcopy(self.inputs),
            "parent_snapshot_id": self.parent_snapshot_id,
            "timestamp": self.timestamp,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        """Construct a Snapshot from a dictionary."""
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            thread_id=str(data["thread_id"]),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            step=int(data.get("step", 0)),
            state=dict(data.get("state") or {}),
            inputs=dict(data.get("inputs") or {}),
            parent_snapshot_id=data.get("parent_snapshot_id"),
            timestamp=str(data.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Checkpoint:
    """A point-in-time snapshot of an agent or workflow run state."""

    thread_id: str
    checkpoint_id: str = field(
        default_factory=lambda: (
            f"cp_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
    )
    parent_checkpoint_id: str | None = None
    node_id: str | None = None
    step: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to a dictionary."""
        return {
            "thread_id": self.thread_id,
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "node_id": self.node_id,
            "step": self.step,
            "state": self.state,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Construct a Checkpoint from a dictionary."""
        return cls(
            thread_id=data["thread_id"],
            checkpoint_id=data["checkpoint_id"],
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            node_id=data.get("node_id"),
            step=int(data.get("step", 0)),
            state=dict(data.get("state") or {}),
            metadata=dict(data.get("metadata") or {}),
            timestamp=data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        )


class BaseCheckpointer(ABC):
    """Abstract interface for storing and retrieving thread checkpoints and snapshots."""

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        """Persist a checkpoint for a thread."""

    @abstractmethod
    def get(self, thread_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        """Load the latest checkpoint for a thread, or a specific checkpoint by ID."""

    @abstractmethod
    def list(self, thread_id: str, limit: int | None = None) -> list[Checkpoint]:
        """List checkpoints for a thread in chronological order."""

    @abstractmethod
    def save_snapshot(self, snapshot: Snapshot) -> str:
        """Persist an immutable snapshot for a thread and return its snapshot_id."""

    @abstractmethod
    def get_snapshot(self, thread_id: str, snapshot_id: str) -> Snapshot | None:
        """Load a specific snapshot by thread ID and snapshot ID."""

    @abstractmethod
    def list_snapshots(
        self, thread_id: str, limit: int | None = None
    ) -> Sequence[Snapshot]:
        """List all snapshots for a thread in chronological order."""

    def fork_from_snapshot(
        self,
        thread_id: str,
        snapshot_id: str,
        new_thread_id: str,
        override_inputs: dict[str, Any] | None = None,
    ) -> Snapshot:
        """Fork a new thread from a historical snapshot with optional overridden inputs."""
        source = self.get_snapshot(thread_id, snapshot_id)
        if source is None:
            raise ValueError(f"Snapshot '{snapshot_id}' not found for thread '{thread_id}'")

        merged_inputs = copy.deepcopy(source.inputs)
        if override_inputs is not None:
            merged_inputs.update(override_inputs)

        meta = copy.deepcopy(source.metadata)
        meta["forked_from_thread"] = thread_id
        meta["forked_from_snapshot"] = snapshot_id

        # Forked thread must generate its own unique checkpoint ID pointing to source checkpoint as parent
        new_cp_id = f"cp_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        forked_state = copy.deepcopy(source.state)
        if override_inputs is not None:
            if "inputs" in forked_state and isinstance(forked_state["inputs"], dict):
                forked_state["inputs"].update(override_inputs)
            else:
                forked_state["inputs"] = copy.deepcopy(merged_inputs)

            new_prompt = override_inputs.get("prompt") or override_inputs.get("goal")
            if new_prompt:
                forked_state["goal"] = new_prompt
                if "messages" in forked_state and isinstance(forked_state["messages"], list):
                    for m in forked_state["messages"]:
                        if isinstance(m, dict) and m.get("role") == "user":
                            m["content"] = new_prompt
                            break
            for k, v in override_inputs.items():
                if k not in ("prompt", "goal", "messages"):
                    forked_state[k] = copy.deepcopy(v)
                elif k == "messages":
                    forked_state["messages"] = copy.deepcopy(v)

        self.save(
            Checkpoint(
                thread_id=new_thread_id,
                checkpoint_id=new_cp_id,
                parent_checkpoint_id=source.checkpoint_id or None,
                step=source.step,
                state=copy.deepcopy(forked_state),
                metadata=copy.deepcopy(meta),
            )
        )

        forked = Snapshot(
            thread_id=new_thread_id,
            checkpoint_id=new_cp_id,
            step=source.step,
            state=copy.deepcopy(forked_state),
            inputs=merged_inputs,
            parent_snapshot_id=source.snapshot_id,
            metadata=meta,
        )
        self.save_snapshot(forked)
        return forked

    def put(
        self,
        thread_id: str,
        state: dict[str, Any],
        *,
        node_id: str | None = None,
        step: int | None = None,
        parent_checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Convenience method to construct and save a new checkpoint."""
        if parent_checkpoint_id is None or step is None:
            latest = self.get(thread_id)
            if latest is not None:
                if parent_checkpoint_id is None:
                    parent_checkpoint_id = latest.checkpoint_id
                if step is None:
                    step = latest.step + 1
            else:
                if step is None:
                    step = 0

        cp = Checkpoint(
            thread_id=thread_id,
            parent_checkpoint_id=parent_checkpoint_id,
            node_id=node_id,
            step=step,
            state=state,
            metadata=metadata or {},
        )
        return self.save(cp)


class MemoryCheckpointer(BaseCheckpointer):
    """Thread-safe, in-memory checkpointer for local runs and unit testing."""

    def __init__(self) -> None:
        self._threads: dict[str, list[Checkpoint]] = {}
        self._snapshots: dict[str, list[Snapshot]] = {}
        self._lock = threading.Lock()

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        with self._lock:
            history = self._threads.setdefault(checkpoint.thread_id, [])
            history.append(checkpoint)
        return checkpoint

    def get(self, thread_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        with self._lock:
            history = self._threads.get(thread_id, [])
            if not history:
                return None
            if checkpoint_id is None:
                return history[-1]
            for cp in reversed(history):
                if cp.checkpoint_id == checkpoint_id:
                    return cp
            return None

    def list(self, thread_id: str, limit: int | None = None) -> list[Checkpoint]:
        with self._lock:
            history = list(self._threads.get(thread_id, []))
            if limit is not None:
                history = history[-limit:]
            return history

    def save_snapshot(self, snapshot: Snapshot) -> str:
        with self._lock:
            history = self._snapshots.setdefault(snapshot.thread_id, [])
            if any(s.snapshot_id == snapshot.snapshot_id for s in history):
                raise SnapshotImmutableError(
                    f"Snapshot '{snapshot.snapshot_id}' already exists and is immutable."
                )
            snap_to_store = Snapshot(
                snapshot_id=snapshot.snapshot_id,
                thread_id=snapshot.thread_id,
                checkpoint_id=snapshot.checkpoint_id,
                step=snapshot.step,
                state=copy.deepcopy(snapshot.state),
                inputs=copy.deepcopy(snapshot.inputs),
                parent_snapshot_id=snapshot.parent_snapshot_id,
                timestamp=snapshot.timestamp,
                metadata=copy.deepcopy(snapshot.metadata),
            )
            history.append(snap_to_store)
            return snap_to_store.snapshot_id

    def get_snapshot(self, thread_id: str, snapshot_id: str) -> Snapshot | None:
        with self._lock:
            history = self._snapshots.get(thread_id, [])
            for s in history:
                if s.snapshot_id == snapshot_id:
                    return Snapshot(
                        snapshot_id=s.snapshot_id,
                        thread_id=s.thread_id,
                        checkpoint_id=s.checkpoint_id,
                        step=s.step,
                        state=copy.deepcopy(s.state),
                        inputs=copy.deepcopy(s.inputs),
                        parent_snapshot_id=s.parent_snapshot_id,
                        timestamp=s.timestamp,
                        metadata=copy.deepcopy(s.metadata),
                    )
            return None

    def list_snapshots(
        self, thread_id: str, limit: int | None = None
    ) -> Sequence[Snapshot]:
        with self._lock:
            history = list(self._snapshots.get(thread_id, []))
            if limit is not None:
                history = history[-limit:]
            return [
                Snapshot(
                    snapshot_id=s.snapshot_id,
                    thread_id=s.thread_id,
                    checkpoint_id=s.checkpoint_id,
                    step=s.step,
                    state=copy.deepcopy(s.state),
                    inputs=copy.deepcopy(s.inputs),
                    parent_snapshot_id=s.parent_snapshot_id,
                    timestamp=s.timestamp,
                    metadata=copy.deepcopy(s.metadata),
                )
                for s in history
            ]

    def clear(self, thread_id: str | None = None) -> None:
        """Clear all checkpoints, or checkpoints for a specific thread."""
        with self._lock:
            if thread_id is not None:
                self._threads.pop(thread_id, None)
                self._snapshots.pop(thread_id, None)
            else:
                self._threads.clear()
                self._snapshots.clear()


class DynamoDBCheckpointer(BaseCheckpointer):
    """DynamoDB-backed persistent checkpointer.

    Compatible with dynavec's single-table schema:
      - Individual checkpoint: pk = "{namespace}#checkpoint#{thread_id}#{checkpoint_id}"
      - Thread manifest: pk = "{namespace}#thread#{thread_id}"
      - Individual snapshot: pk = "{namespace}#snapshot#{thread_id}#{snapshot_id}"
      - Thread snapshots index: pk = "{namespace}#snapshots#{thread_id}"
    """

    def __init__(
        self,
        table: Any,
        *,
        namespace: str = "default",
    ) -> None:
        self.table = table
        self.namespace = namespace

    def _cp_pk(self, thread_id: str, checkpoint_id: str) -> str:
        return f"{self.namespace}#checkpoint#{thread_id}#{checkpoint_id}"

    def _thread_pk(self, thread_id: str) -> str:
        return f"{self.namespace}#thread#{thread_id}"

    def _snapshot_pk(self, thread_id: str, snapshot_id: str) -> str:
        return f"{self.namespace}#snapshot#{thread_id}#{snapshot_id}"

    def _snapshots_index_pk(self, thread_id: str) -> str:
        return f"{self.namespace}#snapshots#{thread_id}"

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        """Save a checkpoint item and update the thread manifest in DynamoDB."""
        item: dict[str, Any] = {
            "pk": self._cp_pk(checkpoint.thread_id, checkpoint.checkpoint_id),
            "ns": self.namespace,
            "type": "checkpoint",
            "thread_id": checkpoint.thread_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
            "node_id": checkpoint.node_id,
            "step": checkpoint.step,
            "state": checkpoint.state,
            "metadata": checkpoint.metadata,
            "timestamp": checkpoint.timestamp,
        }
        self.table.put_item(Item=_to_dynamo(item))

        # Update thread manifest
        manifest_pk = self._thread_pk(checkpoint.thread_id)
        summary = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
            "node_id": checkpoint.node_id,
            "step": checkpoint.step,
            "timestamp": checkpoint.timestamp,
        }

        try:
            self.table.update_item(
                Key={"pk": manifest_pk},
                UpdateExpression=(
                    "SET latest_checkpoint_id = :cid, "
                    "updated_at = :ts, "
                    "#tp = :type, "
                    "history = list_append(if_not_exists(history, :empty_list), :new_entry)"
                ),
                ExpressionAttributeNames={"#tp": "type"},
                ExpressionAttributeValues={
                    ":cid": checkpoint.checkpoint_id,
                    ":ts": checkpoint.timestamp,
                    ":type": "thread_manifest",
                    ":empty_list": [],
                    ":new_entry": [_to_dynamo(summary)],
                },
            )
        except Exception:
            # Fallback for mock environments without full update_item list_append support
            resp = self.table.get_item(Key={"pk": manifest_pk})
            manifest = resp.get("Item") or {
                "pk": manifest_pk,
                "ns": self.namespace,
                "type": "thread_manifest",
                "thread_id": checkpoint.thread_id,
                "history": [],
            }
            manifest["latest_checkpoint_id"] = checkpoint.checkpoint_id
            manifest["updated_at"] = checkpoint.timestamp
            history = list(manifest.get("history") or [])
            history.append(_to_dynamo(summary))
            manifest["history"] = history
            self.table.put_item(Item=_to_dynamo(manifest))

        return checkpoint

    def get(self, thread_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        """Fetch the latest checkpoint for a thread or a specific checkpoint by ID."""
        target_cid = checkpoint_id
        if target_cid is None:
            manifest_resp = self.table.get_item(Key={"pk": self._thread_pk(thread_id)})
            manifest_item = manifest_resp.get("Item")
            if not manifest_item or "latest_checkpoint_id" not in manifest_item:
                return None
            target_cid = str(manifest_item["latest_checkpoint_id"])

        cp_resp = self.table.get_item(Key={"pk": self._cp_pk(thread_id, target_cid)})
        cp_item = cp_resp.get("Item")
        if not cp_item:
            return None

        clean = _from_dynamo(cp_item)
        return Checkpoint.from_dict(clean)

    def list(self, thread_id: str, limit: int | None = None) -> list[Checkpoint]:
        """List checkpoints for a thread in chronological order."""
        manifest_resp = self.table.get_item(Key={"pk": self._thread_pk(thread_id)})
        manifest_item = manifest_resp.get("Item")
        if not manifest_item:
            return []

        history = _from_dynamo(manifest_item.get("history", []))
        if not history:
            return []

        if limit is not None:
            history = history[-limit:]

        checkpoints: list[Checkpoint] = []
        for entry in history:
            cid = entry.get("checkpoint_id")
            if cid:
                cp = self.get(thread_id, cid)
                if cp:
                    checkpoints.append(cp)

        return checkpoints

    def save_snapshot(self, snapshot: Snapshot) -> str:
        """Persist an immutable snapshot item and update thread snapshot index in DynamoDB."""
        item: dict[str, Any] = {
            "pk": self._snapshot_pk(snapshot.thread_id, snapshot.snapshot_id),
            "ns": self.namespace,
            "type": "snapshot",
            "snapshot_id": snapshot.snapshot_id,
            "thread_id": snapshot.thread_id,
            "checkpoint_id": snapshot.checkpoint_id,
            "step": snapshot.step,
            "state": snapshot.state,
            "inputs": snapshot.inputs,
            "parent_snapshot_id": snapshot.parent_snapshot_id,
            "timestamp": snapshot.timestamp,
            "metadata": snapshot.metadata,
        }

        try:
            self.table.put_item(
                Item=_to_dynamo(item),
                ConditionExpression="attribute_not_exists(pk)",
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise SnapshotImmutableError(
                    f"Snapshot '{snapshot.snapshot_id}' already exists and is immutable."
                ) from e
            raise

        index_pk = self._snapshots_index_pk(snapshot.thread_id)
        summary = {
            "snapshot_id": snapshot.snapshot_id,
            "parent_snapshot_id": snapshot.parent_snapshot_id,
            "checkpoint_id": snapshot.checkpoint_id,
            "step": snapshot.step,
            "timestamp": snapshot.timestamp,
        }

        try:
            self.table.update_item(
                Key={"pk": index_pk},
                UpdateExpression=(
                    "SET latest_snapshot_id = :sid, "
                    "updated_at = :ts, "
                    "#tp = :type, "
                    "snapshots = list_append(if_not_exists(snapshots, :empty_list), :new_entry)"
                ),
                ExpressionAttributeNames={"#tp": "type"},
                ExpressionAttributeValues={
                    ":sid": snapshot.snapshot_id,
                    ":ts": snapshot.timestamp,
                    ":type": "thread_snapshots",
                    ":empty_list": [],
                    ":new_entry": [_to_dynamo(summary)],
                },
            )
        except Exception:
            resp = self.table.get_item(Key={"pk": index_pk})
            index_item = resp.get("Item") or {
                "pk": index_pk,
                "ns": self.namespace,
                "type": "thread_snapshots",
                "thread_id": snapshot.thread_id,
                "snapshots": [],
            }
            index_item["latest_snapshot_id"] = snapshot.snapshot_id
            index_item["updated_at"] = snapshot.timestamp
            snapshots_list = list(index_item.get("snapshots") or [])
            snapshots_list.append(_to_dynamo(summary))
            index_item["snapshots"] = snapshots_list
            self.table.put_item(Item=_to_dynamo(index_item))

        return snapshot.snapshot_id

    def get_snapshot(self, thread_id: str, snapshot_id: str) -> Snapshot | None:
        """Fetch a specific snapshot by thread ID and snapshot ID."""
        resp = self.table.get_item(
            Key={"pk": self._snapshot_pk(thread_id, snapshot_id)}
        )
        item = resp.get("Item")
        if not item:
            return None

        clean = _from_dynamo(item)
        return Snapshot.from_dict(clean)

    def list_snapshots(
        self, thread_id: str, limit: int | None = None
    ) -> Sequence[Snapshot]:
        """List snapshots for a thread in chronological order."""
        resp = self.table.get_item(Key={"pk": self._snapshots_index_pk(thread_id)})
        item = resp.get("Item")
        if not item:
            return []

        entries = _from_dynamo(item.get("snapshots", []))
        if not entries:
            return []

        if limit is not None:
            entries = entries[-limit:]

        snapshots: list[Snapshot] = []
        for entry in entries:
            sid = entry.get("snapshot_id") if isinstance(entry, dict) else entry
            if sid:
                snap = self.get_snapshot(thread_id, str(sid))
                if snap:
                    snapshots.append(snap)

        return snapshots
