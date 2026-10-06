"""Run and thread state checkpointer for agent orchestration (dynaflow).

Provides persistent snapshots and time-travel resumability for agent runs
keyed by thread_id and checkpoint_id. Supports both an in-memory checkpointer
for local execution/testing and a serverless DynamoDB-backed checkpointer
for durable, in-account agent execution.
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .stores.dynamodb import _from_dynamo, _to_dynamo


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
    """Abstract interface for storing and retrieving thread checkpoints."""

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        """Persist a checkpoint for a thread."""

    @abstractmethod
    def get(self, thread_id: str, checkpoint_id: str | None = None) -> Checkpoint | None:
        """Load the latest checkpoint for a thread, or a specific checkpoint by ID."""

    @abstractmethod
    def list(self, thread_id: str, limit: int | None = None) -> list[Checkpoint]:
        """List checkpoints for a thread in chronological order."""

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

    def clear(self, thread_id: str | None = None) -> None:
        """Clear all checkpoints, or checkpoints for a specific thread."""
        with self._lock:
            if thread_id is not None:
                self._threads.pop(thread_id, None)
            else:
                self._threads.clear()


class DynamoDBCheckpointer(BaseCheckpointer):
    """DynamoDB-backed persistent checkpointer.

    Compatible with dynavec's single-table schema:
      - Individual checkpoint: pk = "{namespace}#checkpoint#{thread_id}#{checkpoint_id}"
      - Thread manifest: pk = "{namespace}#thread#{thread_id}"
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
