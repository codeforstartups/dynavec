"""Conversation and short-term memory store on DynamoDB (dynaflow).

Complements dynavec's long-term semantic vector memory with short-term
conversation history and scratchpad persisted on DynamoDB, supporting
sliding window, token budget, and summary-on-overflow memory policies.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from .chat.base import Message, Role, ToolCall
from .stores.dynamodb import _from_dynamo, _to_dynamo


def message_to_dict(msg: Message) -> dict[str, Any]:
    """Serialize a Message to a dictionary."""
    return {
        "role": msg.role,
        "content": msg.content,
        "tool_call_id": msg.tool_call_id,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls
        ],
    }


def dict_to_message(d: dict[str, Any]) -> Message:
    """Deserialize a dictionary to a Message."""
    tcs = [
        ToolCall(id=t["id"], name=t["name"], arguments=t["arguments"])
        for t in d.get("tool_calls", [])
    ]
    return Message(
        role=d["role"],
        content=d.get("content"),
        tool_calls=tcs,
        tool_call_id=d.get("tool_call_id"),
    )


# --- Memory Policies ---


class MemoryPolicy(ABC):
    """Abstract policy governing message retention, trimming, and compression."""

    @abstractmethod
    def apply(self, messages: list[Message]) -> list[Message]:
        """Apply policy to a list of messages, returning the retained messages."""


class WindowMemoryPolicy(MemoryPolicy):
    """Retains the most recent k non-system messages, optionally preserving system prompt."""

    def __init__(self, k: int = 10, *, preserve_system: bool = True) -> None:
        self.k = max(1, k)
        self.preserve_system = preserve_system

    def apply(self, messages: list[Message]) -> list[Message]:
        if not messages or len(messages) <= self.k:
            return list(messages)

        system_msg: Message | None = None
        other_messages: list[Message] = []

        for m in messages:
            if m.role == "system" and system_msg is None and self.preserve_system:
                system_msg = m
            else:
                other_messages.append(m)

        trimmed = other_messages[-self.k :]
        if system_msg is not None:
            return [system_msg, *trimmed]
        return trimmed


def default_token_counter(msg: Message) -> int:
    """Approximate token count for a Message (1 token ~= 4 chars)."""
    text = msg.content or ""
    if msg.tool_calls:
        for tc in msg.tool_calls:
            text += f" {tc.name} {tc.arguments}"
    return max(1, len(text) // 4)


class TokenBudgetMemoryPolicy(MemoryPolicy):
    """Trims oldest non-system messages to stay within a maximum token budget."""

    def __init__(
        self,
        max_tokens: int = 2048,
        *,
        token_counter: Callable[[Message], int] | None = None,
        preserve_system: bool = True,
    ) -> None:
        self.max_tokens = max(1, max_tokens)
        self.counter = token_counter or default_token_counter
        self.preserve_system = preserve_system

    def apply(self, messages: list[Message]) -> list[Message]:
        if not messages:
            return []

        system_msg: Message | None = None
        remaining: list[Message] = []

        for m in messages:
            if m.role == "system" and system_msg is None and self.preserve_system:
                system_msg = m
            else:
                remaining.append(m)

        sys_cost = self.counter(system_msg) if system_msg is not None else 0
        budget = max(0, self.max_tokens - sys_cost)

        retained: list[Message] = []
        current_tokens = 0

        # Traverse backwards from newest to oldest
        for m in reversed(remaining):
            cost = self.counter(m)
            if current_tokens + cost <= budget or not retained:
                retained.append(m)
                current_tokens += cost
            else:
                break

        retained.reverse()
        if system_msg is not None:
            return [system_msg, *retained]
        return retained


class SummaryMemoryPolicy(MemoryPolicy):
    """Compacts older messages into a summary message when exceeding max_messages."""

    def __init__(
        self,
        max_messages: int = 6,
        *,
        summarizer: Callable[[list[Message]], str] | None = None,
        preserve_system: bool = True,
    ) -> None:
        self.max_messages = max(2, max_messages)
        self.summarizer = summarizer or self._default_summary
        self.preserve_system = preserve_system

    def _default_summary(self, messages: list[Message]) -> str:
        points = []
        for m in messages:
            content = (m.content or "").strip()
            if len(content) > 120:
                content = content[:117] + "..."
            points.append(f"- {m.role.capitalize()}: {content}")
        return "\n".join(points)

    def apply(self, messages: list[Message]) -> list[Message]:
        if len(messages) <= self.max_messages:
            return list(messages)

        system_msg: Message | None = None
        others: list[Message] = []

        for m in messages:
            if m.role == "system" and system_msg is None and self.preserve_system:
                system_msg = m
            else:
                others.append(m)

        cutoff = len(others) - (self.max_messages // 2)
        to_summarize = others[:cutoff]
        recent = others[cutoff:]

        summary_text = f"Summary of prior conversation:\n{self.summarizer(to_summarize)}"
        summary_msg = Message(role="system", content=summary_text)

        result: list[Message] = []
        if system_msg is not None:
            result.append(system_msg)
        result.append(summary_msg)
        result.extend(recent)
        return result


# --- Memory Stores ---


class BaseMemory(ABC):
    """Abstract interface for managing agent short-term message memory."""

    @abstractmethod
    def append(
        self,
        thread_id: str,
        messages: Message | Sequence[Message] | str,
        *,
        role: Role = "user",
    ) -> list[Message]:
        """Append one or more messages to the thread history."""

    @abstractmethod
    def get_messages(self, thread_id: str) -> list[Message]:
        """Retrieve the current message history for a thread."""

    @abstractmethod
    def clear(self, thread_id: str) -> None:
        """Clear message history and scratchpad for a thread."""

    @abstractmethod
    def trim(self, thread_id: str) -> list[Message]:
        """Apply memory policy and save trimmed history."""

    @abstractmethod
    def get_scratchpad(self, thread_id: str) -> dict[str, Any]:
        """Retrieve the thread scratchpad dictionary."""

    @abstractmethod
    def set_scratchpad(self, thread_id: str, scratchpad: dict[str, Any]) -> None:
        """Set the thread scratchpad dictionary."""


class InMemoryMemoryStore(BaseMemory):
    """Thread-safe, in-memory conversation memory store for tests and local agents."""

    def __init__(self, policy: MemoryPolicy | None = None) -> None:
        self.policy = policy
        self._messages: dict[str, list[Message]] = {}
        self._scratchpads: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _normalize(
        self,
        messages: Message | Sequence[Message] | str,
        role: Role,
    ) -> list[Message]:
        if isinstance(messages, str):
            return [Message(role=role, content=messages)]
        if isinstance(messages, Message):
            return [messages]
        return list(messages)

    def append(
        self,
        thread_id: str,
        messages: Message | Sequence[Message] | str,
        *,
        role: Role = "user",
    ) -> list[Message]:
        new_msgs = self._normalize(messages, role)
        with self._lock:
            history = self._messages.setdefault(thread_id, [])
            history.extend(new_msgs)
            if self.policy is not None:
                history = self.policy.apply(history)
                self._messages[thread_id] = history
            return list(history)

    def get_messages(self, thread_id: str) -> list[Message]:
        with self._lock:
            history = self._messages.get(thread_id, [])
            if self.policy is not None:
                return self.policy.apply(list(history))
            return list(history)

    def clear(self, thread_id: str) -> None:
        with self._lock:
            self._messages.pop(thread_id, None)
            self._scratchpads.pop(thread_id, None)

    def trim(self, thread_id: str) -> list[Message]:
        with self._lock:
            history = self._messages.get(thread_id, [])
            if self.policy is not None:
                history = self.policy.apply(history)
                self._messages[thread_id] = history
            return list(history)

    def get_scratchpad(self, thread_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._scratchpads.get(thread_id, {}))

    def set_scratchpad(self, thread_id: str, scratchpad: dict[str, Any]) -> None:
        with self._lock:
            self._scratchpads[thread_id] = dict(scratchpad)


class DynamoDBMemoryStore(BaseMemory):
    """DynamoDB-backed conversation and scratchpad store.

    Stores conversation state per thread using single-table schema:
      - pk: "{namespace}#memory#{thread_id}"
      - messages: serialized message history
      - scratchpad: agent working memory dict
      - updated_at: ISO UTC timestamp
    """

    def __init__(
        self,
        table: Any,
        *,
        namespace: str = "default",
        policy: MemoryPolicy | None = None,
    ) -> None:
        self.table = table
        self.namespace = namespace
        self.policy = policy

    def _pk(self, thread_id: str) -> str:
        return f"{self.namespace}#memory#{thread_id}"

    def _normalize(
        self,
        messages: Message | Sequence[Message] | str,
        role: Role,
    ) -> list[Message]:
        if isinstance(messages, str):
            return [Message(role=role, content=messages)]
        if isinstance(messages, Message):
            return [messages]
        return list(messages)

    def get_messages(self, thread_id: str) -> list[Message]:
        """Fetch conversation message history for thread_id."""
        resp = self.table.get_item(Key={"pk": self._pk(thread_id)})
        item = resp.get("Item")
        if not item or "messages" not in item:
            return []

        clean_data = _from_dynamo(item["messages"])
        messages = [dict_to_message(m) for m in clean_data]
        if self.policy is not None:
            return self.policy.apply(messages)
        return messages

    def append(
        self,
        thread_id: str,
        messages: Message | Sequence[Message] | str,
        *,
        role: Role = "user",
    ) -> list[Message]:
        """Append messages to thread history and persist to DynamoDB."""
        to_add = self._normalize(messages, role)
        current = self.get_messages(thread_id)
        current.extend(to_add)

        if self.policy is not None:
            current = self.policy.apply(current)

        serialized = [message_to_dict(m) for m in current]
        pk = self._pk(thread_id)

        item: dict[str, Any] = {
            "pk": pk,
            "ns": self.namespace,
            "type": "conversation_memory",
            "thread_id": thread_id,
            "messages": serialized,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Preserve existing scratchpad if already written
        try:
            resp = self.table.get_item(Key={"pk": pk})
            existing = resp.get("Item")
            if existing and "scratchpad" in existing:
                item["scratchpad"] = existing["scratchpad"]
        except Exception:
            pass

        self.table.put_item(Item=_to_dynamo(item))
        return current

    def trim(self, thread_id: str) -> list[Message]:
        """Apply memory policy and save trimmed history."""
        messages = self.get_messages(thread_id)
        if self.policy is not None:
            messages = self.policy.apply(messages)

        serialized = [message_to_dict(m) for m in messages]
        pk = self._pk(thread_id)

        self.table.update_item(
            Key={"pk": pk},
            UpdateExpression="SET #m = :m, updated_at = :ts",
            ExpressionAttributeNames={"#m": "messages"},
            ExpressionAttributeValues={
                ":m": _to_dynamo(serialized),
                ":ts": datetime.now(timezone.utc).isoformat(),
            },
        )
        return messages

    def clear(self, thread_id: str) -> None:
        """Clear memory for a thread."""
        self.table.delete_item(Key={"pk": self._pk(thread_id)})

    def get_scratchpad(self, thread_id: str) -> dict[str, Any]:
        """Retrieve scratchpad data for thread_id."""
        resp = self.table.get_item(Key={"pk": self._pk(thread_id)})
        item = resp.get("Item")
        if not item or "scratchpad" not in item:
            return {}
        return dict(_from_dynamo(item["scratchpad"]))

    def set_scratchpad(self, thread_id: str, scratchpad: dict[str, Any]) -> None:
        """Update scratchpad data for thread_id."""
        pk = self._pk(thread_id)
        self.table.update_item(
            Key={"pk": pk},
            UpdateExpression="SET scratchpad = :s, updated_at = :ts",
            ExpressionAttributeValues={
                ":s": _to_dynamo(scratchpad),
                ":ts": datetime.now(timezone.utc).isoformat(),
            },
        )
