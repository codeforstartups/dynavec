"""Unit tests for conversation and short-term memory store on DynamoDB (Issue #290)."""

from __future__ import annotations

import boto3
from moto import mock_aws

from dynavec.agents import ReActAgent
from dynavec.chat.base import ChatModel, ChatResult, Message, Tool, ToolCall
from dynavec.memory import (
    BaseMemory,
    DynamoDBMemoryStore,
    InMemoryMemoryStore,
    SummaryMemoryPolicy,
    TokenBudgetMemoryPolicy,
    WindowMemoryPolicy,
    dict_to_message,
    message_to_dict,
)


class ScriptedChatModel(ChatModel):
    """Deterministic test double for ChatModel."""

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
                message=Message(role="assistant", content="Done."),
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


def test_message_serialization_round_trip():
    """Verify message dictionary serialization and deserialization."""
    msg = Message(
        role="assistant",
        content="Calculated result.",
        tool_calls=[ToolCall(id="tc-1", name="add", arguments='{"a": 1, "b": 2}')],
        tool_call_id=None,
    )

    d = message_to_dict(msg)
    assert d["role"] == "assistant"
    assert d["content"] == "Calculated result."
    assert len(d["tool_calls"]) == 1
    assert d["tool_calls"][0]["name"] == "add"

    restored = dict_to_message(d)
    assert restored.role == msg.role
    assert restored.content == msg.content
    assert len(restored.tool_calls) == 1
    assert restored.tool_calls[0].arguments == '{"a": 1, "b": 2}'


def test_window_memory_policy():
    """Verify WindowMemoryPolicy retains last k messages while preserving system message."""
    messages = [
        Message(role="system", content="System instruction"),
        Message(role="user", content="msg 1"),
        Message(role="assistant", content="msg 2"),
        Message(role="user", content="msg 3"),
        Message(role="assistant", content="msg 4"),
    ]

    policy = WindowMemoryPolicy(k=2, preserve_system=True)
    trimmed = policy.apply(messages)

    assert len(trimmed) == 3
    assert trimmed[0].role == "system"
    assert trimmed[0].content == "System instruction"
    assert trimmed[1].content == "msg 3"
    assert trimmed[2].content == "msg 4"


def test_token_budget_memory_policy():
    """Verify TokenBudgetMemoryPolicy trims history to remain under token budget."""
    messages = [
        Message(role="system", content="System"),  # ~1-2 tokens
        Message(role="user", content="A" * 80),  # ~20 tokens
        Message(role="assistant", content="B" * 80),  # ~20 tokens
        Message(role="user", content="C" * 20),  # ~5 tokens
    ]

    # Limit budget so only newest user message and system fit
    policy = TokenBudgetMemoryPolicy(max_tokens=15, preserve_system=True)
    trimmed = policy.apply(messages)

    assert len(trimmed) == 2
    assert trimmed[0].role == "system"
    assert trimmed[1].content == "C" * 20


def test_summary_memory_policy():
    """Verify SummaryMemoryPolicy compacts overflow messages into a summary block."""
    messages = [
        Message(role="system", content="You are a helper."),
        Message(role="user", content="What is DynamoDB?"),
        Message(role="assistant", content="It is a NoSQL database."),
        Message(role="user", content="What is S3?"),
        Message(role="assistant", content="It is an object storage."),
        Message(role="user", content="What is dynavec?"),
        Message(role="assistant", content="A serverless vector database."),
    ]

    policy = SummaryMemoryPolicy(max_messages=4, preserve_system=True)
    compacted = policy.apply(messages)

    assert len(compacted) < len(messages)
    assert compacted[0].role == "system"
    assert compacted[0].content == "You are a helper."
    # Second message is the summary
    assert compacted[1].role == "system"
    assert "Summary of prior conversation:" in compacted[1].content
    assert "DynamoDB" in compacted[1].content
    # Tail messages preserved
    assert compacted[-1].content == "A serverless vector database."


def test_in_memory_memory_store():
    """Verify InMemoryMemoryStore append, retrieval, scratchpad, and clear."""
    store = InMemoryMemoryStore(policy=WindowMemoryPolicy(k=2))
    assert isinstance(store, BaseMemory)

    store.append("th-1", "Hello")
    store.append("th-1", Message(role="assistant", content="Hi! How can I help?"))
    store.append("th-1", "Tell me about AWS.")

    msgs = store.get_messages("th-1")
    assert len(msgs) == 2
    assert msgs[0].content == "Hi! How can I help?"
    assert msgs[1].content == "Tell me about AWS."

    # Scratchpad
    store.set_scratchpad("th-1", {"current_step": 3, "intent": "cloud_query"})
    sp = store.get_scratchpad("th-1")
    assert sp["intent"] == "cloud_query"

    store.clear("th-1")
    assert store.get_messages("th-1") == []
    assert store.get_scratchpad("th-1") == {}


def test_dynamodb_memory_store_moto():
    """Verify DynamoDBMemoryStore with mocked AWS DynamoDB table."""
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        table = ddb.create_table(
            TableName="test_agent_memory",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        store = DynamoDBMemoryStore(
            table=table,
            namespace="prod_app",
            policy=WindowMemoryPolicy(k=3, preserve_system=True),
        )

        # Initially empty
        assert store.get_messages("th-ddb-1") == []

        # Append system and conversation messages
        store.append("th-ddb-1", Message(role="system", content="You are a tutor."))
        store.append("th-ddb-1", "Explain vectors.")
        store.append(
            "th-ddb-1",
            Message(role="assistant", content="Vectors are mathematical lists of numbers."),
        )
        store.append("th-ddb-1", "How are they stored?")
        store.append("th-ddb-1", Message(role="assistant", content="In S3 Vectors and DynamoDB."))

        # Retrieve with window policy (k=3 non-system + 1 system)
        messages = store.get_messages("th-ddb-1")
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].content == "Vectors are mathematical lists of numbers."
        assert messages[2].content == "How are they stored?"
        assert messages[3].content == "In S3 Vectors and DynamoDB."

        # Scratchpad
        store.set_scratchpad("th-ddb-1", {"topic": "math", "progress": 0.5, "attempts": 2})
        sp = store.get_scratchpad("th-ddb-1")
        assert sp["topic"] == "math"
        assert sp["progress"] == 0.5

        # Multi-thread isolation
        store.append("th-ddb-2", "Different thread message")
        assert len(store.get_messages("th-ddb-2")) == 1
        assert len(store.get_messages("th-ddb-1")) == 4

        # Clear
        store.clear("th-ddb-2")
        assert store.get_messages("th-ddb-2") == []
        assert len(store.get_messages("th-ddb-1")) == 4


def test_react_agent_memory_multi_turn():
    """Verify ReActAgent maintains multi-turn conversation memory across multiple runs."""
    model = ScriptedChatModel(
        [
            # Turn 1 response
            ChatResult(message=Message(role="assistant", content="Nice to meet you, Alice!")),
            # Turn 2 response
            ChatResult(message=Message(role="assistant", content="Your name is Alice.")),
        ]
    )

    memory = InMemoryMemoryStore()
    agent = ReActAgent(
        model=model,
        system_prompt="Helpful assistant.",
        memory=memory,
    )

    # Turn 1
    res1 = agent.run("My name is Alice.", thread_id="thread-conv-1")
    assert res1.output == "Nice to meet you, Alice!"

    # Verify memory recorded turn 1 (system + user + assistant)
    msgs_t1 = memory.get_messages("thread-conv-1")
    assert len(msgs_t1) == 3
    assert msgs_t1[0].role == "system"
    assert msgs_t1[1].content == "My name is Alice."
    assert msgs_t1[2].content == "Nice to meet you, Alice!"

    # Turn 2
    res2 = agent.run("What is my name?", thread_id="thread-conv-1")
    assert res2.output == "Your name is Alice."

    # Verify agent received prior turn history in turn 2
    assert len(model.received_messages) == 2
    t2_messages = model.received_messages[1]
    assert len(t2_messages) == 4
    assert t2_messages[1].content == "My name is Alice."
    assert t2_messages[2].content == "Nice to meet you, Alice!"
    assert t2_messages[3].content == "What is my name?"

    # Verify memory contains all 5 messages
    final_msgs = memory.get_messages("thread-conv-1")
    assert len(final_msgs) == 5
