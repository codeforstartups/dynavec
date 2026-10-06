"""Built-in standard node library for dynaflow agent orchestrations."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..chat.base import ChatModel, Message
from .base import Node, NodePort, NodeSchema
from .registry import node_registry

# --- 1. RetrieverNode ---


@node_registry.register
class RetrieverNode(Node):
    """Executes dense/hybrid vector retrieval against dynavec vector store."""

    node_type = "retriever"
    schema = NodeSchema(
        node_type="retriever",
        label="Vector Retriever",
        description="Searches dynavec vector index and formats context for LLMs.",
        category="retrieval",
        inputs=[
            NodePort(
                name="query",
                port_type="string",
                description="Natural language query string.",
                required=True,
            ),
            NodePort(
                name="filter",
                port_type="object",
                description="Metadata filter dictionary.",
                required=False,
                default=None,
            ),
            NodePort(
                name="top_k",
                port_type="number",
                description="Maximum number of documents to return.",
                required=False,
                default=5,
            ),
        ],
        outputs=[
            NodePort(
                name="documents",
                port_type="array",
                description="Retrieved document records.",
            ),
            NodePort(
                name="context",
                port_type="string",
                description="Concatenated document text context block.",
            ),
        ],
        config_schema={
            "top_k": {"type": "integer", "default": 5},
            "namespace": {"type": "string", "default": "default"},
        },
    )

    def __init__(
        self,
        retriever_fn: Callable[..., Any] | None = None,
        node_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_id=node_id, config=config)
        self.retriever_fn = retriever_fn

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = str(inputs.get("query", ""))
        top_k = inputs.get("top_k", self.config.get("top_k", 5))
        filter_dict = inputs.get("filter", self.config.get("filter"))

        documents: list[dict[str, Any]] = []
        if self.retriever_fn is not None:
            raw_docs = self.retriever_fn(query, top_k=top_k, filter=filter_dict)
            for d in raw_docs:
                if hasattr(d, "text") and hasattr(d, "id"):
                    documents.append(
                        {
                            "id": d.id,
                            "text": getattr(d, "text", ""),
                            "score": getattr(d, "score", 0.0),
                            "metadata": getattr(d, "metadata", {}),
                        }
                    )
                elif isinstance(d, dict):
                    documents.append(d)
                else:
                    documents.append({"text": str(d)})

        context_parts = [d.get("text", "") for d in documents if d.get("text")]
        context_str = "\n\n".join(context_parts)

        return {
            "documents": documents,
            "context": context_str,
        }


# --- 2. LLMNode ---


@node_registry.register
class LLMNode(Node):
    """Executes a language model prompt invocation."""

    node_type = "llm"
    schema = NodeSchema(
        node_type="llm",
        label="LLM Generation",
        description="Invokes a ChatModel with prompt and contextual grounding.",
        category="model",
        inputs=[
            NodePort(
                name="prompt",
                port_type="string",
                description="The prompt or user question.",
                required=True,
            ),
            NodePort(
                name="context",
                port_type="string",
                description="Optional retrieved reference context.",
                required=False,
                default="",
            ),
            NodePort(
                name="system_prompt",
                port_type="string",
                description="System instruction prompt.",
                required=False,
                default=None,
            ),
        ],
        outputs=[
            NodePort(
                name="response",
                port_type="string",
                description="Generated text output.",
            ),
            NodePort(
                name="message",
                port_type="object",
                description="Full ChatModel output message object.",
            ),
        ],
        config_schema={
            "model_name": {"type": "string"},
            "system_prompt": {"type": "string"},
            "temperature": {"type": "number", "default": 0.0},
        },
    )

    def __init__(
        self,
        model: ChatModel | Callable[[list[Message]], Any],
        node_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_id=node_id, config=config)
        self.model = model

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        prompt = str(inputs.get("prompt", ""))
        context = str(inputs.get("context", ""))
        sys_prompt = inputs.get("system_prompt", self.config.get("system_prompt"))

        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nUser Question:\n{prompt}"

        messages: list[Message] = []
        if sys_prompt:
            messages.append(Message(role="system", content=sys_prompt))
        messages.append(Message(role="user", content=full_prompt))

        if hasattr(self.model, "invoke"):
            chat_res = self.model.invoke(messages)
            out_text = chat_res.message.content or ""
            out_msg = chat_res.message
        elif callable(self.model):
            res = self.model(messages)
            out_text = str(res)
            out_msg = Message(role="assistant", content=out_text)
        else:
            raise TypeError("Model must be a ChatModel or callable.")

        return {
            "response": out_text,
            "message": out_msg,
        }


# --- 3. ToolNode ---


@node_registry.register
class ToolNode(Node):
    """Invokes a callable tool or registered AgentTool with input arguments."""

    node_type = "tool"
    schema = NodeSchema(
        node_type="tool",
        label="Tool Execution",
        description="Invokes external APIs, code functions, or MCP connectors.",
        category="tools",
        inputs=[
            NodePort(
                name="arguments",
                port_type="object",
                description="JSON parameters dictionary for tool invocation.",
                required=False,
                default={},
            ),
            NodePort(
                name="input",
                port_type="any",
                description="Alternative single parameter input.",
                required=False,
                default=None,
            ),
        ],
        outputs=[
            NodePort(
                name="result",
                port_type="any",
                description="Output returned from tool execution.",
            ),
            NodePort(
                name="status",
                port_type="string",
                description="Execution status ('success' or 'error').",
            ),
        ],
        config_schema={
            "tool_name": {"type": "string"},
        },
    )

    def __init__(
        self,
        tool: Callable[..., Any],
        node_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_id=node_id, config=config)
        self.tool = tool

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        args = inputs.get("arguments")
        if args is None and "input" in inputs:
            args = inputs["input"]

        try:
            if hasattr(self.tool, "execute"):
                res = self.tool.execute(args)
            elif isinstance(args, dict):
                res = self.tool(**args)
            elif isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict):
                        res = self.tool(**parsed)
                    else:
                        res = self.tool(args)
                except Exception:
                    res = self.tool(args)
            elif args is not None:
                res = self.tool(args)
            else:
                res = self.tool()
            return {"result": res, "status": "success"}
        except Exception as exc:
            return {"result": str(exc), "status": "error"}


# --- 4. RouterNode ---


@node_registry.register
class RouterNode(Node):
    """Routes execution along one of multiple named branches based on condition."""

    node_type = "router"
    schema = NodeSchema(
        node_type="router",
        label="Router / Branch",
        description="Selects an execution path based on predicate function or condition.",
        category="control_flow",
        inputs=[
            NodePort(
                name="data",
                port_type="any",
                description="Input data payload passing through router.",
                required=True,
            ),
            NodePort(
                name="route",
                port_type="string",
                description="Explicit route key override.",
                required=False,
                default=None,
            ),
        ],
        outputs=[
            NodePort(
                name="selected_route",
                port_type="string",
                description="The route key chosen for downstream branching.",
            ),
            NodePort(
                name="data",
                port_type="any",
                description="Forwarded payload unchanged.",
            ),
        ],
        config_schema={
            "routes": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["default"],
            },
        },
    )

    def __init__(
        self,
        route_fn: Callable[[Any], str] | None = None,
        node_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_id=node_id, config=config)
        self.route_fn = route_fn

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data = inputs.get("data")
        explicit_route = inputs.get("route")

        if explicit_route is not None:
            chosen = str(explicit_route)
        elif self.route_fn is not None:
            chosen = str(self.route_fn(data))
        else:
            chosen = self.config.get("default_route", "default")

        return {
            "selected_route": chosen,
            "data": data,
        }


# --- 5. CodeNode ---


@node_registry.register
class CodeNode(Node):
    """Executes arbitrary user-defined Python logic."""

    node_type = "code"
    schema = NodeSchema(
        node_type="code",
        label="Custom Python Code",
        description="Runs Python transform function over inputs.",
        category="custom",
        inputs=[
            NodePort(
                name="inputs",
                port_type="object",
                description="Dictionary of inputs passed to the code function.",
                required=True,
            ),
        ],
        outputs=[
            NodePort(
                name="outputs",
                port_type="object",
                description="Dictionary returned by the code function.",
            ),
        ],
    )

    def __init__(
        self,
        code_fn: Callable[[dict[str, Any]], dict[str, Any]],
        node_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_id=node_id, config=config)
        self.code_fn = code_fn

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        in_data = inputs.get("inputs", inputs)
        result = self.code_fn(in_data)
        if isinstance(result, dict):
            return result
        return {"outputs": result}


# --- 6. ApprovalNode ---


@node_registry.register
class ApprovalNode(Node):
    """Human-in-the-loop approval or interrupt checkpoint node."""

    node_type = "approval"
    schema = NodeSchema(
        node_type="approval",
        label="Human Approval",
        description="Gates workflow progression on human review or decision.",
        category="control_flow",
        inputs=[
            NodePort(
                name="request",
                port_type="any",
                description="Payload submitted for human review.",
                required=True,
            ),
            NodePort(
                name="approved",
                port_type="boolean",
                description="Review decision (true to continue, false to reject).",
                required=False,
                default=True,
            ),
            NodePort(
                name="notes",
                port_type="string",
                description="Reviewer comments or revision notes.",
                required=False,
                default=None,
            ),
        ],
        outputs=[
            NodePort(
                name="approved",
                port_type="boolean",
                description="Final approval verdict.",
            ),
            NodePort(
                name="data",
                port_type="any",
                description="Payload forwarded upon approval.",
            ),
            NodePort(
                name="notes",
                port_type="string",
                description="Feedback notes from reviewer.",
            ),
        ],
        config_schema={
            "role": {"type": "string", "default": "admin"},
        },
    )

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "approved": bool(inputs.get("approved", True)),
            "data": inputs.get("request"),
            "notes": inputs.get("notes"),
        }


# --- 7. LoopNode ---


@node_registry.register
class LoopNode(Node):
    """Iterates or maps an array of elements through execution."""

    node_type = "loop"
    schema = NodeSchema(
        node_type="loop",
        label="Loop / Map",
        description="Splits an array of items for parallel or sequential processing.",
        category="control_flow",
        inputs=[
            NodePort(
                name="items",
                port_type="array",
                description="List of items to iterate over.",
                required=True,
            ),
        ],
        outputs=[
            NodePort(
                name="items",
                port_type="array",
                description="Processed output elements.",
            ),
            NodePort(
                name="count",
                port_type="number",
                description="Total number of elements processed.",
            ),
        ],
        config_schema={
            "max_iterations": {"type": "integer", "default": 100},
        },
    )

    def __init__(
        self,
        map_fn: Callable[[Any], Any] | None = None,
        node_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_id=node_id, config=config)
        self.map_fn = map_fn

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw_items = list(inputs.get("items", []))
        max_iters = self.config.get("max_iterations", 100)
        items = raw_items[:max_iters]

        if self.map_fn is not None:
            processed = [self.map_fn(item) for item in items]
        else:
            processed = items

        return {
            "items": processed,
            "count": len(processed),
        }


# --- 8. WebhookNode ---


@node_registry.register
class WebhookNode(Node):
    """Trigger / entrypoint node ingesting external webhook payloads."""

    node_type = "webhook"
    schema = NodeSchema(
        node_type="webhook",
        label="Webhook Trigger",
        description="Entrypoint receiving external HTTP webhooks and event triggers.",
        category="triggers",
        inputs=[
            NodePort(
                name="payload",
                port_type="object",
                description="Incoming JSON request body.",
                required=True,
            ),
            NodePort(
                name="headers",
                port_type="object",
                description="HTTP request headers.",
                required=False,
                default={},
            ),
        ],
        outputs=[
            NodePort(
                name="payload",
                port_type="object",
                description="Sanitized event payload to start workflow.",
            ),
        ],
        config_schema={
            "path": {"type": "string", "default": "/webhook"},
        },
    )

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"payload": inputs.get("payload", {})}


# --- 9. SubGraphNode ---


@node_registry.register
class SubGraphNode(Node):
    """Encapsulates a nested graph or composite workflow as a single node."""

    node_type = "subgraph"
    schema = NodeSchema(
        node_type="subgraph",
        label="Sub-Graph",
        description="Executes a modular reusable sub-workflow.",
        category="composite",
        inputs=[
            NodePort(
                name="inputs",
                port_type="object",
                description="Input dictionary for sub-graph execution.",
                required=True,
            ),
        ],
        outputs=[
            NodePort(
                name="outputs",
                port_type="object",
                description="Outputs resulting from sub-graph execution.",
            ),
        ],
        config_schema={
            "subgraph_id": {"type": "string"},
        },
    )

    def __init__(
        self,
        subgraph: Any = None,
        node_id: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_id=node_id, config=config)
        self.subgraph = subgraph

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        in_data = inputs.get("inputs", inputs)
        if hasattr(self.subgraph, "run"):
            res = self.subgraph.run(in_data)
        elif callable(self.subgraph):
            res = self.subgraph(in_data)
        else:
            res = in_data
        return {"outputs": res}
