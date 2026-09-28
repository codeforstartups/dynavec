"""Framework-agnostic retrieval tool factory.

Most agent frameworks (OpenAI Assistants, LangGraph, CrewAI, Strands, plain
function-calling) just need a **callable** that takes a query string and
returns text. This module provides ``make_retriever_fn`` plus first-class
adapters for OpenAI Assistants, LangChain, and CrewAI.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from ..client import Dynavec
from ..models import SearchResult
from ..namespace import NamespaceView
from ..retrievers import QueryExpansionRetriever


def make_retriever_fn(
    source: Dynavec | NamespaceView | QueryExpansionRetriever,
    *,
    top_k: int = 4,
    namespace: str = "default",
    filter: dict[str, Any] | None = None,
    rescore: str | dict[str, float] | None = None,
    join: str = "\n\n",
    include_scores: bool = False,
) -> Callable[[str], str]:
    """Return ``fn(query: str) -> str`` — the lowest common denominator tool.

    Works as-is in LangGraph nodes, CrewAI tools, Strands tools, or any
    function-calling agent.
    """
    if isinstance(source, QueryExpansionRetriever) and rescore is not None:
        raise ValueError("rescore is not supported with query-expansion retrievers")

    def _search(query: str) -> list[SearchResult]:
        if isinstance(source, QueryExpansionRetriever):
            return source.search(query, top_k=top_k, filter=filter)
        if isinstance(source, NamespaceView):
            return source.search(query, top_k=top_k, filter=filter, rescore=rescore)
        return source.search(
            query, top_k=top_k, namespace=namespace, filter=filter, rescore=rescore
        )

    def retrieve(query: str) -> str:
        hits = _search(query)
        parts = []
        for h in hits:
            body = h.text or ""
            parts.append(f"[{h.score:.3f}] {body}" if include_scores else body)
        return join.join(parts)

    retrieve.__name__ = "dynavec_retrieve"
    retrieve.__doc__ = (
        "Search the dynavec knowledge base and return the most relevant passages "
        "for a natural-language query."
    )
    return retrieve


class OpenAIAssistantTool:
    """OpenAI Assistants and Function Calling tool adapter backed by Dynavec.

    Provides OpenAI-compatible function calling schemas, single-tool-call execution,
    and batch execution for Assistant run loops without requiring external SDKs.
    """

    def __init__(
        self,
        retriever_fn: Callable[[str], str],
        name: str = "dynavec_search",
        description: str | None = None,
    ) -> None:
        self.retriever_fn = retriever_fn
        self.name = name
        self.description = description or (
            "Search the knowledge base and return the most relevant passages for a query."
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return the standard OpenAI Tool definition dictionary."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to retrieve relevant passages.",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """Return the OpenAI Tool definition dictionary (alias for ``schema``)."""
        return self.schema

    def to_dict(self) -> dict[str, Any]:
        """Return the OpenAI Tool definition dictionary."""
        return self.schema

    def __call__(self, query: str) -> str:
        """Directly invoke the retriever with a query string."""
        return self.retriever_fn(query)

    def handle_tool_call(self, tool_call: Any) -> dict[str, str]:
        """Execute retrieval for a single OpenAI ToolCall and return output dict.

        Accepts either an OpenAI SDK ``ToolCall`` instance (or Pydantic/Namespace model)
        or a dictionary format:
        ``{"id": "call_123", "function": {"name": "dynavec_search", "arguments": "{\\"query\\": \\"...\\"}"}}``

        Returns:
            A dictionary formatted for OpenAI ``submit_tool_outputs``:
            ``{"tool_call_id": "<id>", "output": "<retrieved text>"}``
        """
        call_id = getattr(tool_call, "id", None)
        if call_id is None and isinstance(tool_call, dict):
            call_id = tool_call.get("id") or tool_call.get("tool_call_id") or ""
        call_id = str(call_id or "")

        fn_obj = getattr(tool_call, "function", None)
        if fn_obj is None and isinstance(tool_call, dict):
            fn_obj = tool_call.get("function")

        raw_args = getattr(fn_obj, "arguments", None)
        if raw_args is None and isinstance(fn_obj, dict):
            raw_args = fn_obj.get("arguments")
        if raw_args is None and isinstance(tool_call, dict):
            raw_args = tool_call.get("arguments")

        try:
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    parsed_args = {"query": raw_args}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            elif raw_args is None:
                parsed_args = {}
            else:
                parsed_args = {"query": str(raw_args)}

            if parsed_args is None:
                query = ""
            elif isinstance(parsed_args, dict):
                query_value: Any = parsed_args.get("query")
                if query_value is None:
                    for fallback_key in ("q", "input", "search_query", "text", "prompt"):
                        if fallback_key in parsed_args:
                            query_value = parsed_args[fallback_key]
                            break
                if query_value is None and parsed_args:
                    query_value = next((v for v in parsed_args.values() if isinstance(v, str)), "")
                query = str(query_value or "")
            elif isinstance(parsed_args, (list, tuple, set)):
                query = " ".join(str(x) for x in parsed_args)
            else:
                query = str(parsed_args)

            output = self.retriever_fn(query)
        except Exception as exc:
            output = f"Error executing tool '{self.name}': {exc}"

        return {"tool_call_id": call_id, "output": output}

    def submit_tool_outputs(self, tool_calls: Sequence[Any] | None) -> list[dict[str, str]]:
        """Batch-process multiple tool calls from an assistant run loop.

        Filters calls matching ``self.name``, executes retrieval for each, and
        returns a list of ``{"tool_call_id": ..., "output": ...}`` objects
        ready to be passed to ``client.beta.threads.runs.submit_tool_outputs()``.
        """
        if not tool_calls:
            return []

        outputs: list[dict[str, str]] = []
        for call in tool_calls:
            fn_obj = getattr(call, "function", None)
            if fn_obj is None and isinstance(call, dict):
                fn_obj = call.get("function")

            fn_name = getattr(fn_obj, "name", None)
            if fn_name is None and isinstance(fn_obj, dict):
                fn_name = fn_obj.get("name")
            if fn_name is None and isinstance(call, dict):
                fn_name = call.get("name")

            if fn_name and fn_name != self.name:
                continue

            outputs.append(self.handle_tool_call(call))
        return outputs


def as_openai_tool(
    source: Dynavec | NamespaceView,
    *,
    name: str = "dynavec_search",
    description: str | None = None,
    **kw: Any,
) -> OpenAIAssistantTool:
    """Wrap a Dynavec client or NamespaceView as an OpenAI Assistant tool.

    Requires zero third-party dependencies. Returns an ``OpenAIAssistantTool``
    ready to be registered on an Assistant or passed into chat completions.
    """
    fn = make_retriever_fn(source, **kw)
    return OpenAIAssistantTool(fn, name=name, description=description)


def as_langchain_tool(
    source: Dynavec | NamespaceView | QueryExpansionRetriever,
    *,
    name: str = "dynavec_search",
    **kw: Any,
) -> Any:
    """Wrap the retriever as a LangChain ``StructuredTool`` (requires langchain-core)."""
    from ..exceptions import MissingDependencyError

    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("as_langchain_tool", "langchain-core", "langchain") from exc

    fn = make_retriever_fn(source, **kw)
    return StructuredTool.from_function(func=fn, name=name, description=fn.__doc__)


def as_crewai_tool(
    source: Dynavec | NamespaceView,
    *,
    name: str = "dynavec_search",
    **kw: Any,
) -> Any:
    """Wrap the retriever as a CrewAI tool (requires crewai)."""
    from ..exceptions import MissingDependencyError

    try:
        from crewai.tools import tool as crewai_tool
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("as_crewai_tool", "crewai", "all") from exc

    fn = make_retriever_fn(source, **kw)

    def _tool(query: str) -> str:
        """Search the dynavec knowledge base for relevant passages."""
        return fn(query)

    return crewai_tool(name)(_tool)
