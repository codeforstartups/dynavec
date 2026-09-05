"""Framework-agnostic retrieval tool factory.

Most agent frameworks (LangGraph, CrewAI, Strands, plain function-calling) just
need a **callable** that takes a query string and returns text. This builds one
from a dynavec client / namespace, plus thin wrappers that register it as a
native tool in LangChain and CrewAI when those are installed.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from ..client import Dynavec
from ..namespace import NamespaceView


def make_retriever_fn(
    source: Union[Dynavec, NamespaceView],
    *,
    top_k: int = 4,
    namespace: str = "default",
    filter: Optional[dict] = None,
    rescore=None,
    join: str = "\n\n",
    include_scores: bool = False,
) -> Callable[[str], str]:
    """Return ``fn(query: str) -> str`` — the lowest common denominator tool.

    Works as-is in LangGraph nodes, CrewAI tools, Strands tools, or any
    function-calling agent.
    """

    def _search(query: str):
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


def as_langchain_tool(source, *, name: str = "dynavec_search", **kw) -> Any:
    """Wrap the retriever as a LangChain ``StructuredTool`` (requires langchain-core)."""
    from ..exceptions import MissingDependencyError

    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("as_langchain_tool", "langchain-core", "langchain") from exc

    fn = make_retriever_fn(source, **kw)
    return StructuredTool.from_function(
        func=fn, name=name, description=fn.__doc__
    )


def as_crewai_tool(source, *, name: str = "dynavec_search", **kw) -> Any:
    """Wrap the retriever as a CrewAI tool (requires crewai)."""
    from ..exceptions import MissingDependencyError

    try:
        from crewai.tools import tool as crewai_tool
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("as_crewai_tool", "crewai", "all") from exc

    fn = make_retriever_fn(source, **kw)

    @crewai_tool(name)
    def _tool(query: str) -> str:
        """Search the dynavec knowledge base for relevant passages."""
        return fn(query)

    return _tool
