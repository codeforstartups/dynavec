"""FastMCP server exposing dynavec search and GraphRAG to MCP clients."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from typing import Any

from ..client import Dynavec
from ..config import DynavecConfig
from ..exceptions import ConfigurationError, MissingDependencyError


def _resolve_embedder(env: Mapping[str, str]):
    """Instantiate an embedder from environment variables or return None."""
    embedder_type = (env.get("DYNAVEC_EMBEDDER") or "").strip().lower()
    model = env.get("DYNAVEC_EMBEDDER_MODEL")

    # Auto-detect if not explicitly set
    if not embedder_type:
        if env.get("OPENAI_API_KEY"):
            embedder_type = "openai"
        elif env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY"):
            embedder_type = "gemini"
        elif env.get("VOYAGE_API_KEY"):
            embedder_type = "voyage"

    if not embedder_type or embedder_type in ("none", "false", "0"):
        return None

    if embedder_type == "openai":
        from ..embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(model=model or "text-embedding-3-small", api_key=env.get("OPENAI_API_KEY"))
    if embedder_type == "gemini":
        from ..embeddings.gemini import GeminiEmbedder

        return GeminiEmbedder(
            model=model or "text-embedding-004",
            api_key=env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY"),
        )
    if embedder_type == "voyage":
        from ..embeddings.voyage import VoyageEmbedder

        return VoyageEmbedder(model=model or "voyage-4", api_key=env.get("VOYAGE_API_KEY"))
    if embedder_type == "bedrock":
        from ..embeddings.bedrock import BedrockEmbedder

        region = env.get("DYNAVEC_REGION") or env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
        return BedrockEmbedder(model_id=model or "amazon.titan-embed-text-v2:0", region=region)
    if embedder_type in ("sentence-transformers", "local"):
        from ..embeddings.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model=model or "all-MiniLM-L6-v2")

    raise ConfigurationError(f"Unknown embedder type: '{embedder_type}'")


def client_from_env(env: Mapping[str, str] | None = None) -> Dynavec:
    """Build a Dynavec client configured from environment variables."""
    e = env if env is not None else os.environ

    bucket = e.get("DYNAVEC_VECTOR_BUCKET") or e.get("DYNAVEC_BUCKET")
    index = e.get("DYNAVEC_INDEX")
    table = e.get("DYNAVEC_TABLE")

    missing = []
    if not bucket:
        missing.append("DYNAVEC_VECTOR_BUCKET (or DYNAVEC_BUCKET)")
    if not index:
        missing.append("DYNAVEC_INDEX")
    if not table:
        missing.append("DYNAVEC_TABLE")
    if missing:
        raise ConfigurationError(f"Missing required environment variable(s): {', '.join(missing)}")

    embedder = _resolve_embedder(e)

    dim_raw = e.get("DYNAVEC_DIMENSION")
    if dim_raw:
        dimension = int(dim_raw)
    elif embedder is not None and getattr(embedder, "dimension", None):
        dimension = int(embedder.dimension)
    else:
        dimension = 1536

    fk_raw = e.get("DYNAVEC_FILTERABLE_KEYS")
    filterable_keys = [k.strip() for k in fk_raw.split(",") if k.strip()] if fk_raw else None

    region = e.get("DYNAVEC_REGION") or e.get("AWS_REGION") or e.get("AWS_DEFAULT_REGION")
    distance_metric = e.get("DYNAVEC_DISTANCE_METRIC", "cosine")

    cfg = DynavecConfig(
        vector_bucket=str(bucket),
        index=str(index),
        table=str(table),
        dimension=dimension,
        distance_metric=distance_metric,  # type: ignore[arg-type]
        region=region,
        filterable_keys=filterable_keys,
    )
    return Dynavec(cfg, embedder=embedder)


def _format_hits(hits: list[Any], query: str, context_label: str = "") -> str:
    """Format search results into human/agent readable text."""
    if not hits:
        return f"No results found for query '{query}'{context_label}."

    lines = [f"Found {len(hits)} results for query '{query}'{context_label}:\n"]
    for i, h in enumerate(hits, 1):
        meta_str = f" | metadata: {json.dumps(h.metadata, default=str)}" if h.metadata else ""
        lines.append(f"[{i}] id: {h.id} (score: {h.score:.4f}{meta_str})\n{h.text or ''}\n")
    return "\n".join(lines).strip()


def create_mcp_server(db: Dynavec | None = None, name: str = "dynavec"):
    """Create and configure a FastMCP server exposing dynavec search and graph search tools."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("create_mcp_server", "mcp", "mcp") from exc

    mcp = FastMCP(name)

    def _get_db() -> Dynavec:
        return db if db is not None else client_from_env()

    @mcp.tool()
    def dynavec_search(
        query: str,
        top_k: int = 5,
        namespace: str = "default",
        filter_json: str | None = None,
        rescore: str | None = None,
        rerank: str | None = None,
    ) -> str:
        """Search the dynavec vector database for semantic similarity.

        Parameters
        ----------
        query:
            Natural language query text.
        top_k:
            Number of matching documents to retrieve (default: 5).
        namespace:
            Tenant or collection namespace (default: "default").
        filter_json:
            Optional JSON string for metadata pre-filtering (e.g. '{"topic": "tech"}').
        rescore:
            Optional rescoring metric ("cosine", "dot", "euclidean", "manhattan").
        rerank:
            Optional reranking strategy (e.g. "mmr" for diversity).
        """
        filter_dict = None
        if filter_json:
            try:
                filter_dict = json.loads(filter_json)
            except Exception as exc:
                return f"Error parsing filter_json: {exc}"

        client = _get_db()
        hits = client.search(
            query=query,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            rescore=rescore,
            rerank=rerank,
        )
        return _format_hits(hits, query, f" in namespace '{namespace}'")

    @mcp.tool()
    def dynavec_graph_search(
        query: str,
        seed_entities: list[str],
        hops: int = 2,
        top_k: int = 5,
        namespace: str = "default",
    ) -> str:
        """Perform GraphRAG search by traversing related entities before vector ranking.

        Parameters
        ----------
        query:
            Search query to rank documents associated with traversed entities.
        seed_entities:
            List of starting entity IDs for the graph traversal.
        hops:
            Number of relationship hops to traverse from seed entities (default: 2).
        top_k:
            Number of documents to return (default: 5).
        namespace:
            Tenant or collection namespace (default: "default").
        """
        client = _get_db()
        hits = client.graph_search(
            query=query,
            seed_entities=seed_entities,
            hops=hops,
            top_k=top_k,
            namespace=namespace,
        )
        return _format_hits(hits, query, f" (seeds: {seed_entities}, hops: {hops})")

    return mcp


def main(argv: list[str] | None = None) -> int:
    """Run the FastMCP server with stdio or SSE transport."""
    parser = argparse.ArgumentParser(prog="dynavec mcp", description="Run the dynavec FastMCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    args = parser.parse_args(argv)

    server = create_mcp_server()
    if args.transport == "sse":
        server.settings.port = args.port
        server.run(transport="sse")
    else:
        server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
