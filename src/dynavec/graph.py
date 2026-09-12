"""Knowledge-graph / entity-relationship layer over DynamoDB.

This is the "meaning attached to embeddings" layer. Alongside the vector index,
dynavec keeps a lightweight graph in DynamoDB:

    (entity) --[relation]--> (entity)
        |
        └── linked to --> document ids  ---> S3 Vectors embeddings

Because DynamoDB is a superb adjacency store, you can **traverse the graph first**
(cheap, single-digit-ms key lookups) to gather a candidate set of documents that
are *semantically related by structure*, then rank only those against the query
embedding in S3 Vectors. That's the DynamoDB→S3-Vectors pointer/reference join:
graph edges narrow and guide the vector search instead of scanning everything.

Storage (single-table, works with the existing pk-only schema):
    node:  pk = "{ns}#node#{entity_id}"  attrs: ntype, props, edges[], docs[]
    edges are embedded adjacency lists: [{"relation": r, "target": entity_id}, ...]

Note: embedded adjacency keeps a node's fan-out in one 400KB item. Very high
fan-out entities want a sort-key adjacency design (roadmap).
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any

from .config import DynavecConfig
from .utils import KEY_SEPARATOR, decode_key_component, encode_key_component, retry

Props = dict[str, Any]

EXPORT_FORMATS = ("mermaid", "dot")

# A bare Mermaid flowchart node id: letters/digits/underscore, not keyword-shaped.
_MERMAID_SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# An unquoted Mermaid edge label may not carry the `|` delimiter or spaces.
_MERMAID_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_]+$")
# Flowchart keywords. Used bare they are parsed as syntax, not as a node.
_MERMAID_RESERVED = frozenset(
    {
        "call",
        "class",
        "classdef",
        "click",
        "direction",
        "end",
        "flowchart",
        "graph",
        "href",
        "linkstyle",
        "style",
        "subgraph",
    }
)


def _mermaid_text(value: str) -> str:
    """Escape a Mermaid label. Mermaid uses `#nnn;` entities, not backslashes."""
    return (
        value.replace("#", "#35;")  # first: later replacements introduce `#`
        .replace('"', "#quot;")
        .replace("<", "#lt;")
        .replace(">", "#gt;")
    )


def _mermaid_label(relation: str) -> str:
    """Render an edge label, quoting it only when it cannot stand bare."""
    if _MERMAID_SAFE_LABEL.match(relation):
        return relation
    return f'"{_mermaid_text(relation)}"'


def _mermaid_aliases(nodes: list[str]) -> dict[str, str]:
    """Map entity id -> Mermaid node id.

    Ids that are already valid Mermaid identifiers are used verbatim, so the
    common case stays readable. Anything else (spaces, quotes, hyphens, a
    reserved keyword) gets an `nN` alias carrying the real id as its label.
    Alias numbering follows sorted id order, so it is stable across exports.
    """
    bare = {
        n for n in nodes if _MERMAID_SAFE_ID.match(n) and n.lower() not in _MERMAID_RESERVED
    }
    aliases: dict[str, str] = {}
    counter = 0
    for node in nodes:
        if node in bare:
            aliases[node] = node
            continue
        candidate = f"n{counter}"
        while candidate in bare:  # never shadow a real id spelled `n0`
            counter += 1
            candidate = f"n{counter}"
        aliases[node] = candidate
        counter += 1
    return aliases


def _render_mermaid(nodes: list[str], edges: list[tuple[str, str, str]]) -> str:
    aliases = _mermaid_aliases(nodes)
    linked = {n for src, _, dst in edges for n in (src, dst)}
    lines = ["graph LR"]
    for node in nodes:
        alias = aliases[node]
        if alias != node:
            lines.append(f'  {alias}["{_mermaid_text(node)}"]')
        elif node not in linked:
            lines.append(f"  {node}")  # isolated: declare it or it vanishes
    for src, relation, dst in edges:
        arrow = f"-->|{_mermaid_label(relation)}|" if relation else "-->"
        lines.append(f"  {aliases[src]} {arrow} {aliases[dst]}")
    return "\n".join(lines) + "\n"


def _dot_text(value: str) -> str:
    """Escape a DOT quoted string. Every id is quoted, so this is all it takes."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _render_dot(nodes: list[str], edges: list[tuple[str, str, str]]) -> str:
    linked = {n for src, _, dst in edges for n in (src, dst)}
    lines = ["digraph G {"]
    for node in nodes:
        if node not in linked:
            lines.append(f'  "{_dot_text(node)}";')
    for src, relation, dst in edges:
        attrs = f' [label="{_dot_text(relation)}"]' if relation else ""
        lines.append(f'  "{_dot_text(src)}" -> "{_dot_text(dst)}"{attrs};')
    lines.append("}")
    return "\n".join(lines) + "\n"


_RENDERERS = {"mermaid": _render_mermaid, "dot": _render_dot}


class GraphStore:
    """DynamoDB-backed property graph sharing the dynavec document table."""

    def __init__(self, config: DynavecConfig, boto_session=None) -> None:
        import boto3

        session = boto_session or boto3.Session()
        self._config = config
        resource_kwargs: dict[str, object] = {"region_name": config.region}
        botocore_config = config.botocore_config()
        if botocore_config is not None:
            resource_kwargs["config"] = botocore_config
        self._ddb = session.resource("dynamodb", **resource_kwargs)  # type: ignore[arg-type]
        self._table = self._ddb.Table(config.table)

    @staticmethod
    def _node_pk(ns: str, entity_id: str) -> str:
        return (
            f"{encode_key_component(ns)}{KEY_SEPARATOR}node{KEY_SEPARATOR}"
            f"{encode_key_component(entity_id)}"
        )

    # --------------------------------------------------------------- mutations
    @retry()
    def add_node(
        self, ns: str, entity_id: str, ntype: str | None = None, props: Props | None = None
    ) -> None:
        self._table.update_item(
            Key={"pk": self._node_pk(ns, entity_id)},
            UpdateExpression=(
                "SET kind = :k, ns = :ns, entity_id = :eid, ntype = :t, props = :p, "
                "edges = if_not_exists(edges, :empty), docs = if_not_exists(docs, :empty)"
            ),
            ExpressionAttributeValues={
                ":k": "node",
                ":ns": ns,
                ":eid": entity_id,
                ":t": ntype,
                ":p": props or {},
                ":empty": [],
            },
        )

    @retry()
    def add_edge(self, ns: str, src: str, relation: str, dst: str) -> None:
        # ensure both endpoints exist, then append the edge to src's adjacency
        self.add_node(ns, src)
        self.add_node(ns, dst)
        self._table.update_item(
            Key={"pk": self._node_pk(ns, src)},
            UpdateExpression="SET edges = list_append(if_not_exists(edges, :empty), :e)",
            ExpressionAttributeValues={
                ":e": [{"relation": relation, "target": dst}],
                ":empty": [],
            },
        )

    @retry()
    def link_docs(self, ns: str, entity_id: str, doc_ids: list[str]) -> None:
        self.add_node(ns, entity_id)
        self._table.update_item(
            Key={"pk": self._node_pk(ns, entity_id)},
            UpdateExpression="SET docs = list_append(if_not_exists(docs, :empty), :d)",
            ExpressionAttributeValues={":d": list(doc_ids), ":empty": []},
        )

    # ------------------------------------------------------------------ reads
    @retry()
    def get_node(self, ns: str, entity_id: str) -> dict | None:
        resp = self._table.get_item(Key={"pk": self._node_pk(ns, entity_id)})
        return resp.get("Item")

    def neighbors(self, ns: str, entity_id: str, relation: str | None = None) -> list[str]:
        node = self.get_node(ns, entity_id)
        if not node:
            return []
        out = []
        for edge in node.get("edges", []):
            if relation is None or edge.get("relation") == relation:
                out.append(edge["target"])
        return out

    def get_docs(self, ns: str, entity_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for eid in entity_ids:
            node = self.get_node(ns, eid)
            if not node:
                continue
            for doc_id in node.get("docs", []):
                if doc_id not in seen:
                    seen.add(doc_id)
                    ordered.append(doc_id)
        return ordered

    @retry()
    def list_node_ids(self, ns: str) -> list[str]:
        """Every entity id stored under ``ns``, sorted.

        The namespace lives in the node key, so — exactly like
        :meth:`~dynavec.client.Dynavec.list_vectors`, which scopes on the
        decoded vector key because ``ListVectors`` takes no server-side
        ``filter`` — the scope here is a prefix match on ``pk`` rather than a
        query on the ``_dv_ns`` tag. ``encode_key_component`` makes the prefix
        unambiguous: a namespace can never be a prefix of another one's key.

        This is a table scan. Fine for the export/debugging graphs this exists
        for; not something to put on a hot path.
        """
        prefix = f"{encode_key_component(ns)}{KEY_SEPARATOR}node{KEY_SEPARATOR}"
        params: dict[str, Any] = {
            "FilterExpression": "begins_with(pk, :prefix)",
            "ExpressionAttributeValues": {":prefix": prefix},
            "ProjectionExpression": "pk, entity_id",
        }
        ids: list[str] = []
        while True:
            resp = self._table.scan(**params)
            for item in resp.get("Items", []):
                entity_id = item.get("entity_id")
                if entity_id is None:  # pre-``entity_id`` item: recover from the key
                    entity_id = decode_key_component(item["pk"][len(prefix) :])
                ids.append(entity_id)
            start_key = resp.get("LastEvaluatedKey")
            if not start_key:
                return sorted(ids)
            params["ExclusiveStartKey"] = start_key

    # ----------------------------------------------------------------- export
    def graph_export(
        self,
        fmt: str,
        ns: str,
        *,
        roots: list[str] | None = None,
        relation: str | None = None,
    ) -> str:
        """Render the ``ns`` subgraph as Mermaid or Graphviz DOT source.

        Returns the diagram as a string — nothing is written or printed, so the
        caller decides whether it lands in a docs page, a log line, or a file.
        No new dependency: the text is emitted directly.

        Parameters
        ----------
        fmt:
            ``"mermaid"`` or ``"dot"``. Anything else raises ``ValueError``.
        ns:
            Namespace to export. Only this subgraph is walked, never the whole
            table.
        roots:
            Start the walk at these entities and follow edges outwards. The
            default (``None``) seeds from every node in the namespace, which is
            the whole subgraph including nodes nothing points at.
        relation:
            Only traverse and emit edges carrying this relation.

        Output is deterministic: nodes and edges are sorted, and each edge is
        emitted exactly once even when the graph contains cycles.
        """
        render = _RENDERERS.get(fmt)
        if render is None:
            supported = ", ".join(repr(f) for f in EXPORT_FORMATS)
            raise ValueError(f"Unknown export format {fmt!r}. Supported formats: {supported}.")
        nodes, edges = self._collect_subgraph(ns, roots, relation)
        return render(nodes, edges)

    def _collect_subgraph(
        self, ns: str, roots: list[str] | None, relation: str | None
    ) -> tuple[list[str], list[tuple[str, str, str]]]:
        """Breadth-first walk returning ``(sorted nodes, sorted unique edges)``.

        ``visited`` is what makes a cyclic graph terminate; ``edges`` is a set,
        so a duplicated adjacency entry still renders one arrow.
        """
        seeds = self.list_node_ids(ns) if roots is None else list(dict.fromkeys(roots))
        visited: set[str] = set(seeds)
        edges: set[tuple[str, str, str]] = set()
        queue = deque(seeds)
        while queue:
            entity_id = queue.popleft()
            node = self.get_node(ns, entity_id)
            if not node:
                continue
            for edge in node.get("edges", []):
                edge_relation = edge.get("relation") or ""
                if relation is not None and edge_relation != relation:
                    continue
                target = edge["target"]
                edges.add((entity_id, edge_relation, target))
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
        return sorted(visited), sorted(edges)
