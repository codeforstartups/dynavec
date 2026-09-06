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

from typing import Any

from .config import DynavecConfig
from .utils import KEY_SEPARATOR, encode_key_component, retry

Props = dict[str, Any]


class GraphStore:
    """DynamoDB-backed property graph sharing the dynavec document table."""

    def __init__(self, config: DynavecConfig, boto_session=None) -> None:
        import boto3

        session = boto_session or boto3.Session()
        self._config = config
        self._ddb = session.resource("dynamodb", region_name=config.region)
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
