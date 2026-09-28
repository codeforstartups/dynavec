"""Launch the dynavec observability dashboard on REAL telemetry — no AWS needed.

This runs actual ``dynavec`` searches (embed -> ANN -> hydrate -> cache -> rank)
against small in-memory stand-ins for S3 Vectors + DynamoDB, records real
telemetry, and serves the dashboard. Every number you see comes from real calls.

    python examples/dashboard_demo.py         # then open http://127.0.0.1:8779

For real production telemetry, do the same with your real client:

    from dynavec.telemetry import TelemetryRecorder
    from dynavec.dashboard import serve
    rec = TelemetryRecorder()
    db = Dynavec(cfg, embedder=emb, cache=SemanticCache(), telemetry=rec)
    ...  # your app runs searches
    serve(rec, port=8779)
"""

from __future__ import annotations

import math
import random
import threading
import time

import dynavec.client as cm
from dynavec import Document, Dynavec, DynavecConfig, SemanticCache
from dynavec.dashboard import serve
from dynavec.embeddings.base import Embedder
from dynavec.telemetry import TelemetryRecorder

DIM = 16


class _HashEmbedder(Embedder):
    def __init__(self, dimension: int = DIM) -> None:
        self.dimension = dimension

    def embed_documents(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dimension
            for i, c in enumerate(t):
                v[i % self.dimension] += (ord(c) % 17) / 17.0
            out.append(v)
        return out


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return 1 - dot / (na * nb)


class _FakeS3(cm.S3VectorsStore):
    def __init__(self, config, boto_session=None):
        self.config = config
        self._store = {}

    def put_vectors(self, vectors):
        for k, v, m in vectors:
            self._store[k] = (list(v), dict(m))

    def _matches(self, meta, flt):
        if not flt:
            return True
        for clause in (flt["$and"] if "$and" in flt else [flt]):
            for k, v in clause.items():
                if meta.get(k) != v:
                    return False
        return True

    def query(self, query_vector, top_k, filter=None, return_metadata=True, return_distance=True):
        scored = [
            (k, _cos(query_vector, vec), meta)
            for k, (vec, meta) in self._store.items()
            if self._matches(meta, filter)
        ]
        scored.sort(key=lambda x: x[1])
        return [{"key": k, "distance": d, "metadata": m} for k, d, m in scored[:top_k]]

    def query_pages(self, query_vector, top_k, filter=None, return_metadata=True, return_distance=True):
        hits = self.query(query_vector, top_k, filter)
        for i in range(0, len(hits), 2):
            yield hits[i : i + 2]

    def get_vectors(self, keys, return_metadata=False):
        return {k: {"vector": self._store[k][0], "metadata": self._store[k][1]} for k in keys if k in self._store}

    def delete_vectors(self, keys):
        for k in keys:
            self._store.pop(k, None)


class _FakeDDB(cm.DynamoDBStore):
    def __init__(self, config, boto_session=None):
        self.config = config
        self._store = {}

    def put_many(self, ns, items):
        for i, t, m in items:
            self._store[(ns, i)] = {"text": t, "metadata": dict(m)}

    def get_many(self, ns, ids):
        return {i: self._store[(ns, i)] for i in ids if (ns, i) in self._store}

    def delete_many(self, ns, ids):
        for i in ids:
            self._store.pop((ns, i), None)


class _FakeGraph:
    def __init__(self, config, boto_session=None):
        self.config = config
        self._nodes = {}  # (ns, entity_id) -> list of edges
        
    def add_node(self, ns, entity_id, ntype=None, props=None):
        if (ns, entity_id) not in self._nodes:
            self._nodes[(ns, entity_id)] = []
            
    def add_edge(self, ns, src, relation, dst):
        self.add_node(ns, src)
        self.add_node(ns, dst)
        self._nodes[(ns, src)].append({"relation": relation, "target": dst})
        
    def list_node_ids(self, ns):
        return sorted({eid for n, eid in self._nodes.keys() if n == ns})
        
    def _collect_subgraph(self, ns, roots, relation):
        seeds = self.list_node_ids(ns) if roots is None else roots
        visited = set(seeds)
        edges = set()
        queue = list(seeds)
        while queue:
            eid = queue.pop(0)
            for edge in self._nodes.get((ns, eid), []):
                rel = edge.get("relation") or ""
                if relation is not None and rel != relation:
                    continue
                tgt = edge["target"]
                edges.add((eid, rel, tgt))
                if tgt not in visited:
                    visited.add(tgt)
                    queue.append(tgt)
        return sorted(visited), sorted(edges)



TOPICS = ["food", "space", "ai", "aws", "bio"]
QUERIES = [
    "apple pie recipe", "rocket to mars", "serverless vectors on aws",
    "how do cells work", "transformer attention", "cheap vector database",
    "dynamodb latency", "embedding models",
]
NAMESPACES = ["kb", "tenant-a", "tenant-b"]


def _one_search(db):
    ns = random.choice(NAMESPACES)
    kw = {}
    r = random.random()
    if r < 0.3:
        kw["filter"] = {"topic": random.choice(TOPICS)}
    if r > 0.7:
        kw["rerank"] = "mmr"
    elif r > 0.5:
        kw["rescore"] = "cosine"
    try:
        db.search(random.choice(QUERIES), top_k=random.choice([5, 10, 10, 20]), namespace=ns, **kw)
    except Exception:  # noqa: BLE001 - demo resilience
        pass


def main() -> None:
    cm.S3VectorsStore = _FakeS3
    cm.DynamoDBStore = _FakeDDB
    import dynavec.graph
    from dynavec.models import IndexInfo
    dynavec.graph.GraphStore = _FakeGraph
    cm.GraphStore = _FakeGraph
    rec = TelemetryRecorder(capture_text=True)
    cfg = DynavecConfig(vector_bucket="dynavec-demo-bucket", index="hnsw-demo", table="dynavec-demo-table", dimension=DIM)
    db = Dynavec(cfg, embedder=_HashEmbedder(), cache=SemanticCache(threshold=0.9), telemetry=rec)
    db.describe = lambda: IndexInfo(
        vector_bucket=cfg.vector_bucket,
        index=cfg.index,
        dimension=cfg.dimension,
        distance_metric="cosine",
        table=cfg.table,
        table_status="ACTIVE",
        item_count=len(db._docs._store) if hasattr(db._docs, '_store') else 4200
    )
    db.list_namespaces = lambda: NAMESPACES

    for ns in NAMESPACES:
        db.upsert(
            [
                Document(id=f"{ns}-{i}", text=f"{random.choice(TOPICS)} document {i} about vectors",
                         metadata={"topic": random.choice(TOPICS)})
                for i in range(40)
            ],
            namespace=ns,
        )
        db.graph_add_edge("Serverless", "reduces", "Cost", namespace=ns)
        db.graph_add_edge("Dynavec", "runs_on", "AWS", namespace=ns)
        db.graph_add_edge("Dynavec", "uses", "DynamoDB", namespace=ns)
        db.graph_add_edge("Dynavec", "uses", "S3", namespace=ns)
        db.graph_add_edge("DynamoDB", "stores", "Metadata", namespace=ns)
        db.graph_add_edge("DynamoDB", "stores", "Graph", namespace=ns)
        db.graph_add_edge("S3", "stores", "Vectors", namespace=ns)
        db.graph_add_edge("Graph", "powers", "GraphRAG", namespace=ns)

    for _ in range(150):  # seed real history
        _one_search(db)
    print(f"seeded {len(rec.snapshot())} real telemetry events")

    def workload():
        while True:
            _one_search(db)
            time.sleep(random.uniform(0.05, 0.25))

    threading.Thread(target=workload, daemon=True).start()
    serve(rec, port=8779, db=db)


if __name__ == "__main__":
    main()
