import pytest
from test_client_inmemory import FakeDDB, FakeGraph, FakeS3

import dynavec.client as client_mod
from dynavec import (
    BM25HybridRetriever,
    BM25Index,
    BM25Retriever,
    Document,
    Dynavec,
    DynavecConfig,
    FitResult,
)
from dynavec.bm25 import default_tokenize
from dynavec.embeddings.base import Embedder

# Test embeddings in 4D
E0 = [1.0, 0.0, 0.0, 0.0]
E1 = [0.0, 1.0, 0.0, 0.0]
E2 = [0.0, 0.0, 1.0, 0.0]
E3 = [0.0, 0.0, 0.0, 1.0]


class FakeEmbedder(Embedder):
    dimension = 4

    def __init__(self):
        self.query_calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        if "laptop" in text.lower():
            return E1
        if "server" in text.lower():
            return E2
        return E0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        res = []
        for t in texts:
            tl = t.lower()
            if "laptop" in tl:
                res.append(E1)
            elif "server" in tl:
                res.append(E2)
            else:
                res.append(E0)
        return res


class FakeS3WithPages(FakeS3):
    def list_pages(self, return_data=False, return_metadata=True, page_size=None):
        keys = list(self._store.keys())
        chunk_size = page_size or 2
        for i in range(0, len(keys), chunk_size):
            page = []
            for k in keys[i : i + chunk_size]:
                vec, meta = self._store[k]
                item = {"key": k}
                if return_data:
                    item["data"] = {"float32": vec}
                if return_metadata:
                    item["metadata"] = meta
                page.append(item)
            yield page


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(client_mod, "S3VectorsStore", FakeS3WithPages)
    monkeypatch.setattr(client_mod, "DynamoDBStore", FakeDDB)
    monkeypatch.setattr(client_mod, "GraphStore", FakeGraph)

    cfg = DynavecConfig(vector_bucket="b", index="i", table="t", dimension=4, region="us-east-1")
    client = Dynavec(cfg, embedder=FakeEmbedder())

    docs = [
        Document(
            id="doc1",
            text="Dell XPS-13-9310 ultrabook laptop with Intel Core i7",
            vector=E1,
            metadata={"category": "hardware", "sku": "SKU-9310-XPS"},
        ),
        Document(
            id="doc2",
            text="Lenovo ThinkPad X1 Carbon laptop for business travel",
            vector=E1,
            metadata={"category": "hardware", "sku": "SKU-X1-CARB"},
        ),
        Document(
            id="doc3",
            text="High performance rack server Dell PowerEdge R750 with dual Xeon",
            vector=E2,
            metadata={"category": "server", "sku": "SKU-R750-PWR"},
        ),
        Document(
            id="doc4",
            text="Python troubleshooting guide: handling ConnectionResetError during TLS handshake",
            vector=E0,
            metadata={"category": "software", "code": "ERR-104"},
        ),
    ]
    client.upsert(docs)
    return client


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


def test_default_tokenize_basic():
    tokens = default_tokenize("The quick brown fox jumps over the lazy dog.")
    # Stopwords ("the", "over") filtered out
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "dog" in tokens
    assert "the" not in tokens


def test_default_tokenize_compound_identifiers():
    tokens = default_tokenize("Order item SKU-892-XZ and Dell XPS-13-9310")
    # Exact compound preservation
    assert "sku-892-xz" in tokens
    assert "xps-13-9310" in tokens
    # Sub-token preservation
    assert "sku" in tokens
    assert "892" in tokens
    assert "xz" in tokens
    assert "xps" in tokens
    assert "13" in tokens
    assert "9310" in tokens


def test_default_tokenize_custom_stopwords():
    tokens = default_tokenize("apple banana cherry", stopwords=frozenset({"banana"}))
    assert tokens == ["apple", "cherry"]

    tokens_no_removal = default_tokenize("apple banana", remove_stopwords=False)
    assert "banana" in tokens_no_removal


def test_default_tokenize_empty_and_special():
    assert default_tokenize("") == []
    assert default_tokenize("   ") == []
    assert default_tokenize("!@#$%^&*()") == []


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


def test_bm25_index_validation():
    with pytest.raises(ValueError, match="k1 must be non-negative"):
        BM25Index(k1=-1.0)
    with pytest.raises(ValueError, match="b must be in"):
        BM25Index(b=1.5)
    with pytest.raises(ValueError, match="b must be in"):
        BM25Index(b=-0.1)


def test_bm25_index_crud():
    index = BM25Index()
    assert len(index) == 0

    with pytest.raises(ValueError, match="doc_id cannot be empty"):
        index.add_document("", "some text")

    index.add_document("doc1", "first document with keyword python")
    index.add_document("doc2", "second document with keyword rust")
    assert len(index) == 2

    # Search keyword
    res = index.search("python")
    assert len(res) == 1
    assert res[0].id == "doc1"
    assert res[0].score > 0

    # Overwrite doc1
    index.add_document("doc1", "updated document without keyword")
    assert len(index) == 2
    res_after = index.search("python")
    assert len(res_after) == 0

    # Remove document
    assert index.remove_document("doc1") is True
    assert index.remove_document("nonexistent") is False
    assert len(index) == 1

    # Clear
    index.clear()
    assert len(index) == 0
    assert index.search("rust") == []


def test_bm25_index_batch_add():
    index = BM25Index()
    docs = [
        Document(id="d1", text="alpha beta", metadata={"cat": "a"}),
        {"id": "d2", "text": "beta gamma", "metadata": {"cat": "b"}},
        ("d3", "gamma delta"),
    ]
    index.add_documents(docs)
    assert len(index) == 3

    with pytest.raises(TypeError, match="Unsupported document type"):
        index.add_documents([123])  # type: ignore


def test_bm25_index_filtering():
    index = BM25Index()
    index.add_document("d1", "database query optimization", {"type": "db", "priority": 1})
    index.add_document("d2", "database index design", {"type": "db", "priority": 5})
    index.add_document("d3", "database backup procedures", {"type": "ops", "priority": 3})

    # Exact filter
    hits = index.search("database", filter={"type": "db"})
    hit_ids = {h.id for h in hits}
    assert hit_ids == {"d1", "d2"}

    # Operator filter: $in
    hits_in = index.search("database", filter={"type": {"$in": ["ops"]}})
    assert [h.id for h in hits_in] == ["d3"]

    # Operator filter: $gte
    hits_gte = index.search("database", filter={"priority": {"$gte": 3}})
    hit_gte_ids = {h.id for h in hits_gte}
    assert hit_gte_ids == {"d2", "d3"}

    # Operator filter: $eq and $ne
    hits_ne = index.search("database", filter={"type": {"$ne": "db"}})
    assert [h.id for h in hits_ne] == ["d3"]


def test_bm25_index_empty_query():
    index = BM25Index()
    index.add_document("d1", "sample text")
    assert index.search("") == []
    assert index.search("   ") == []
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        index.search("sample", top_k=0)


# ---------------------------------------------------------------------------
# BM25Retriever tests
# ---------------------------------------------------------------------------


def test_bm25_retriever_basic(db):
    retriever = BM25Retriever(db)
    retriever.index_documents(
        [
            Document(id="doc1", text="Dell XPS-13-9310 laptop"),
            Document(id="doc2", text="Lenovo ThinkPad laptop"),
        ]
    )

    results = retriever.search("XPS-13-9310")
    assert len(results) >= 1
    assert results[0].id == "doc1"
    assert "XPS-13-9310" in results[0].text

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.search("")
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        retriever.search("test", top_k=-1)


@pytest.mark.asyncio
async def test_bm25_retriever_async(db):
    retriever = BM25Retriever(db)
    retriever.index_documents(
        [
            Document(id="d1", text="ConnectionResetError encountered"),
            Document(id="d2", text="HTTP 500 internal server error"),
        ]
    )

    results = await retriever.asearch("ConnectionResetError")
    assert len(results) == 1
    assert results[0].id == "d1"


def test_bm25_retriever_populate_from_store(db):
    retriever = BM25Retriever(db, populate_from_store=True)
    # The fake store has 4 documents seeded
    assert len(retriever.index) == 4

    results = retriever.search("XPS-13-9310")
    assert len(results) >= 1
    assert results[0].id == "doc1"


# ---------------------------------------------------------------------------
# BM25HybridRetriever tests
# ---------------------------------------------------------------------------


def test_hybrid_retriever_initialization_and_weights(db):
    retriever = BM25HybridRetriever(db, dense_weight=1.2, sparse_weight=0.6)
    assert retriever.dense_weight == 1.2
    assert retriever.sparse_weight == 0.6

    with pytest.raises(ValueError, match="weights must contain exactly 2 elements"):
        BM25HybridRetriever(db, weights=[1.0])

    with pytest.raises(ValueError, match="weights must be positive"):
        BM25HybridRetriever(db, weights=[-1.0, 1.0])

    # FitResult compatibility
    fit_res = FitResult(weights=[0.8, 0.4], score=0.95, method="grid", n_evaluations=10)
    retriever_fitted = BM25HybridRetriever(db, weights=fit_res)
    assert retriever_fitted.dense_weight == 0.8
    assert retriever_fitted.sparse_weight == 0.4


def test_hybrid_retriever_search_lexical_boost(db):
    # Seed BM25 retriever with same docs
    hybrid = BM25HybridRetriever(db, dense_weight=1.0, sparse_weight=1.5)
    hybrid.bm25_retriever.populate_from_store()

    # Query for exact model: "XPS-13-9310"
    # S3 vector search maps "XPS-13-9310" to E0 (ranking doc4 top),
    # but BM25 strongly ranks doc1 top. Fused RRF elevates doc1 to rank 1.
    results = hybrid.search("XPS-13-9310", top_k=2)
    assert len(results) > 0
    assert results[0].id == "doc1"


def test_hybrid_retriever_weights_override_and_fit_result(db):
    hybrid = BM25HybridRetriever(db, dense_weight=1.0, sparse_weight=0.5)
    hybrid.bm25_retriever.populate_from_store()

    fit_res = FitResult(weights=[1.5, 0.5], score=0.92, method="grid", n_evaluations=5)
    results = hybrid.search("laptop", weights=fit_res, top_k=3)
    assert len(results) > 0

    with pytest.raises(ValueError, match="weights must contain exactly 2 elements"):
        hybrid.search("laptop", weights=[1.0, 2.0, 3.0])


@pytest.mark.asyncio
async def test_hybrid_retriever_asearch(db):
    hybrid = BM25HybridRetriever(db)
    hybrid.bm25_retriever.populate_from_store()

    results = await hybrid.asearch("server Dell PowerEdge", top_k=2)
    assert len(results) > 0
    assert results[0].id == "doc3"


# ---------------------------------------------------------------------------
# Client & NamespaceView Ergonomics tests
# ---------------------------------------------------------------------------


def test_client_retriever_ergonomics(db):
    # as_bm25_retriever
    bm25 = db.as_bm25_retriever(populate_from_store=True)
    assert isinstance(bm25, BM25Retriever)
    assert len(bm25.index) == 4

    # as_hybrid_retriever
    hybrid = db.as_hybrid_retriever(dense_weight=1.0, sparse_weight=0.9, bm25_retriever=bm25)
    assert isinstance(hybrid, BM25HybridRetriever)
    assert hybrid.sparse_weight == 0.9

    # hybrid_search directly on client
    res = db.hybrid_search("laptop Dell XPS", bm25_retriever=bm25, top_k=2)
    assert len(res) <= 2
    assert res[0].id in ("doc1", "doc2")


def test_namespace_view_retriever_ergonomics(db):
    ns = db.namespace("default")

    # as_bm25_retriever on namespace
    bm25 = ns.as_bm25_retriever(populate_from_store=True)
    assert isinstance(bm25, BM25Retriever)
    assert bm25.namespace == "default"

    # as_hybrid_retriever on namespace
    hybrid = ns.as_hybrid_retriever(dense_weight=1.0, sparse_weight=0.7, bm25_retriever=bm25)
    assert isinstance(hybrid, BM25HybridRetriever)
    assert hybrid.namespace == "default"

    # hybrid_search on namespace
    res = ns.hybrid_search("ConnectionResetError", bm25_retriever=bm25, top_k=1)
    assert len(res) == 1
    assert res[0].id == "doc4"
