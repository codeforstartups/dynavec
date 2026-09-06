"""Tests for chunking + the MCP resource source (pure, no AWS)."""

from dynavec.ingest import MCPResourceSource, Record, chunk_text, ingest
from dynavec.models import UpsertResult


def test_chunk_text_windows_with_overlap():
    text = "abcdefghij"  # length 10
    chunks = list(chunk_text(text, chunk_size=4, overlap=1))
    # step = 3 -> starts at 0,3,6,9
    assert chunks[0] == "abcd"
    assert chunks[1] == "defg"
    assert all(len(c) <= 4 for c in chunks)


def test_chunk_text_whitespace_only_returns_no_chunks():
    assert list(chunk_text("   \n\t")) == []


def test_chunk_text_shorter_than_chunk_size_returns_one_chunk():
    assert list(chunk_text("hello", chunk_size=100, overlap=10)) == ["hello"]


def test_chunk_text_empty_and_validation():
    assert list(chunk_text("")) == []
    import pytest

    with pytest.raises(ValueError):
        list(chunk_text("x", chunk_size=0))
    with pytest.raises(ValueError):
        list(chunk_text("x", chunk_size=4, overlap=4))


# ---- fake MCP session mirroring the SDK's list_resources / read_resource ----
class _Res:
    def __init__(self, uri, name):
        self.uri = uri
        self.name = name


class _Part:
    def __init__(self, text):
        self.text = text


class _Contents:
    def __init__(self, text):
        self.contents = [_Part(text)]


class FakeMCPSession:
    def __init__(self, docs):
        self._docs = docs  # {uri: (name, text)}

    def list_resources(self):
        return [_Res(uri, name) for uri, (name, _t) in self._docs.items()]

    def read_resource(self, uri):
        return _Contents(self._docs[uri][1])


def test_mcp_resource_source_yields_records():
    session = FakeMCPSession(
        {
            "notion://page/1": ("Roadmap", "Q1 plans and OKRs"),
            "confluence://doc/2": ("Runbook", "How to deploy the service"),
        }
    )
    records = list(MCPResourceSource(session))
    assert len(records) == 2
    assert all(isinstance(r, Record) for r in records)
    by_uri = {r.id: r for r in records}
    assert by_uri["notion://page/1"].text == "Q1 plans and OKRs"
    assert by_uri["notion://page/1"].metadata["source"] == "mcp"


def test_mcp_uri_filter():
    session = FakeMCPSession(
        {"notion://a": ("A", "x"), "confluence://b": ("B", "y")}
    )
    records = list(MCPResourceSource(session, uri_filter=lambda u: u.startswith("notion")))
    assert [r.id for r in records] == ["notion://a"]


class FakeIngestDB:
    def __init__(self):
        self.documents = []

    def upsert(self, documents, **kwargs):
        self.documents.extend(documents)
        return UpsertResult(count=len(documents), ids=[document.id for document in documents])


def test_ingest_deduplicates_identical_chunks_within_run():
    db = FakeIngestDB()
    source = [
        Record(id="doc-a", text="duplicate text"),
        Record(id="doc-b", text="duplicate text"),
        Record(id="doc-c", text="unique text"),
    ]

    count = ingest(db, source, chunk_size=100, overlap=0)

    assert count == 2
    assert [document.text for document in db.documents] == [
        "duplicate text",
        "unique text",
    ]