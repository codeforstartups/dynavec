"""Tests for chunking + MCP/PDF ingestion sources (pure, no AWS)."""

import sys
from types import SimpleNamespace

from dynavec.exceptions import MissingDependencyError
from dynavec.ingest import MCPResourceSource, PDFSource, URLSource, Record, chunk_text, ingest
from dynavec.models import UpsertResult

import requests


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


class _PDFPage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _PDFReader:
    def __init__(self, path):
        self.path = path
        self.pages = [
            _PDFPage("First page text"),
            _PDFPage("   \n"),
            _PDFPage("Third page text"),
        ]


def test_pdf_source_yields_page_records(monkeypatch):
    fake_pypdf = SimpleNamespace(PdfReader=_PDFReader)
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    records = list(PDFSource("docs/sample.pdf"))

    assert len(records) == 2

    assert records[0].id == "docs/sample.pdf#page1"
    assert records[0].text == "First page text"
    assert records[0].metadata == {
        "source": "pdf",
        "path": "docs/sample.pdf",
        "page": 1,
    }

    assert records[1].id == "docs/sample.pdf#page3"
    assert records[1].text == "Third page text"
    assert records[1].metadata["page"] == 3


def test_pdf_source_missing_dependency(monkeypatch):
    import pytest

    monkeypatch.setitem(sys.modules, "pypdf", None)

    with pytest.raises(MissingDependencyError, match=r"dynavec\[ingest\]"):
        PDFSource("sample.pdf")



def test_url_source_yields_readable_text(monkeypatch):
    class FakeResponse:
        text = """
        <html>
            <head>
                <script>alert("ignore me")</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <h1>Hello Dynavec</h1>
                <p>This is useful content.</p>
            </body>
        </html>
        """

        def raise_for_status(self):
            pass

    def fake_get(url, timeout, headers):
        assert url == "https://example.com"
        assert timeout == 10
        assert headers["User-Agent"] == "dynavec/1.0"
        return FakeResponse()

    monkeypatch.setattr("dynavec.ingest.requests.get", fake_get)

    records = list(URLSource("https://example.com"))

    assert len(records) == 1
    assert records[0].id == "https://example.com"
    assert "Hello Dynavec" in records[0].text
    assert "This is useful content." in records[0].text
    assert "alert" not in records[0].text
    assert "color: red" not in records[0].text
    assert records[0].metadata == {
        "source": "url",
        "url": "https://example.com",
    }


def test_url_source_raises_for_http_error(monkeypatch):
    import pytest
    class FakeResponse:
        text = ""

        def raise_for_status(self):
            raise requests.HTTPError("404 Not Found")

    def fake_get(url, timeout, headers):
        return FakeResponse()

    monkeypatch.setattr("dynavec.ingest.requests.get", fake_get)

    with pytest.raises(requests.HTTPError):
        list(URLSource("https://example.com/missing"))


def test_url_source_skips_empty_pages(monkeypatch):
    class FakeResponse:
        text = "<html><body></body></html>"

        def raise_for_status(self):
            pass

    def fake_get(url, timeout, headers):
        return FakeResponse()

    monkeypatch.setattr("dynavec.ingest.requests.get", fake_get)

    records = list(URLSource("https://example.com"))

    assert records == []


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
