"""Tests for chunking + MCP/PDF ingestion sources (pure, no AWS)."""

import sys
from types import SimpleNamespace

import requests

from dynavec.exceptions import MissingDependencyError
from dynavec.ingest import (
    DocxSource,
    MCPResourceSource,
    PDFSource,
    PptxSource,
    Record,
    URLSource,
    XlsxSource,
    chunk_text,
    ingest,
)
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



class _FakeSoup:
    def __init__(self, text, parser):
        self._text = text

    def __call__(self, tags):
        return []

    def get_text(self, separator=" ", strip=True):
        if "Hello Dynavec" in self._text:
            return "Hello Dynavec This is useful content."
        return ""

def test_url_source_yields_readable_text(monkeypatch):
    monkeypatch.setitem(sys.modules, "bs4", SimpleNamespace(BeautifulSoup=_FakeSoup))

    class FakeResponse:
        text = """
        <html>
            <body>
                <h1>Hello Dynavec</h1>
                <p>This is useful content.</p>
            </body>
        </html>
        """

        def raise_for_status(self):
            pass

    fake_requests = SimpleNamespace(get=lambda url, timeout, headers: FakeResponse())
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    records = list(URLSource("https://example.com"))

    assert len(records) == 1
    assert records[0].id == "https://example.com"
    assert "Hello Dynavec" in records[0].text
    assert "This is useful content." in records[0].text
    assert records[0].metadata == {
        "source": "url",
        "url": "https://example.com",
    }


def test_url_source_raises_for_http_error(monkeypatch):
    import pytest
    monkeypatch.setitem(sys.modules, "bs4", SimpleNamespace(BeautifulSoup=_FakeSoup))

    class FakeHTTPError(Exception):
        pass

    class FakeResponse:
        text = ""

        def raise_for_status(self):
            raise FakeHTTPError("404 Not Found")

    fake_requests = SimpleNamespace(
        get=lambda url, timeout, headers: FakeResponse(),
        HTTPError=FakeHTTPError,
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    with pytest.raises(FakeHTTPError):
        list(URLSource("https://example.com/missing"))


def test_url_source_skips_empty_pages(monkeypatch):
    monkeypatch.setitem(sys.modules, "bs4", SimpleNamespace(BeautifulSoup=_FakeSoup))

    class FakeResponse:
        text = "<html><body></body></html>"

        def raise_for_status(self):
            pass

    fake_requests = SimpleNamespace(get=lambda url, timeout, headers: FakeResponse())
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

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


def test_docx_source_yields_records(monkeypatch):
    class _Paragraph:
        def __init__(self, text):
            self.text = text

    class _DocxDocument:
        def __init__(self, path):
            self.paragraphs = [
                _Paragraph("Heading 1"),
                _Paragraph("   "),
                _Paragraph("Main content paragraph."),
            ]

    fake_docx = SimpleNamespace(Document=_DocxDocument)
    monkeypatch.setitem(sys.modules, "docx", fake_docx)

    records = list(DocxSource("docs/sample.docx"))
    assert len(records) == 1
    assert records[0].id == "docs/sample.docx"
    assert "Heading 1" in records[0].text
    assert "Main content paragraph." in records[0].text
    assert records[0].metadata == {
        "source": "docx",
        "path": "docs/sample.docx",
    }


def test_docx_source_missing_dependency(monkeypatch):
    import pytest

    monkeypatch.setitem(sys.modules, "docx", None)
    with pytest.raises(MissingDependencyError, match=r"dynavec\[ingest\]"):
        DocxSource("sample.docx")


def test_pptx_source_yields_slide_records(monkeypatch):
    class _Shape:
        def __init__(self, text):
            self.text = text

    class _Slide:
        def __init__(self, texts):
            self.shapes = [_Shape(t) for t in texts]

    class _Presentation:
        def __init__(self, path):
            self.slides = [
                _Slide(["Title Slide", "Subtitle"]),
                _Slide([]),
                _Slide(["Content Slide"]),
            ]

    fake_pptx = SimpleNamespace(Presentation=_Presentation)
    monkeypatch.setitem(sys.modules, "pptx", fake_pptx)

    records = list(PptxSource("docs/sample.pptx"))
    assert len(records) == 2
    assert records[0].id == "docs/sample.pptx#slide1"
    assert records[0].text == "Title Slide\nSubtitle"
    assert records[0].metadata["slide"] == 1
    assert records[1].id == "docs/sample.pptx#slide3"
    assert records[1].text == "Content Slide"
    assert records[1].metadata["slide"] == 3


def test_pptx_source_missing_dependency(monkeypatch):
    import pytest

    monkeypatch.setitem(sys.modules, "pptx", None)
    with pytest.raises(MissingDependencyError, match=r"dynavec\[ingest\]"):
        PptxSource("sample.pptx")


def test_xlsx_source_yields_sheet_records(monkeypatch):
    class _Sheet:
        def __init__(self, rows):
            self._rows = rows

        def iter_rows(self, values_only=True):
            return self._rows

    class _Workbook:
        def __init__(self):
            self.sheetnames = ["Summary", "EmptySheet", "Data"]
            self._sheets = {
                "Summary": _Sheet([("Header 1", "Header 2"), ("Val A", 100)]),
                "EmptySheet": _Sheet([]),
                "Data": _Sheet([("Row 1", None, "Col 3")]),
            }

        def __getitem__(self, item):
            return self._sheets[item]

    def fake_load_workbook(path, data_only=True):
        return _Workbook()

    fake_openpyxl = SimpleNamespace(load_workbook=fake_load_workbook)
    monkeypatch.setitem(sys.modules, "openpyxl", fake_openpyxl)

    records = list(XlsxSource("docs/sample.xlsx"))
    assert len(records) == 2
    assert records[0].id == "docs/sample.xlsx#Summary"
    assert "Header 1 | Header 2" in records[0].text
    assert "Val A | 100" in records[0].text
    assert records[0].metadata == {
        "source": "xlsx",
        "path": "docs/sample.xlsx",
        "sheet": "Summary",
    }
    assert records[1].id == "docs/sample.xlsx#Data"
    assert records[1].text == "Row 1 | Col 3"


def test_xlsx_source_missing_dependency(monkeypatch):
    import pytest

    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(MissingDependencyError, match=r"dynavec\[ingest\]"):
        XlsxSource("sample.xlsx")

