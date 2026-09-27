"""Ingestion layer: pull external content, chunk it, embed + store it.

The point of this module is to make "sucking in" data from anywhere trivial. A
**Source** is just an iterable of records ``{id, text, metadata}``. dynavec
chunks, embeds (via your configured embedder), and upserts them.

MCP connector
-------------
:class:`MCPResourceSource` turns **any MCP server** into a dynavec source by
walking its *resources* primitive — so a Notion / Confluence / Google-Drive /
Slack MCP server (or your own) becomes an embeddable corpus with no bespoke code.
It's duck-typed against the MCP Python SDK session API
(``list_resources`` / ``read_resource``), so you pass a live session and dynavec
consumes it. Tools and prompts primitives can be adapted the same way.
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import Dynavec
from .exceptions import MissingDependencyError
from .models import Document
from .transforms import Transform, TransformPipeline
from .utils import chunked

Metadata = dict[str, Any]


def _normalize_front_matter(value: Any) -> Any:
    """Recursively normalize parsed YAML front-matter for storage.

    ``yaml.safe_load()`` converts unquoted dates (``date: 2026-09-24``) into
    ``datetime.date`` / ``datetime.datetime`` objects, which DynamoDB's
    ``TypeSerializer`` rejects. Convert those to ISO 8601 strings, recurse
    through mappings and sequences, and leave strings, ints, floats, and
    booleans untouched.
    """
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _normalize_front_matter(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_front_matter(v) for v in value]
    return value


@dataclass
class Record:
    """One source document before chunking."""

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> Iterator[str]:
    """Sliding-window character chunks (a generator — memory stays flat)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    text = text or ""
    if not text:
        return
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size]
        if piece.strip():
            yield piece
        if start + chunk_size >= len(text):
            break


class IterableSource:
    """Wrap a list/iterable of records or dicts as a Source."""

    def __init__(self, records: Iterable[Record | dict[str, Any]]) -> None:
        self._records = records

    def __iter__(self) -> Iterator[Record]:
        for r in self._records:
            yield r if isinstance(r, Record) else Record(**r)


class PDFSource:
    """Yield one Record per text-bearing page in a PDF."""

    def __init__(self, path: str | Path) -> None:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise MissingDependencyError(
                "PDFSource",
                "pypdf",
                "ingest",
            ) from exc

        self._path = Path(path)
        self._reader_cls = PdfReader

    def __iter__(self) -> Iterator[Record]:
        reader = self._reader_cls(self._path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            path_str = self._path.as_posix()
            yield Record(
                id=f"{path_str}#page{page_number}",
                text=text,
                metadata={
                    "source": "pdf",
                    "path": path_str,
                    "page": page_number,
                },
            )


class DocxSource:
    """Yield one Record containing readable paragraph text from a Word document (.docx)."""

    def __init__(self, path: str | Path) -> None:
        try:
            import docx
        except ImportError as exc:
            raise MissingDependencyError(
                "DocxSource",
                "python-docx",
                "ingest",
            ) from exc

        self._path = Path(path)
        self._document_cls = docx.Document

    def __iter__(self) -> Iterator[Record]:
        doc = self._document_cls(str(self._path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        if not paragraphs:
            return

        text = "\n\n".join(paragraphs)
        path_str = self._path.as_posix()
        yield Record(
            id=path_str,
            text=text,
            metadata={
                "source": "docx",
                "path": path_str,
            },
        )


class PptxSource:
    """Yield one Record per slide in a PowerPoint presentation (.pptx)."""

    def __init__(self, path: str | Path) -> None:
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise MissingDependencyError(
                "PptxSource",
                "python-pptx",
                "ingest",
            ) from exc

        self._path = Path(path)
        self._presentation_cls = Presentation

    def __iter__(self) -> Iterator[Record]:
        prs = self._presentation_cls(str(self._path))
        path_str = self._path.as_posix()

        for slide_num, slide in enumerate(prs.slides, start=1):
            text_runs = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text and shape.text.strip():
                    text_runs.append(shape.text.strip())

            if not text_runs:
                continue

            slide_text = "\n".join(text_runs)
            yield Record(
                id=f"{path_str}#slide{slide_num}",
                text=slide_text,
                metadata={
                    "source": "pptx",
                    "path": path_str,
                    "slide": slide_num,
                },
            )


class XlsxSource:
    """Yield one Record per data row in each worksheet of an Excel workbook (.xlsx).

    The first row of each sheet is treated as the header; each subsequent
    row is rendered as ``"column: value"`` pairs (skipping blank cells) so
    a chunk still stands on its own once split off from the rest.
    """

    def __init__(self, path: str | Path) -> None:
        try:
            import openpyxl
        except ImportError as exc:
            raise MissingDependencyError(
                "XlsxSource",
                "openpyxl",
                "ingest",
            ) from exc

        self._path = Path(path)
        self._load_workbook = openpyxl.load_workbook

    def __iter__(self) -> Iterator[Record]:
        wb = self._load_workbook(self._path, data_only=True)
        path_str = self._path.as_posix()

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            row_iter = iter(sheet.iter_rows(values_only=True))

            try:
                header = next(row_iter)
            except StopIteration:
                continue
            header = [str(cell).strip() if cell is not None else "" for cell in header]

            for row_number, row in enumerate(row_iter, start=1):
                pairs = [
                    f"{col}: {val}"
                    for col, val in zip(header, row)
                    if val is not None and str(val).strip()
                ]
                text = ", ".join(pairs)

                if not text:
                    continue

                yield Record(
                    id=f"{path_str}#{sheet_name}#row{row_number}",
                    text=text,
                    metadata={
                        "source": "xlsx",
                        "path": path_str,
                        "sheet": sheet_name,
                        "row": row_number,
                    },
                )


class CsvSource:
    """Yield one Record per data row in a CSV file (first row = header).

    Each row is rendered as ``"column: value"`` pairs (skipping blank
    cells), mirroring XlsxSource's row format.
    """

    def __init__(self, path: str | Path, *, delimiter: str = ",") -> None:
        self._path = Path(path)
        self._delimiter = delimiter

    def __iter__(self) -> Iterator[Record]:
        import csv

        path_str = self._path.as_posix()

        with self._path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh, delimiter=self._delimiter)

            try:
                header = next(reader)
            except StopIteration:
                return

            for row_number, row in enumerate(reader, start=1):
                pairs = [f"{col}: {val}" for col, val in zip(header, row) if val and val.strip()]
                text = ", ".join(pairs)

                if not text:
                    continue

                yield Record(
                    id=f"{path_str}#row{row_number}",
                    text=text,
                    metadata={
                        "source": "csv",
                        "path": path_str,
                        "row": row_number,
                    },
                )


class URLSource:
    """Yield one Record containing readable text extracted from a URL."""

    def __init__(self, url: str, timeout: float = 10) -> None:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise MissingDependencyError(
                "URLSource",
                "requests",
                "ingest",
            ) from exc
        self._url = url
        self._timeout = timeout
        self._requests = requests
        self._parser_cls = BeautifulSoup

    def __iter__(self) -> Iterator[Record]:
        response = self._requests.get(
            self._url,
            timeout=self._timeout,
            headers={"User-Agent": "dynavec/1.0"},
        )
        response.raise_for_status()
        soup = self._parser_cls(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        page_text = soup.get_text(separator=" ", strip=True)
        if not page_text:
            return
        yield Record(
            id=self._url,
            text=page_text,
            metadata={
                "source": "url",
                "url": self._url,
            },
        )


class MarkdownSource:
    """Read UTF-8 Markdown and text files from a directory.

    ``glob`` is relative to ``root`` and defaults to recursive discovery.
    Records use root-relative POSIX paths as IDs. Markdown YAML front matter
    becomes metadata and is excluded from the text. The ``source`` and ``path``
    metadata fields are reserved for file provenance. Install ``dynavec[ingest]``
    to read front matter; files without it need no optional dependencies.
    """

    def __init__(self, root: str | Path, *, glob: str = "**/*") -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.glob = glob

    @staticmethod
    def _front_matter(text: str, path: Path) -> tuple[str, Metadata]:
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            return text, {}
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            raise ValueError(f"Unclosed YAML front matter in {path}")
        try:
            import yaml
        except ImportError as exc:
            raise MissingDependencyError("Markdown front matter", "PyYAML", "ingest") from exc

        try:
            metadata = yaml.safe_load("".join(lines[1:end]))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML front matter in {path}: {exc}") from exc
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict) or any(not isinstance(k, str) for k in metadata):
            raise ValueError(f"Front matter in {path} must be a mapping with string keys")
        # yaml.safe_load() turns unquoted dates into datetime.date/datetime
        # objects, which storage backends (e.g. DynamoDB) cannot serialize.
        return "".join(lines[end + 1 :]), _normalize_front_matter(metadata)

    def __iter__(self) -> Iterator[Record]:
        for path in sorted(self.root.glob(self.glob)):
            if not path.is_file() or path.suffix.lower() not in (".md", ".txt"):
                continue
            text = path.read_text(encoding="utf-8-sig")
            metadata: Metadata = {}
            if path.suffix.lower() == ".md":
                text, metadata = self._front_matter(text, path)
            relative_path = path.relative_to(self.root).as_posix()
            yield Record(
                id=relative_path,
                text=text,
                metadata={**metadata, "source": "file", "path": relative_path},
            )


class MCPResourceSource:
    """Adapt an MCP server's *resources* into dynavec records.

    Parameters
    ----------
    session:
        A connected MCP client session exposing ``list_resources()`` and
        ``read_resource(uri)`` (the standard MCP primitives). Duck-typed so it
        works with the official SDK or a compatible wrapper.
    uri_filter:
        Optional predicate ``(uri) -> bool`` to select which resources to pull.
    """

    def __init__(
        self,
        session: Any,
        uri_filter: Callable[[str], bool] | None = None,
    ) -> None:
        self._session = session
        self._uri_filter = uri_filter

    @staticmethod
    def _extract_text(contents: Any) -> str:
        # MCP read_resource returns an object/list of content parts; grab text.
        parts = getattr(contents, "contents", contents)
        if isinstance(parts, (list, tuple)):
            texts = []
            for p in parts:
                t = getattr(p, "text", None)
                if t is None and isinstance(p, dict):
                    t = p.get("text")
                if t:
                    texts.append(t)
            return "\n\n".join(texts)
        return getattr(parts, "text", "") or ""

    def __iter__(self) -> Iterator[Record]:
        listing = self._session.list_resources()
        resources = getattr(listing, "resources", listing)
        for res in resources:
            uri = getattr(res, "uri", None) or (res.get("uri") if isinstance(res, dict) else None)
            if uri is None:
                continue
            if self._uri_filter and not self._uri_filter(str(uri)):
                continue
            name = getattr(res, "name", None) or (
                res.get("name") if isinstance(res, dict) else None
            )
            contents = self._session.read_resource(uri)
            text = self._extract_text(contents)
            if not text:
                continue
            yield Record(
                id=str(uri),
                text=text,
                metadata={"source": "mcp", "uri": str(uri), "name": name},
            )


class S3Source:
    """Ingest documents stored in an Amazon S3 bucket.

    Discovers objects in ``bucket`` under ``prefix``, streaming and decoding
    them into dynavec records according to file extension or content-type.

    Supported formats:
    - Text / Markdown (``.txt``, ``.md``, ``text/plain``, ``text/markdown``)
      (Markdown YAML front matter is extracted into metadata)
    - CSV (``.csv``, ``text/csv``)
    - PDF (``.pdf``, ``application/pdf``) — requires ``pypdf``
    - Word documents (``.docx``) — requires ``python-docx``
    - PowerPoint presentations (``.pptx``) — requires ``python-pptx``
    - Excel spreadsheets (``.xlsx``) — requires ``openpyxl``

    Parameters
    ----------
    bucket:
        Name of the S3 bucket.
    prefix:
        Key prefix to filter objects (default: ``""``).
    suffix:
        Optional file extension or tuple of extensions to include (e.g. ``".md"`` or ``(".md", ".txt")``).
    boto_session:
        Optional custom ``boto3.Session``.
    s3_client:
        Optional pre-configured S3 client (useful for dependency injection or testing).
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        suffix: str | tuple[str, ...] | None = None,
        boto_session: Any = None,
        s3_client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.suffix = suffix
        self._session = boto_session
        self._client = s3_client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import boto3

        session = self._session or boto3.Session()
        return session.client("s3")

    def _matches_suffix(self, key: str) -> bool:
        if self.suffix is None:
            return True
        key_lower = key.lower()
        if isinstance(self.suffix, str):
            return key_lower.endswith(self.suffix.lower())
        return any(key_lower.endswith(s.lower()) for s in self.suffix)

    def _parse_object(self, key: str, data: bytes, content_type: str) -> Iterator[Record]:
        import csv
        import io

        ext = Path(key).suffix.lower()
        record_id = f"s3://{self.bucket}/{key}"
        base_meta: Metadata = {"source": "s3", "bucket": self.bucket, "key": key}

        # 1. Markdown
        if ext == ".md" or "text/markdown" in content_type:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                return
            clean_text, front_meta = MarkdownSource._front_matter(text, Path(key))
            if clean_text.strip():
                yield Record(
                    id=record_id,
                    text=clean_text,
                    metadata={**base_meta, **front_meta, "format": "md"},
                )
            return

        # 2. CSV
        if ext == ".csv" or "text/csv" in content_type:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                return
            lines = text.splitlines()
            csv_reader = csv.reader(lines)
            try:
                header = next(csv_reader)
            except StopIteration:
                return
            for row_number, row in enumerate(csv_reader, start=1):
                pairs = [f"{col}: {val}" for col, val in zip(header, row) if val and val.strip()]
                row_text = ", ".join(pairs)
                if row_text:
                    yield Record(
                        id=f"{record_id}#row{row_number}",
                        text=row_text,
                        metadata={**base_meta, "format": "csv", "row": row_number},
                    )
            return

        # 3. PDF
        if ext == ".pdf" or "application/pdf" in content_type:
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise MissingDependencyError("S3Source PDF parsing", "pypdf", "ingest") from exc

            pdf_reader = PdfReader(io.BytesIO(data))
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    yield Record(
                        id=f"{record_id}#page{page_number}",
                        text=page_text,
                        metadata={**base_meta, "format": "pdf", "page": page_number},
                    )
            return

        # 4. Word (.docx)
        if ext == ".docx" or "wordprocessingml.document" in content_type:
            try:
                import docx
            except ImportError as exc:
                raise MissingDependencyError(
                    "S3Source DOCX parsing", "python-docx", "ingest"
                ) from exc

            doc = docx.Document(io.BytesIO(data))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
            if paragraphs:
                yield Record(
                    id=record_id,
                    text="\n\n".join(paragraphs),
                    metadata={**base_meta, "format": "docx"},
                )
            return

        # 5. PowerPoint (.pptx)
        if ext == ".pptx" or "presentationml.presentation" in content_type:
            try:
                from pptx import Presentation
            except ImportError as exc:
                raise MissingDependencyError(
                    "S3Source PPTX parsing", "python-pptx", "ingest"
                ) from exc

            prs = Presentation(io.BytesIO(data))
            for slide_num, slide in enumerate(prs.slides, start=1):
                text_runs = [
                    s.text.strip()
                    for s in slide.shapes
                    if hasattr(s, "text") and s.text and s.text.strip()
                ]
                if text_runs:
                    yield Record(
                        id=f"{record_id}#slide{slide_num}",
                        text="\n".join(text_runs),
                        metadata={**base_meta, "format": "pptx", "slide": slide_num},
                    )
            return

        # 6. Excel (.xlsx)
        if ext == ".xlsx" or "spreadsheetml.sheet" in content_type:
            try:
                import openpyxl
            except ImportError as exc:
                raise MissingDependencyError("S3Source XLSX parsing", "openpyxl", "ingest") from exc

            wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                row_iter = iter(sheet.iter_rows(values_only=True))
                try:
                    header = next(row_iter)
                except StopIteration:
                    continue
                header = [str(c).strip() if c is not None else "" for c in header]
                for row_number, row in enumerate(row_iter, start=1):
                    pairs = [
                        f"{col}: {val}"
                        for col, val in zip(header, row)
                        if val is not None and str(val).strip()
                    ]
                    row_text = ", ".join(pairs)
                    if row_text:
                        yield Record(
                            id=f"{record_id}#{sheet_name}#row{row_number}",
                            text=row_text,
                            metadata={
                                **base_meta,
                                "format": "xlsx",
                                "sheet": sheet_name,
                                "row": row_number,
                            },
                        )
            return

        # 7. Plain text or generic fallback
        try:
            text = data.decode("utf-8-sig")
            if text and text.strip():
                yield Record(
                    id=record_id,
                    text=text,
                    metadata={**base_meta, "format": "text"},
                )
        except UnicodeDecodeError:
            pass

    def __iter__(self) -> Iterator[Record]:
        client = self._get_client()
        kwargs: dict[str, Any] = {"Bucket": self.bucket}
        if self.prefix:
            kwargs["Prefix"] = self.prefix

        continuation_token: str | None = None
        while True:
            req_kwargs = dict(kwargs)
            if continuation_token:
                req_kwargs["ContinuationToken"] = continuation_token

            resp = client.list_objects_v2(**req_kwargs)
            for obj in resp.get("Contents", []):
                key = obj.get("Key", "")
                if not key or key.endswith("/"):
                    continue
                if not self._matches_suffix(key):
                    continue

                obj_resp = client.get_object(Bucket=self.bucket, Key=key)
                content_type = (obj_resp.get("ContentType") or "").lower()
                body = obj_resp["Body"]
                data = body.read() if hasattr(body, "read") else bytes(body)

                yield from self._parse_object(key, data, content_type)

            if resp.get("IsTruncated"):
                continuation_token = resp.get("NextContinuationToken")
                if not continuation_token:
                    break
            else:
                break


def ingest(
    db: Dynavec,
    source: Iterable[Record | dict[str, Any]],
    *,
    namespace: str = "default",
    chunk_size: int = 1000,
    overlap: int = 150,
    batch_size: int = 256,
    auto_metadata: bool = True,
    transform: TransformPipeline | Transform | Iterable[Transform] | None = None,
) -> int:
    """Pull records from ``source``, chunk, embed, and upsert. Returns #chunks.

    Chunk ids are ``"{record_id}#chunk{n}"`` and each carries ``source_id`` /
    ``chunk`` metadata so you can group or delete a whole document later.
    """

    seen_hashes: set[str] = set()

    def _documents() -> Iterator[Document]:
        for rec in source:
            rec = rec if isinstance(rec, Record) else Record(**rec)
            for n, piece in enumerate(chunk_text(rec.text, chunk_size, overlap)):
                content_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                yield Document(
                    id=f"{rec.id}#chunk{n}",
                    text=piece,
                    metadata={**rec.metadata, "source_id": rec.id, "chunk": n},
                )

    total = 0
    for batch in chunked(_documents(), batch_size):
        res = db.upsert(
            batch, namespace=namespace, auto_metadata=auto_metadata, transform=transform
        )
        total += res.count
    return total


__all__ = [
    "Record",
    "chunk_text",
    "ingest",
    "IterableSource",
    "PDFSource",
    "DocxSource",
    "PptxSource",
    "XlsxSource",
    "CsvSource",
    "URLSource",
    "MarkdownSource",
    "MCPResourceSource",
    "S3Source",
]
