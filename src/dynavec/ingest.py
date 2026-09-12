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

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .client import Dynavec
from .exceptions import MissingDependencyError
from .models import Document
from .utils import chunked

Metadata = dict[str, Any]


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

    def __init__(self, records: Iterable) -> None:
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

class URLSource:
    """Yield one Record containing readable text extracted from a URL."""
    def __init__(self, url: str, timeout: float=10)->None:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise MissingDependencyError(
                "URLSource",
                "beautifulsoup4",
                "ingest"
            ) from exc
        self._url= url
        self._timeout= timeout
        self._parser_cls= BeautifulSoup
    
    def __iter__(self)-> Iterator[Record]:
        response=requests.get(
            self._url,
            timeout=self._timeout,
            headers={"User-Agent": "dynavec/1.0"},
        )
        response.raise_for_status()
        soup=self._parser_cls(response.text,"html.parser")  
        for tag in soup(["script","style"]):
            tag.decompose()
        page_text=soup.get_text(separator=" ",strip=True)
        if not page_text:
            return
        yield Record(
            id= self._url,
            text= page_text,
            metadata={
                "source": "url",
                "url": self._url
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
            raise MissingDependencyError(
                "Markdown front matter", "PyYAML", "ingest"
            ) from exc

        try:
            metadata = yaml.safe_load("".join(lines[1:end]))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML front matter in {path}: {exc}") from exc
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict) or any(not isinstance(k, str) for k in metadata):
            raise ValueError(f"Front matter in {path} must be a mapping with string keys")
        return "".join(lines[end + 1 :]), metadata

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

    def __init__(self, session, uri_filter=None) -> None:
        self._session = session
        self._uri_filter = uri_filter

    @staticmethod
    def _extract_text(contents) -> str:
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
            name = getattr(res, "name", None) or (res.get("name") if isinstance(res, dict) else None)
            contents = self._session.read_resource(uri)
            text = self._extract_text(contents)
            if not text:
                continue
            yield Record(
                id=str(uri),
                text=text,
                metadata={"source": "mcp", "uri": str(uri), "name": name},
            )


def ingest(
    db: Dynavec,
    source: Iterable,
    *,
    namespace: str = "default",
    chunk_size: int = 1000,
    overlap: int = 150,
    batch_size: int = 256,
    auto_metadata: bool = True,
    transform=None,
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
