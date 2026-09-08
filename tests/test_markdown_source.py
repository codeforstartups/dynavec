import sys
from unittest.mock import Mock

import pytest

from dynavec.exceptions import MissingDependencyError
from dynavec.ingest import MarkdownSource, ingest
from dynavec.models import UpsertResult


def test_recursive_discovery_and_relative_ids(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (tmp_path / "notes" / "guide.md").write_text("Nested guide", encoding="utf-8")
    (tmp_path / "notes" / "plain.txt").write_text("Plain text", encoding="utf-8")
    (tmp_path / "ignore.html").write_text("<p>Ignored</p>", encoding="utf-8")
    (tmp_path / "directory.md").mkdir()

    records = list(MarkdownSource(tmp_path))

    assert [r.id for r in records] == ["guide.md", "notes/guide.md", "notes/plain.txt"]
    assert [r.text for r in records] == ["# Guide\n", "Nested guide", "Plain text"]
    assert records[1].metadata == {"source": "file", "path": "notes/guide.md"}
    assert list(MarkdownSource(tmp_path)) == records


def test_glob_restricts_discovery(tmp_path):
    (tmp_path / "notes").mkdir()
    for name in ("root.md", "notes/a.md", "notes/b.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    records = list(MarkdownSource(tmp_path, glob="notes/*.md"))

    assert [r.id for r in records] == ["notes/a.md"]


def test_front_matter_becomes_metadata(tmp_path):
    (tmp_path / "guide.md").write_text(
        "---\ntitle: Guide\ntags: [aws, rag]\nyear: 2026\npublished: true\n"
        "source: forged\npath: elsewhere\n---\n# Hello\n\nBody\n---\n",
        encoding="utf-8-sig",
    )

    record = next(iter(MarkdownSource(tmp_path)))

    assert record.text == "# Hello\n\nBody\n---\n"
    assert record.metadata == {
        "title": "Guide",
        "tags": ["aws", "rag"],
        "year": 2026,
        "published": True,
        "source": "file",
        "path": "guide.md",
    }


def test_text_file_does_not_parse_front_matter(tmp_path):
    text = "---\nthis is plain text\n---\n"
    (tmp_path / "notes.txt").write_text(text, encoding="utf-8")

    record = next(iter(MarkdownSource(tmp_path)))

    assert record.text == text
    assert record.metadata == {"source": "file", "path": "notes.txt"}


def test_unicode_and_empty_front_matter(tmp_path):
    (tmp_path / "guide.MD").write_bytes("---\r\n---\r\nCafé 日本語\r\n".encode())

    record = next(iter(MarkdownSource(tmp_path)))

    assert record.text == "Café 日本語\n"
    assert record.metadata == {"source": "file", "path": "guide.MD"}


@pytest.mark.parametrize(
    "text, message",
    [
        ("---\ntitle: missing delimiter\n", "Unclosed YAML"),
        ("---\ntitle: [broken\n---\nBody", "Invalid YAML"),
        ("---\n- a\n- b\n---\nBody", "mapping with string keys"),
        ("---\n42: value\n---\nBody", "mapping with string keys"),
        ("---\n!!python/object:builtins.object {}\n---\nBody", "Invalid YAML"),
    ],
)
def test_invalid_front_matter_reports_filename(tmp_path, text, message):
    (tmp_path / "broken.md").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as exc:
        list(MarkdownSource(tmp_path))

    assert "broken.md" in str(exc.value)


def test_yaml_dependency_is_only_needed_for_front_matter(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    path = tmp_path / "guide.md"
    path.write_text("# Plain Markdown", encoding="utf-8")
    assert next(iter(MarkdownSource(tmp_path))).text == "# Plain Markdown"

    path.write_text("---\ntitle: Guide\n---\nBody", encoding="utf-8")
    with pytest.raises(MissingDependencyError, match=r"dynavec\[ingest\]"):
        list(MarkdownSource(tmp_path))


def test_root_must_be_an_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        MarkdownSource(tmp_path / "missing")
    path = tmp_path / "file.txt"
    path.write_text("Text", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        MarkdownSource(path)


def test_empty_directory_yields_no_records(tmp_path):
    assert list(MarkdownSource(tmp_path)) == []


def test_markdown_source_flows_through_ingest(tmp_path):
    (tmp_path / "guide.md").write_text("---\ntopic: aws\n---\nabcdefghij", encoding="utf-8")
    db = Mock()
    db.upsert.side_effect = lambda docs, **kw: UpsertResult(
        count=len(docs), ids=[doc.id for doc in docs]
    )

    count = ingest(db, MarkdownSource(tmp_path), namespace="docs", chunk_size=6, overlap=2)

    assert count == 2
    docs = db.upsert.call_args.args[0]
    assert [doc.id for doc in docs] == ["guide.md#chunk0", "guide.md#chunk1"]
    assert [doc.text for doc in docs] == ["abcdef", "efghij"]
    assert docs[1].metadata == {
        "topic": "aws",
        "source": "file",
        "path": "guide.md",
        "source_id": "guide.md",
        "chunk": 1,
    }
    assert db.upsert.call_args.kwargs["namespace"] == "docs"
