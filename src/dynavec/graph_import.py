"""Prevalidate graph records and files before the storage layer writes anything."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any

from boto3.dynamodb.types import Binary, TypeSerializer

from .stores.dynamodb import _to_dynamo
from .utils import KEY_SEPARATOR, encode_key_component


@dataclass(frozen=True)
class GraphImportResult:
    """Successful import counts; processed nodes may already have existed."""

    nodes_processed: int
    edges_appended: int


@dataclass
class _PreparedImport:
    nodes: dict[str, dict[str, Any]]
    edges: dict[str, list[dict[str, Any]]]
    edge_count: int


def _dynamo_value(value: Any, depth: int = 0) -> Any:
    """Copy supported values, converting floats (including in sets) to Decimal."""
    if isinstance(value, dict):
        if depth > 32:
            raise ValueError("DynamoDB documents cannot exceed 32 nested levels")
        if any(not isinstance(key, str) for key in value):
            raise ValueError("property map keys must be strings")
        for key in value:
            key.encode("utf-8")
        return {key: _dynamo_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if depth > 32:
            raise ValueError("DynamoDB documents cannot exceed 32 nested levels")
        return [_dynamo_value(item, depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        if not value:
            raise ValueError("DynamoDB does not support empty sets")
        # Booleans are valid scalars, but are not DynamoDB number-set members.
        if any(isinstance(item, bool) for item in value):
            raise ValueError("DynamoDB does not support boolean set members")
        return {_dynamo_value(item) for item in value}
    if isinstance(value, Binary):
        return Binary(bytes(value.value))
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        value.encode("utf-8")
    value = _to_dynamo(value)
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError("numbers must be finite")
    return value


def _properties(value: Any, depth: int) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("props must be a dictionary or None")
    copied: dict[str, Any] = _dynamo_value(value, depth)
    # Use boto3's actual serializer to check numeric precision/range, unsupported
    # objects and mixed sets. Never silently stringify unsupported objects.
    TypeSerializer().serialize(copied)
    return copied


def _identifier(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    value.encode("utf-8")
    return value


def _node_key(ns: str, entity_id: str) -> None:
    key = KEY_SEPARATOR.join((encode_key_component(ns), "node", encode_key_component(entity_id)))
    if len(key.encode("utf-8")) > 2048:
        raise ValueError("encoded node partition key exceeds DynamoDB's 2048-byte limit")


def _records(value: Any, name: str) -> Sequence[Any]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of records")
    return value


def _prepare_import(
    ns: str,
    nodes: Any,
    edges: Any,
    batch_size: int,
    *,
    node_contexts: list[str] | None = None,
    edge_contexts: list[str] | None = None,
) -> _PreparedImport:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(ns, str) or not ns:
        raise ValueError("namespace must be a non-empty string")
    prepared_nodes: dict[str, dict[str, Any]] = {}
    grouped_edges: dict[str, list[dict[str, Any]]] = {}
    count = 0
    for kind, records, contexts in (
        ("nodes", _records(nodes, "nodes"), node_contexts),
        ("edges", _records(edges, "edges"), edge_contexts),
    ):
        for index, record in enumerate(records):
            context = contexts[index] if contexts is not None else f"{kind}[{index}]"
            try:
                if not isinstance(record, dict):
                    raise ValueError("record must be a dictionary")
                if kind == "nodes":
                    entity_id = _identifier(record, "id")
                    _node_key(ns, entity_id)
                    ntype = record.get("ntype")
                    if ntype is not None and not isinstance(ntype, str):
                        raise ValueError("ntype must be a string or None")
                    if ntype is not None:
                        ntype.encode("utf-8")
                    props = _properties(record.get("props"), depth=1)
                    attrs = prepared_nodes.setdefault(entity_id, {})
                    # Collapse repeated ids without losing sequential semantics:
                    # the last supplied attribute wins; None leaves it alone.
                    if ntype is not None:
                        attrs["ntype"] = ntype
                    if props is not None:
                        attrs["props"] = props
                else:
                    src = _identifier(record, "src")
                    relation = _identifier(record, "relation")
                    dst = _identifier(record, "dst")
                    weight = record.get("weight", 1.0)
                    if isinstance(weight, bool) or not isinstance(weight, (int, float, Decimal)):
                        raise ValueError("weight must be a finite number")
                    weight = _dynamo_value(weight)
                    TypeSerializer().serialize(weight)
                    props = _properties(record.get("props"), depth=3)
                    for endpoint in (src, dst):
                        _node_key(ns, endpoint)
                        prepared_nodes.setdefault(endpoint, {})
                    grouped_edges.setdefault(src, []).append(
                        {
                            "relation": relation,
                            "target": dst,
                            "weight": weight,
                            "props": props or {},
                        }
                    )
                    count += 1
            except (ValueError, TypeError, DecimalException, OverflowError, RecursionError) as exc:
                raise ValueError(f"{context}: {exc}") from exc
    return _PreparedImport(prepared_nodes, grouped_edges, count)


def _load_json(path: str | Path) -> tuple[list[Any], list[Any], list[str], list[str]]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle, parse_float=Decimal)
    except (ValueError, DecimalException, RecursionError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: JSON must contain an object with nodes/edges arrays")
    for field in ("nodes", "edges"):
        if not isinstance(data.get(field, []), list):
            raise ValueError(f"{path}: {field} must be an array")
    nodes, edges = data.get("nodes", []), data.get("edges", [])
    return (
        nodes,
        edges,
        [f"{path}: nodes[{i}]" for i in range(len(nodes))],
        [f"{path}: edges[{i}]" for i in range(len(edges))],
    )


def _load_csv(path: str | Path, kind: str) -> tuple[list[dict[str, Any]], list[str]]:
    required = {"id"} if kind == "nodes" else {"src", "relation", "dst"}
    records: list[dict[str, Any]] = []
    contexts: list[str] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        try:
            headers = reader.fieldnames or []
            missing = required.difference(headers)
            if missing:
                raise ValueError(f"missing required headers: {', '.join(sorted(missing))}")
            if len(set(headers)) != len(headers):
                raise ValueError("duplicate CSV headers")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("CSV row has a different number of cells than its header")
                record: dict[str, Any] = {field: row[field] for field in required}
                if kind == "nodes" and row.get("ntype", "") != "":
                    record["ntype"] = row["ntype"]
                if row.get("props", "") != "":
                    record["props"] = json.loads(row["props"], parse_float=Decimal)
                    if not isinstance(record["props"], dict):
                        raise ValueError("props cell must contain a JSON object")
                if kind == "edges" and row.get("weight", "") != "":
                    record["weight"] = Decimal(row["weight"])
                records.append(record)
                contexts.append(f"{path}: CSV row {reader.line_num}")
        except (ValueError, DecimalException, csv.Error, RecursionError) as exc:
            # DictReader.line_num is only advanced after a successful row;
            # the underlying reader also reports lines consumed on parse errors.
            raise ValueError(f"{path}: CSV row {reader.reader.line_num}: {exc}") from exc
    return records, contexts


def _load_files(
    json_file: str | Path | None,
    nodes_csv: str | Path | None,
    edges_csv: str | Path | None,
) -> tuple[list[Any], list[Any], list[str], list[str]]:
    if json_file is not None:
        if nodes_csv is not None or edges_csv is not None:
            raise ValueError("supply json_file or CSV files, not both formats")
        return _load_json(json_file)
    if nodes_csv is None and edges_csv is None:
        raise ValueError("supply json_file, nodes_csv, or edges_csv")
    nodes, node_contexts = _load_csv(nodes_csv, "nodes") if nodes_csv is not None else ([], [])
    edges, edge_contexts = _load_csv(edges_csv, "edges") if edges_csv is not None else ([], [])
    return nodes, edges, node_contexts, edge_contexts
