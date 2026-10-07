"""Bulk graph import: real updates against Moto, recorded before execution."""

import copy
import csv
import json
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from dynavec import Dynavec, GraphImportResult
from dynavec.config import DynavecConfig
from dynavec.graph import GraphStore


@pytest.fixture
def store():
    with mock_aws():
        session = boto3.Session(
            aws_access_key_id="testing", aws_secret_access_key="testing", region_name="us-east-1"
        )
        session.client("dynamodb").create_table(
            TableName="test-table",
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        config = DynavecConfig(
            vector_bucket="test-bucket", index="test-index", table="test-table", dimension=4
        )
        yield GraphStore(config, boto_session=session)


@pytest.fixture
def updates(store, monkeypatch):
    calls = []
    real_update = store._table.update_item

    def record(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return real_update(**kwargs)

    monkeypatch.setattr(store._table, "update_item", record)
    return calls


def edge(src="a", dst="b", **kw):
    return {"src": src, "relation": "next", "dst": dst, **kw}


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("batch_size,lengths", [(100, [3]), (2, [2, 1]), (1, [1, 1, 1])])
def test_grouped_updates_and_chunk_boundaries(store, updates, batch_size, lengths):
    """Three edges initialize four nodes once and append chunks in input order."""
    result = store.import_graph(
        "ns", edges=[edge(dst=dst) for dst in ("b", "c", "d")], batch_size=batch_size
    )
    assert result == GraphImportResult(nodes_processed=4, edges_appended=3)
    init = [call for call in updates if ":eid" in call["ExpressionAttributeValues"]]
    appends = [call for call in updates if ":e" in call["ExpressionAttributeValues"]]
    assert len(init) == 4
    assert {call["Key"]["pk"] for call in init} == {f"ns#node#{n}" for n in "abcd"}
    assert [len(call["ExpressionAttributeValues"][":e"]) for call in appends] == lengths
    assert all(call["Key"] == {"pk": "ns#node#a"} for call in appends)
    assert all("list_append" in call["UpdateExpression"] for call in appends)
    assert len(updates) == 4 + len(lengths)
    assert store.neighbors("ns", "a") == ["b", "c", "d"]
    assert store.list_node_ids("ns") == list("abcd")
    assert all(store.get_node("ns", n)["edges"] == [] for n in "bcd")


def test_individual_edge_baseline_uses_nine_updates(store, updates):
    """Measure the existing single-edge path, rather than assume request savings."""
    for dst in "bcd":
        store.add_edge("ns", "a", "next", dst)
    assert len(updates) == 9
    assert store.neighbors("ns", "a") == list("bcd")


def test_records_weights_nested_props_and_input_immutability(store):
    """Float copies become Decimal; strings and supported native values survive."""
    timestamp = "2026-10-07T12:34:56Z"
    nodes = [{"id": "a", "ntype": "Person", "props": {"scores": [2.5, {"x": 0.2}]}}]
    edges = [
        edge(props={"created_at": timestamp, "nested": [{"score": 3.25}]}),
        edge(dst="c", weight=-2.5, props={"flags": [True, None], "set": {1.5, 2.5}}),
        edge(dst="d", weight=Decimal("4.75"), props={"raw": bytearray(b"data")}),
        edge(dst="e", weight=0),
    ]
    original = copy.deepcopy((nodes, edges))
    assert store.import_graph("ns", nodes=nodes, edges=edges) == GraphImportResult(5, 4)
    assert (nodes, edges) == original
    node = store.get_node("ns", "a")
    assert node["props"] == {"scores": [Decimal("2.5"), {"x": Decimal("0.2")}]}
    assert [e["weight"] for e in node["edges"]] == [1, Decimal("-2.5"), Decimal("4.75"), 0]
    assert node["edges"][0]["props"] == {
        "created_at": timestamp,
        "nested": [{"score": Decimal("3.25")}],
    }
    assert isinstance(node["edges"][0]["props"]["created_at"], str)
    assert node["edges"][1]["props"] == {
        "flags": [True, None],
        "set": {Decimal("1.5"), Decimal("2.5")},
    }
    assert node["edges"][2]["props"]["raw"] == b"data"
    assert node["edges"][3]["props"] == {}
    assert store.neighbors("ns", "a", with_weights=True) == [
        ("d", 4.75),
        ("b", 1.0),
        ("e", 0.0),
        ("c", -2.5),
    ]
    assert store.neighbors("ns", "a", min_weight=1) == ["b", "d"]


@pytest.mark.parametrize("attrs", [{}, {"ntype": None, "props": None}])
def test_existing_nodes_and_inferred_endpoints_preserve_metadata(store, attrs):
    """Omitted/None attributes and endpoint creation preserve edges, docs, metadata."""
    store.add_node("ns", "a", "Person", {"role": "admin"})
    store.add_node("ns", "b", "Team", {"name": "Eng"})
    store.add_edge("ns", "a", "old", "b")
    store.link_docs("ns", "a", ["doc-1", "doc-2"])
    store.import_graph("ns", nodes=[{"id": "a", **attrs}], edges=[edge()])
    node = store.get_node("ns", "a")
    assert node["ntype"] == "Person" and node["props"] == {"role": "admin"}
    assert node["docs"] == ["doc-1", "doc-2"]
    assert [e["relation"] for e in node["edges"]] == ["old", "next"]
    assert store.get_node("ns", "b")["ntype"] == "Team"
    assert store.get_node("ns", "b")["props"] == {"name": "Eng"}


@pytest.mark.parametrize("props", [{"new": 2}, {}])
def test_explicit_props_replace_instead_of_merge(store, props):
    """Explicit maps replace the full old map; {} clears it without clearing docs."""
    store.add_node("ns", "a", "Old", {"old": 1})
    store.link_docs("ns", "a", ["doc"])
    store.import_graph("ns", nodes=[{"id": "a", "props": props}])
    node = store.get_node("ns", "a")
    assert node["props"] == props
    assert node["ntype"] == "Old" and node["docs"] == ["doc"]


def test_repeated_node_records_have_sequential_semantics(store, updates):
    """Later supplied attrs win independently; None leaves earlier attrs intact."""
    nodes = [
        {"id": "a", "ntype": "First", "props": {"old": 1}},
        {"id": "a", "props": {}},
        {"id": "a", "ntype": "Last"},
        {"id": "a", "ntype": None, "props": None},
    ]
    assert store.import_graph("ns", nodes=nodes, edges=[edge(dst="a")]) == GraphImportResult(1, 1)
    assert len(updates) == 2
    node = store.get_node("ns", "a")
    assert node["ntype"] == "Last" and node["props"] == {}
    assert store.neighbors("ns", "a") == ["a"]


def test_interleaved_sources_duplicates_and_repeated_imports(store, updates):
    """Grouping preserves per-source order and never deduplicates appended edges."""
    edges = [edge(), edge("b", "c"), edge(dst="c"), edge(), edge("b", "a")]
    for _ in range(2):
        assert store.import_graph("ns", edges=edges, batch_size=2) == GraphImportResult(3, 5)
    assert store.neighbors("ns", "a") == ["b", "c", "b", "b", "c", "b"]
    assert store.neighbors("ns", "b") == ["c", "a", "c", "a"]
    assert len(updates) == 12  # each import: 3 nodes + 2 chunks for a + 1 for b


def test_namespace_key_encoding_and_identifiers_are_unchanged(store):
    """Namespaces remain isolated; separators escape and spaces/Unicode survive."""
    for ns in ("tenant#one", "tenant", "default"):
        assert store.import_graph(ns, edges=[edge(" a#%é ", " b ")]) == GraphImportResult(2, 1)
    assert store.get_node("tenant#one", " a#%é ")["pk"] == "tenant%23one#node# a%23%25é "
    assert store.neighbors("tenant#one", " a#%é ") == [" b "]
    store.import_graph("tenant", edges=[edge(" a#%é ", "other")])
    assert store.neighbors("tenant#one", " a#%é ") == [" b "]
    assert store.neighbors("default", " a#%é ") == [" b "]


@pytest.mark.parametrize("kw", [{}, {"nodes": [], "edges": []}, {"nodes": (), "edges": ()}])
def test_empty_imports_do_not_write(store, updates, kw):
    """Omitted and empty collections return zero counts and perform zero writes."""
    assert store.import_graph("ns", **kw) == GraphImportResult(0, 0)
    assert updates == []


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "2", None])
def test_invalid_batch_sizes_do_not_write(store, updates, batch_size):
    """A positive integer is required; booleans cannot masquerade as batch sizes."""
    with pytest.raises(ValueError, match="batch_size"):
        store.import_graph("ns", nodes=[{"id": "valid"}], batch_size=batch_size)
    assert updates == []


@pytest.mark.parametrize(
    "kind,bad,message",
    [
        ("nodes", {}, "id"),
        ("nodes", {"id": ""}, "id"),
        ("nodes", {"id": 5}, "id"),
        ("nodes", {"id": "a", "ntype": 4}, "ntype"),
        ("nodes", {"id": "a", "props": []}, "props"),
        ("nodes", None, "dictionary"),
        ("edges", {"src": "a", "dst": "b"}, "relation"),
        ("edges", {"relation": "r", "dst": "b"}, "src"),
        ("edges", {"src": "a", "relation": "r"}, "dst"),
        ("edges", edge(src=""), "src"),
        ("edges", edge(dst=""), "dst"),
        ("edges", {**edge(), "relation": ""}, "relation"),
        ("edges", edge(dst=None), "dst"),
        ("edges", "bad", "dictionary"),
        ("edges", edge(props="bad"), "props"),
        ("nodes", {"id": "x" * 2049}, "partition key"),
    ],
)
def test_invalid_records_report_index_and_write_nothing(store, updates, kind, bad, message):
    """Even a late invalid record must fail before writing earlier valid records."""
    valid = {"id": "valid"} if kind == "nodes" else edge()
    with pytest.raises(ValueError) as exc:
        store.import_graph("ns", **{kind: [valid, bad]})
    assert f"{kind}[1]" in str(exc.value) and message in str(exc.value)
    assert updates == []


@pytest.mark.parametrize(
    "weight",
    [
        True,
        None,
        "2.5",
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("NaN"),
        Decimal("-Infinity"),
        Decimal("1e999"),
        10**40,
    ],
)
def test_invalid_weights_are_prevalidated(store, updates, weight):
    """Reject nonnumeric, nonfinite and numbers outside DynamoDB's precision/range."""
    with pytest.raises(ValueError, match=r"edges\[1\]"):
        store.import_graph("ns", nodes=[{"id": "valid"}], edges=[edge(), edge(weight=weight)])
    assert updates == []


@pytest.mark.parametrize(
    "value",
    [
        object(),
        datetime(2026, 10, 7),
        {1: "bad-key"},
        {"nested": float("nan")},
        {"nested": Decimal("1e999")},
        set(),
        {"str", 1},
        {True},
        [object()],
    ],
    ids=[
        "object",
        "datetime",
        "map-key",
        "nan",
        "number-range",
        "empty-set",
        "mixed-set",
        "bool-set",
        "nested-object",
    ],
)
@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_unsupported_properties_are_prevalidated(store, updates, kind, value):
    """Run real DynamoDB serialization validation on both node and edge properties."""
    bad = (
        {"id": "bad", "props": {"value": value}}
        if kind == "nodes"
        else edge(props={"value": value})
    )
    with pytest.raises(ValueError, match=rf"{kind}\[0\]"):
        store.import_graph("ns", **{kind: [bad]})
    assert updates == []


@pytest.mark.parametrize("kind", ["nodes", "edges"])
def test_excessive_property_nesting_is_prevalidated(store, updates, kind):
    """Boto3 accepts deeply nested maps, but DynamoDB's document-depth limit does not."""
    nested = {}
    for _ in range(33):
        nested = {"child": nested}
    record = {"id": "a", "props": nested} if kind == "nodes" else edge(props=nested)
    with pytest.raises(ValueError, match="32 nested levels"):
        store.import_graph("ns", **{kind: [record]})
    assert updates == []


def test_all_records_are_validated_even_when_later_nodes_replace_props(store, updates):
    """Invalid earlier attributes cannot be hidden by a later valid replacement."""
    nodes = [{"id": "a", "props": {"bad": object()}}, {"id": "a", "props": {}}]
    with pytest.raises(ValueError, match=r"nodes\[0\]"):
        store.import_graph("ns", nodes=nodes)
    assert updates == []


def test_valid_nodes_and_late_invalid_edges_write_nothing(store, updates):
    """Validation spans both collections before initializing even valid nodes."""
    nodes = [{"id": "a", "props": {"nested": [1.25]}}]
    edges = [edge(), edge(weight=float("nan"))]
    original = copy.deepcopy(nodes)
    with pytest.raises(ValueError, match=r"edges\[1\]"):
        store.import_graph("ns", nodes=nodes, edges=edges)
    assert updates == [] and nodes == original


@pytest.mark.parametrize("kind", ["nodes", "edges"])
@pytest.mark.parametrize("container", ["bad", {}, 3, {"a", "b"}])
def test_invalid_collection_types_do_not_write(store, updates, kind, container):
    """Reject scalar, mapping and unordered collections instead of treating them as records."""
    with pytest.raises(ValueError, match=kind):
        store.import_graph("ns", **{kind: container})
    assert updates == []


@pytest.mark.parametrize(
    "data,counts",
    [
        ({}, (0, 0)),
        ({"nodes": [], "edges": []}, (0, 0)),
        ({"nodes": [{"id": "a"}]}, (1, 0)),
        ({"edges": [edge()]}, (2, 1)),
        (
            {
                "nodes": [{"id": "a", "ntype": "Person", "props": {"x": 1.25}}],
                "edges": [edge(weight=3.5, props={"created_at": "2026-10-07"})],
            },
            (2, 1),
        ),
    ],
)
def test_json_loading(store, tmp_path, data, counts):
    """Combined, single-collection and empty UTF-8 JSON feed the shared importer."""
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert store.import_graph_file("ns", json_file=path) == GraphImportResult(*counts)
    if data.get("edges"):
        stored = store.get_node("ns", "a")["edges"][0]
        assert stored["target"] == "b"
        assert stored["weight"] == Decimal(str(data["edges"][0].get("weight", 1)))
        assert stored["props"] == data["edges"][0].get("props", {})
    if data.get("nodes") and data["nodes"][0].get("props"):
        assert store.get_node("ns", "a")["props"] == {"x": Decimal("1.25")}


@pytest.mark.parametrize("mode", ["nodes", "edges", "both"])
def test_csv_loading_standard_quoting_and_blank_defaults(store, tmp_path, mode):
    """Node/edge CSVs work separately and together, including quoted commas/newlines."""
    nodes = write_csv(
        tmp_path / "nodes.csv",
        ["id", "ntype", "props"],
        [
            ["a, one", "Person", json.dumps({"label": 'comma, newline\nquote"', "x": 2.5})],
            [" b ", "", ""],
        ],
    )
    edges = write_csv(
        tmp_path / "edges.csv",
        ["src", "relation", "dst", "weight", "props"],
        [
            ["a, one", "next", " b ", "", ""],
            [
                "a, one",
                "next",
                " b ",
                "-2.5",
                json.dumps({"created_at": "2026-10-07", "x": [1.25]}),
            ],
        ],
    )
    kw = {}
    if mode in ("nodes", "both"):
        kw["nodes_csv"] = nodes
    if mode in ("edges", "both"):
        kw["edges_csv"] = edges
    assert store.import_graph_file("ns", **kw) == GraphImportResult(2, 0 if mode == "nodes" else 2)
    a = store.get_node("ns", "a, one")
    if mode in ("nodes", "both"):
        assert a["ntype"] == "Person"
        assert a["props"] == {"label": 'comma, newline\nquote"', "x": Decimal("2.5")}
    else:
        assert a["ntype"] is None and a["props"] == {}
    b = store.get_node("ns", " b ")
    assert b["ntype"] is None and b["props"] == {} and b["docs"] == []
    if mode != "nodes":
        assert [e["weight"] for e in a["edges"]] == [1, Decimal("-2.5")]
        assert a["edges"][0]["props"] == {}
        assert a["edges"][1]["props"] == {"created_at": "2026-10-07", "x": [Decimal("1.25")]}


def test_csv_required_headers_only(store, tmp_path):
    """Optional headers may be absent, with normal type/props/weight defaults."""
    nodes = write_csv(tmp_path / "n.csv", ["id"], [["a"]])
    edges = write_csv(tmp_path / "e.csv", ["src", "relation", "dst"], [["a", "r", "b"]])
    assert store.import_graph_file("ns", nodes_csv=nodes, edges_csv=edges) == GraphImportResult(
        2, 1
    )
    assert store.get_node("ns", "a")["props"] == {}
    assert store.get_node("ns", "a")["edges"] == [
        {"target": "b", "relation": "r", "weight": 1, "props": {}}
    ]


@pytest.mark.parametrize(
    "text,message",
    [
        ("{broken", "graph.json"),
        ("[]", "object"),
        ('{"nodes": {}}', "nodes"),
        ('{"edges": null}', "edges"),
        ('{"nodes": [null]}', "nodes[0]"),
        ('{"nodes": [{"id": "a"}], "edges": [{}]}', "edges[0]"),
        ('{"edges": [{"src":"a", "relation":"r", "dst":"b", "weight":NaN}]}', "edges[0]"),
    ],
)
def test_invalid_json_reports_file_and_never_writes(store, updates, tmp_path, text, message):
    """Malformed JSON, wrong arrays and bad records include filename and record context."""
    path = tmp_path / "graph.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        store.import_graph_file("ns", json_file=path)
    assert str(path) in str(exc.value) and message in str(exc.value)
    assert updates == []


@pytest.mark.parametrize("kind", ["json", "csv"])
def test_file_properties_use_shared_serialization_validation(store, updates, tmp_path, kind):
    """Parsed properties still pass the same finite-value check as Python records."""
    path = tmp_path / f"bad.{kind}"
    if kind == "json":
        path.write_text('{"nodes": [{"id": "a", "props": {"score": NaN}}]}', encoding="utf-8")
        kw = {"json_file": path}
        context = "nodes[0]"
    else:
        write_csv(path, ["id", "props"], [["a", '{"score": NaN}']])
        kw = {"nodes_csv": path}
        context = "CSV row 2"
    with pytest.raises(ValueError) as exc:
        store.import_graph_file("ns", **kw)
    assert str(path) in str(exc.value) and context in str(exc.value)
    assert updates == []


@pytest.mark.parametrize("kind", ["json", "csv"])
def test_non_utf8_files_do_not_write(store, updates, tmp_path, kind):
    """Encoding errors identify their file and never reach the write phase."""
    path = tmp_path / f"bad.{kind}"
    path.write_bytes(b"id\n\xff\n")
    with pytest.raises(ValueError) as exc:
        store.import_graph_file("ns", **{"json_file" if kind == "json" else "nodes_csv": path})
    assert str(path) in str(exc.value)
    assert updates == []


@pytest.mark.parametrize("kind", ["json", "csv"])
def test_parser_recursion_errors_include_filename(store, updates, tmp_path, kind):
    """Extremely nested file data fails with file context before serialization or writes."""
    path = tmp_path / f"deep.{kind}"
    nested = "[" * 1100 + "0" + "]" * 1100
    if kind == "json":
        path.write_text('{"nodes": ' + nested + "}", encoding="utf-8")
        kw = {"json_file": path}
    else:
        write_csv(path, ["id", "props"], [["a", '{"nested":' + nested + "}"]])
        kw = {"nodes_csv": path}
    with pytest.raises(ValueError) as exc:
        store.import_graph_file("ns", **kw)
    assert str(path) in str(exc.value)
    assert updates == []


@pytest.mark.parametrize(
    "kind,text,row",
    [
        ("nodes", "ntype\nPerson\n", 1),
        ("edges", "src,dst\na,b\n", 1),
        ("nodes", "id,props\na,{broken\n", 2),
        ("nodes", "id,props\na,[]\n", 2),
        ("nodes", 'id\nvalid\n""\n', 3),
        ("edges", "src,relation,dst,weight\na,r,b,2\na,r,c,bad\n", 3),
        ("edges", "src,relation,dst,weight\na,r,b,Infinity\n", 2),
        ("edges", "src,relation,dst\na,,b\n", 2),
        ("nodes", 'id,props\na,"unterminated\n', 2),
        ("nodes", "id,props\na,{},extra\n", 2),
        ("nodes", "id,props\na\n", 2),
        ("nodes", "id,id\na,b\n", 1),
    ],
)
def test_invalid_csv_reports_row_and_validates_both_files(
    store, updates, tmp_path, kind, text, row
):
    """Both files must pass before writing, including parser and semantic errors."""
    bad = tmp_path / "bad.csv"
    bad.write_text(text, encoding="utf-8")
    good_nodes = write_csv(tmp_path / "good-n.csv", ["id"], [["valid"]])
    good_edges = write_csv(tmp_path / "good-e.csv", ["src", "relation", "dst"], [["a", "r", "b"]])
    with pytest.raises(ValueError) as exc:
        store.import_graph_file(
            "ns",
            nodes_csv=bad if kind == "nodes" else good_nodes,
            edges_csv=bad if kind == "edges" else good_edges,
        )
    assert str(bad) in str(exc.value) and f"CSV row {row}" in str(exc.value)
    assert updates == []


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"json_file": "g.json", "nodes_csv": "n.csv"},
        {"json_file": "g.json", "edges_csv": "e.csv"},
    ],
)
def test_file_argument_errors_do_not_write(store, updates, kw):
    """Require one format and at least one file before trying to open any file."""
    with pytest.raises(ValueError):
        store.import_graph_file("ns", **kw)
    assert updates == []


@pytest.mark.parametrize("namespace", [None, "", 3])
def test_invalid_namespace_does_not_write(store, updates, namespace):
    """Reject unusable namespaces before constructing or writing DynamoDB keys."""
    with pytest.raises(ValueError, match="namespace"):
        store.import_graph(namespace, nodes=[{"id": "a"}])
    assert updates == []


def test_client_forwards_records_and_files_and_returns_result():
    """Thin client wrappers pass exact arguments and return the graph result unchanged."""
    graph = Mock()
    result = GraphImportResult(2, 1)
    graph.import_graph.return_value = result
    graph.import_graph_file.return_value = result
    client = Dynavec.__new__(Dynavec)
    client._graph_store = graph
    nodes, edges = [{"id": "a"}], [edge()]
    assert client.graph_import(nodes=nodes, edges=edges) is result
    graph.import_graph.assert_called_with("default", nodes=nodes, edges=edges, batch_size=100)
    assert client.graph_import(nodes=nodes, edges=edges, namespace="tenant", batch_size=2) is result
    graph.import_graph.assert_called_with("tenant", nodes=nodes, edges=edges, batch_size=2)
    assert client.graph_import_file(json_file="g.json") is result
    graph.import_graph_file.assert_called_with(
        "default", json_file="g.json", nodes_csv=None, edges_csv=None, batch_size=100
    )
    assert (
        client.graph_import_file(
            nodes_csv="n.csv", edges_csv="e.csv", namespace="tenant", batch_size=3
        )
        is result
    )
    graph.import_graph_file.assert_called_with(
        "tenant", json_file=None, nodes_csv="n.csv", edges_csv="e.csv", batch_size=3
    )


def test_client_import_uses_real_store(store, tmp_path):
    """Exercise wrapper -> importer -> stored graph with both public APIs."""
    client = Dynavec.__new__(Dynavec)
    client._graph_store = store
    assert client.graph_import(nodes=[{"id": "a"}], edges=[edge()]) == GraphImportResult(2, 1)
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"edges": [edge(dst="c")]}), encoding="utf-8")
    assert client.graph_import_file(
        json_file=path, namespace="tenant", batch_size=1
    ) == GraphImportResult(2, 1)
    assert store.neighbors("default", "a") == ["b"]
    assert store.neighbors("tenant", "a") == ["c"]


def test_failed_append_does_not_replay_completed_chunks(store, monkeypatch):
    """Simulate an ambiguous retryable failure after commit; completed appends stay once."""
    real_update = store._table.update_item
    appends = []

    def fail_second_append(**kwargs):
        result = real_update(**kwargs)
        if ":e" in kwargs["ExpressionAttributeValues"]:
            appends.append(kwargs)
            if len(appends) == 2:
                raise ClientError({"Error": {"Code": "InternalServerError"}}, "UpdateItem")
        return result

    monkeypatch.setattr(store._table, "update_item", fail_second_append)
    with pytest.raises(ClientError):
        store.import_graph("ns", edges=[edge(dst=dst) for dst in "bcd"], batch_size=1)
    assert len(appends) == 2
    assert store.neighbors("ns", "a") == ["b", "c"]
    assert store.get_node("ns", "d") is not None  # nodes prepared before appends


def test_node_initialization_keeps_existing_retry_policy(store, monkeypatch):
    """Safe node preparation may retry without restarting the import's append phase."""
    real_update = store._table.update_item
    calls = []

    def throttled_once(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ClientError({"Error": {"Code": "ThrottlingException"}}, "UpdateItem")
        return real_update(**kwargs)

    monkeypatch.setattr(store._table, "update_item", throttled_once)
    monkeypatch.setattr("dynavec.utils.time.sleep", lambda _: None)
    assert store.import_graph("ns", edges=[edge()]) == GraphImportResult(2, 1)
    assert len(calls) == 4  # one failed init + two successful inits + one append
    assert store.neighbors("ns", "a") == ["b"]
