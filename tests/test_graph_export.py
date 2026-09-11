"""Mermaid / Graphviz export of a namespace subgraph.

Same convention as the other graph tests: subclass the real ``GraphStore`` and
swap the DynamoDB calls for a dict, so the traversal and the renderers under
test are the real ones.
"""

import boto3
import pytest
from botocore.stub import Stubber

from dynavec.config import DynavecConfig
from dynavec.graph import GraphStore


class FakeGraph(GraphStore):
    """In-memory stand-in: only the two DynamoDB reads are replaced."""

    def __init__(self):
        self._nodes = {}  # (ns, entity_id) -> {"edges": [...], "docs": [...]}

    def _node(self, ns, entity_id):
        return self._nodes.setdefault((ns, entity_id), {"edges": [], "docs": []})

    def add_node(self, ns, entity_id, ntype=None, props=None):
        self._node(ns, entity_id)

    def add_edge(self, ns, src, relation, dst):
        self._node(ns, src)["edges"].append({"relation": relation, "target": dst})
        self._node(ns, dst)

    def get_node(self, ns, entity_id):
        return self._nodes.get((ns, entity_id))

    def list_node_ids(self, ns):
        return sorted(eid for node_ns, eid in self._nodes if node_ns == ns)


@pytest.fixture
def graph():
    g = FakeGraph()
    g.add_edge("corp", "acme", "competes_with", "globex")
    g.add_edge("corp", "globex", "supplies", "initech")
    # a second namespace that must never leak into a "corp" export
    g.add_edge("other", "hooli", "acquires", "piedpiper")
    return g


def test_mermaid_export_renders_the_namespace_subgraph(graph):
    assert graph.graph_export("mermaid", "corp") == (
        "graph LR\n"
        "  acme -->|competes_with| globex\n"
        "  globex -->|supplies| initech\n"
    )


def test_dot_export_renders_the_namespace_subgraph(graph):
    assert graph.graph_export("dot", "corp") == (
        "digraph G {\n"
        '  "acme" -> "globex" [label="competes_with"];\n'
        '  "globex" -> "initech" [label="supplies"];\n'
        "}\n"
    )


def test_export_is_scoped_to_one_namespace(graph):
    assert "hooli" not in graph.graph_export("mermaid", "corp")
    assert "acme" not in graph.graph_export("mermaid", "other")


# --------------------------------------------------------------------- cycles
def test_cycle_terminates_and_emits_each_edge_once():
    g = FakeGraph()
    g.add_edge("ns", "a", "next", "b")
    g.add_edge("ns", "b", "next", "c")
    g.add_edge("ns", "c", "next", "a")  # closes the loop
    g.add_edge("ns", "a", "next", "b")  # duplicate adjacency entry

    assert g.graph_export("mermaid", "ns") == (
        "graph LR\n"
        "  a -->|next| b\n"
        "  b -->|next| c\n"
        "  c -->|next| a\n"
    )


def test_cycle_terminates_when_walking_from_a_root():
    g = FakeGraph()
    g.add_edge("ns", "a", "next", "b")
    g.add_edge("ns", "b", "next", "a")
    g.add_node("ns", "orphan")  # unreachable from "a", so absent

    assert g.graph_export("mermaid", "ns", roots=["a"]) == (
        "graph LR\n"
        "  a -->|next| b\n"
        "  b -->|next| a\n"
    )


def test_self_loop_terminates():
    g = FakeGraph()
    g.add_edge("ns", "a", "cites", "a")

    assert g.graph_export("dot", "ns") == (
        "digraph G {\n"
        '  "a" -> "a" [label="cites"];\n'
        "}\n"
    )


# -------------------------------------------------------------------- escaping
@pytest.fixture
def awkward():
    g = FakeGraph()
    g.add_edge("ns", "acme corp", "competes with", "end")  # space; reserved word
    g.add_edge("ns", 'say "hi"', "knows", "acme corp")  # quotes
    g.add_edge("ns", "#hash", "tag", "multi-word")  # `#` entity; hyphen
    return g


def test_mermaid_escapes_ids_and_labels(awkward):
    # None of these ids can stand bare in Mermaid, so each gets an `nN` alias
    # carrying the real id as an escaped label.
    assert awkward.graph_export("mermaid", "ns") == (
        "graph LR\n"
        '  n0["#35;hash"]\n'
        '  n1["acme corp"]\n'
        '  n2["end"]\n'
        '  n3["multi-word"]\n'
        '  n4["say #quot;hi#quot;"]\n'
        "  n0 -->|tag| n3\n"
        '  n1 -->|"competes with"| n2\n'
        "  n4 -->|knows| n1\n"
    )


def test_dot_escapes_ids_and_labels(awkward):
    # DOT quotes every id, so only the backslash/quote pair needs escaping.
    assert awkward.graph_export("dot", "ns") == (
        "digraph G {\n"
        '  "#hash" -> "multi-word" [label="tag"];\n'
        '  "acme corp" -> "end" [label="competes with"];\n'
        '  "say \\"hi\\"" -> "acme corp" [label="knows"];\n'
        "}\n"
    )


def test_mermaid_alias_never_shadows_a_real_id():
    g = FakeGraph()
    g.add_edge("ns", "n0", "points_at", "zz top")  # "n0" is a legal bare id

    assert g.graph_export("mermaid", "ns") == (
        "graph LR\n"
        '  n1["zz top"]\n'
        "  n0 -->|points_at| n1\n"
    )


def test_dot_escapes_backslashes():
    g = FakeGraph()
    g.add_node("ns", "c:\\temp")

    assert g.graph_export("dot", "ns") == 'digraph G {\n  "c:\\\\temp";\n}\n'


# ------------------------------------------------------------- degenerate input
def test_unknown_format_raises_value_error(graph):
    with pytest.raises(ValueError) as excinfo:
        graph.graph_export("graphml", "corp")

    message = str(excinfo.value)
    assert "'mermaid'" in message and "'dot'" in message


def test_empty_subgraph_returns_an_empty_document(graph):
    assert graph.graph_export("mermaid", "nothing-here") == "graph LR\n"
    assert graph.graph_export("dot", "nothing-here") == "digraph G {\n}\n"


def test_isolated_nodes_are_declared():
    g = FakeGraph()
    g.add_node("ns", "lonely")
    g.add_edge("ns", "a", "next", "b")

    assert g.graph_export("mermaid", "ns") == (
        "graph LR\n"
        "  lonely\n"
        "  a -->|next| b\n"
    )
    assert g.graph_export("dot", "ns") == (
        "digraph G {\n"
        '  "lonely";\n'
        '  "a" -> "b" [label="next"];\n'
        "}\n"
    )


def test_relation_filter_keeps_only_matching_edges():
    g = FakeGraph()
    g.add_edge("ns", "a", "next", "b")
    g.add_edge("ns", "a", "owns", "c")

    assert g.graph_export("mermaid", "ns", roots=["a"], relation="next") == (
        "graph LR\n"
        "  a -->|next| b\n"
    )


# ------------------------------------------------------- list_node_ids (stubbed)
def _stubbed_store():
    """A real ``GraphStore`` whose DynamoDB calls go through a botocore stub.

    Same trick as ``test_list_vectors.py``: the stub validates our request
    parameters against the actual DynamoDB service model, which a hand-rolled
    fake table cannot.
    """
    session = boto3.Session(
        aws_access_key_id="testing", aws_secret_access_key="testing", region_name="us-east-1"
    )
    ddb = session.resource("dynamodb", region_name="us-east-1")
    store = GraphStore.__new__(GraphStore)
    store._config = DynavecConfig(
        vector_bucket="test-bucket", index="test-index", table="test-table", dimension=4
    )
    store._ddb = ddb
    store._table = ddb.Table("test-table")
    return store, Stubber(ddb.meta.client)


def test_list_node_ids_scans_the_namespace_prefix_and_paginates():
    store, stubber = _stubbed_store()
    base = {
        "TableName": "test-table",
        "FilterExpression": "begins_with(pk, :prefix)",
        # the namespace is escaped into the key, so the prefix is unambiguous
        "ExpressionAttributeValues": {":prefix": "corp#node#"},
        "ProjectionExpression": "pk, entity_id",
    }
    with stubber as stub:
        stub.add_response(
            "scan",
            {
                "Items": [
                    {"pk": {"S": "corp#node#globex"}, "entity_id": {"S": "globex"}},
                    {"pk": {"S": "corp#node#acme"}, "entity_id": {"S": "acme"}},
                ],
                "LastEvaluatedKey": {"pk": {"S": "corp#node#acme"}},
            },
            base,
        )
        stub.add_response(
            "scan",
            # no entity_id attribute: recovered by decoding the key
            {"Items": [{"pk": {"S": "corp#node#initech%23eu"}}]},
            {**base, "ExclusiveStartKey": {"pk": "corp#node#acme"}},
        )

        assert store.list_node_ids("corp") == ["acme", "globex", "initech#eu"]
        stub.assert_no_pending_responses()


def test_list_node_ids_escapes_the_namespace_into_the_prefix():
    store, stubber = _stubbed_store()
    with stubber as stub:
        stub.add_response(
            "scan",
            {"Items": []},
            {
                "TableName": "test-table",
                "FilterExpression": "begins_with(pk, :prefix)",
                "ExpressionAttributeValues": {":prefix": "tenant%23one#node#"},
                "ProjectionExpression": "pk, entity_id",
            },
        )

        assert store.list_node_ids("tenant#one") == []
        stub.assert_no_pending_responses()
