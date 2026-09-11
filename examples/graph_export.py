"""Export a dynavec knowledge-graph namespace as Mermaid / Graphviz DOT.

Builds a tiny supply-chain graph that deliberately contains a cycle
(acme -> globex -> initech -> acme), then prints the same subgraph in both
formats. ``graph_export`` returns a string, so paste the Mermaid block straight
into a README or pipe the DOT at ``dot -Tsvg``.

Runs against YOUR AWS account — dynavec auto-provisions the DynamoDB table that
backs the graph on first use. No embedder and no vector search involved.

Run
---
    python examples/graph_export.py

Requires AWS credentials with dynamodb + s3vectors permissions.
"""

from __future__ import annotations

import os

from dynavec import Dynavec, DynavecConfig

REGION = os.environ.get("AWS_REGION", "us-east-1")
NAMESPACE = "supply-chain"

cfg = DynavecConfig(
    vector_bucket="dynavec-demo",
    index="graph-export",
    table="dynavec_graph_export",
    dimension=8,  # unused here: this example never embeds anything
    region=REGION,
    auto_provision=True,
)

db = Dynavec(cfg)

# (src, relation, dst) — the first three edges form a cycle.
EDGES = [
    ("acme", "competes_with", "globex"),
    ("globex", "supplies", "initech"),
    ("initech", "supplies", "acme"),
    ("acme", "owns", "Acme Logistics GmbH"),  # spaces: needs escaping
    ("globex", "spun off", "end-of-life unit"),  # reserved word + hyphen
]

for src, relation, dst in EDGES:
    db.graph_add_edge(src, relation, dst, namespace=NAMESPACE)

print("--- mermaid ---")
print(db.graph.graph_export("mermaid", NAMESPACE))

print("--- graphviz dot ---")
print(db.graph.graph_export("dot", NAMESPACE))

# Narrow the walk instead of exporting the whole namespace: start at one entity
# and follow a single relation. (Seeding at "globex" alone would still reach
# everything here — the acme/globex/initech cycle makes the graph strongly
# connected, and the walk terminates on it either way.)
print("--- mermaid, 'supplies' chain from globex ---")
print(db.graph.graph_export("mermaid", NAMESPACE, roots=["globex"], relation="supplies"))
