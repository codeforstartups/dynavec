"""Example: Defining and Composing Custom Dynaflow Nodes via the SDK.

Demonstrates:
1. Defining a custom node by subclassing `Node` and registering with `@node_registry.register`.
2. Defining a custom node using the functional `@custom_node` decorator.
3. Wiring custom nodes alongside built-in primitives in a `SimpleGraph`.
4. Type-safe edge port validation (compatible vs. incompatible ports).
5. Serializing the graph to ReactFlow-compatible versioned JSON and deserializing back.

Run:
---
    python examples/custom_nodes_sdk.py
"""

from __future__ import annotations

from typing import Any

from dynavec import (
    CodeNode,
    Node,
    NodePort,
    NodeSchema,
    PortCompatibilityError,
    SimpleGraph,
    custom_node,
    node_registry,
)

# ============================================================================
# Approach 1: Class-Based Custom Node via Node Subclassing
# ============================================================================


@node_registry.register
class TextSanitizerNode(Node):
    """Custom node that normalizes and strips unwanted characters from input text."""

    node_type = "text_sanitizer"
    schema = NodeSchema(
        node_type="text_sanitizer",
        label="Text Sanitizer",
        description="Normalizes casing and strips surrounding whitespace.",
        category="transform",
        inputs=[
            NodePort(name="raw_text", port_type="string", description="Raw unparsed text.")
        ],
        outputs=[
            NodePort(
                name="clean_text", port_type="string", description="Sanitized clean text."
            )
        ],
        config_schema={"lowercase": {"type": "boolean", "default": True}},
    )

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = str(inputs.get("raw_text", "")).strip()
        if self.config.get("lowercase", True):
            text = text.lower()
        return {"clean_text": text}


# ============================================================================
# Approach 2: Functional Custom Node via @custom_node Decorator
# ============================================================================


@custom_node(
    node_type="keyword_analyzer",
    label="Keyword Analyzer",
    description="Detects high-value keywords and computes relevance score.",
    category="nlp",
    inputs=[("clean_text", "string")],
    outputs=[("keywords", "array"), ("relevance_score", "number")],
    config_schema={"min_length": {"type": "integer", "default": 4}},
)
def analyze_keywords(inputs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    min_len = config.get("min_length", 4)
    words = [w for w in inputs.get("clean_text", "").split() if len(w) >= min_len]
    unique_words = sorted(set(words))
    score = min(1.0, len(unique_words) * 0.25)
    return {
        "keywords": unique_words,
        "relevance_score": score,
    }


def main() -> None:
    print("=== Dynaflow Custom Node SDK Demo ===")

    # 1. Instantiate nodes
    sanitizer = TextSanitizerNode(
        node_id="sanitizer_1", config={"lowercase": True}
    )
    analyzer = analyze_keywords(
        node_id="analyzer_1", config={"min_length": 4}
    )
    formatter = CodeNode(
        code_fn=lambda inp: {
            "summary": (
                f"Found {len(inp.get('keywords', []))} keywords "
                f"with relevance {inp.get('relevance_score', 0):.2f}"
            )
        },
        node_id="formatter_1",
    )

    # 2. Build execution graph
    graph = SimpleGraph()
    graph.add_node(sanitizer)
    graph.add_node(analyzer)
    graph.add_node(formatter)

    # Connect ports with automatic type compatibility validation
    # sanitizer.clean_text (string) -> analyzer.clean_text (string)
    graph.add_edge(
        source_id="sanitizer_1",
        target_id="analyzer_1",
        source_port="clean_text",
        target_port="clean_text",
    )

    # analyzer.outputs -> formatter (passthrough)
    graph.add_edge(
        source_id="analyzer_1",
        target_id="formatter_1",
    )

    # 3. Validate entire graph topology
    graph.validate()
    print("✓ Graph topology and port types validated successfully.")

    # 4. Demonstrate port compatibility enforcement
    try:
        # Attempt to connect number (relevance_score) to string (raw_text)
        graph.add_edge(
            source_id="analyzer_1",
            target_id="sanitizer_1",
            source_port="relevance_score",
            target_port="raw_text",
        )
    except PortCompatibilityError as err:
        print(f"✓ PortCompatibilityError caught as expected: {err}")

    # 5. Run the pipeline
    test_input = {"raw_text": "  Dynavec delivers Serverless Vector Search on AWS  "}
    result = graph.run(test_input)
    print("\n--- Execution Output ---")
    print(f"Clean text: {result.get('clean_text')}")
    print(f"Keywords:   {result.get('keywords')}")
    print(f"Score:      {result.get('relevance_score')}")
    print(f"Summary:    {result.get('summary')}")

    # 6. Graph JSON Round-Trip
    print("\n--- JSON Serialization Round-Trip ---")
    json_repr = graph.to_json(indent=2)
    print(f"Serialized graph size: {len(json_repr)} bytes")

    # Deserialize back from JSON and re-run
    reloaded_graph = SimpleGraph.from_json(json_repr)
    # Re-inject callable for CodeNode
    reloaded_graph.nodes["formatter_1"].code_fn = formatter.code_fn  # type: ignore[attr-defined]

    reloaded_result = reloaded_graph.run(test_input)
    assert reloaded_result["summary"] == result["summary"]
    print("✓ Reconstructed graph executed identically after JSON round-trip!")


if __name__ == "__main__":
    main()
