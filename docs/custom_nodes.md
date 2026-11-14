# Custom Nodes & Typed Graph Schema (`dynaflow`)

`dynaflow` provides a composable, type-safe execution graph where workflow primitives, models, retrievers, and user-defined logic run as typed nodes.

This guide explains how to define custom node types using the Python SDK, enforce port type compatibility across connections, and serialize graphs to versioned JSON for the visual canvas.

---

## 1. Defining Custom Nodes

Dynavec provides two ways to author custom nodes:

### Approach A: Subclassing `Node` (Object-Oriented)

For complex nodes with internal state, custom initialization, or explicit schemas:

```python
from typing import Any
from dynavec import Node, NodePort, NodeSchema, node_registry

@node_registry.register
class TextSanitizerNode(Node):
    """Custom node that normalizes and lowercases text."""

    node_type = "text_sanitizer"
    schema = NodeSchema(
        node_type="text_sanitizer",
        label="Text Sanitizer",
        description="Strips whitespace and optionally converts to lowercase.",
        category="transform",
        inputs=[NodePort(name="raw_text", port_type="string", description="Raw text.")],
        outputs=[NodePort(name="clean_text", port_type="string", description="Clean text.")],
        config_schema={"lowercase": {"type": "boolean", "default": True}},
    )

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = str(inputs.get("raw_text", "")).strip()
        if self.config.get("lowercase", True):
            text = text.lower()
        return {"clean_text": text}
```

Key requirements for subclassing:
1. Define a unique `node_type: str` on the class.
2. Define a `schema: NodeSchema` specifying port types, labels, and config options.
3. Decorate with `@node_registry.register` (or call `node_registry.register(MyNode)`).
4. Implement `execute(self, inputs: dict[str, Any]) -> dict[str, Any]`.

---

### Approach B: Functional `@custom_node` Decorator

For concise, function-based transforms without boilerplate class declarations:

```python
from typing import Any
from dynavec import custom_node

@custom_node(
    node_type="keyword_analyzer",
    label="Keyword Analyzer",
    description="Extracts unique keywords matching minimum length.",
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
```

The `@custom_node` decorator accepts port definitions either as `NodePort` objects or shorthand tuples:
- `("port_name", "port_type")`
- `("port_name", "port_type", "description")`

---

## 2. Port Typing & Compatibility Validation

Every node port specifies a `PortType`:
`"string"`, `"number"`, `"boolean"`, `"array"`, `"object"`, or `"any"`.

### Compatibility Rules

| Source Port Type | Target Port Type | Compatible? | Notes |
| :--- | :--- | :--- | :--- |
| `any` | *any type* | **Yes** | Wildcard / dynamic passthrough |
| *any type* | `any` | **Yes** | Wildcard / dynamic passthrough |
| `T` | `T` | **Yes** | Identical primitive types |
| `string` | `object` | **Yes** | Allows serialized JSON strings from LLMs into objects |
| `object` | `string` | **Yes** | Allows object payloads to flow into string ports |
| `number` | `string` | **No** | Raises `PortCompatibilityError` |
| `boolean` | `number` | **No** | Raises `PortCompatibilityError` |
| `array` | `boolean` | **No** | Raises `PortCompatibilityError` |

When connecting edges via `graph.add_edge(...)` or running `graph.validate()`, dynaflow automatically enforces compatibility:

```python
from dynavec import SimpleGraph, PortCompatibilityError

graph = SimpleGraph()
graph.add_node(sanitizer)
graph.add_node(analyzer)

# Valid connection: string -> string
graph.add_edge(
    source_id="sanitizer_1",
    target_id="analyzer_1",
    source_port="clean_text",
    target_port="clean_text",
)

# Incompatible connection: number -> string (raises error)
try:
    graph.add_edge(
        source_id="analyzer_1",
        target_id="sanitizer_1",
        source_port="relevance_score",  # number
        target_port="raw_text",         # string
    )
except PortCompatibilityError as err:
    print(f"Connection blocked: {err}")
```

---

## 3. Versioned Graph JSON Round-Trip

Graphs serialize to a versioned, ReactFlow-compatible JSON structure:

```python
# Serialize to JSON string
json_data = graph.to_json(indent=2)

# Deserialize back into an executable SimpleGraph
restored_graph = SimpleGraph.from_json(json_data)
result = restored_graph.run({"raw_text": "dynavec on AWS"})
```

### JSON Graph Schema (`version: "1.0"`)

```json
{
  "version": "1.0",
  "nodes": [
    {
      "id": "sanitizer_1",
      "type": "text_sanitizer",
      "data": {
        "label": "Text Sanitizer",
        "category": "transform",
        "config": {"lowercase": true},
        "status": "pending"
      },
      "position": {"x": 0, "y": 0}
    }
  ],
  "edges": [
    {
      "id": "sanitizer_1:clean_text->analyzer_1:clean_text",
      "source": "sanitizer_1",
      "target": "analyzer_1",
      "source_port": "clean_text",
      "target_port": "clean_text",
      "mapping": {},
      "is_error_edge": false
    }
  ],
  "execution_order": ["sanitizer_1", "analyzer_1"]
}
```

---

## 4. Visual Canvas Palette Export

Custom nodes registered in `node_registry` automatically appear in visual builder component palettes:

```python
from dynavec import node_registry

palette = node_registry.export_canvas_palette()
# Returns a list of ReactFlow component specifications:
# [
#   {
#     "type": "text_sanitizer",
#     "label": "Text Sanitizer",
#     "category": "transform",
#     "description": "...",
#     "inputPorts": ["raw_text"],
#     "outputPorts": ["clean_text"],
#     "config": {"lowercase": {"type": "boolean", "default": true}}
#   },
#   ...
# ]
```

---

## 5. End-to-End Runnable Example

See [examples/custom_nodes_sdk.py](file:///d:/DEVELOPMENT/dynavec-local/examples/custom_nodes_sdk.py) for the complete executable script.
