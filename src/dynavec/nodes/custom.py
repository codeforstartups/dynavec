"""Custom node SDK utilities for user-defined dynaflow nodes."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from .base import Node, NodePort, NodeSchema, PortType, RetryPolicy
from .registry import NodeRegistry, node_registry


def _normalize_port(
    port: NodePort | tuple[str, PortType] | tuple[str, PortType, str] | str,
) -> NodePort:
    """Convert various shorthand port representations into a NodePort object."""
    if isinstance(port, NodePort):
        return port
    if isinstance(port, tuple):
        if len(port) == 2:
            name, p_type = port
            return NodePort(name=name, port_type=p_type)
        if len(port) >= 3:
            name, p_type, desc = port[:3]
            return NodePort(name=name, port_type=p_type, description=desc)
    if isinstance(port, str):
        return NodePort(name=port, port_type="any")
    raise TypeError(f"Unsupported port specification: {port!r}")


def custom_node(
    node_type: str,
    label: str | None = None,
    description: str = "",
    category: str = "custom",
    inputs: list[NodePort | tuple[str, PortType] | tuple[str, PortType, str] | str]
    | None = None,
    outputs: list[NodePort | tuple[str, PortType] | tuple[str, PortType, str] | str]
    | None = None,
    config_schema: dict[str, Any] | None = None,
    registry: NodeRegistry | None = None,
    register: bool = True,
) -> Callable[[Callable[..., dict[str, Any]]], type[Node]]:
    """Decorator to define and register a custom dynaflow Node from a python function.

    Allows defining nodes either with standard `NodePort` objects or shorthand
    tuples e.g. `("text", "string")`.

    Example:
    -------
    ```python
    @custom_node(
        node_type="sentiment_classifier",
        label="Sentiment Classifier",
        description="Classifies input text sentiment.",
        inputs=[("text", "string")],
        outputs=[("sentiment", "string"), ("score", "number")],
        config_schema={"threshold": {"type": "number", "default": 0.5}},
    )
    def sentiment_node(inputs: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("text", "")
        threshold = config.get("threshold", 0.5)
        is_pos = "great" in text.lower() or "good" in text.lower()
        return {
            "sentiment": "positive" if is_pos else "neutral",
            "score": 0.9 if is_pos else 0.5,
        }
    ```
    """
    normalized_inputs = [_normalize_port(p) for p in (inputs or [])]
    normalized_outputs = [_normalize_port(p) for p in (outputs or [])]
    node_label = label or node_type.replace("_", " ").title()

    schema = NodeSchema(
        node_type=node_type,
        label=node_label,
        description=description,
        category=category,
        inputs=normalized_inputs,
        outputs=normalized_outputs,
        config_schema=dict(config_schema or {}),
    )

    def decorator(fn: Callable[..., dict[str, Any]]) -> type[Node]:
        sig = inspect.signature(fn)
        params_count = len(sig.parameters)

        class CustomNode(Node):
            def __init__(
                self,
                node_id: str | None = None,
                config: dict[str, Any] | None = None,
                retry_policy: RetryPolicy | None = None,
            ) -> None:
                super().__init__(
                    node_id=node_id,
                    config=config,
                    retry_policy=retry_policy,
                )

            def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
                if params_count >= 2:
                    return fn(inputs, self.config)
                return fn(inputs)

        CustomNode.node_type = schema.node_type
        CustomNode.schema = schema
        CustomNode.__name__ = f"CustomNode_{node_type}"
        CustomNode.__doc__ = fn.__doc__ or description or f"Custom node: {node_type}"

        if register:
            target_registry = registry or node_registry
            target_registry.register(CustomNode)

        return CustomNode

    return decorator
