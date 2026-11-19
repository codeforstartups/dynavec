"""Code-to-visual graph round-trip generator and safe AST parser for dynaflow."""

from __future__ import annotations

import ast
import json
import keyword
import re
from typing import Any, cast

from .base import Node, NodeStatus, RetryPolicy
from .builtin import (
    ApprovalNode,
    CodeNode,
    LLMNode,
    LoopNode,
    RetrieverNode,
    RouterNode,
    SubGraphNode,
    ToolNode,
    WebhookNode,
)
from .custom import custom_node
from .graph import SimpleGraph
from .registry import NodeRegistry, node_registry

BUILTIN_CLASS_MAP: dict[str, type[Node]] = {
    "ApprovalNode": ApprovalNode,
    "CodeNode": CodeNode,
    "LLMNode": LLMNode,
    "LoopNode": LoopNode,
    "RetrieverNode": RetrieverNode,
    "RouterNode": RouterNode,
    "SubGraphNode": SubGraphNode,
    "ToolNode": ToolNode,
    "WebhookNode": WebhookNode,
}

BUILTIN_NODE_CLASSES: dict[str, str] = {
    "approval": "ApprovalNode",
    "code": "CodeNode",
    "llm": "LLMNode",
    "loop": "LoopNode",
    "retriever": "RetrieverNode",
    "router": "RouterNode",
    "subgraph": "SubGraphNode",
    "tool": "ToolNode",
    "webhook": "WebhookNode",
}


def _format_literal(val: Any, indent_level: int = 1) -> str:
    """Format Python values into clean, valid Python literal code."""
    if isinstance(val, str):
        return json.dumps(val)
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, (int, float)):
        return repr(val)
    if val is None:
        return "None"
    if isinstance(val, dict):
        if not val:
            return "{}"
        items_str = ", ".join(
            f"{json.dumps(k)}: {_format_literal(v, indent_level + 1)}" for k, v in val.items()
        )
        repr_str = "{" + items_str + "}"
        if len(repr_str) <= 60 and "\n" not in repr_str:
            return repr_str
        lines: list[str] = ["{"]
        sub_indent = "    " * (indent_level + 1)
        close_indent = "    " * indent_level
        for k, v in val.items():
            formatted_v = _format_literal(v, indent_level + 1)
            lines.append(f"{sub_indent}{json.dumps(k)}: {formatted_v},")
        lines.append(f"{close_indent}}}")
        return "\n".join(lines)
    if isinstance(val, list):
        if not val:
            return "[]"
        items_str = ", ".join(_format_literal(item, indent_level + 1) for item in val)
        repr_str = "[" + items_str + "]"
        if len(repr_str) <= 60 and "\n" not in repr_str:
            return repr_str
        lines = ["["]
        sub_indent = "    " * (indent_level + 1)
        close_indent = "    " * indent_level
        for item in val:
            formatted_item = _format_literal(item, indent_level + 1)
            lines.append(f"{sub_indent}{formatted_item},")
        lines.append(f"{close_indent}]")
        return "\n".join(lines)
    return repr(val)


def export_graph_to_code(graph: SimpleGraph) -> str:
    """Generate clean, idiomatic, runnable Python SDK code defining the SimpleGraph."""
    lines: list[str] = [
        '"""Generated dynaflow pipeline graph."""',
        "",
        "from __future__ import annotations",
        "",
    ]

    imported_dynavec: set[str] = {"SimpleGraph"}

    # Determine node ordering
    ordered_node_ids: list[str] = []
    for nid in graph._execution_order:
        if nid in graph.nodes and nid not in ordered_node_ids:
            ordered_node_ids.append(nid)
    for nid in graph.nodes:
        if nid not in ordered_node_ids:
            ordered_node_ids.append(nid)

    # Determine imports and generate unique variable names
    node_var_map: dict[str, str] = {}
    used_vars: set[str] = {"graph"}
    custom_type_nodes: dict[str, str] = {}

    for nid in ordered_node_ids:
        node = graph.nodes[nid]
        cls_name = BUILTIN_NODE_CLASSES.get(node.node_type)
        if cls_name is not None and node.__class__.__name__ == cls_name:
            imported_dynavec.add(cls_name)
        else:
            raw_cls_name = node.__class__.__name__
            if raw_cls_name in BUILTIN_CLASS_MAP:
                imported_dynavec.add(raw_cls_name)
            elif node_registry.has(node.node_type):
                imported_dynavec.add("node_registry")
                custom_type_nodes[nid] = f'node_registry.get("{node.node_type}")'
            else:
                custom_type_nodes[nid] = raw_cls_name

        if node.retry_policy is not None:
            imported_dynavec.add("RetryPolicy")

        base_var = re.sub(r"\W", "_", nid)
        if not base_var.isidentifier() or keyword.iskeyword(base_var):
            base_var = f"node_{base_var}"
        candidate = base_var
        counter = 1
        while candidate in used_vars:
            counter += 1
            candidate = f"{base_var}_{counter}"
        used_vars.add(candidate)
        node_var_map[nid] = candidate

    # Format imports
    dynavec_imports = sorted(imported_dynavec)
    if len(dynavec_imports) == 1:
        lines.append(f"from dynavec import {dynavec_imports[0]}")
    else:
        lines.append("from dynavec import (")
        for item in dynavec_imports:
            lines.append(f"    {item},")
        lines.append(")")
    lines.append("")
    lines.append("")

    # Initialize graph
    lines.append("# Initialize execution graph")
    lines.append("graph = SimpleGraph()")

    if ordered_node_ids:
        lines.append("")
        lines.append("# Node definitions")
        for nid in ordered_node_ids:
            node = graph.nodes[nid]
            var_name = node_var_map[nid]
            cls_expr = custom_type_nodes.get(
                nid, BUILTIN_NODE_CLASSES.get(node.node_type, node.__class__.__name__)
            )

            args: list[str] = [f"node_id={json.dumps(node.id)}"]
            if node.config:
                formatted_cfg = _format_literal(node.config, indent_level=1)
                args.append(f"config={formatted_cfg}")
            if node.retry_policy is not None:
                rp = node.retry_policy
                rp_str = (
                    f"RetryPolicy("
                    f"max_attempts={rp.max_attempts}, "
                    f"base_delay={rp.base_delay}, "
                    f"max_delay={rp.max_delay}, "
                    f"jitter={rp.jitter}"
                    f")"
                )
                args.append(f"retry_policy={rp_str}")

            lines.append(f"{var_name} = {cls_expr}(")
            for arg in args:
                lines.append(f"    {arg},")
            lines.append(")")
            lines.append(f"graph.add_node({var_name})")
            if node.status != "pending":
                lines.append(f"{var_name}.status = {json.dumps(node.status)}")
            lines.append("")

    if graph.edges:
        lines.append("# Edges")
        for edge in graph.edges:
            edge_args: list[str] = [
                f"source_id={json.dumps(edge.source_node)}",
                f"target_id={json.dumps(edge.target_node)}",
            ]
            if edge.source_port != "output" or edge.target_port != "input":
                edge_args.append(f"source_port={json.dumps(edge.source_port)}")
                edge_args.append(f"target_port={json.dumps(edge.target_port)}")
            if edge.mapping:
                edge_args.append(f"mapping={_format_literal(edge.mapping, indent_level=1)}")
            if edge.is_error_edge:
                edge_args.append("is_error_edge=True")

            lines.append("graph.add_edge(")
            for e_arg in edge_args:
                lines.append(f"    {e_arg},")
            lines.append(")")
        lines.append("")

    if list(graph._execution_order) != ordered_node_ids:
        lines.append(f"graph._execution_order = {json.dumps(list(graph._execution_order))}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _safe_eval_literal(node: ast.AST) -> Any:
    """Safely evaluate literal AST expression without code execution."""
    try:
        return ast.literal_eval(node)
    except Exception as exc:
        raise ValueError(f"Invalid literal expression in AST: {ast.dump(node)}") from exc


def _resolve_node_id(expr: ast.AST, var_nodes: dict[str, Node]) -> str:
    """Resolve node identifier string from literal, name, or attribute expression."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Name):
        if expr.id in var_nodes:
            return var_nodes[expr.id].id
        return expr.id
    if isinstance(expr, ast.Attribute) and expr.attr == "id":
        if isinstance(expr.value, ast.Name) and expr.value.id in var_nodes:
            return var_nodes[expr.value.id].id
    try:
        val = ast.literal_eval(expr)
        if isinstance(val, str):
            return val
    except Exception:
        pass
    raise ValueError(f"Could not resolve node identifier from AST: {ast.dump(expr)}")


def _parse_retry_policy(expr: ast.AST) -> RetryPolicy | None:
    """Parse RetryPolicy constructor call from AST."""
    if isinstance(expr, ast.Constant) and expr.value is None:
        return None
    if isinstance(expr, ast.Call):
        kw_dict = {
            kw.arg: _safe_eval_literal(kw.value)
            for kw in expr.keywords
            if kw.arg is not None
        }
        args = [_safe_eval_literal(arg) for arg in expr.args]
        max_attempts = kw_dict.get("max_attempts", args[0] if len(args) > 0 else 3)
        base_delay = kw_dict.get("base_delay", args[1] if len(args) > 1 else 0.05)
        max_delay = kw_dict.get("max_delay", args[2] if len(args) > 2 else 2.0)
        jitter = kw_dict.get("jitter", args[3] if len(args) > 3 else True)
        return RetryPolicy(
            max_attempts=int(max_attempts),
            base_delay=float(base_delay),
            max_delay=float(max_delay),
            jitter=bool(jitter),
        )
    return None


def _resolve_node_class(
    expr: ast.AST,
    reg: NodeRegistry,
    custom_node_classes: dict[str, type[Node]],
    imports_map: dict[str, str],
) -> type[Node] | None:
    """Resolve a Node class from an AST expression."""
    if isinstance(expr, ast.Name):
        name = imports_map.get(expr.id, expr.id)
        if name in custom_node_classes:
            return custom_node_classes[name]
        if name in BUILTIN_CLASS_MAP:
            return BUILTIN_CLASS_MAP[name]
        for t in reg.list_types():
            cls = reg.get(t)
            if cls is not None and cls.__name__ == name:
                return cls
        if reg.has(name.lower()):
            return reg.get(name.lower())
        if name.lower().endswith("node"):
            prefix = name.lower()[:-4]
            if reg.has(prefix):
                return reg.get(prefix)
        return None

    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Attribute) and expr.func.attr == "get":
            if (
                expr.args
                and isinstance(expr.args[0], ast.Constant)
                and isinstance(expr.args[0].value, str)
            ):
                node_type = expr.args[0].value
                cls = reg.get(node_type)
                if cls is not None:
                    return cls
        return None

    return None


def _try_parse_node_call(
    call: ast.Call,
    reg: NodeRegistry,
    custom_node_classes: dict[str, type[Node]],
    imports_map: dict[str, str],
    target_var_name: str | None = None,
) -> Node | None:
    """Attempt to parse a node instantiation AST call."""
    node_cls = _resolve_node_class(call.func, reg, custom_node_classes, imports_map)
    if node_cls is None:
        if isinstance(call.func, ast.Name) and call.func.id.endswith("Node"):
            raise ValueError(
                f"Unknown node type or class {call.func.id!r}. "
                f"Register it with node_registry before importing."
            )
        return None

    kw_dict = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
    args = call.args

    # node_id
    if "node_id" in kw_dict:
        node_id: str | None = str(_safe_eval_literal(kw_dict["node_id"]))
    elif len(args) >= 1 and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
        node_id = str(args[0].value)
    elif target_var_name:
        node_id = target_var_name
    else:
        node_id = None

    # config
    if "config" in kw_dict:
        config = dict(_safe_eval_literal(kw_dict["config"]))
    elif len(args) >= 2 and isinstance(args[1], ast.Dict):
        config = dict(_safe_eval_literal(args[1]))
    else:
        config = {}

    # retry_policy
    if "retry_policy" in kw_dict:
        retry_policy = _parse_retry_policy(kw_dict["retry_policy"])
    else:
        retry_policy = None

    node = node_cls(node_id=node_id, config=config, retry_policy=retry_policy)

    if "status" in kw_dict:
        node.status = cast(NodeStatus, str(_safe_eval_literal(kw_dict["status"])))

    return node


def _parse_edge_call(
    call: ast.Call,
    graph: SimpleGraph,
    var_nodes: dict[str, Node],
    nodes_by_id: dict[str, Node],
    is_error: bool = False,
    validate_ports: bool = False,
) -> None:
    """Parse graph.add_edge(...) or graph.add_error_edge(...) AST call."""
    kw_dict = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
    args = call.args

    if "source_id" in kw_dict:
        source_id = _resolve_node_id(kw_dict["source_id"], var_nodes)
    elif "source" in kw_dict:
        source_id = _resolve_node_id(kw_dict["source"], var_nodes)
    elif "source_node" in kw_dict:
        source_id = _resolve_node_id(kw_dict["source_node"], var_nodes)
    elif len(args) >= 1:
        source_id = _resolve_node_id(args[0], var_nodes)
    else:
        raise ValueError("Missing source node identifier in add_edge call.")

    if "target_id" in kw_dict:
        target_id = _resolve_node_id(kw_dict["target_id"], var_nodes)
    elif "target" in kw_dict:
        target_id = _resolve_node_id(kw_dict["target"], var_nodes)
    elif "target_node" in kw_dict:
        target_id = _resolve_node_id(kw_dict["target_node"], var_nodes)
    elif len(args) >= 2:
        target_id = _resolve_node_id(args[1], var_nodes)
    else:
        raise ValueError("Missing target node identifier in add_edge call.")

    default_source_port = "error" if is_error else "output"
    default_target_port = "error" if is_error else "input"

    if "source_port" in kw_dict:
        source_port = str(_safe_eval_literal(kw_dict["source_port"]))
    elif len(args) >= 3:
        source_port = str(_safe_eval_literal(args[2]))
    else:
        source_port = default_source_port

    if "target_port" in kw_dict:
        target_port = str(_safe_eval_literal(kw_dict["target_port"]))
    elif len(args) >= 4:
        target_port = str(_safe_eval_literal(args[3]))
    else:
        target_port = default_target_port

    if "mapping" in kw_dict:
        mapping = dict(_safe_eval_literal(kw_dict["mapping"]))
    elif len(args) >= 5:
        mapping = dict(_safe_eval_literal(args[4]))
    else:
        mapping = {}

    if is_error:
        is_error_edge = True
    elif "is_error_edge" in kw_dict:
        is_error_edge = bool(_safe_eval_literal(kw_dict["is_error_edge"]))
    elif len(args) >= 6:
        is_error_edge = bool(_safe_eval_literal(args[5]))
    else:
        is_error_edge = False

    # Auto-add referenced nodes to graph if instantiated earlier
    if source_id in nodes_by_id and source_id not in graph.nodes:
        graph.add_node(nodes_by_id[source_id])
    if target_id in nodes_by_id and target_id not in graph.nodes:
        graph.add_node(nodes_by_id[target_id])

    graph.add_edge(
        source_id=source_id,
        target_id=target_id,
        source_port=source_port,
        target_port=target_port,
        mapping=mapping,
        is_error_edge=is_error_edge,
        validate_ports=validate_ports,
    )


def _collect_statements(nodes: list[ast.stmt]) -> list[ast.stmt]:
    """Recursively collect statements within blocks and functions in order."""
    stmts: list[ast.stmt] = []
    for stmt in nodes:
        stmts.append(stmt)
        if isinstance(stmt, (ast.If, ast.Try)):
            stmts.extend(_collect_statements(stmt.body))
            stmts.extend(_collect_statements(stmt.orelse))
        elif isinstance(stmt, ast.With):
            stmts.extend(_collect_statements(stmt.body))
        elif isinstance(stmt, ast.FunctionDef):
            stmts.extend(_collect_statements(stmt.body))
    return stmts


def _unroll_calls(expr: ast.AST) -> list[ast.Call]:
    """Unroll chained method calls into sequential execution order."""
    calls: list[ast.Call] = []
    current: ast.AST | None = expr
    while isinstance(current, ast.Call):
        calls.append(current)
        if isinstance(current.func, ast.Attribute):
            current = current.func.value
        else:
            break
    calls.reverse()
    return calls


def import_graph_from_code(
    code_str: str,
    registry: NodeRegistry | None = None,
    validate_ports: bool = False,
) -> SimpleGraph:
    """Statically parse Python SDK source code into a SimpleGraph without code execution."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax in graph code: {exc}") from exc

    reg = registry if registry is not None else node_registry
    graph = SimpleGraph()
    var_nodes: dict[str, Node] = {}
    nodes_by_id: dict[str, Node] = {}
    custom_node_classes: dict[str, type[Node]] = {}
    imports_map: dict[str, str] = {}

    # 1. Scan for custom_node decorator definitions
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            for dec in stmt.decorator_list:
                if isinstance(dec, ast.Call):
                    dec_name = (
                        dec.func.id
                        if isinstance(dec.func, ast.Name)
                        else getattr(dec.func, "attr", "")
                    )
                    if dec_name == "custom_node":
                        dec_kw = {
                            kw.arg: _safe_eval_literal(kw.value)
                            for kw in dec.keywords
                            if kw.arg is not None
                        }
                        node_type = dec_kw.get("node_type")
                        if node_type:
                            label = dec_kw.get("label", node_type.replace("_", " ").title())
                            description = dec_kw.get("description", stmt.name)
                            category = dec_kw.get("category", "custom")
                            inputs = dec_kw.get("inputs", [])
                            outputs = dec_kw.get("outputs", [])
                            config_schema = dec_kw.get("config_schema", {})

                            def _dummy_custom_fn(
                                inputs: dict[str, Any],
                                config: dict[str, Any] | None = None,
                            ) -> dict[str, Any]:
                                return {}

                            custom_cls = custom_node(
                                node_type=node_type,
                                label=label,
                                description=description,
                                category=category,
                                inputs=inputs,
                                outputs=outputs,
                                config_schema=config_schema,
                                registry=reg,
                                register=True,
                            )(_dummy_custom_fn)
                            custom_node_classes[stmt.name] = custom_cls
                            custom_node_classes[custom_cls.__name__] = custom_cls

    all_stmts = _collect_statements(tree.body)

    for stmt in all_stmts:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.asname:
                    imports_map[alias.asname] = alias.name
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                actual_name = alias.name
                used_name = alias.asname or alias.name
                imports_map[used_name] = actual_name

        elif isinstance(stmt, ast.Assign):
            # Check target
            if len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    var_name = target.id
                    if isinstance(stmt.value, ast.Call):
                        node = _try_parse_node_call(
                            stmt.value,
                            reg,
                            custom_node_classes,
                            imports_map,
                            target_var_name=var_name,
                        )
                        if node is not None:
                            var_nodes[var_name] = node
                            nodes_by_id[node.id] = node
                elif isinstance(target, ast.Attribute):
                    # var.status = "..."
                    if (
                        isinstance(target.value, ast.Name)
                        and target.value.id in var_nodes
                        and target.attr == "status"
                    ):
                        var_nodes[target.value.id].status = cast(
                            NodeStatus, str(_safe_eval_literal(stmt.value))
                        )
                    elif target.attr == "_execution_order":
                        order = _safe_eval_literal(stmt.value)
                        if isinstance(order, list):
                            graph._execution_order = [str(x) for x in order]

        elif isinstance(stmt, ast.Expr):
            calls = _unroll_calls(stmt.value)
            for call in calls:
                if isinstance(call.func, ast.Attribute):
                    attr_name = call.func.attr
                    if attr_name == "add_node":
                        if call.args:
                            arg0 = call.args[0]
                            if isinstance(arg0, ast.Name):
                                if arg0.id in var_nodes:
                                    n = var_nodes[arg0.id]
                                    graph.add_node(n)
                            elif isinstance(arg0, ast.Call):
                                inline_node = _try_parse_node_call(
                                    arg0,
                                    reg,
                                    custom_node_classes,
                                    imports_map,
                                )
                                if inline_node is not None:
                                    nodes_by_id[inline_node.id] = inline_node
                                    graph.add_node(inline_node)
                    elif attr_name in ("add_edge", "add_error_edge"):
                        _parse_edge_call(
                            call,
                            graph,
                            var_nodes,
                            nodes_by_id,
                            is_error=(attr_name == "add_error_edge"),
                            validate_ports=validate_ports,
                        )

    return graph
