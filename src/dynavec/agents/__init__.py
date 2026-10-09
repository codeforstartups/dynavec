"""Agent primitives, execution loops, and graph orchestration for dynaflow."""

from __future__ import annotations

from ..checkpoint import (
    BaseCheckpointer,
    Checkpoint,
    DynamoDBCheckpointer,
    MemoryCheckpointer,
)
from .base import (
    AgentResult,
    AgentStep,
    AgentTool,
    Plan,
    PlanStep,
    tool,
)
from .connectors import (
    create_http_tool,
    create_python_code_tool,
    mcp_tools_from_session,
    mcp_tools_from_session_async,
)
from .graph import (
    AgentGraph,
    AgentState,
    BaseNode,
    Edge,
    FlowGraph,
    FlowState,
    FunctionNode,
    GraphCycleError,
    GraphError,
    GraphResult,
    GraphState,
    GraphValidationError,
    MaxIterationsExceededError,
    NodeTrace,
    append_reducer,
    default_reducer,
)
from .planner import Planner
from .react import ReActAgent
from .registry import ToolRegistry, default_registry

__all__ = [
    # Existing agent primitives
    "AgentResult",
    "AgentStep",
    "AgentTool",
    "Plan",
    "PlanStep",
    "Planner",
    "ReActAgent",
    "tool",
    # Tool registry & connectors
    "ToolRegistry",
    "create_http_tool",
    "create_python_code_tool",
    "default_registry",
    "mcp_tools_from_session",
    "mcp_tools_from_session_async",
    # Checkpointer
    "Checkpoint",
    "BaseCheckpointer",
    "MemoryCheckpointer",
    "DynamoDBCheckpointer",
    # Graph orchestration engine (#277 / #278)
    "AgentGraph",
    "AgentState",
    "BaseNode",
    "Edge",
    "FlowGraph",
    "FlowState",
    "FunctionNode",
    "GraphCycleError",
    "GraphError",
    "GraphResult",
    "GraphState",
    "GraphValidationError",
    "MaxIterationsExceededError",
    "NodeTrace",
    "append_reducer",
    "default_reducer",
]
