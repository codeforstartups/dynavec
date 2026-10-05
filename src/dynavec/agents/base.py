"""Core data models and tool primitives for dynaflow agents."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, overload

from ..chat.base import Tool, ToolCall


@dataclass
class AgentStep:
    """A single execution step in an agent's reasoning loop."""

    step_number: int
    thought: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """The final result of an agent run."""

    output: str
    steps: list[AgentStep] = field(default_factory=list)
    finished: bool = True
    termination_reason: str = (
        "completed"  # "completed", "max_steps_reached", "error", "interrupted"
    )
    total_steps: int = 0
    tool_calls_count: int = 0
    interrupt_payload: dict[str, Any] | None = None


@dataclass
class PlanStep:
    """A single step in a decomposed task plan."""

    step_number: int
    description: str
    tool_hint: str | None = None


@dataclass
class Plan:
    """An ordered decomposition of a goal into actionable steps."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)


def _python_type_to_json_type(py_type: Any) -> str:
    """Map standard Python types and type annotation strings to JSON Schema data types."""
    if py_type in (str, "str", "string"):
        return "string"
    if py_type in (int, "int", "integer"):
        return "integer"
    if py_type in (float, "float", "number"):
        return "number"
    if py_type in (bool, "bool", "boolean"):
        return "boolean"
    if py_type in (list, tuple, set, "list", "tuple", "set") or (
        isinstance(py_type, str) and py_type.startswith(("list[", "Sequence[", "tuple["))
    ):
        return "array"
    if py_type in (dict, Any, "dict", "dict[str, Any]", "Mapping") or (
        isinstance(py_type, str) and py_type.startswith("dict[")
    ):
        return "object"
    return "string"


def _generate_json_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Generate a JSON schema parameters dictionary from a function signature."""
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = param.annotation
        json_type = (
            _python_type_to_json_type(param_type)
            if param_type is not inspect.Parameter.empty
            else "string"
        )

        properties[param_name] = {
            "type": json_type,
            "description": f"Parameter '{param_name}'",
        }

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


class AgentTool:
    """An executable tool wrapped with a typed JSON schema for agent use."""

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        # Extract name directly or fall back to function attribute
        self.name = (
            name or getattr(fn, "name", None) or getattr(fn, "__name__", None) or "unnamed_tool"
        )
        self.description = description or (fn.__doc__ or f"Execute {self.name}").strip()
        self.parameters = parameters if parameters is not None else _generate_json_schema(fn)

    def to_chat_tool(self) -> Tool:
        """Convert to a dynavec.chat.Tool schema."""
        return Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def execute(self, arguments: dict[str, Any] | str | None = None) -> str:
        """Execute the wrapped function and return a string observation."""
        from ..exceptions import NodeInterrupt

        parsed_args: dict[str, Any] = {}
        if isinstance(arguments, str):
            if arguments.strip():
                try:
                    loaded = json.loads(arguments)
                    if isinstance(loaded, dict):
                        parsed_args = loaded
                    else:
                        parsed_args = {"input": loaded}
                except Exception:
                    parsed_args = {"input": arguments}
        elif isinstance(arguments, dict):
            parsed_args = arguments

        try:
            # Check if function accepts kwargs or positional
            sig = inspect.signature(self.fn)
            params = sig.parameters

            if not params:
                result = self.fn()
            elif len(params) == 1 and list(params.keys())[0] not in parsed_args:
                # If single param expected and keys don't match, pass the first val or raw dict
                first_val = next(iter(parsed_args.values())) if parsed_args else arguments
                result = self.fn(first_val)
            else:
                # Filter only valid keyword arguments
                valid_args = {k: v for k, v in parsed_args.items() if k in params}
                result = self.fn(**valid_args)

            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)

        except NodeInterrupt:
            raise

        except Exception as exc:  # noqa: BLE001
            return f"Error executing tool {self.name!r}: {exc}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)











@overload
def tool(fn: Callable[..., Any], /) -> AgentTool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], AgentTool]: ...


def tool(
    fn_or_name: Callable[..., Any] | str | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> AgentTool | Callable[[Callable[..., Any]], AgentTool]:
    """Decorator to convert a standard Python function into an AgentTool."""
    custom_name = name if name is not None else (fn_or_name if isinstance(fn_or_name, str) else None)

    if callable(fn_or_name):
        fn = fn_or_name
        tool_name = custom_name or getattr(fn, "__name__", "unnamed_tool")
        return AgentTool(
            fn=fn,
            name=tool_name,
            description=description,
            parameters=parameters,
        )

    def decorator(fn: Callable[..., Any]) -> AgentTool:
        if isinstance(fn, AgentTool):
            if custom_name:
                fn.name = custom_name
            if description:
                fn.description = description
            if parameters:
                fn.parameters = parameters
            return fn

        tool_name = custom_name or getattr(fn, "__name__", "unnamed_tool")
        return AgentTool(
            fn=fn,
            name=tool_name,
            description=description,
            parameters=parameters,
        )

    return decorator


def interrupt(thread_id: str, node_id: str, payload: dict[str, Any] | None = None) -> None:
    """Pause current run and wait for human input/approval."""
    from ..exceptions import NodeInterrupt

    raise NodeInterrupt(thread_id=thread_id, node_id=node_id, payload=payload)
