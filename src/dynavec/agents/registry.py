"""Tool registry for discovering and executing agent tools."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from ..chat.base import Tool
from .base import AgentTool, tool


class ToolRegistry:
    """A registry for managing and executing agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(
        self,
        fn_or_tool: AgentTool | Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> AgentTool:
        """Register a new tool or function."""
        if isinstance(fn_or_tool, AgentTool):
            t = fn_or_tool
            # Override if explicitly provided
            if name:
                t.name = name
            if description:
                t.description = description
            if tags is not None:
                t.tags = tags
        else:
            t = tool(name=name, description=description, tags=tags)(fn_or_tool)

        self._tools[t.name] = t
        return t

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name. Returns True if removed."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> AgentTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools

    def list_tools(self, tag: str | None = None) -> list[AgentTool]:
        """List registered tools, optionally filtered by tag."""
        if tag is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if tag in getattr(t, "tags", [])]

    def to_chat_tools(self, tag: str | None = None) -> list[Tool]:
        """Export tools to chat schemas."""
        return [t.to_chat_tool() for t in self.list_tools(tag=tag)]

    def execute(self, name: str, arguments: dict[str, Any] | str | None = None) -> str:
        """Execute a tool by name synchronously."""
        t = self.get(name)
        if not t:
            return f"Error: Tool {name!r} is not registered."
        return t.execute(arguments)

    async def aexecute(self, name: str, arguments: dict[str, Any] | str | None = None) -> str:
        """Execute a tool by name asynchronously."""
        t = self.get(name)
        if not t:
            return f"Error: Tool {name!r} is not registered."
        return await t.aexecute(arguments)

    def register_all(self, tools: list[AgentTool] | ToolRegistry) -> None:
        """Register multiple tools at once."""
        if isinstance(tools, ToolRegistry):
            tools = tools.list_tools()
        for t in tools:
            self.register(t)

    def __getitem__(self, name: str) -> AgentTool:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)


default_registry = ToolRegistry()
