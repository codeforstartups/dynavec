"""Tests for the ToolRegistry and built-in connectors."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from dynavec.agents import (
    ToolRegistry,
    create_http_tool,
    create_python_code_tool,
    mcp_tools_from_session,
    tool,
)


def test_registry_crud() -> None:
    """Test registry CRUD operations and tags."""
    registry = ToolRegistry()

    @tool(name="calc", tags=["math"])
    def calc_tool(a: int, b: int) -> int:
        return a + b

    # Register
    registry.register(calc_tool)
    assert registry.has("calc")
    assert len(registry) == 1

    # Get
    retrieved = registry.get("calc")
    assert retrieved is not None
    assert retrieved.name == "calc"
    assert "math" in retrieved.tags

    # Execute
    assert registry.execute("calc", {"a": 2, "b": 3}) == "5"

    # List and filter
    assert len(registry.list_tools()) == 1
    assert len(registry.list_tools(tag="math")) == 1
    assert len(registry.list_tools(tag="api")) == 0

    # Dict-like access
    assert "calc" in registry
    assert registry["calc"].name == "calc"
    assert list(registry) == ["calc"]

    # Unregister
    assert registry.unregister("calc")
    assert not registry.has("calc")
    assert not registry.unregister("missing")


@pytest.mark.asyncio
async def test_registry_aexecute() -> None:
    """Test async execution through the registry."""
    registry = ToolRegistry()

    @tool(name="async_math")
    async def async_math(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    registry.register(async_math)
    result = await registry.aexecute("async_math", {"x": 21})
    assert result == "42"


def test_http_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the HTTP connector tool."""
    http = create_http_tool(base_url="https://api.example.com")

    # Mock urllib.request.urlopen
    class MockResponse:
        def __init__(self, data: str) -> None:
            self.data = data

        def read(self) -> bytes:
            return self.data.encode("utf-8")

        def __enter__(self) -> MockResponse:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def mock_urlopen(req: Any, **kwargs: Any) -> MockResponse:
        assert req.full_url == "https://api.example.com/test?q=hello"
        assert req.method == "GET"
        return MockResponse('{"status": "ok"}')

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    res = http.execute({"url": "/test", "params": {"q": "hello"}})
    assert '"status": "ok"' in res


def test_python_code_tool() -> None:
    """Test the Python execution sandbox."""
    py_tool = create_python_code_tool()

    # Basic arithmetic and print
    res = py_tool.execute({"code": "print('hello')\n2 + 3"})
    assert "hello\n" in res
    assert "5" in res

    # Syntax error
    err_res = py_tool.execute({"code": "def func("})
    assert "SyntaxError" in err_res

    # Runtime exception
    err_res2 = py_tool.execute({"code": "1 / 0"})
    assert "ZeroDivisionError" in err_res2


def test_mcp_adapter() -> None:
    """Test adapting an MCP session to dynaflow tools."""
    mock_session = MagicMock()

    # Mock list_tools
    class MockTool:
        name = "mcp_math"
        description = "MCP math"
        inputSchema = {"type": "object", "properties": {"a": {"type": "integer"}}}

    mock_listing = MagicMock()
    mock_listing.tools = [MockTool()]
    mock_session.list_tools.return_value = mock_listing

    # Mock call_tool
    class MockResult:
        content = [{"text": "success: 42"}]
        isError = False

    mock_session.call_tool.return_value = MockResult()

    tools = mcp_tools_from_session(mock_session, tags=["mcp"])
    assert len(tools) == 1
    assert tools[0].name == "mcp_math"
    assert tools[0].description == "MCP math"
    assert "mcp" in tools[0].tags

    # Execute
    res = tools[0].execute({"a": 42})
    assert res == "success: 42"
    mock_session.call_tool.assert_called_with("mcp_math", {"a": 42})
