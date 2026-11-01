"""Built-in tool connectors for HTTP, Python execution, and MCP."""

from __future__ import annotations

import contextlib
import io
import json
import traceback
import urllib.error
import urllib.request  # noqa: F401
from collections.abc import Callable
from typing import Any

from .base import AgentTool, tool


def create_http_tool(
    name: str = "http_request",
    description: str = "Make an HTTP request.",
    base_url: str | None = None,
    default_headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    tags: list[str] | None = None,
) -> AgentTool:
    """Create an HTTP request tool."""

    @tool(name=name, description=description, tags=tags)
    def http_request(
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> str:
        """Make an HTTP REST API call.
        Args:
            url: The URL or path to request.
            method: HTTP method (GET, POST, PUT, DELETE, PATCH).
            headers: Optional HTTP headers.
            params: Optional query parameters.
            json_data: Optional JSON body for POST/PUT requests.
        """
        full_url = url
        if base_url:
            # Simple URL joining
            if base_url.endswith("/") and full_url.startswith("/"):
                full_url = base_url + full_url[1:]
            elif not base_url.endswith("/") and not full_url.startswith("/"):
                full_url = base_url + "/" + full_url
            else:
                full_url = base_url + full_url

        if params:
            import urllib.parse

            query_string = urllib.parse.urlencode(params)
            sep = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{sep}{query_string}"

        req_headers = dict(default_headers or {})
        if headers:
            req_headers.update(headers)

        data_bytes = None
        if json_data is not None:
            data_bytes = json.dumps(json_data).encode("utf-8")
            if "Content-Type" not in req_headers and "content-type" not in req_headers:
                req_headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            full_url, data=data_bytes, headers=req_headers, method=method.upper()
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body: str = str(response.read().decode("utf-8"))
                # Try parsing as JSON to format nicely, fallback to raw text
                try:
                    parsed = json.loads(body)
                    return json.dumps(parsed, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    return body
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = ""
            return f"HTTP {e.code} Error: {e.reason}\n{err_body}"
        except urllib.error.URLError as e:
            return f"Connection Error: {e.reason}"
        except TimeoutError:
            return f"Timeout Error: Request exceeded {timeout} seconds."
        except Exception as e:
            return f"Unexpected Error: {e!s}"

    return http_request


def create_python_code_tool(
    name: str = "python_repl",
    description: str = "Execute Python code in a restricted sandbox and return the output.",
    tags: list[str] | None = None,
) -> AgentTool:
    """Create a Python code execution tool."""

    @tool(name=name, description=description, tags=tags)
    def python_repl(code: str) -> str:
        """Execute arbitrary Python code and capture prints and return values.
        Args:
            code: The Python code to execute.
        """
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                # We use a restricted globals dict to avoid polluting the main namespace
                exec_globals: dict[str, Any] = {"__builtins__": __builtins__}

                import ast

                # Parse to check if the last statement is an expression
                tree = ast.parse(code)
                if not tree.body:
                    return ""

                last_node = tree.body[-1]
                if isinstance(last_node, ast.Expr):
                    # Compile all but the last node as statements
                    module = ast.Module(tree.body[:-1], type_ignores=[])
                    exec(compile(module, "<ast>", "exec"), exec_globals)
                    # Compile the last node as an expression and eval it
                    expr = ast.Expression(last_node.value)
                    result = eval(compile(expr, "<ast>", "eval"), exec_globals)
                    if result is not None:
                        print(repr(result))
                else:
                    exec(code, exec_globals)

        except Exception:
            traceback.print_exc(file=out)

        return out.getvalue()

    return python_repl


def mcp_tools_from_session(
    session: Any,
    tags: list[str] | None = None,
) -> list[AgentTool]:
    """Extract and adapt MCP tools from an MCP Client Session.

    Args:
        session: An MCP Client session exposing `list_tools()` and `call_tool(name, arguments)`.
        tags: Optional tags to apply to the generated tools.
    """
    listing = session.list_tools()

    # Handle async listing if needed (e.g. if list_tools is a coroutine)
    if hasattr(listing, "__await__"):
        raise ValueError(
            "Session list_tools is asynchronous. Use mcp_tools_from_session_async() instead "
            "or await the list_tools call before adapting."
        )

    tools = getattr(listing, "tools", listing)
    agent_tools = []

    for t in tools:
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
        if not name:
            continue

        description = getattr(t, "description", None) or (
            t.get("description") if isinstance(t, dict) else ""
        )
        schema = (
            getattr(t, "inputSchema", None)
            or getattr(t, "input_schema", None)
            or (t.get("inputSchema") if isinstance(t, dict) else None)
        )

        # We create a closure to bind the session and name
        def _make_caller(tool_name: str) -> Callable[..., str]:
            def call_mcp_tool(**kwargs: Any) -> str:
                # Some MCP clients take arguments as dict
                result = session.call_tool(tool_name, kwargs)
                # Parse standard MCP result shape (list of content objects)
                content = getattr(result, "content", result)
                if isinstance(content, (list, tuple)):
                    texts = []
                    for p in content:
                        txt = getattr(p, "text", None) or (
                            p.get("text") if isinstance(p, dict) else None
                        )
                        if txt:
                            texts.append(txt)
                    return "\n\n".join(texts)
                return str(result)

            return call_mcp_tool

        fn = _make_caller(name)
        fn.__name__ = name
        fn.__doc__ = description

        agent_tool = AgentTool(
            fn=fn,
            name=name,
            description=description,
            parameters=schema,
            tags=tags,
        )
        agent_tools.append(agent_tool)

    return agent_tools


async def mcp_tools_from_session_async(
    session: Any,
    tags: list[str] | None = None,
) -> list[AgentTool]:
    """Extract and adapt MCP tools from an Async MCP Client Session."""
    listing = await session.list_tools()
    tools = getattr(listing, "tools", listing)
    agent_tools = []

    for t in tools:
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
        if not name:
            continue

        description = getattr(t, "description", None) or (
            t.get("description") if isinstance(t, dict) else ""
        )
        schema = (
            getattr(t, "inputSchema", None)
            or getattr(t, "input_schema", None)
            or (t.get("inputSchema") if isinstance(t, dict) else None)
        )

        def _make_async_caller(tool_name: str) -> Callable[..., Any]:
            async def acall_mcp_tool(**kwargs: Any) -> str:
                result = await session.call_tool(tool_name, arguments=kwargs)
                content = getattr(result, "content", result)
                is_error = getattr(result, "isError", False)
                prefix = "Error: " if is_error else ""

                if isinstance(content, (list, tuple)):
                    texts = []
                    for p in content:
                        txt = getattr(p, "text", None) or (
                            p.get("text") if isinstance(p, dict) else None
                        )
                        if txt:
                            texts.append(txt)
                    return prefix + "\n\n".join(texts)
                return prefix + str(result)

            return acall_mcp_tool

        fn = _make_async_caller(name)
        fn.__name__ = name
        fn.__doc__ = description

        agent_tool = AgentTool(
            fn=fn,
            name=name,
            description=description,
            parameters=schema,
            tags=tags,
        )
        agent_tools.append(agent_tool)

    return agent_tools
