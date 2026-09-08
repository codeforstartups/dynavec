"""Model Context Protocol (MCP) server for dynavec.

Allows any MCP client (Claude Desktop, Cursor, Antigravity, custom agents)
to query dynavec over FastMCP tools.
"""

from __future__ import annotations

from .server import client_from_env, create_mcp_server, main

__all__ = ["client_from_env", "create_mcp_server", "main"]

