"""
MCP (Model Context Protocol) integration for Forla.

This module provides integration with MCP servers, allowing agents to use
tools from any MCP-compliant server as if they were native Forla tools.

Example:
    ```python
    from forla.tools import create_mcp_tools, StdioServerConfig

    # Configure MCP server
    config = StdioServerConfig(
        server_id="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    # Create tools
    manager, tools = await create_mcp_tools([config])

    # Use with agent
    agent = Agent(name="mcp_agent", tools=tools, ...)
    ```
"""

from ._config import HTTPServerConfig, MCPServerConfig, StdioServerConfig, TransportType
from ._integration import create_mcp_tools
from ._tool import MCPTool

try:
    from ._client import MCPClientManager

    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCPClientManager = None  # type: ignore
    MCP_CLIENT_AVAILABLE = False

__all__ = [
    "MCPTool",
    "MCPClientManager",
    "MCPServerConfig",
    "StdioServerConfig",
    "HTTPServerConfig",
    "TransportType",
    "create_mcp_tools",
    "MCP_CLIENT_AVAILABLE",
]
