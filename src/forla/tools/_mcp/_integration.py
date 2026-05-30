"""
Helper functions for easy MCP integration with Forla.

This module provides high-level utilities for quickly setting up
MCP servers and creating tools for use with agents.
"""

from typing import Any, Callable, List, Tuple, Union

from .._base import BaseTool
from ._client import MCPClientManager
from ._config import MCPServerConfig


async def create_mcp_tools(
    server_configs: List[MCPServerConfig],
    auto_connect: bool = True,
) -> Tuple[MCPClientManager, List[Union[BaseTool, Callable[..., Any]]]]:
    """
    Create MCP tools from server configurations.

    This is the main entry point for integrating MCP servers
    with Forla. It handles connection setup and tool discovery.

    Args:
        server_configs: List of MCP server configurations
        auto_connect: If True, connect immediately and discover tools.
                     If False, servers are registered but not connected.

    Returns:
        Tuple of (client_manager, discovered_tools)

    Example:
        ```python
        from forla.tools import create_mcp_tools, StdioServerConfig

        # Configure MCP servers
        github_config = StdioServerConfig(
            server_id="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "your-token"}
        )

        filesystem_config = StdioServerConfig(
            server_id="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )

        # Create tools
        manager, tools = await create_mcp_tools([
            github_config,
            filesystem_config,
        ])

        # Use with agent
        agent = Agent(
            name="mcp_agent",
            tools=tools,  # MCP tools work like native tools
            model_client=model_client,
        )

        # Run agent
        result = await agent.run("List files in /tmp")

        # Cleanup when done
        await manager.disconnect_all()
        ```

    Note:
        Always call `await manager.disconnect_all()` when done to clean up
        server connections and release resources.
    """
    manager = MCPClientManager()

    # Register all servers
    for config in server_configs:
        manager.add_server(config)

    # Connect and discover tools if requested
    if auto_connect:
        for config in server_configs:
            await manager.connect(config.server_id)

    # Get all discovered tools
    all_tools = manager.get_tools()

    return manager, all_tools
