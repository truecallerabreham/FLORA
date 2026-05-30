"""
MCPClientManager - Manages connections to MCP servers.

This module handles server lifecycle, tool discovery, and session management
for multiple MCP servers across different transports.
"""

from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Union

from mcp import ClientSession

from .._base import BaseTool
from ._config import MCPServerConfig
from ._tool import MCPTool
from ._transports import connect_to_server


class MCPClientManager:
    """
    Manages connections to MCP servers and provides tool discovery.

    This class handles:
    - Connecting to multiple MCP servers (stdio, SSE, HTTP)
    - Discovering available tools from each server
    - Creating MCPTool instances for discovered tools
    - Lifecycle management (startup/shutdown)
    - Session caching and reuse

    Example:
        ```python
        manager = MCPClientManager()

        # Add server configurations
        manager.add_server(StdioServerConfig(
            server_id="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        ))

        # Connect and discover tools
        await manager.connect("filesystem")

        # Get discovered tools
        tools = manager.get_tools("filesystem")

        # Use tools with agents
        agent = Agent(name="file_agent", tools=tools, ...)

        # Cleanup
        await manager.disconnect_all()
        ```
    """

    def __init__(self):
        """Initialize an empty client manager."""
        self._servers: Dict[str, MCPServerConfig] = {}
        self._sessions: Dict[str, ClientSession] = {}
        self._tools: Dict[str, List[MCPTool]] = {}
        self._client_contexts: Dict[str, Any] = {}

    def add_server(self, config: MCPServerConfig) -> None:
        """
        Register an MCP server configuration.

        The server is not connected until connect() is called.

        Args:
            config: Server configuration with transport details

        Raises:
            ValueError: If a server with this ID is already registered
        """
        if config.server_id in self._servers:
            raise ValueError(f"Server '{config.server_id}' is already registered")

        self._servers[config.server_id] = config

    async def connect(self, server_id: str) -> None:
        """
        Connect to an MCP server and discover its tools.

        This method:
        1. Establishes connection using the configured transport
        2. Initializes the MCP session
        3. Discovers available tools
        4. Creates MCPTool instances for each tool

        Args:
            server_id: ID of the server to connect to

        Raises:
            ValueError: If server_id is not registered
            ConnectionError: If connection fails
        """
        if server_id not in self._servers:
            raise ValueError(f"Unknown server: {server_id}")

        # Skip if already connected
        if server_id in self._sessions:
            return

        config = self._servers[server_id]

        try:
            # Connect using appropriate transport
            read, write, client_context = await connect_to_server(config)
            self._client_contexts[server_id] = client_context

            # Create and initialize session
            session = ClientSession(read, write)
            await session.__aenter__()
            self._sessions[server_id] = session

            # Initialize the MCP connection
            await session.initialize()

            # Discover tools
            await self._discover_tools(server_id)

        except Exception as e:
            # Cleanup on failure
            if server_id in self._sessions:
                try:
                    await self._sessions[server_id].__aexit__(None, None, None)
                except Exception:
                    pass
                del self._sessions[server_id]

            if server_id in self._client_contexts:
                try:
                    await self._client_contexts[server_id].__aexit__(None, None, None)
                except Exception:
                    pass
                del self._client_contexts[server_id]

            raise ConnectionError(
                f"Failed to connect to MCP server '{server_id}': {e}"
            ) from e

    async def _discover_tools(self, server_id: str) -> None:
        """
        Discover available tools from an MCP server.

        Args:
            server_id: Server to discover tools from
        """
        session = self._sessions[server_id]

        # List available tools
        tools_response = await session.list_tools()

        # Create MCPTool instances
        mcp_tools = []
        for tool in tools_response.tools:
            mcp_tool = MCPTool(
                mcp_tool_name=tool.name,
                mcp_tool_description=tool.description or "",
                mcp_tool_schema=tool.inputSchema,
                client_manager=self,
                server_id=server_id,
            )
            mcp_tools.append(mcp_tool)

        self._tools[server_id] = mcp_tools

    async def get_session(self, server_id: str) -> ClientSession:
        """
        Get the MCP client session for a server.

        Automatically connects if not already connected.

        Args:
            server_id: Server ID

        Returns:
            ClientSession for the server

        Raises:
            ValueError: If server_id is not registered
        """
        if server_id not in self._sessions:
            await self.connect(server_id)
        return self._sessions[server_id]

    def get_tools(
        self, server_id: Optional[str] = None
    ) -> List[Union[BaseTool, Callable[..., Any]]]:
        """
        Get tools from MCP servers.

        Args:
            server_id: If provided, return tools from specific server.
                      If None, return tools from all connected servers.

        Returns:
            List of tools compatible with Agent's tools parameter.
            Returns Union type to match Agent signature exactly.
        """
        if server_id:
            # Cast to match Agent's expected type
            tools: List[Union[BaseTool, Callable[..., Any]]] = list(
                self._tools.get(server_id, [])
            )
            return tools

        # Return tools from all servers
        all_tools: List[Union[BaseTool, Callable[..., Any]]] = []
        for tools_list in self._tools.values():
            all_tools.extend(tools_list)
        return all_tools

    def list_servers(self) -> List[str]:
        """
        List all registered server IDs.

        Returns:
            List of server IDs
        """
        return list(self._servers.keys())

    def is_connected(self, server_id: str) -> bool:
        """
        Check if a server is currently connected.

        Args:
            server_id: Server ID to check

        Returns:
            True if connected, False otherwise
        """
        return server_id in self._sessions

    async def disconnect(self, server_id: str) -> None:
        """
        Disconnect from an MCP server.

        Cleans up sessions, contexts, and cached tools.

        Args:
            server_id: Server to disconnect from
        """
        # Close session
        if server_id in self._sessions:
            try:
                session = self._sessions.pop(server_id)
                await session.__aexit__(None, None, None)
            except Exception:
                pass  # Best effort cleanup

        # Close client context
        if server_id in self._client_contexts:
            try:
                context = self._client_contexts.pop(server_id)
                await context.__aexit__(None, None, None)
            except Exception:
                pass

        # Clear cached tools
        if server_id in self._tools:
            del self._tools[server_id]

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        server_ids = list(self._sessions.keys())
        for server_id in server_ids:
            await self.disconnect(server_id)

    @asynccontextmanager
    async def managed_connection(self, server_id: str):
        """
        Context manager for automatic connection/disconnection.

        Ensures proper cleanup even if errors occur.

        Args:
            server_id: Server to connect to

        Yields:
            The manager instance

        Example:
            ```python
            async with manager.managed_connection("github"):
                tools = manager.get_tools("github")
                # Use tools...
            # Automatically disconnected
            ```
        """
        await self.connect(server_id)
        try:
            yield self
        finally:
            await self.disconnect(server_id)

    def __repr__(self) -> str:
        connected = [sid for sid in self._servers if self.is_connected(sid)]
        return (
            f"MCPClientManager("
            f"servers={len(self._servers)}, "
            f"connected={len(connected)}, "
            f"tools={sum(len(t) for t in self._tools.values())})"
        )
