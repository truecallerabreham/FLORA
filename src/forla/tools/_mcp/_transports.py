"""
Transport layer implementations for connecting to MCP servers.

Supports stdio, SSE, and streamable HTTP transports.
"""

from typing import Any, Tuple

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from ._config import HTTPServerConfig, MCPServerConfig, StdioServerConfig


async def connect_to_server(config: MCPServerConfig) -> Tuple[Any, Any, Any]:
    """
    Connect to an MCP server using the appropriate transport.

    Args:
        config: Server configuration

    Returns:
        Tuple of (read_stream, write_stream, client_context)

    Raises:
        ValueError: If transport type is not supported
        ImportError: If required transport dependencies are missing
    """
    if isinstance(config, StdioServerConfig):
        return await _connect_stdio(config)
    elif isinstance(config, HTTPServerConfig):
        if config.transport == "sse":
            return await _connect_sse(config)
        else:  # streamable-http
            return await _connect_http(config)
    else:
        raise ValueError(f"Unknown transport: {config.transport}")


async def _connect_stdio(config: StdioServerConfig) -> Tuple[Any, Any, Any]:
    """
    Connect using stdio transport.

    Spawns a subprocess and communicates via stdin/stdout.
    """
    server_params = StdioServerParameters(
        command=config.command,
        args=config.args,
        env=config.env or {},
    )

    client_context = stdio_client(server_params)
    read, write = await client_context.__aenter__()
    return read, write, client_context


async def _connect_sse(config: HTTPServerConfig) -> Tuple[Any, Any, Any]:
    """
    Connect using SSE (Server-Sent Events) transport.

    Suitable for one-way streaming from server to client.
    """
    try:
        from mcp.client.sse import sse_client
    except ImportError as e:
        raise ImportError(
            "SSE transport requires mcp[sse] to be installed. "
            "Install with: pip install mcp[sse]"
        ) from e

    client_context = sse_client(
        url=config.url,
        headers=config.headers or {},
    )

    read, write = await client_context.__aenter__()
    return read, write, client_context


async def _connect_http(config: HTTPServerConfig) -> Tuple[Any, Any, Any]:
    """
    Connect using streamable HTTP transport.

    The recommended transport for production deployments.
    Supports both stateful and stateless operation modes.
    """
    try:
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as e:
        raise ImportError(
            "Streamable HTTP transport requires mcp to be installed. "
            "Install with: pip install mcp"
        ) from e

    client_context = streamablehttp_client(
        url=config.url,
        headers=config.headers or {},
    )

    read, write, _ = await client_context.__aenter__()
    return read, write, client_context
