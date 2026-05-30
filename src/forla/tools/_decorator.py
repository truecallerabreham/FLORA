"""
Tool decorator for creating tools from functions with approval support.
"""

from functools import wraps
from typing import Callable, Optional, TypeVar, Union

from ._base import ApprovalMode, FunctionTool

T = TypeVar("T")


def tool(
    func: Optional[Callable[..., T]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    approval_mode: Union[str, ApprovalMode] = "never_require",
) -> Union[FunctionTool, Callable[[Callable[..., T]], FunctionTool]]:
    """
    Decorator to create a tool from a function with approval support.

    Args:
        func: The function to wrap (when used without parentheses)
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        approval_mode: When to require approval ("always_require" or "never_require")

    Returns:
        FunctionTool or decorator function

    Example:
        # Without approval
        @tool
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Weather in {city}: Sunny"

        # With approval required
        @tool(approval_mode="always_require")
        def delete_file(path: str) -> str:
            '''Delete a file from the filesystem.'''
            os.remove(path)
            return f"Deleted {path}"

        # With custom name
        @tool(name="weather_tool", description="Gets weather info")
        def my_weather_func(city: str) -> str:
            return f"Weather in {city}"
    """

    def decorator(f: Callable[..., T]) -> FunctionTool:
        # Convert string to enum if needed
        mode = (
            ApprovalMode(approval_mode)
            if isinstance(approval_mode, str)
            else approval_mode
        )

        tool_name = name or f.__name__
        tool_desc = description or f.__doc__ or ""

        return FunctionTool(
            func=f,
            name=tool_name,
            description=tool_desc,
            approval_mode=mode,
        )

    # If func is provided, we're being used without parentheses
    if func is not None:
        return decorator(func)

    # Otherwise, we're being used with parentheses
    return decorator