from __future__ import annotations
import asyncio
import inspect
from typing import Any, Callable, Dict, Optional, get_type_hints
from ._base import BaseTool, ToolResult


class FunctionTool(BaseTool):
    """Wraps a plain Python function as a tool.
    
    The 'magic': it inspects the function's signature, type hints, and docstring
    to automatically generate the JSON schema for the LLM.
    
    You write this:
        def get_weather(location: str) -> str:
            '''Get weather for a location.'''
            return f"Sunny in {location}"
    
    And get this automatically:
        {
          "name": "get_weather",
          "description": "Get weather for a location.",
          "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
          }
        }
    
    This works for both synchronous and asynchronous functions.
    """

    def __init__(self, func: Callable):
        # Extract name and description from the function
        name = func.__name__
        description = inspect.getdoc(func) or f"Tool: {name}"
        super().__init__(name=name, description=description)
        
        self._func = func
        self._parameters_schema = self._build_schema_from_function(func)

    def _build_schema_from_function(self, func: Callable) -> Dict[str, Any]:
        """Inspect the function's type hints to build a JSON Schema.
        
        Handles: str, int, float, bool, and Optional[X] types.
        For production: consider using a library like 'docstring_parser'
        for more sophisticated parameter descriptions.
        """
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        
        properties: Dict[str, Any] = {}
        required: list = []
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' (for methods) and 'context' (workflow context parameter)
            if param_name in ("self", "context"):
                continue
            
            type_hint = hints.get(param_name, str)
            json_type = self._python_type_to_json_schema_type(type_hint)
            
            properties[param_name] = {"type": json_type}
            
            # If the parameter has no default value, it is required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _python_type_to_json_schema_type(self, python_type) -> str:
        """Map Python type annotations to JSON Schema type strings."""
        # Direct mapping for basic types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        
        # Handle Optional[X] → extract the inner type (ignore None)
        if hasattr(python_type, "__args__"):
            args = [a for a in python_type.__args__ if a is not type(None)]
            if args:
                return self._python_type_to_json_schema_type(args[0])
        
        return type_map.get(python_type, "string")

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters_schema

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the wrapped function, handling both sync and async cases."""
        try:
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(**parameters)
            else:
                result = self._func(**parameters)
            return ToolResult(success=True, result=result)
        except TypeError as e:
            # Wrong parameters passed
            return ToolResult(success=False, error=f"Invalid parameters: {e}")
        except Exception as e:
            # Any other error from the function
            return ToolResult(success=False, error=str(e))
