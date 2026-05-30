from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Union


@dataclass
class MiddlewareContext:
    """Complete context of the operation being intercepted.
    
    'operation' tells you what kind of operation this is:
    - "model_call": About to call the LLM API
    - "tool_call": About to execute a tool
    
    'data' is the input:
    - For model_call: the list of messages being sent to the LLM
    - For tool_call: the parameters being sent to the tool
    
    'metadata' is a shared dict you can use to pass information
    between process_request and process_response (e.g., start timestamp).
    """
    operation: str                          # "model_call" or "tool_call"
    agent_name: str
    data: Any                               # Messages or tool parameters
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_name: Optional[str] = None         # Set for tool_call operations


class BaseMiddleware(ABC):
    """Intercepts operations in the agent pipeline.
    
    WHY async generators (yield) instead of async functions (return)?
    
    Middleware needs to do multiple things:
    1. Check/modify the request (before the operation)
    2. Optionally emit events (log entries, approval requests)
    3. Pass control to the next middleware or the actual operation
    4. Check/modify the response (after the operation)
    
    With async generators, a middleware can yield events BEFORE yielding
    the context, which lets the caller see those events in real time.
    
    Example of middleware that logs AND passes through:
    
        async def process_request(self, ctx):
            print(f"Starting {ctx.operation}...")   # Before
            yield ctx                                # Pass through (required)
    
    Example of middleware that BLOCKS:
    
        async def process_request(self, ctx):
            if "malicious" in str(ctx.data):
                raise ValueError("Blocked!")        # Never reaches yield
            yield ctx                               # Only yields if not blocked
    """

    @abstractmethod
    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Union[MiddlewareContext, Any], None]:
        """Called BEFORE the model call or tool execution.
        
        To ALLOW: yield context (possibly modified)
        To BLOCK: raise an exception (never yield)
        To EMIT EVENTS: yield event objects before yielding context
        """
        yield context   # Default: pass through unchanged

    @abstractmethod
    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Any, None]:
        """Called AFTER the operation completes successfully."""
        yield result    # Default: pass through unchanged

    @abstractmethod
    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Any, None]:
        """Called when the operation raises an exception."""
        raise error     # Default: re-raise the exception
