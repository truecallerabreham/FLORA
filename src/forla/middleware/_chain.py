from __future__ import annotations
from typing import Any, Callable, List
from ._base import BaseMiddleware, MiddlewareContext


class MiddlewareChain:
    """Executes middleware in sequence, threading context through each one.
    
    HOW IT WORKS:
    For a chain [Middleware1, Middleware2, Middleware3]:
    
    Forward pass (before operation):
    context → M1.process_request → M2.process_request → M3.process_request → operation
    
    Backward pass (after operation):
    result → M3.process_response → M2.process_response → M1.process_response → caller
    
    Each middleware in the forward pass can see and modify the context.
    Each middleware in the backward pass can see and modify the result.
    Any middleware can block by raising an exception.
    
    WHY this order? The last middleware added is closest to the actual operation.
    Ordering matters: you want security checks BEFORE logging (don't log blocked requests).
    """

    def __init__(self, middlewares: List[BaseMiddleware]):
        self._middlewares = middlewares

    async def execute_stream(
        self,
        operation: str,
        agent_name: str,
        data: Any,
        func: Callable,
        tool_name: str = None,
    ):
        """Execute func through the middleware chain, yielding events along the way.
        
        This is an async generator. It yields:
        1. Events from middleware (logs, approval requests, etc.)
        2. The final result from func
        """
        # Build the context for this operation
        context = MiddlewareContext(
            operation=operation,
            agent_name=agent_name,
            data=data,
            tool_name=tool_name,
        )

        # === Forward pass: process_request through all middleware ===
        for mw in self._middlewares:
            async for item in mw.process_request(context):
                if isinstance(item, MiddlewareContext):
                    context = item    # Updated context — use it for the next middleware
                else:
                    yield item        # It's an event — pass it to the caller

        # === Execute the actual operation ===
        try:
            result = await func(context.data)
        except Exception as error:
            # Backward pass through errors
            for mw in reversed(self._middlewares):
                async for item in mw.process_error(context, error):
                    yield item
            return

        # === Backward pass: process_response through all middleware (reversed) ===
        for mw in reversed(self._middlewares):
            async for item in mw.process_response(context, result):
                result = item  # Allow middleware to transform the result

        yield result  # Yield the final (possibly transformed) result
