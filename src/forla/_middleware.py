"""
Middleware system for forla.

This module provides the middleware infrastructure for intercepting and processing
agent operations like model calls, tool calls, and memory access.

Middleware uses async generators to support:
- Event emission (for observability)
- Approval requests (pausing execution)
- Streaming transformations
- Error recovery
"""

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .context import AgentContext
from .messages import Message

if TYPE_CHECKING:
    from .types import AgentEvent


class MiddlewareContext(BaseModel):
    """Context passed through middleware chain."""

    operation: str = Field(
        description="Operation type: 'model_call', 'tool_call', 'memory_access'"
    )
    agent_name: str = Field(description="Name of the agent executing the operation")
    agent_context: AgentContext = Field(description="The agent's context")
    data: Any = Field(
        description="Operation-specific data (messages, tool request, etc)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for middleware use"
    )


class BaseMiddleware(ABC):
    """
    Abstract base class for middleware components.

    Middleware intercepts agent operations and can:
    - Emit events for observability (logs, metrics, approval requests)
    - Transform requests/responses
    - Pause execution (e.g., for approval)
    - Handle errors with recovery

    All middleware methods are async generators that yield events and/or results.
    """

    @abstractmethod
    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Union[MiddlewareContext, "AgentEvent"], None]:
        """
        Process before the operation executes.

        Yields:
            - AgentEvent: Events for observability (logs, approval requests, etc)
            - MiddlewareContext: Final yield MUST be the context (modified or original)

        To pause execution for approval:
            ```python
            yield ToolApprovalEvent(approval_request=req)
            return  # Don't yield context - agent will pause
            ```

        Example:
            ```python
            async def process_request(self, context):
                # Emit log event
                yield LogEvent(message="Processing request")

                # Check if approval needed
                if needs_approval:
                    yield ToolApprovalEvent(approval_request=req)
                    return  # PAUSE - don't yield context

                # Continue execution
                yield context
            ```

        Raises:
            Exception: To abort the operation (caught by process_error)
        """
        yield context

    @abstractmethod
    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """
        Process after the operation completes successfully.

        Args:
            context: The middleware context
            result: The operation result

        Yields:
            - AgentEvent: Events for observability
            - Any: Final yield MUST be the result (modified or original)

        Example:
            ```python
            async def process_response(self, context, result):
                # Emit metric event
                yield MetricEvent(duration=elapsed)

                # Modify result
                result.metadata["processed"] = True
                yield result
            ```
        """
        yield result

    @abstractmethod
    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """
        Handle errors from the operation.

        Args:
            context: The middleware context
            error: The exception that occurred

        Yields:
            - AgentEvent: Events for observability
            - Any: Recovery value to use instead of raising error

        To propagate the error (no recovery):
            ```python
            async def process_error(self, context, error):
                yield ErrorEvent(error=str(error))
                raise error
            ```

        To recover from error:
            ```python
            async def process_error(self, context, error):
                yield ErrorEvent(error=str(error))
                yield default_value  # Use this instead of raising
            ```
        """
        if False:  # Type checker hint - never executed
            yield
        raise error


class MiddlewareChain:
    """Executes a chain of middleware as async generator pipeline."""

    def __init__(self, middlewares: Optional[List[BaseMiddleware]] = None):
        """
        Initialize the middleware chain.

        Args:
            middlewares: List of middleware to execute in order
        """
        self.middlewares = middlewares or []

    def add(self, middleware: BaseMiddleware) -> None:
        """Add a middleware to the chain."""
        self.middlewares.append(middleware)

    def remove(self, middleware: BaseMiddleware) -> None:
        """Remove a middleware from the chain."""
        if middleware in self.middlewares:
            self.middlewares.remove(middleware)

    async def execute_stream(
        self,
        operation: str,
        agent_name: str,
        agent_context: AgentContext,
        data: Any,
        func: Callable,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """
        Execute the middleware chain with streaming support.

        This method processes requests through middleware, executes the operation,
        and processes responses, all while yielding events for observability.

        Args:
            operation: Type of operation being performed
            agent_name: Name of the agent
            agent_context: Agent's context
            data: Operation-specific data
            func: The actual operation to execute
            metadata: Optional metadata to pass to middleware

        Yields:
            - AgentEvent: Events from middleware (logs, metrics, approval requests)
            - Any: Final result from the operation

        The execution flow:
            1. Pre-process through middleware (forward order) - yields events
            2. Execute the operation
            3. Post-process through middleware (reverse order) - yields events
            4. Yield final result
        """
        # Create middleware context
        ctx = MiddlewareContext(
            operation=operation,
            agent_name=agent_name,
            agent_context=agent_context,
            data=data,
            metadata=metadata or {},
        )

        # PHASE 1: Pre-process through all middleware (forward order)
        for middleware in self.middlewares:
            try:
                final_ctx = None
                async for item in middleware.process_request(ctx):
                    # Import here to avoid circular dependency
                    from .types import (
                        ErrorEvent,
                        FatalErrorEvent,
                        MemoryRetrievalEvent,
                        MemoryUpdateEvent,
                        ModelCallEvent,
                        ModelResponseEvent,
                        ModelStreamChunkEvent,
                        TaskCompleteEvent,
                        TaskStartEvent,
                        ToolApprovalEvent,
                        ToolCallEvent,
                        ToolCallResponseEvent,
                        ToolValidationEvent,
                    )

                    if isinstance(item, MiddlewareContext):
                        # This is the final context
                        final_ctx = item
                    elif isinstance(
                        item,
                        (
                            TaskStartEvent,
                            TaskCompleteEvent,
                            ModelCallEvent,
                            ModelResponseEvent,
                            ModelStreamChunkEvent,
                            ToolCallEvent,
                            ToolCallResponseEvent,
                            ToolApprovalEvent,
                            ToolValidationEvent,
                            MemoryUpdateEvent,
                            MemoryRetrievalEvent,
                            ErrorEvent,
                            FatalErrorEvent,
                        ),
                    ):
                        # Middleware emitted an event
                        yield item

                        # Check if middleware requested approval (pause execution)
                        if isinstance(item, ToolApprovalEvent):
                            return  # STOP - wait for approval

                if final_ctx is None:
                    # Middleware didn't yield context - it must have paused
                    return

                ctx = final_ctx

            except Exception as e:
                # Middleware raised exception - try error handlers (reverse order)
                recovered = False
                for error_mw in reversed(self.middlewares):
                    try:
                        async for item in error_mw.process_error(ctx, e):
                            from .types import (
                                ErrorEvent,
                                FatalErrorEvent,
                                MemoryRetrievalEvent,
                                MemoryUpdateEvent,
                                ModelCallEvent,
                                ModelResponseEvent,
                                ModelStreamChunkEvent,
                                TaskCompleteEvent,
                                TaskStartEvent,
                                ToolApprovalEvent,
                                ToolCallEvent,
                                ToolCallResponseEvent,
                                ToolValidationEvent,
                            )

                            if isinstance(
                                item,
                                (
                                    TaskStartEvent,
                                    TaskCompleteEvent,
                                    ModelCallEvent,
                                    ModelResponseEvent,
                                    ModelStreamChunkEvent,
                                    ToolCallEvent,
                                    ToolCallResponseEvent,
                                    ToolApprovalEvent,
                                    ToolValidationEvent,
                                    MemoryUpdateEvent,
                                    MemoryRetrievalEvent,
                                    ErrorEvent,
                                    FatalErrorEvent,
                                ),
                            ):
                                yield item
                            else:
                                # Middleware recovered - yield recovery value and stop
                                yield item
                                recovered = True
                                return
                    except Exception:
                        continue

                if not recovered:
                    raise e

        # PHASE 2: Execute actual operation
        # result is guaranteed to be assigned before PHASE 3 due to exception handling
        result: Any
        try:
            result = await func(ctx.data)
        except Exception as e:
            # Error handling through middleware (reverse order)
            recovered = False
            for middleware in reversed(self.middlewares):
                try:
                    async for item in middleware.process_error(ctx, e):
                        from .types import (
                            ErrorEvent,
                            FatalErrorEvent,
                            MemoryRetrievalEvent,
                            MemoryUpdateEvent,
                            ModelCallEvent,
                            ModelResponseEvent,
                            ModelStreamChunkEvent,
                            TaskCompleteEvent,
                            TaskStartEvent,
                            ToolApprovalEvent,
                            ToolCallEvent,
                            ToolCallResponseEvent,
                            ToolValidationEvent,
                        )

                        if isinstance(
                            item,
                            (
                                TaskStartEvent,
                                TaskCompleteEvent,
                                ModelCallEvent,
                                ModelResponseEvent,
                                ModelStreamChunkEvent,
                                ToolCallEvent,
                                ToolCallResponseEvent,
                                ToolApprovalEvent,
                                ToolValidationEvent,
                                MemoryUpdateEvent,
                                MemoryRetrievalEvent,
                                ErrorEvent,
                                FatalErrorEvent,
                            ),
                        ):
                            yield item
                        else:
                            # Middleware recovered - yield recovery value
                            yield item
                            recovered = True
                            return
                except Exception:
                    continue

            if not recovered:
                raise e
            # If we reach here, middleware recovered but didn't return (shouldn't happen)
            # This path is unreachable in practice, but we assign result for type checker
            raise RuntimeError("Middleware recovery logic error")  # pragma: no cover

        # PHASE 3: Post-process through middleware (reverse order)
        for middleware in reversed(self.middlewares):
            try:
                final_result = None
                async for item in middleware.process_response(ctx, result):
                    # Import event types to check
                    from .types import (
                        ErrorEvent,
                        FatalErrorEvent,
                        MemoryRetrievalEvent,
                        MemoryUpdateEvent,
                        ModelCallEvent,
                        ModelResponseEvent,
                        ModelStreamChunkEvent,
                        TaskCompleteEvent,
                        TaskStartEvent,
                        ToolApprovalEvent,
                        ToolCallEvent,
                        ToolCallResponseEvent,
                        ToolValidationEvent,
                    )

                    # Check if item is an event type (avoid isinstance with Union)
                    if isinstance(
                        item,
                        (
                            TaskStartEvent,
                            TaskCompleteEvent,
                            ModelCallEvent,
                            ModelResponseEvent,
                            ModelStreamChunkEvent,
                            ToolCallEvent,
                            ToolCallResponseEvent,
                            ToolApprovalEvent,
                            ToolValidationEvent,
                            MemoryUpdateEvent,
                            MemoryRetrievalEvent,
                            ErrorEvent,
                            FatalErrorEvent,
                        ),
                    ):
                        yield item
                    else:
                        final_result = item

                result = final_result if final_result is not None else result

            except Exception as e:
                # Error in response processing
                for error_mw in reversed(self.middlewares):
                    try:
                        async for item in error_mw.process_error(ctx, e):
                            # Import event types
                            from .types import (
                                ErrorEvent,
                                FatalErrorEvent,
                                MemoryRetrievalEvent,
                                MemoryUpdateEvent,
                                ModelCallEvent,
                                ModelResponseEvent,
                                ModelStreamChunkEvent,
                                TaskCompleteEvent,
                                TaskStartEvent,
                                ToolApprovalEvent,
                                ToolCallEvent,
                                ToolCallResponseEvent,
                                ToolValidationEvent,
                            )

                            if isinstance(
                                item,
                                (
                                    TaskStartEvent,
                                    TaskCompleteEvent,
                                    ModelCallEvent,
                                    ModelResponseEvent,
                                    ModelStreamChunkEvent,
                                    ToolCallEvent,
                                    ToolCallResponseEvent,
                                    ToolApprovalEvent,
                                    ToolValidationEvent,
                                    MemoryUpdateEvent,
                                    MemoryRetrievalEvent,
                                    ErrorEvent,
                                    FatalErrorEvent,
                                ),
                            ):
                                yield item
                            else:
                                yield item
                                return
                    except Exception:
                        continue
                raise e

        # Yield final result
        yield result


# Example Middleware Implementations


class LoggingMiddleware(BaseMiddleware):
    """Logs all agent operations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Union[MiddlewareContext, "AgentEvent"], None]:
        """Log operation start."""
        self.logger.info(
            f"[{context.agent_name}] Starting {context.operation}",
            extra={
                "agent": context.agent_name,
                "operation": context.operation,
                "session_id": context.agent_context.session_id,
            },
        )
        context.metadata["start_time"] = time.time()
        yield context

    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """Log operation completion."""
        duration = time.time() - context.metadata.get("start_time", 0)
        self.logger.info(
            f"[{context.agent_name}] Completed {context.operation} in {duration:.2f}s",
            extra={
                "agent": context.agent_name,
                "operation": context.operation,
                "duration": duration,
                "session_id": context.agent_context.session_id,
            },
        )
        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """Log operation error."""
        self.logger.error(
            f"[{context.agent_name}] Error in {context.operation}: {error}",
            extra={
                "agent": context.agent_name,
                "operation": context.operation,
                "error_type": type(error).__name__,
                "session_id": context.agent_context.session_id,
            },
        )
        if False:  # Type checker hint
            yield
        raise error


class RateLimitMiddleware(BaseMiddleware):
    """Rate limits operations per agent."""

    def __init__(self, max_calls_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_calls_per_minute: Maximum operations allowed per minute
        """
        self.max_calls = max_calls_per_minute
        self.call_times: List[float] = []  # Stateful tracking of call times

    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Union[MiddlewareContext, "AgentEvent"], None]:
        """Check and enforce rate limit."""
        now = time.time()

        # Remove calls outside the 60-second window
        self.call_times = [t for t in self.call_times if now - t < 60]

        # Check if we've hit the limit
        if len(self.call_times) >= self.max_calls:
            # Calculate how long to wait
            oldest_call = self.call_times[0]
            wait_time = 60 - (now - oldest_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                now = time.time()

        # Record this call
        self.call_times.append(now)
        yield context

    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """No response processing needed."""
        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """No error recovery."""
        if False:  # Type checker hint
            yield
        raise error


class PIIRedactionMiddleware(BaseMiddleware):
    """Redacts personally identifiable information from inputs and outputs."""

    def __init__(self, patterns: Optional[Dict[str, str]] = None):
        """
        Initialize PII redactor.

        Args:
            patterns: Custom patterns for PII detection (regex -> replacement)
        """
        self.patterns = patterns or {
            # SSN
            r"\b\d{3}-\d{2}-\d{4}\b": "[SSN-REDACTED]",
            # Email
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "[EMAIL-REDACTED]",
            # Phone
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b": "[PHONE-REDACTED]",
            # Credit card
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b": "[CC-REDACTED]",
        }

    def _redact_text(self, text: str) -> str:
        """Apply redaction patterns to text."""
        for pattern, replacement in self.patterns.items():
            text = re.sub(pattern, replacement, text)
        return text

    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Union[MiddlewareContext, "AgentEvent"], None]:
        """Redact PII from inputs."""
        if context.operation == "model_call" and isinstance(context.data, list):
            # Create new messages with redacted content (since messages are frozen)
            redacted_messages = []
            for msg in context.data:
                if hasattr(msg, "content"):
                    # Create a new message with redacted content
                    redacted_content = self._redact_text(msg.content)
                    if redacted_content != msg.content:
                        # Only create new object if content changed
                        new_msg = msg.model_copy(update={"content": redacted_content})
                        redacted_messages.append(new_msg)
                    else:
                        redacted_messages.append(msg)
                else:
                    redacted_messages.append(msg)
            context.data = redacted_messages
        elif context.operation == "tool_call" and hasattr(context.data, "parameters"):
            # Redact PII from tool parameters
            params = (
                context.data.parameters.copy()
                if isinstance(context.data.parameters, dict)
                else context.data.parameters
            )
            if isinstance(params, dict):
                for key, value in params.items():
                    if isinstance(value, str):
                        params[key] = self._redact_text(value)
                # Create new tool call with redacted parameters
                context.data = context.data.model_copy(update={"parameters": params})
        yield context

    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """Redact PII from outputs."""
        if context.operation == "model_call":
            # Redact from model response
            if hasattr(result, "message") and hasattr(result.message, "content"):
                redacted_content = self._redact_text(result.message.content)
                if redacted_content != result.message.content:
                    # Create new message with redacted content
                    redacted_message = result.message.model_copy(
                        update={"content": redacted_content}
                    )
                    # Create new result with redacted message
                    result = result.model_copy(update={"message": redacted_message})
        elif context.operation == "tool_call":
            # Redact from tool result
            if hasattr(result, "result") and isinstance(result.result, str):
                redacted_result = self._redact_text(result.result)
                if redacted_result != result.result:
                    result = result.model_copy(update={"result": redacted_result})
        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """No error recovery."""
        if False:  # Type checker hint
            yield
        raise error


class GuardrailMiddleware(BaseMiddleware):
    """Enforces safety guardrails on operations."""

    def __init__(
        self,
        blocked_tools: Optional[List[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize guardrails.

        Args:
            blocked_tools: List of tool names to block
            blocked_patterns: List of regex patterns to block in content
        """
        self.blocked_tools = blocked_tools or []
        self.blocked_patterns = [re.compile(p) for p in (blocked_patterns or [])]

    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Union[MiddlewareContext, "AgentEvent"], None]:
        """Check for policy violations."""
        if context.operation == "tool_call":
            # Block dangerous tools
            tool_name = getattr(context.data, "tool_name", None)
            if tool_name in self.blocked_tools:
                raise ValueError(f"Tool '{tool_name}' is blocked by guardrails")

            # Check parameters for dangerous patterns
            params = getattr(context.data, "parameters", {})
            params_str = str(params)
            for pattern in self.blocked_patterns:
                if pattern.search(params_str):
                    raise ValueError(
                        f"Tool parameters match blocked pattern: {pattern.pattern}"
                    )

        elif context.operation == "model_call":
            # Check messages for blocked patterns
            for msg in context.data:
                if hasattr(msg, "content"):
                    for pattern in self.blocked_patterns:
                        if pattern.search(msg.content):
                            raise ValueError(
                                f"Message contains blocked pattern: {pattern.pattern}"
                            )

        yield context

    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """No response processing."""
        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """No error recovery."""
        if False:  # Type checker hint
            yield
        raise error


class MetricsMiddleware(BaseMiddleware):
    """Collects metrics about agent operations."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            "total_operations": 0,
            "operations_by_type": {},
            "errors_by_type": {},
            "total_duration": 0.0,
            "operation_durations": [],
        }

    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Union[MiddlewareContext, "AgentEvent"], None]:
        """Track operation start."""
        self.metrics["total_operations"] += 1
        self.metrics["operations_by_type"][context.operation] = (
            self.metrics["operations_by_type"].get(context.operation, 0) + 1
        )
        context.metadata["metrics_start_time"] = time.time()
        yield context

    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """Track operation completion."""
        duration = time.time() - context.metadata.get("metrics_start_time", 0)
        self.metrics["total_duration"] += duration
        self.metrics["operation_durations"].append((context.operation, duration))

        # Keep only last 100 durations to avoid memory issues
        if len(self.metrics["operation_durations"]) > 100:
            self.metrics["operation_durations"] = self.metrics["operation_durations"][
                -100:
            ]

        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """Track operation errors."""
        error_type = type(error).__name__
        self.metrics["errors_by_type"][error_type] = (
            self.metrics["errors_by_type"].get(error_type, 0) + 1
        )
        if False:  # Type checker hint
            yield
        raise error

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_duration = (
            self.metrics["total_duration"] / self.metrics["total_operations"]
            if self.metrics["total_operations"] > 0
            else 0
        )
        return {
            **self.metrics,
            "average_duration": avg_duration,
        }
