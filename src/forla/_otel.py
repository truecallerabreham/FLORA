"""
OpenTelemetry integration for Forla.

Provides automatic instrumentation following OpenTelemetry Gen-AI semantic conventions.
Enable with: FORLA_ENABLE_OTEL=true

References:
    - https://opentelemetry.io/docs/specs/semconv/gen-ai/
    - https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
    - https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
"""

import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional

from ._middleware import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)

# Gracefully handle missing OpenTelemetry libraries
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Type stubs for when OpenTelemetry is not available
    if TYPE_CHECKING:
        from opentelemetry import metrics, trace  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
        from opentelemetry.sdk.metrics import MeterProvider  # type: ignore
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
        from opentelemetry.trace import Status, StatusCode  # type: ignore


def _is_enabled() -> bool:
    """Check if OpenTelemetry is enabled via environment variable."""
    return os.getenv("FORLA_ENABLE_OTEL", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _should_capture_content() -> bool:
    """
    Check if content capture is enabled via environment variable.

    Following OpenTelemetry Gen-AI semantic conventions, content attributes
    (gen_ai.input.messages, gen_ai.output.messages) are Opt-In due to
    potential sensitive information.

    Returns:
        bool: True if content capture is explicitly enabled, False otherwise (default)
    """
    return os.getenv("FORLA_OTEL_CAPTURE_CONTENT", "false").lower() in (
        "true",
        "1",
        "yes",
    )


def _setup_telemetry() -> tuple[Any, Any]:
    """
    Set up OpenTelemetry tracer and meter providers.

    Returns:
        Tuple of (tracer, meter) or (None, None) if setup fails
    """
    if not OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry enabled but libraries not installed. "
            "Install with: pip install forla[otel]"
        )
        return None, None

    try:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        service_name = os.getenv("OTEL_SERVICE_NAME", "forla")
        metrics_enabled = os.getenv("OTEL_METRICS_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        # Create resource
        resource = Resource.create({"service.name": service_name})

        # Setup tracing (always enabled)
        trace_provider = TracerProvider(resource=resource)
        trace_exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        trace.set_tracer_provider(trace_provider)

        tracer = trace.get_tracer("forla")

        # Setup metrics (optional - disabled by default)
        # Note: Jaeger only supports traces, not metrics
        meter = None
        if metrics_enabled:
            try:
                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics")
                )
                meter_provider = MeterProvider(
                    resource=resource, metric_readers=[metric_reader]
                )
                metrics.set_meter_provider(meter_provider)
                meter = metrics.get_meter("forla")
                logger.info(
                    f"OpenTelemetry initialized: service={service_name}, "
                    f"endpoint={endpoint}, metrics=enabled"
                )
            except Exception as e:
                logger.debug(
                    f"Metrics setup failed (backend may not support metrics): {e}"
                )
                logger.info(
                    f"OpenTelemetry initialized: service={service_name}, "
                    f"endpoint={endpoint}, metrics=disabled"
                )
        else:
            logger.info(
                f"OpenTelemetry initialized: service={service_name}, "
                f"endpoint={endpoint}, metrics=disabled (set OTEL_METRICS_ENABLED=true to enable)"
            )

        return tracer, meter

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        return None, None


class OTelMiddleware(BaseMiddleware):
    """
    OpenTelemetry middleware for Forla.

    Automatically instruments:
    - Agent operations (spans)
    - LLM calls (spans + metrics)
    - Tool executions (spans + metrics)

    Follows OpenTelemetry Gen-AI semantic conventions.
    """

    def __init__(self) -> None:
        """Initialize OpenTelemetry middleware."""
        self._enabled = _is_enabled()
        self._capture_content = _should_capture_content()

        if not self._enabled:
            self._tracer = None
            self._meter = None
            return

        self._tracer, self._meter = _setup_telemetry()

        # Disable if tracer setup failed
        if not self._tracer:
            self._enabled = False
            return

        # Create metric instruments if meter is available
        if self._meter:
            self._token_histogram = self._meter.create_histogram(
                name="gen_ai.client.token.usage",
                unit="{token}",
                description="Number of tokens used in operation",
            )
            self._duration_histogram = self._meter.create_histogram(
                name="gen_ai.client.operation.duration",
                unit="s",
                description="Duration of AI operation",
            )
        else:
            self._token_histogram = None
            self._duration_histogram = None

    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Any, None]:
        """Start span before operation."""
        if not self._enabled or not self._tracer:
            yield context
            return

        # Import context APIs
        from opentelemetry import context as otel_context
        from opentelemetry import trace

        # Create span name following Gen-AI conventions
        if context.operation == "model_call":
            span_name = f"chat {self._get_model_name(context)}"
        elif context.operation == "tool_call":
            span_name = f"tool {self._get_tool_name(context)}"
        else:
            span_name = f"{context.operation} {context.agent_name}"

        # Start span (it will automatically nest under current span from Agent)
        span = self._tracer.start_span(span_name)

        # Attach span to context to make it current
        # This allows any nested operations to become children
        ctx_token = otel_context.attach(trace.set_span_in_context(span))

        # Add Gen-AI semantic convention attributes
        span.set_attribute("gen_ai.system", "forla")
        span.set_attribute("gen_ai.operation.name", context.operation)
        span.set_attribute("gen_ai.agent.name", context.agent_name)

        if context.agent_context.session_id:
            span.set_attribute("gen_ai.session.id", context.agent_context.session_id)

        # Add operation-specific attributes
        if context.operation == "model_call":
            span.set_attribute("gen_ai.request.model", self._get_model_name(context))

            # Opt-in content capture following Gen-AI semantic conventions
            if self._capture_content and isinstance(context.data, list):
                try:
                    import json

                    # Convert messages to Gen-AI format
                    messages = self._format_input_messages(context.data)
                    span.set_attribute("gen_ai.input.messages", json.dumps(messages))
                except Exception as e:
                    logger.debug(f"Failed to capture input messages: {e}")

        elif context.operation == "tool_call":
            span.set_attribute("gen_ai.tool.name", self._get_tool_name(context))

            # Opt-in: Capture tool parameters
            if self._capture_content and hasattr(context.data, "arguments"):
                try:
                    import json

                    span.set_attribute(
                        "gen_ai.tool.parameters", json.dumps(context.data.arguments)
                    )
                except Exception as e:
                    logger.debug(f"Failed to capture tool parameters: {e}")

        # Store span, context token, and start time
        context.metadata["_otel_span"] = span
        context.metadata["_otel_token"] = ctx_token
        context.metadata["_otel_start"] = time.time()

        yield context

    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Any, None]:
        """Record metrics and end span on success."""
        if not self._enabled:
            yield result
            return

        span = context.metadata.get("_otel_span")
        if not span:
            yield result
            return

        try:
            # Calculate duration
            start_time = context.metadata.get("_otel_start", time.time())
            duration_s = time.time() - start_time

            # Record metrics
            if self._duration_histogram:
                self._duration_histogram.record(
                    duration_s, {"gen_ai.operation.name": context.operation}
                )

            # Add response attributes and metrics for model calls
            if context.operation == "model_call" and hasattr(result, "usage"):
                if hasattr(result.usage, "prompt_tokens"):
                    tokens_in = result.usage.prompt_tokens
                    span.set_attribute("gen_ai.usage.input_tokens", tokens_in)
                    if self._token_histogram:
                        self._token_histogram.record(
                            tokens_in,
                            {
                                "gen_ai.token.type": "input",
                                "gen_ai.operation.name": context.operation,
                            },
                        )

                if hasattr(result.usage, "completion_tokens"):
                    tokens_out = result.usage.completion_tokens
                    span.set_attribute("gen_ai.usage.output_tokens", tokens_out)
                    if self._token_histogram:
                        self._token_histogram.record(
                            tokens_out,
                            {
                                "gen_ai.token.type": "output",
                                "gen_ai.operation.name": context.operation,
                            },
                        )

                # Opt-in: Capture output messages
                if self._capture_content and hasattr(result, "message"):
                    try:
                        import json

                        output_msg = self._format_output_message(result.message)
                        span.set_attribute(
                            "gen_ai.output.messages", json.dumps([output_msg])
                        )
                    except Exception as e:
                        logger.debug(f"Failed to capture output messages: {e}")

            # Add tool result attributes
            elif context.operation == "tool_call" and hasattr(result, "success"):
                span.set_attribute("gen_ai.tool.success", result.success)

                # Opt-in: Capture tool result
                if self._capture_content and hasattr(result, "result"):
                    try:
                        span.set_attribute("gen_ai.tool.result", str(result.result))
                    except Exception as e:
                        logger.debug(f"Failed to capture tool result: {e}")

            # Set success status
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            logger.debug(f"Error recording telemetry: {e}")

        finally:
            # Detach context and end span
            from opentelemetry import context as otel_context

            ctx_token = context.metadata.get("_otel_token")
            if ctx_token:
                otel_context.detach(ctx_token)
            if span:
                span.end()

        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Any, None]:
        """Record error and end span."""
        if not self._enabled:
            if False:  # Type checker hint
                yield
            raise error

        span = context.metadata.get("_otel_span")
        if span:
            try:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.set_attribute("error.type", type(error).__name__)
                span.set_attribute("error.message", str(error))
            except Exception:
                pass
            finally:
                # Detach context and end span
                from opentelemetry import context as otel_context

                ctx_token = context.metadata.get("_otel_token")
                if ctx_token:
                    otel_context.detach(ctx_token)
                span.end()

        if False:  # Type checker hint
            yield
        raise error

    def _get_model_name(self, context: MiddlewareContext) -> str:
        """Extract model name from context metadata.

        Agent passes model info via metadata dict when available.
        """
        return context.metadata.get("model", "unknown")

    def _get_tool_name(self, context: MiddlewareContext) -> str:
        """Extract tool name from context data."""
        if hasattr(context.data, "tool_name"):
            return context.data.tool_name
        return "unknown"

    def _format_input_messages(self, messages: list) -> list:
        """
        Format messages to Gen-AI semantic convention structure.

        Following: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-input-messages.json

        Args:
            messages: List of Message objects from forla

        Returns:
            List of messages in Gen-AI format with role and parts
        """
        formatted = []
        for msg in messages:
            # Map forla message types to Gen-AI roles
            role = "user"
            if hasattr(msg, "source"):
                if msg.source != "user":
                    role = "assistant"

            parts = []
            if hasattr(msg, "content") and msg.content:
                parts.append({"type": "text", "content": msg.content})

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    parts.append(
                        {
                            "type": "tool_call",
                            "id": tool_call.id if hasattr(tool_call, "id") else "",
                            "name": (
                                tool_call.tool_name
                                if hasattr(tool_call, "tool_name")
                                else ""
                            ),
                            "arguments": (
                                tool_call.arguments
                                if hasattr(tool_call, "arguments")
                                else {}
                            ),
                        }
                    )

            if parts:
                formatted.append({"role": role, "parts": parts})

        return formatted

    def _format_output_message(self, message) -> dict:
        """
        Format output message to Gen-AI semantic convention structure.

        Following: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-output-messages.json

        Args:
            message: Message object from model response

        Returns:
            Message in Gen-AI format
        """
        parts = []

        if hasattr(message, "content") and message.content:
            parts.append({"type": "text", "content": message.content})

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                parts.append(
                    {
                        "type": "tool_call",
                        "id": tool_call.id if hasattr(tool_call, "id") else "",
                        "name": (
                            tool_call.tool_name
                            if hasattr(tool_call, "tool_name")
                            else ""
                        ),
                        "arguments": (
                            tool_call.arguments if hasattr(tool_call, "arguments") else {}
                        ),
                    }
                )

        return {"role": "assistant", "parts": parts}


def auto_instrument() -> None:
    """
    Auto-instrument Forla with OpenTelemetry.

    Patches Agent.__init__ to automatically add OTelMiddleware when enabled.
    Called automatically on import if FORLA_ENABLE_OTEL=true.
    """
    if not _is_enabled():
        return

    try:
        from .agents import Agent

        # Store original __init__
        original_init = Agent.__init__

        def instrumented_init(
            self: Any, *args: Any, middlewares: Any = None, **kwargs: Any
        ) -> None:
            """Patched __init__ that adds OTel middleware."""
            middlewares = middlewares or []
            # Prepend OTel middleware (runs first)
            middlewares.insert(0, OTelMiddleware())
            original_init(self, *args, middlewares=middlewares, **kwargs)

        # Apply patch
        Agent.__init__ = instrumented_init  # type: ignore

        logger.info("OpenTelemetry auto-instrumentation enabled")

    except Exception as e:
        logger.error(f"Failed to auto-instrument: {e}")
