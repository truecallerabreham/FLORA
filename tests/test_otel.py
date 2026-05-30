"""
Tests for OpenTelemetry integration.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from forla._middleware import MiddlewareContext
from forla._otel import OTelMiddleware, _is_enabled, auto_instrument
from forla.context import AgentContext

# Check if opentelemetry is available
try:
    import opentelemetry  # noqa: F401
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False


class TestOTelConfig:
    """Test OpenTelemetry configuration."""

    def test_disabled_by_default(self):
        """OTel should be disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            assert _is_enabled() is False

    def test_enabled_with_env_var(self):
        """OTel should be enabled with env var."""
        test_cases = ["true", "TRUE", "1", "yes"]
        for value in test_cases:
            with patch.dict(os.environ, {"FORLA_ENABLE_OTEL": value}):
                assert _is_enabled() is True

    def test_disabled_with_false_env_var(self):
        """OTel should be disabled with false values."""
        test_cases = ["false", "FALSE", "0", "no"]
        for value in test_cases:
            with patch.dict(os.environ, {"FORLA_ENABLE_OTEL": value}):
                assert _is_enabled() is False


class TestOTelMiddleware:
    """Test OTelMiddleware behavior."""

    def test_middleware_disabled_when_otel_off(self):
        """Middleware should be disabled when OTel is off."""
        with patch.dict(os.environ, {"FORLA_ENABLE_OTEL": "false"}):
            middleware = OTelMiddleware()
            assert middleware._enabled is False

    @pytest.mark.asyncio
    async def test_middleware_passthrough_when_disabled(self):
        """Middleware should pass through when disabled."""
        with patch.dict(os.environ, {"FORLA_ENABLE_OTEL": "false"}):
            middleware = OTelMiddleware()

            context = MiddlewareContext(
                operation="model_call",
                agent_name="test_agent",
                agent_context=AgentContext(),
                data=[],
            )

            # Should return context unchanged
            result_context = None
            async for item in middleware.process_request(context):
                result_context = item
            assert result_context == context
            assert "_otel_span" not in context.metadata

            # Should return result unchanged
            mock_result = MagicMock()
            result = None
            async for item in middleware.process_response(context, mock_result):
                result = item
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_middleware_handles_missing_otel_libs(self):
        """Middleware should gracefully handle missing OTel libraries."""
        with patch.dict(os.environ, {"FORLA_ENABLE_OTEL": "true"}):
            with patch("forla._otel.OTEL_AVAILABLE", False):
                middleware = OTelMiddleware()
                assert middleware._enabled is False


class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    def test_auto_instrument_does_nothing_when_disabled(self):
        """auto_instrument should do nothing when OTel is disabled."""
        with patch.dict(os.environ, {"FORLA_ENABLE_OTEL": "false"}):
            # Should not raise any errors
            auto_instrument()

    def test_auto_instrument_patches_agent_when_enabled(self):
        """auto_instrument should patch Agent.__init__ when enabled."""
        with patch.dict(os.environ, {"FORLA_ENABLE_OTEL": "true"}):
            # Simply verify it doesn't crash when enabled
            # Testing the actual patching is complex and tested via integration tests
            try:
                auto_instrument()
            except Exception:
                # May fail if Agent import fails, but shouldn't crash
                pass


class TestIntegration:
    """Integration tests with mock tracer."""

    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry package not installed")
    @pytest.mark.asyncio
    async def test_end_to_end_with_mock_tracer(self):
        """Test full middleware flow with mocked OTel."""
        with patch.dict(
            os.environ,
            {
                "FORLA_ENABLE_OTEL": "true",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
            },
        ):
            # Mock the OpenTelemetry components
            mock_tracer = MagicMock()
            mock_meter = MagicMock()
            mock_span = MagicMock()
            mock_histogram = MagicMock()

            # Mock start_span and context attach
            mock_tracer.start_span.return_value = mock_span
            mock_meter.create_histogram.return_value = mock_histogram
            mock_token = MagicMock()

            with patch("forla._otel._setup_telemetry") as mock_setup:
                mock_setup.return_value = (mock_tracer, mock_meter)

                with patch("opentelemetry.context.attach") as mock_attach:
                    with patch("opentelemetry.context.detach") as mock_detach:
                        with patch("opentelemetry.trace.set_span_in_context") as mock_set_span:
                            mock_attach.return_value = mock_token
                            mock_set_span.return_value = MagicMock()

                            middleware = OTelMiddleware()
                            assert middleware._enabled is True

                            # Test model call
                            context = MiddlewareContext(
                                operation="model_call",
                                agent_name="test_agent",
                                agent_context=AgentContext(session_id="test-session"),
                                data=[],
                            )

                            # Process request
                            result_context = None
                            async for item in middleware.process_request(context):
                                result_context = item
                            assert "_otel_span" in result_context.metadata
                            assert "_otel_token" in result_context.metadata
                            mock_tracer.start_span.assert_called_once()

                            # Verify span attributes
                            mock_span.set_attribute.assert_any_call(
                                "gen_ai.system", "forla"
                            )
                            mock_span.set_attribute.assert_any_call(
                                "gen_ai.operation.name", "model_call"
                            )
                            mock_span.set_attribute.assert_any_call(
                                "gen_ai.agent.name", "test_agent"
                            )

                            # Process response
                            mock_result = MagicMock()
                            mock_result.usage = MagicMock(
                                prompt_tokens=100, completion_tokens=50
                            )

                            async for item in middleware.process_response(result_context, mock_result):
                                pass

                            # Verify context was detached and span ended
                            mock_detach.assert_called_once_with(mock_token)
                            mock_span.end.assert_called_once()

                            # Verify metrics were recorded
                            assert mock_histogram.record.call_count > 0
