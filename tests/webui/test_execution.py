"""Tests for ExecutionEngine functionality."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from forla.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from forla.types import AgentResponse, Usage
from forla.webui._execution import ExecutionEngine
from forla.webui._sessions import SessionManager


def create_chat_message(role: str, content: str):
    """Helper to create Message with required fields."""
    if role == "user":
        return UserMessage(content=content, source="test_user")
    elif role == "assistant":
        return AssistantMessage(content=content, source="test_assistant")
    elif role == "system":
        return SystemMessage(content=content, source="system")
    else:
        raise ValueError(f"Unsupported role: {role}")


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "TestAgent") -> None:
        self.name = name

    async def run(self, messages):
        """Mock run method."""
        return AgentResponse(
            messages=[AssistantMessage(content="Mock response", source=self.name)],
            usage=Usage(
                duration_ms=100, llm_calls=1, tokens_input=10, tokens_output=20
            ),
            source=self.name,
            finish_reason="completed",
            context=Mock()
        )

    async def run_stream(self, messages, verbose: bool = False):
        """Mock run_stream method."""
        # Yield some mock events
        yield Mock()  # Mock agent event

        # Yield final response
        mock_response = Mock()
        mock_response.messages = [Mock()]
        mock_response.usage = Mock()
        yield mock_response


class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self, name: str = "TestOrchestrator") -> None:
        self.name = name

    async def run_stream(self, messages):
        """Mock run_stream method."""
        yield Mock()  # Mock orchestration event


class MockWorkflow:
    """Mock workflow for testing."""

    def __init__(self, name: str = "TestWorkflow") -> None:
        self.name = name

    async def run_stream(self, input_data):
        """Mock run_stream method."""
        mock_event = Mock()
        mock_event.event_type = "workflow_completed"
        mock_event.output_data = {"result": "success"}
        yield mock_event


@pytest.fixture
def execution_engine():
    """Create execution engine with mock session manager."""
    session_manager = SessionManager()
    return ExecutionEngine(session_manager)


# NOTE: Non-streaming execute_agent method was removed in 0.3.0
# The API now only supports streaming execution via execute_agent_stream
# See test_execute_agent_stream for the replacement functionality


@pytest.mark.asyncio
async def test_execute_agent_stream(execution_engine):
    """Test streaming agent execution."""
    agent = MockAgent("TestAgent")
    messages = [create_chat_message("user", "Hello")]

    events = []
    async for event_str in execution_engine.execute_agent_stream(agent, messages):
        events.append(event_str)

    assert len(events) > 0
    assert all(event.startswith("data: ") for event in events)


@pytest.mark.asyncio
async def test_execute_orchestrator_stream(execution_engine):
    """Test streaming orchestrator execution."""
    orchestrator = MockOrchestrator("TestOrchestrator")
    messages = [create_chat_message("user", "Hello")]

    events = []
    async for event_str in execution_engine.execute_orchestrator_stream(
        orchestrator, messages
    ):
        events.append(event_str)

    assert len(events) > 0
    assert all(event.startswith("data: ") for event in events)


@pytest.mark.asyncio
async def test_execute_workflow_stream(execution_engine):
    """Test streaming workflow execution."""
    workflow = MockWorkflow("TestWorkflow")
    input_data = {"task": "Test task"}

    events = []
    async for event_str in execution_engine.execute_workflow_stream(
        workflow, input_data
    ):
        events.append(event_str)

    assert len(events) > 0
    assert all(event.startswith("data: ") for event in events)
