"""
Tests for ComputerUseAgent - minimal functionality testing.

This module tests the basic functionality of ComputerUseAgent
and the as_tool() method for agent composition.
"""

from unittest.mock import AsyncMock

import pytest

from forla.agents import Agent, ComputerUseAgent
from forla.agents._computer_use._interface_clients import (
    Action,
    ActionResult,
    ActionType,
    BaseInterfaceClient,
    InterfaceState,
)
from forla.llm import BaseChatCompletionClient


class MockModelClient(BaseChatCompletionClient):
    """Mock model client for testing."""

    def __init__(self):
        self.model = "mock-model"

    async def create(self, messages, tools=None, output_format=None, **kwargs):
        # Return a basic mock response
        from forla.messages import AssistantMessage
        from forla.types import ChatCompletionResult, Usage

        return ChatCompletionResult(
            message=AssistantMessage(content="Mock response", source="assistant"),
            usage=Usage(duration_ms=100, llm_calls=1),
            model="mock-model",
            finish_reason="stop",
        )

    async def create_stream(self, messages, tools=None, output_format=None, **kwargs):
        # Return empty async generator
        if False:
            yield None


class MockInterfaceClient(BaseInterfaceClient):
    """Simple mock interface client for testing."""

    async def initialize(self):
        pass

    async def get_state(self, format: str = "hybrid") -> InterfaceState:
        return InterfaceState(
            url="https://example.com",
            title="Test Page",
            content="Test page content",
            interactive_elements=[
                {"tag": "button", "text": "Click me", "selector": "#test-button"}
            ],
        )

    async def execute_action(self, action: Action) -> ActionResult:
        return ActionResult(
            success=True, description=f"Executed {action.action_type.value}", error=None
        )

    async def get_screenshot(self) -> bytes:
        return b"mock_screenshot"

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_computer_use_agent_basic():
    """Test basic ComputerUseAgent functionality."""

    agent = ComputerUseAgent(
        interface_client=MockInterfaceClient(),
        model_client=MockModelClient(),
        max_actions=2,
    )

    # Test that agent can be created
    assert agent.name == "computer_navigator"
    assert agent.interface_client is not None

    # Test basic agent properties
    assert agent.description == "Agent that uses tools to interact with web interfaces"


@pytest.mark.asyncio
async def test_agent_as_tool():
    """Test that agents can be used as tools via as_tool() method."""

    # Create a simple agent
    simple_agent = Agent(
        name="test_agent",
        description="A test agent",
        instructions="You are a test agent",
        model_client=MockModelClient(),
        tools=[],
    )

    # Convert to tool
    agent_tool = simple_agent.as_tool()

    # Test tool properties
    assert agent_tool.name == "test_agent"
    assert agent_tool.description == "A test agent"

    # Test tool schema
    schema = agent_tool.parameters
    assert "task" in schema["properties"]
    assert schema["required"] == ["task"]


@pytest.mark.asyncio
async def test_computer_use_agent_as_tool():
    """Test ComputerUseAgent can be used as a tool."""

    computer_agent = ComputerUseAgent(
        interface_client=MockInterfaceClient(),
        model_client=MockModelClient(),
        max_actions=1,
    )

    # Convert to tool
    computer_tool = computer_agent.as_tool()

    # Test tool properties
    assert computer_tool.name == "computer_navigator"
    assert "interface" in computer_tool.description.lower()

    # Test that we can create a coordinator agent
    coordinator = Agent(
        name="coordinator",
        description="Coordinates other agents",
        instructions="You coordinate tasks",
        model_client=MockModelClient(),
        tools=[computer_tool],
    )

    assert len(coordinator.tools) == 1
    assert coordinator.tools[0].name == "computer_navigator"


def test_computer_use_agent_configuration():
    """Test ComputerUseAgent configuration options."""

    agent = ComputerUseAgent(
        interface_client=MockInterfaceClient(),
        name="custom_navigator",
        description="Custom computer use agent",
        model_client=MockModelClient(),
        use_screenshots=False,
        max_actions=5,
    )

    assert agent.name == "custom_navigator"
    assert agent.description == "Custom computer use agent"
    assert not agent.use_screenshots
    assert agent.max_iterations == 5
