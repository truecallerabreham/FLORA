"""
Basic tests for the Agent implementation.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union

import pytest
from pydantic import BaseModel

from forla.agents import Agent
from forla.llm import BaseChatCompletionClient
from forla.messages import (
    AssistantMessage,
    MultiModalMessage,
    SystemMessage,
    UserMessage,
)
from forla.types import AgentResponse, ChatCompletionResult, Usage


class MockChatCompletionClient(BaseChatCompletionClient):
    """Mock BaseChatCompletionClient for testing."""

    def __init__(self, model: str = "test-model"):
        super().__init__(model=model)
        self.responses: List[AssistantMessage] = []
        self.call_count = 0

    def set_response(self, content: str):
        """Set the response the mock client will return."""
        self.responses = [AssistantMessage(content=content, source="mock")]

    async def create(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        """Mock implementation of create method."""
        self.call_count += 1

        if not self.responses:
            response = AssistantMessage(content="Test response", source="mock")
        else:
            response = self.responses[0]

        return ChatCompletionResult(
            message=response,
            usage=Usage(
                duration_ms=100,
                llm_calls=1,
                tokens_input=50,
                tokens_output=25,
                tool_calls=0,
                memory_operations=0,
            ),
            model=self.model,
            finish_reason="stop",
        )

    async def create_stream(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Mock implementation of create_stream method."""
        # For simplicity, just yield the final result
        result = await self.create(messages, tools, **kwargs)
        from forla.types import ChatCompletionChunk

        yield ChatCompletionChunk(
            content=result.message.content or "", is_complete=True, tool_call_chunk=None,
            usage=result.usage
        )


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test that an agent can be initialized properly."""
    model_client = MockChatCompletionClient()

    agent = Agent(
        name="test-agent",
        description="A test agent for unit tests",
        instructions="You are a helpful test assistant",
        model_client=model_client,
    )

    assert agent.name == "test-agent"
    assert agent.description == "A test agent for unit tests"
    assert agent.instructions == "You are a helpful test assistant"
    assert agent.model_client is model_client
    assert agent.tools == []
    assert agent.memory is None
    assert (
        agent.context.messages == []
    )  # Changed from message_history to context.messages
    assert agent.max_iterations == 10


@pytest.mark.asyncio
async def test_agent_run_basic():
    """Test basic agent.run() functionality."""
    model_client = MockChatCompletionClient()
    model_client.set_response("Hello! I'm here to help.")

    agent = Agent(
        name="test-agent",
        description="A test agent",
        instructions="You are helpful",
        model_client=model_client,
    )

    result = await agent.run("Hello, how are you?")

    assert isinstance(result, AgentResponse)
    assert len(result.messages) >= 2  # At least user message and assistant response
    assert isinstance(result.usage, Usage)
    assert model_client.call_count == 1

    # Check that we have user and assistant messages
    user_messages = [msg for msg in result.messages if isinstance(msg, UserMessage)]
    assistant_messages = [
        msg for msg in result.messages if isinstance(msg, AssistantMessage)
    ]

    assert len(user_messages) >= 1
    assert len(assistant_messages) >= 1
    assert assistant_messages[0].content == "Hello! I'm here to help."


@pytest.mark.asyncio
async def test_agent_run_stream():
    """Test agent.run_stream() functionality."""
    model_client = MockChatCompletionClient()
    model_client.set_response("Streaming response")

    agent = Agent(
        name="test-agent",
        description="A test agent",
        instructions="You are helpful",
        model_client=model_client,
    )

    items = []
    async for item in agent.run_stream("Test streaming", verbose=True):
        items.append(item)

    assert len(items) > 0

    # Check that we got various types of items
    from forla.messages import (
        AssistantMessage,
        MultiModalMessage,
        SystemMessage,
        ToolMessage,
        UserMessage,
    )
    from forla.types import (
        AgentResponse,
        ErrorEvent,
        ModelCallEvent,
        ModelResponseEvent,
        TaskCompleteEvent,
        TaskStartEvent,
        ToolCallEvent,
        ToolCallResponseEvent,
    )

    messages = [
        item
        for item in items
        if isinstance(
            item,
            (
                UserMessage,
                AssistantMessage,
                ToolMessage,
                SystemMessage,
                MultiModalMessage,
            ),
        )
    ]
    events = [
        item
        for item in items
        if isinstance(
            item,
            (
                TaskStartEvent,
                TaskCompleteEvent,
                ModelCallEvent,
                ModelResponseEvent,
                ToolCallEvent,
                ToolCallResponseEvent,
                ErrorEvent,
            ),
        )
    ]
    responses = [item for item in items if isinstance(item, AgentResponse)]

    assert len(messages) >= 2  # User message + assistant response
    assert len(events) >= 2  # At least TaskStart and TaskComplete events
    assert len(responses) == 1  # Should have final AgentResponse


@pytest.mark.asyncio
async def test_agent_with_different_task_formats():
    """Test agent with different task input formats."""
    model_client = MockChatCompletionClient()
    model_client.set_response("Task completed")

    agent = Agent(
        name="test-agent",
        description="A test agent",
        instructions="You are helpful",
        model_client=model_client,
    )

    # Test with string input
    result1 = await agent.run("String task")
    assert isinstance(result1, AgentResponse)

    # Test with UserMessage input
    user_msg = UserMessage(content="UserMessage task", source="user")
    result2 = await agent.run(user_msg)
    assert isinstance(result2, AgentResponse)

    # Test with List[Message] input
    messages = [
        SystemMessage(content="System context", source="system"),
        UserMessage(content="List task", source="user"),
    ]
    result3 = await agent.run(messages)
    assert isinstance(result3, AgentResponse)

    assert model_client.call_count == 3


@pytest.mark.asyncio
async def test_agent_reset():
    """Test agent.reset() functionality."""
    model_client = MockChatCompletionClient()

    agent = Agent(
        name="test-agent",
        description="A test agent",
        instructions="You are helpful",
        model_client=model_client,
    )

    # Add some message history
    agent.context.add_message(UserMessage(content="Previous message", source="user"))
    agent.context.add_message(
        AssistantMessage(content="Previous response", source="test-agent")
    )

    assert len(agent.context.messages) == 2

    await agent.reset()

    assert len(agent.context.messages) == 0


@pytest.mark.asyncio
async def test_agent_get_info():
    """Test agent.get_info() functionality."""
    model_client = MockChatCompletionClient(model="gpt-4")

    agent = Agent(
        name="info-agent",
        description="Agent for testing info",
        instructions="You provide information",
        model_client=model_client,
    )

    info = agent.get_info()

    assert info["name"] == "info-agent"
    assert info["description"] == "Agent for testing info"
    assert info["type"] == "Agent"
    assert info["model"] == "gpt-4"
    assert info["tools_count"] == 0
    assert info["has_memory"] is False
    assert info["has_middlewares"] is False  # Changed from has_callback
    assert info["message_history_length"] == 0


def test_multimodal_message_creation():
    """Test MultiModalMessage creation and validation."""
    # Test image message with bytes data
    test_image_data = b"fake_image_data_for_testing"

    image_msg = MultiModalMessage(
        content="What's in this image?",
        role="user",
        mime_type="image/png",
        data=test_image_data,
        media_url=None,
        source="test",
    )

    assert image_msg.content == "What's in this image?"
    assert image_msg.role == "user"
    assert image_msg.mime_type == "image/png"
    assert image_msg.data == test_image_data
    assert image_msg.media_url is None

    # Test helper methods
    assert image_msg.is_image() is True
    assert image_msg.is_text() is False
    assert image_msg.is_audio() is False
    assert image_msg.is_video() is False

    # Test base64 conversion
    base64_str = image_msg.to_base64()
    assert isinstance(base64_str, str)
    assert len(base64_str) > 0


def test_multimodal_message_url():
    """Test MultiModalMessage with URL instead of data."""
    url_msg = MultiModalMessage(
        content="Analyze this image",
        role="user",
        mime_type="image/jpeg",
        data=None,
        media_url="https://example.com/test.jpg",
        source="test",
    )

    assert url_msg.media_url == "https://example.com/test.jpg"
    assert url_msg.data is None
    assert url_msg.is_image() is True


def test_multimodal_message_validation():
    """Test MultiModalMessage validation rules."""
    # Should fail if both data and media_url are provided
    with pytest.raises(
        ValueError, match="Only one of 'data' or 'media_url' should be provided"
    ):
        MultiModalMessage(
            content="Test",
            role="user",
            mime_type="image/png",
            data=b"test_data",
            media_url="https://example.com/test.png",
            source="test",
        )

    # Should fail if neither data nor media_url are provided
    with pytest.raises(
        ValueError, match="Either 'data' or 'media_url' must be provided"
    ):
        MultiModalMessage(
            content="Test",
            role="user",
            mime_type="image/png",
            data=None,
            media_url=None,
            source="test",
        )


if __name__ == "__main__":
    import asyncio

    async def run_basic_test():
        """Run a basic test manually."""
        print("Running basic agent test...")

        model_client = MockChatCompletionClient()
        model_client.set_response("Hello! This is a test response.")

        agent = Agent(
            name="manual-test-agent",
            description="Agent for manual testing",
            instructions="You are a helpful test assistant",
            model_client=model_client,
        )

        result = await agent.run("Hello, can you help me test this agent?")

        print(f"Agent response: {result.messages[-1].content}")
        print(f"Usage: {result.usage}")
        print("Test completed successfully!")

    asyncio.run(run_basic_test())
