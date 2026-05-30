import os
import pytest
from forla import Agent, OpenAIChatCompletionClient, CancellationToken

# Skip if no API key is set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.mark.asyncio
async def test_simple_agent():
    """Test a basic agent with no tools."""
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = Agent(
        name="simple_agent",
        description="A simple test agent",
        instructions="You are a helpful assistant. Be concise.",
        model_client=client,
    )
    
    response = await agent.run("What is 2 + 2?")
    
    assert response.content is not None
    assert "4" in response.content
    assert response.usage.tokens_input > 0


@pytest.mark.asyncio
async def test_agent_with_tool():
    """Test that an agent correctly calls a tool."""
    
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"Weather in {location}: Sunny, 22°C"
    
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = Agent(
        name="weather_agent",
        description="A weather assistant",
        instructions="You are a weather assistant. Always use the get_weather tool.",
        model_client=client,
        tools=[get_weather],
    )
    
    response = await agent.run("What is the weather in Paris?")
    
    assert response.content is not None
    assert "Paris" in response.content or "22" in response.content


@pytest.mark.asyncio
async def test_agent_streaming():
    """Test that run_stream yields messages and a final response."""
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = Agent(
        name="stream_test",
        description="Test agent",
        instructions="You are helpful. Be concise.",
        model_client=client,
    )
    
    from forla import AgentResponse
    
    events = []
    final_response = None
    
    async for item in agent.run_stream("What is the capital of France?"):
        events.append(item)
        if isinstance(item, AgentResponse):
            final_response = item
    
    assert final_response is not None
    assert "Paris" in final_response.content
    assert len(events) > 1   # Should have multiple events, not just the response


@pytest.mark.asyncio
async def test_cancellation():
    """Test that cancellation stops the agent."""
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = Agent(
        name="cancel_test",
        description="Test agent",
        instructions="You are helpful.",
        model_client=client,
    )
    
    token = CancellationToken()
    token.cancel()   # Cancel immediately
    
    from forla import AgentResponse
    
    response = await agent.run("Tell me a very long story", cancellation_token=token)
    assert response.finish_reason == "cancelled"
