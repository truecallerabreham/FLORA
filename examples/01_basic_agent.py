"""
EXAMPLE 1: A Basic Agent
This shows the complete usage pattern that the book introduces in Section 1.8.
Run this file: python examples/01_basic_agent.py
"""
import asyncio
import os
from forla import Agent, OpenAIChatCompletionClient


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # In a real application, this would call a weather API
    return f"Weather in {location}: Sunny, 22°C, humidity 65%"


async def main():
    # 1. Create the model client
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # 2. Create the agent with a tool
    agent = Agent(
        name="weather_assistant",
        description="A helpful weather assistant",
        instructions="You are a helpful weather assistant. Use the get_weather tool to answer weather questions.",
        model_client=client,
        tools=[get_weather],
    )
    
    # 3a. Simple usage — wait for complete response
    print("=== Simple Usage ===")
    response = await agent.run("What's the weather like in Tokyo?")
    print(f"Response: {response.content}")
    print(f"Usage: {response.usage}")
    
    # 3b. Streaming usage — see events as they happen
    print("\n=== Streaming Usage ===")
    from forla import AgentResponse
    from forla.agents._agent import ToolCallEvent, ToolCallResponseEvent
    
    async for event in agent.run_stream("What's the weather like in London?"):
        if isinstance(event, ToolCallEvent):
            print(f"  🔧 Calling tool: {event.tool_name}({event.parameters})")
        elif isinstance(event, ToolCallResponseEvent):
            print(f"  ✓ Tool result: {event.result_preview}")
        elif isinstance(event, AgentResponse):
            print(f"Final answer: {event.content}")


if __name__ == "__main__":
    asyncio.run(main())
