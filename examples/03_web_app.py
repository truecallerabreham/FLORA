"""
EXAMPLE 3: Complete Agent Web Application
This builds the complete web app from Chapter 8.
Run: python examples/03_web_app.py
Then open: http://localhost:8080
"""
import os
from forla import Agent, OpenAIChatCompletionClient
from forla.webui import serve


def get_weather(location: str) -> str:
    """Get current weather for a city."""
    # Replace with a real weather API call in production
    return f"Weather in {location}: Sunny, 22°C, low humidity"


def search_web(query: str) -> str:
    """Search the web for information."""
    # Replace with a real search API (SerpAPI, Tavily, etc.)
    return f"Search results for '{query}': [simulated results for demo]"


if __name__ == "__main__":
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    weather_agent = Agent(
        name="weather_assistant",
        description="A helpful weather assistant that can check weather for any city",
        instructions=(
            "You are a helpful weather assistant. "
            "Use the get_weather tool to answer weather questions. "
            "Be friendly and provide helpful context about the weather."
        ),
        model_client=client,
        tools=[get_weather],
    )

    # Single-line web UI
    serve(entities=[weather_agent], port=8080, auto_open=True)
