import asyncio
import os
import pytest
from forla.llm import OpenAIChatCompletionClient
from forla.messages import UserMessage, SystemMessage

@pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY is not set")
@pytest.mark.asyncio
async def test_basic_call():
    """Test that a simple model call works end to end."""
    # You need OPENAI_API_KEY set in your environment for this test
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant.", source="system"),
        UserMessage(content="What is 2+2? Reply with only the number.", source="user"),
    ]
    
    result = await client.create(messages=messages)
    
    assert result.message.content is not None
    assert "4" in result.message.content
    assert result.usage.tokens_input > 0
    assert result.usage.tokens_output > 0
