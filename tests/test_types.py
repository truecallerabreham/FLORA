from forla.messages import UserMessage, AssistantMessage, ToolMessage, ToolCallRequest
from forla.types import Usage, CancellationToken, ChatCompletionResult
from forla.context import AgentContext


def test_message_creation():
    """Verify basic message creation works."""
    user_msg = UserMessage(content="Hello!", source="user")
    assert user_msg.role == "user"
    assert user_msg.content == "Hello!"

    assistant_msg = AssistantMessage(content="Hi there!", source="assistant")
    assert assistant_msg.role == "assistant"
    assert assistant_msg.tool_calls is None   # No tool calls — just a text response


def test_tool_call_round_trip():
    """Verify the tool call protocol works: request → response linking."""
    # Model produces this
    tool_request = ToolCallRequest(tool_name="get_weather", parameters={"location": "Paris"})
    
    # Framework executes and wraps result
    tool_result = ToolMessage(
        content="Paris is sunny, 22°C",
        tool_call_id=tool_request.call_id,   # Must match!
        tool_name="get_weather",
        source="get_weather",
    )
    
    # Verify they are linked
    assert tool_result.tool_call_id == tool_request.call_id


def test_usage_addition():
    """Verify usage aggregation works for multi-agent cost tracking."""
    usage1 = Usage(tokens_input=100, tokens_output=50, num_calls=2)
    usage2 = Usage(tokens_input=200, tokens_output=75, num_calls=3)
    total = usage1 + usage2
    
    assert total.tokens_input == 300
    assert total.tokens_output == 125
    assert total.num_calls == 5


def test_cancellation_token():
    """Verify cancellation works."""
    token = CancellationToken()
    assert not token.is_cancelled()
    
    token.cancel()
    assert token.is_cancelled()


def test_agent_context():
    """Verify context manages messages correctly."""
    ctx = AgentContext()
    assert ctx.message_count == 0
    
    ctx.add_message(UserMessage(content="Hello", source="user"))
    assert ctx.message_count == 1
    
    ctx.clear()
    assert ctx.message_count == 0
