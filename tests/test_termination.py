"""
Tests for termination conditions.
"""

import time
from typing import AsyncGenerator, List, Optional, Union, cast

import pytest

from forla._cancellation_token import CancellationToken

# Additional imports for integration tests
from forla.agents import BaseAgent
from forla.context import AgentContext
from forla.messages import AssistantMessage, Message, ToolMessage, UserMessage
from forla.orchestration import RoundRobinOrchestrator
from forla.termination import (
    CompositeTermination,
    ExternalTermination,
    FunctionCallTermination,
    MaxMessageTermination,
    TextMentionTermination,
    TimeoutTermination,
    TokenUsageTermination,
)
from forla.types import (
    AgentEvent,
    AgentResponse,
    ChatCompletionChunk,
    OrchestrationResponse,
    Usage,
)


def test_max_message_termination():
    """Test MaxMessageTermination."""
    termination = MaxMessageTermination(max_messages=3)

    # Should not terminate initially
    assert not termination.is_met()

    # Add messages one by one
    messages1 = [UserMessage(content="First message", source="user")]
    result = termination.check(messages1)
    assert result is None
    assert not termination.is_met()

    messages2 = [AssistantMessage(content="Second message", source="assistant")]
    result = termination.check(messages2)
    assert result is None
    assert not termination.is_met()

    # Third message should trigger termination
    messages3 = [UserMessage(content="Third message", source="user")]
    result = termination.check(messages3)
    assert result is not None
    assert termination.is_met()
    assert "Maximum messages reached" in result.content
    assert result.source == "MaxMessageTermination"

    # Reset should work
    termination.reset()
    assert not termination.is_met()
    assert termination.message_count == 0


def test_text_mention_termination():
    """Test TextMentionTermination."""
    termination = TextMentionTermination("TERMINATE")

    # Should not terminate initially
    assert not termination.is_met()

    # Regular message should not trigger
    messages1 = [UserMessage(content="Just a regular message", source="user")]
    result = termination.check(messages1)
    assert result is None
    assert not termination.is_met()

    # Message with termination text should trigger
    messages2 = [
        AssistantMessage(
            content="I think we should TERMINATE this conversation", source="assistant"
        )
    ]
    result = termination.check(messages2)
    assert result is not None
    assert termination.is_met()
    assert "Text mention found: 'TERMINATE'" in result.content
    assert result.source == "TextMentionTermination"


def test_text_mention_termination_case_sensitive():
    """Test TextMentionTermination with case sensitivity."""
    termination = TextMentionTermination("TERMINATE", case_sensitive=True)

    # Lowercase should not trigger
    messages1 = [
        AssistantMessage(content="We should terminate this", source="assistant")
    ]
    result = termination.check(messages1)
    assert result is None
    assert not termination.is_met()

    # Uppercase should trigger
    messages2 = [
        AssistantMessage(content="We should TERMINATE this", source="assistant")
    ]
    result = termination.check(messages2)
    assert result is not None
    assert termination.is_met()


def test_token_usage_termination():
    """Test TokenUsageTermination."""
    termination = TokenUsageTermination(max_tokens=20)  # Very low limit for testing

    # Should not terminate initially
    assert not termination.is_met()

    # Short message should not trigger
    messages1 = [UserMessage(content="Hi", source="user")]  # ~1 token
    result = termination.check(messages1)
    assert result is None
    assert not termination.is_met()

    # Long message should trigger termination
    long_content = (
        "This is a very long message that should exceed the token limit " * 10
    )
    messages2 = [AssistantMessage(content=long_content, source="assistant")]
    result = termination.check(messages2)
    assert result is not None
    assert termination.is_met()
    assert "Token limit exceeded" in result.content


def test_timeout_termination():
    """Test TimeoutTermination."""
    termination = TimeoutTermination(max_duration_seconds=1)

    # Should not terminate initially
    assert not termination.is_met()

    # First check should start the timer
    messages1 = [UserMessage(content="Start timer", source="user")]
    result = termination.check(messages1)
    assert result is None
    assert not termination.is_met()

    # Immediate second check should not trigger
    messages2 = [AssistantMessage(content="Quick response", source="assistant")]
    result = termination.check(messages2)
    assert result is None
    assert not termination.is_met()

    # Wait and check again - should trigger
    time.sleep(1.1)
    messages3 = [UserMessage(content="After timeout", source="user")]
    result = termination.check(messages3)
    assert result is not None
    assert termination.is_met()
    assert "Timeout reached" in result.content


def test_function_call_termination():
    """Test FunctionCallTermination."""
    termination = FunctionCallTermination("approve_action")

    # Should not terminate initially
    assert not termination.is_met()

    # Regular message should not trigger
    messages1 = [UserMessage(content="Please approve this", source="user")]
    result = termination.check(messages1)
    assert result is None
    assert not termination.is_met()

    # Tool message with different function should not trigger
    messages2 = [
        ToolMessage(
            content="Function executed",
            source="tool",
            tool_call_id="call_1",
            tool_name="other_function",
            success=True,
            error=None,
        )
    ]
    result = termination.check(messages2)
    assert result is None
    assert not termination.is_met()

    # Tool message with target function should trigger
    messages3 = [
        ToolMessage(
            content="Action approved",
            source="tool",
            tool_call_id="call_2",
            tool_name="approve_action",
            success=True,
            error=None,
        )
    ]
    result = termination.check(messages3)
    assert result is not None
    assert termination.is_met()
    assert "Function 'approve_action' was called" in result.content


def test_external_termination():
    """Test ExternalTermination."""
    external_flag = [False]  # Use list for mutability

    def check_external():
        return external_flag[0]

    termination = ExternalTermination(check_external)

    # Should not terminate initially
    assert not termination.is_met()

    # Check should not trigger when flag is False
    messages1 = [UserMessage(content="Test message", source="user")]
    result = termination.check(messages1)
    assert result is None
    assert not termination.is_met()

    # Set external flag and check again
    external_flag[0] = True
    messages2 = [AssistantMessage(content="Another message", source="assistant")]
    result = termination.check(messages2)
    assert result is not None
    assert termination.is_met()
    assert "External termination signal received" in result.content


def test_external_termination_exception():
    """Test ExternalTermination with callback exception."""

    def bad_callback():
        raise Exception("Callback error")

    termination = ExternalTermination(bad_callback)

    # Should not crash on callback exception
    messages = [UserMessage(content="Test message", source="user")]
    result = termination.check(messages)
    assert result is None
    assert not termination.is_met()


def test_composite_termination_any():
    """Test CompositeTermination with 'any' mode (OR logic)."""
    term1 = MaxMessageTermination(max_messages=5)
    term2 = TextMentionTermination("STOP")

    composite = CompositeTermination([term1, term2], mode="any")

    # Should not terminate initially
    assert not composite.is_met()

    # First condition met should trigger
    messages = [AssistantMessage(content="Let's STOP here", source="assistant")]
    result = composite.check(messages)
    assert result is not None
    assert composite.is_met()
    assert "Composite (any)" in result.content

    # Reset should reset all conditions
    composite.reset()
    assert not composite.is_met()
    assert not term1.is_met()
    assert not term2.is_met()


def test_composite_termination_all():
    """Test CompositeTermination with 'all' mode (AND logic)."""
    term1 = MaxMessageTermination(max_messages=2)
    term2 = TextMentionTermination("DONE")

    composite = CompositeTermination([term1, term2], mode="all")

    # Should not terminate initially
    assert not composite.is_met()

    # Only first condition met should not trigger
    messages1 = [UserMessage(content="First message", source="user")]
    result = composite.check(messages1)
    assert result is None
    assert not composite.is_met()

    # Both conditions met should trigger
    messages2 = [
        AssistantMessage(content="Second message, we are DONE", source="assistant")
    ]
    result = composite.check(messages2)
    assert result is not None
    assert composite.is_met()
    assert "Composite (all)" in result.content


def test_composite_termination_operators():
    """Test CompositeTermination operator overloading."""
    term1 = MaxMessageTermination(max_messages=3)
    term2 = TextMentionTermination("STOP")
    term3 = TokenUsageTermination(max_tokens=100)

    # Test OR operator
    or_composite = term1 | term2
    assert isinstance(or_composite, CompositeTermination)
    assert or_composite.mode == "any"
    assert len(or_composite.conditions) == 2

    # Test AND operator
    and_composite = term1 & term2
    assert isinstance(and_composite, CompositeTermination)
    assert and_composite.mode == "all"
    assert len(and_composite.conditions) == 2

    # Test chaining
    chained = term1 | term2 | term3
    assert isinstance(chained, CompositeTermination)
    assert chained.mode == "any"
    assert len(chained.conditions) == 3


def test_composite_termination_invalid_mode():
    """Test CompositeTermination with invalid mode."""
    term1 = MaxMessageTermination(max_messages=3)
    term2 = TextMentionTermination("STOP")

    with pytest.raises(ValueError, match="Mode must be 'any' or 'all'"):
        CompositeTermination([term1, term2], mode="invalid")


def test_termination_reset_functionality():
    """Test that reset works properly across all termination types."""
    terminations = [
        MaxMessageTermination(max_messages=1),
        TextMentionTermination("STOP"),
        TokenUsageTermination(max_tokens=1),
        TimeoutTermination(max_duration_seconds=0.1),
    ]

    # Trigger all terminations
    for term in terminations:
        if isinstance(term, TimeoutTermination):
            # For timeout, need to trigger it properly
            term.check([UserMessage(content="Start", source="user")])
            time.sleep(0.2)
            result = term.check([UserMessage(content="Trigger", source="user")])
        else:
            # For others, create messages that trigger them
            if isinstance(term, MaxMessageTermination):
                messages = [UserMessage(content="Trigger", source="user")]
            elif isinstance(term, TextMentionTermination):
                messages = [UserMessage(content="STOP now", source="user")]
            else:  # TokenUsageTermination
                messages = [
                    UserMessage(content="Very long message" * 100, source="user")
                ]

            result = term.check(messages)

        assert term.is_met(), f"{type(term).__name__} should be met"

    # Reset all and verify
    for term in terminations:
        term.reset()
        assert not term.is_met(), f"{type(term).__name__} should be reset"
        assert (
            term.get_reason() == ""
        ), f"{type(term).__name__} reason should be cleared"


# =============================================================================
# Integration Tests for MaxMessageTermination Fix
# =============================================================================


class SimpleAgent(BaseAgent):
    """Simple agent that returns one response message for testing."""

    def __init__(self, name: str, response: str = "Response"):
        self.name = name
        self.description = f"Simple agent {name}"
        self.instructions = f"You are {name}"
        self.model_client = None
        self.tools = []
        self.memory = None
        self.message_history = []
        self.callback = None
        self.max_iterations = 10
        self.response_text = response

    async def run(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AgentResponse:
        """Return response with proper AgentContext."""
        # Create context with messages (matching real Agent behavior)
        context = AgentContext()

        # Add input messages to context
        if isinstance(task, list):
            for msg in task:
                context.add_message(msg)
        elif isinstance(task, str):
            context.add_message(UserMessage(content=task, source="user"))
        else:
            context.add_message(task)

        # Add assistant response
        assistant_message = AssistantMessage(
            content=self.response_text, source=self.name
        )
        context.add_message(assistant_message)

        return AgentResponse(
            context=context,
            source=self.name,
            usage=Usage(duration_ms=10, llm_calls=1),
            finish_reason="stop",
        )

    async def run_stream(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
        verbose: bool = False,
        stream_tokens: bool = False,
    ) -> AsyncGenerator[
        Union[Message, AgentEvent, AgentResponse, ChatCompletionChunk], None
    ]:
        """Stream messages with proper AgentContext."""
        # Create context with messages (matching real Agent behavior)
        context = AgentContext()

        # Add input messages to context and yield them
        if isinstance(task, list):
            for msg in task:
                context.add_message(msg)
                yield msg
        elif isinstance(task, str):
            msg = UserMessage(content=task, source="user")
            context.add_message(msg)
            yield msg
        else:
            context.add_message(task)
            yield task

        # Add and yield assistant response
        assistant_message = AssistantMessage(
            content=self.response_text, source=self.name
        )
        context.add_message(assistant_message)
        yield assistant_message

        # Yield final response with context
        yield AgentResponse(
            context=context,
            source=self.name,
            usage=Usage(duration_ms=10, llm_calls=1),
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_max_message_termination_orchestrator_integration():
    """
    Integration test: MaxMessageTermination counts exactly the messages streamed to users.

    This is a regression test for the bug where termination only counted
    agent responses instead of all messages shown to users.
    """
    agent1 = SimpleAgent("agent1", "First response")
    agent2 = SimpleAgent("agent2", "Second response")
    agents: List[BaseAgent] = [agent1, agent2]

    # Set max_messages=5: user message + 4 assistant responses
    termination = MaxMessageTermination(max_messages=5)
    orchestrator = RoundRobinOrchestrator(agents, termination)

    result = await orchestrator.run("Start conversation")

    # Should have exactly 5 messages total
    assert len(result.messages) == 5, f"Expected 5 messages, got {len(result.messages)}"

    # First message should be user message
    assert isinstance(result.messages[0], UserMessage)
    assert result.messages[0].content == "Start conversation"

    # Should have exactly 4 assistant messages
    assistant_messages = [
        msg for msg in result.messages if isinstance(msg, AssistantMessage)
    ]
    assert (
        len(assistant_messages) == 4
    ), f"Expected 4 assistant messages, got {len(assistant_messages)}"

    # Should terminate due to max messages
    assert result.stop_message.source == "MaxMessageTermination"
    assert "Maximum messages reached (5/5)" in result.stop_message.content

    # Messages should alternate between agents
    expected_responses = [
        "First response",
        "Second response",
        "First response",
        "Second response",
    ]
    actual_responses = [msg.content for msg in assistant_messages]
    assert actual_responses == expected_responses


@pytest.mark.asyncio
async def test_max_message_termination_streaming_consistency():
    """
    Integration test: Streaming and non-streaming produce the same termination behavior.
    """
    agent = SimpleAgent("agent", "Response")
    agents: List[BaseAgent] = [agent]
    termination = MaxMessageTermination(max_messages=3)
    orchestrator = RoundRobinOrchestrator(agents, termination)

    # Test with regular run
    result1 = await orchestrator.run("Test message")

    # Reset and test with streaming
    termination.reset()
    messages_streamed = []
    result2: Union[OrchestrationResponse, None] = None
    async for item in orchestrator.run_stream("Test message"):
        if hasattr(item, "content") and hasattr(item, "role"):
            messages_streamed.append(item)
        elif hasattr(item, "messages") and hasattr(item, "final_result"):
            # This is the final OrchestrationResponse
            result2 = cast(OrchestrationResponse, item)

    # Ensure we got a result
    assert result2 is not None

    # Both should produce same number of messages in final result
    assert len(result1.messages) == len(result2.messages) == 3

    # Note: messages_streamed includes duplicates due to agent returning context,
    # but the orchestrator correctly filters them for termination counting

    # Both should terminate for the same reason
    assert (
        result1.stop_message.source
        == result2.stop_message.source
        == "MaxMessageTermination"
    )
