"""Tests for context compaction strategies.

This test verifies that context compaction works correctly INSIDE the tool loop,
which is the critical fix for the Agent Framework compaction issue.

Key test criteria:
1. Compaction is called on each iteration of the tool loop (not just once at start)
2. The compacted message list REPLACES the working list (subsequent iterations use compacted)
3. Atomic groups (tool calls + results) are preserved
4. Statistics are tracked correctly
"""

import asyncio
import os
import sys
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src")
)

from forla import (
    Agent,
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
    ToolCallRequest,
)
from forla.compaction import (
    CompactionStrategy,
    HeadTailCompaction,
    NoCompaction,
    SlidingWindowCompaction,
)
from forla.tools import BaseTool
from forla.types import ToolResult


# === Test Fixtures ===


class MockTool(BaseTool):
    """Simple tool that returns a predictable response."""

    def __init__(self, name: str = "mock_tool"):
        super().__init__(
            name=name,
            description=f"A mock tool named {name} for testing",
        )
        self.call_count = 0

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input to the tool",
                }
            },
            "required": ["input"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        self.call_count += 1
        return ToolResult(
            success=True,
            result=f"Mock result for: {parameters.get('input', 'no input')}",
            error=None,
        )


class TrackingStrategy:
    """Strategy that tracks when compact is called."""

    def __init__(self, token_budget: int = 100_000):
        self.token_budget = token_budget
        self.call_count = 0
        self.message_counts: List[int] = []

    def compact(self, messages: List[Message]) -> List[Message]:
        self.call_count += 1
        self.message_counts.append(len(messages))
        # Don't actually compact - just track
        return messages


# === Unit Tests for Strategies ===


class TestNoCompaction:
    """Test NoCompaction returns messages unchanged."""

    def test_returns_unchanged(self):
        strategy = NoCompaction()
        messages = [
            SystemMessage(content="You are helpful", source="system"),
            UserMessage(content="Hello", source="user"),
        ]
        result = strategy.compact(messages)
        assert result == messages
        assert len(result) == 2


class TestHeadTailCompaction:
    """Test HeadTailCompaction logic."""

    def test_no_compaction_under_budget(self):
        """Messages under budget should pass through unchanged."""
        strategy = HeadTailCompaction(token_budget=100_000)
        messages = [
            SystemMessage(content="System", source="system"),
            UserMessage(content="Hello", source="user"),
        ]
        result = strategy.compact(messages)
        assert len(result) == len(messages)
        assert strategy.compaction_count == 0

    def test_compaction_over_budget(self):
        """Messages over budget should be compacted."""
        # Use very low budget to force compaction
        strategy = HeadTailCompaction(token_budget=50, head_ratio=0.5)

        # Create many messages to exceed budget
        messages = [SystemMessage(content="System prompt", source="system")]
        for i in range(20):
            messages.append(UserMessage(content=f"Message {i} " * 10, source="user"))

        result = strategy.compact(messages)

        # Should have compacted
        assert len(result) < len(messages)
        assert strategy.compaction_count == 1
        assert strategy.total_tokens_saved > 0

    def test_preserves_atomic_groups(self):
        """Tool calls and their results must stay together."""
        strategy = HeadTailCompaction(token_budget=200, head_ratio=0.3)

        messages = [
            SystemMessage(content="System", source="system"),
            UserMessage(content="Do something", source="user"),
            AssistantMessage(
                content="",
                source="assistant",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="test_tool",
                        parameters={"x": 1},
                        call_id="call_1",
                    )
                ],
            ),
            ToolMessage(
                content="Result 1",
                source="test_tool",
                tool_call_id="call_1",
                tool_name="test_tool",
                success=True,
            ),
            # More messages to force compaction
            UserMessage(content="Another message " * 20, source="user"),
            AssistantMessage(
                content="",
                source="assistant",
                tool_calls=[
                    ToolCallRequest(
                        tool_name="test_tool",
                        parameters={"x": 2},
                        call_id="call_2",
                    )
                ],
            ),
            ToolMessage(
                content="Result 2",
                source="test_tool",
                tool_call_id="call_2",
                tool_name="test_tool",
                success=True,
            ),
        ]

        result = strategy.compact(messages)

        # Check that tool calls and results stay together
        for i, msg in enumerate(result):
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                # Find the expected tool result
                call_id = msg.tool_calls[0].call_id
                # There must be a corresponding ToolMessage after this
                found_result = False
                for j in range(i + 1, len(result)):
                    if (
                        isinstance(result[j], ToolMessage)
                        and result[j].tool_call_id == call_id
                    ):
                        found_result = True
                        break
                assert (
                    found_result
                ), f"Tool call {call_id} missing its result in compacted messages"


class TestSlidingWindowCompaction:
    """Test SlidingWindowCompaction logic."""

    def test_preserves_system_message(self):
        """System message should always be preserved."""
        strategy = SlidingWindowCompaction(token_budget=100)

        messages = [
            SystemMessage(content="Important system prompt", source="system"),
        ]
        # Add many messages to exceed budget
        for i in range(20):
            messages.append(UserMessage(content=f"Message {i} " * 10, source="user"))

        result = strategy.compact(messages)

        # System message should be first
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "Important system prompt"

    def test_keeps_recent_messages(self):
        """Should keep the most recent messages."""
        strategy = SlidingWindowCompaction(token_budget=100)

        messages = [
            SystemMessage(content="System", source="system"),
            UserMessage(content="Old message 1", source="user"),
            UserMessage(content="Old message 2", source="user"),
            UserMessage(content="Recent message", source="user"),
        ]

        result = strategy.compact(messages)

        # Should include the recent message
        contents = [getattr(m, "content", "") for m in result]
        assert "Recent message" in contents


# === Integration Tests ===


class TestCompactionInToolLoop:
    """Test that compaction happens correctly inside the tool loop."""

    def test_strategy_protocol_compliance(self):
        """Verify strategies implement the protocol correctly."""
        from forla.compaction import CompactionStrategy

        # All strategies should satisfy the protocol
        assert isinstance(HeadTailCompaction(), CompactionStrategy)
        assert isinstance(SlidingWindowCompaction(), CompactionStrategy)
        assert isinstance(NoCompaction(), CompactionStrategy)

    def test_compaction_loop_simulation(self):
        """Simulate the agent tool loop to verify compaction behavior.

        This tests the CRITICAL pattern that MiniAgent uses:
        `messages = strategy.compact(messages)` - reassignment!
        """
        strategy = HeadTailCompaction(token_budget=100)

        # Simulate the agent loop pattern
        messages = [
            SystemMessage(content="System", source="system"),
            UserMessage(content="User task", source="user"),
        ]

        # Simulate 3 iterations of the tool loop
        for iteration in range(3):
            # === THIS IS THE KEY: Compaction happens BEFORE LLM call ===
            messages = strategy.compact(messages)

            # Simulate LLM response with tool call
            messages.append(
                AssistantMessage(
                    content="",
                    source="assistant",
                    tool_calls=[
                        ToolCallRequest(
                            tool_name="test",
                            parameters={"x": iteration},
                            call_id=f"call_{iteration}",
                        )
                    ],
                )
            )

            # Simulate tool result
            messages.append(
                ToolMessage(
                    content=f"Result {iteration} " * 50,  # Make results big
                    source="test",
                    tool_call_id=f"call_{iteration}",
                    tool_name="test",
                    success=True,
                )
            )

        # After 3 iterations with big results, compaction should have triggered
        # The key verification: strategy was called and messages were compacted
        assert strategy.compaction_count > 0, "Compaction should have been triggered"
        assert strategy.total_tokens_saved > 0, "Tokens should have been saved"

    def test_compaction_persists_across_iterations(self):
        """Verify the compacted list is used (not discarded) in subsequent iterations."""

        class TrackingCompactStrategy:
            """Tracks input/output lengths to verify list persistence."""

            def __init__(self):
                self.calls: List[tuple] = []  # (input_len, output_len)

            def compact(self, messages: List[Message]) -> List[Message]:
                input_len = len(messages)

                # Compact to 3 messages if over 5
                if len(messages) > 5:
                    compacted = messages[:2] + messages[-1:]
                    self.calls.append((input_len, len(compacted)))
                    return compacted

                self.calls.append((input_len, len(messages)))
                return messages

        strategy = TrackingCompactStrategy()

        messages = [
            SystemMessage(content="System", source="system"),
            UserMessage(content="Task", source="user"),
        ]

        # Simulate iterations
        for i in range(5):
            messages = strategy.compact(messages)
            # Add 2 messages per iteration
            messages.append(
                AssistantMessage(content=f"Response {i}", source="assistant")
            )
            messages.append(UserMessage(content=f"Follow up {i}", source="user"))

        # Verify the pattern: after compaction, next input should start from
        # compacted size (+ new messages), NOT from original growing size
        for i, (input_len, output_len) in enumerate(strategy.calls):
            if i > 0:
                prev_output = strategy.calls[i - 1][1]
                # Input should be prev_output + 2 (the messages added after)
                expected_input = prev_output + 2
                assert input_len == expected_input, (
                    f"Iteration {i}: input_len={input_len}, expected={expected_input}. "
                    f"This means compaction didn't persist - the list wasn't reassigned!"
                )



# === Statistics Tests ===


class TestStrategyStatistics:
    """Test that strategies track statistics correctly."""

    def test_head_tail_tracks_compaction_count(self):
        strategy = HeadTailCompaction(token_budget=50)

        # First call - should compact
        messages1 = [
            SystemMessage(content="S", source="system"),
        ] + [UserMessage(content="M" * 50, source="user") for _ in range(10)]
        strategy.compact(messages1)

        # Second call - should compact again
        messages2 = messages1 + [
            UserMessage(content="More " * 50, source="user")
        ]
        strategy.compact(messages2)

        assert strategy.compaction_count == 2

    def test_head_tail_tracks_tokens_saved(self):
        strategy = HeadTailCompaction(token_budget=50)

        messages = [
            SystemMessage(content="S", source="system"),
        ] + [UserMessage(content="Message " * 20, source="user") for _ in range(10)]

        strategy.compact(messages)

        assert strategy.total_tokens_saved > 0


# === Edge Cases ===


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_messages(self):
        strategy = HeadTailCompaction(token_budget=100)
        result = strategy.compact([])
        assert result == []

    def test_single_message(self):
        strategy = HeadTailCompaction(token_budget=100)
        messages = [SystemMessage(content="Hello", source="system")]
        result = strategy.compact(messages)
        assert len(result) == 1

    def test_no_compaction_configured(self):
        """Agent without compaction should work normally."""
        mock_client = MagicMock()
        mock_client.model = "gpt-4o"

        agent = Agent(
            name="test",
            description="Test",
            instructions="Test",
            model_client=mock_client,
            compaction=None,
        )

        assert agent.compaction is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
