"""
Tests for orchestration patterns.
"""

import asyncio
from typing import AsyncGenerator, List, Optional, Union, cast

import pytest

from forla._cancellation_token import CancellationToken
from forla.agents import BaseAgent
from forla.context import AgentContext
from forla.messages import AssistantMessage, Message, UserMessage
from forla.orchestration import (
    MaxMessageTermination,
    RoundRobinOrchestrator,
    TextMentionTermination,
)
from forla.types import AgentEvent, AgentResponse, ChatCompletionChunk, Usage


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, name: str, response: str = "Mock response"):
        # Create minimal mock agent
        self.name = name
        self.description = f"Mock agent {name}"
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
        """Mock agent run that returns predefined response."""
        # Check for cancellation
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()

        # Simulate agent processing
        await asyncio.sleep(0.01)

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
            usage=Usage(duration_ms=10, llm_calls=1, tokens_input=10, tokens_output=5),
            finish_reason="stop",
        )

    async def run_stream(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
        verbose: bool = False,
        stream_tokens: bool = False,
    ) -> AsyncGenerator[Union[Message, AgentResponse], None]:
        """Mock agent streaming that yields the same result as run()."""
        # Check for cancellation
        if cancellation_token and cancellation_token.is_cancelled():
            raise asyncio.CancelledError()

        # Simulate agent processing
        await asyncio.sleep(0.01)

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
            usage=Usage(duration_ms=10, llm_calls=1, tokens_input=10, tokens_output=5),
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_round_robin_orchestrator_basic():
    """Test basic round-robin orchestration."""
    # Create mock agents
    agent1 = MockAgent("agent1", "Response from agent 1")
    agent2 = MockAgent("agent2", "Response from agent 2")
    agents: List[BaseAgent] = [agent1, agent2]

    # Create termination condition
    termination = MaxMessageTermination(max_messages=5)

    # Create orchestrator
    orchestrator = RoundRobinOrchestrator(agents, termination)

    # Run orchestration
    result = await orchestrator.run("Test task")

    # Verify result
    assert isinstance(result.final_result, str)
    assert len(result.messages) >= 3  # User + at least 2 assistant responses
    assert result.usage.duration_ms > 0
    assert result.stop_message.source == "MaxMessageTermination"

    # Verify pattern metadata
    metadata = result.pattern_metadata
    assert metadata["pattern"] == "RoundRobinOrchestrator"
    assert metadata["agents_count"] == 2
    assert "cycles_completed" in metadata


@pytest.mark.asyncio
async def test_round_robin_orchestrator_agent_selection():
    """Test that round-robin properly cycles through agents."""
    agent1 = MockAgent("agent1", "Response 1")
    agent2 = MockAgent("agent2", "Response 2")
    agent3 = MockAgent("agent3", "Response 3")
    agents: List[BaseAgent] = [agent1, agent2, agent3]

    termination = MaxMessageTermination(max_messages=7)  # User + 6 responses (2 cycles)
    orchestrator = RoundRobinOrchestrator(agents, termination)

    result = await orchestrator.run("Cycle through agents")

    # Check that we have responses from all agents in order
    assistant_messages = [
        msg for msg in result.messages if isinstance(msg, AssistantMessage)
    ]

    # Should have 6 assistant messages (2 full cycles)
    assert len(assistant_messages) == 6

    # Verify round-robin order: agent1, agent2, agent3, agent1, agent2, agent3
    expected_responses = [
        "Response 1",
        "Response 2",
        "Response 3",
        "Response 1",
        "Response 2",
        "Response 3",
    ]
    actual_responses = [msg.content for msg in assistant_messages]
    assert actual_responses == expected_responses


@pytest.mark.asyncio
async def test_round_robin_orchestrator_text_termination():
    """Test orchestrator with text mention termination."""
    agent1 = MockAgent("agent1", "Let's continue")
    agent2 = MockAgent("agent2", "TERMINATE the conversation")
    agents: List[BaseAgent] = [agent1, agent2]

    termination = TextMentionTermination("TERMINATE")
    orchestrator = RoundRobinOrchestrator(agents, termination)

    result = await orchestrator.run("Test termination")

    # Should stop when agent2 says TERMINATE
    assert result.stop_message.source == "TextMentionTermination"
    assert "Text mention found" in result.stop_message.content

    # Should have exactly 3 messages: user + agent1 + agent2
    assert len(result.messages) == 3
    assistant_messages = [
        msg for msg in result.messages if isinstance(msg, AssistantMessage)
    ]
    assert len(assistant_messages) == 2
    assert assistant_messages[1].content == "TERMINATE the conversation"


@pytest.mark.asyncio
async def test_round_robin_orchestrator_cancellation():
    """Test orchestrator cancellation."""
    agent1 = MockAgent("agent1", "Response 1")
    agent2 = MockAgent("agent2", "Response 2")
    agents: List[BaseAgent] = [agent1, agent2]

    termination = MaxMessageTermination(max_messages=10)  # High limit
    orchestrator = RoundRobinOrchestrator(agents, termination)

    # Create cancellation token
    cancellation_token = CancellationToken()

    # Start orchestration
    task = asyncio.create_task(
        orchestrator.run("Test cancellation", cancellation_token)
    )

    # Cancel after a short delay
    await asyncio.sleep(0.05)
    cancellation_token.cancel()

    # Should raise CancelledError
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_round_robin_orchestrator_streaming():
    """Test orchestrator streaming functionality."""
    agent1 = MockAgent("agent1", "Stream response 1")
    agent2 = MockAgent("agent2", "Stream response 2")
    agents: List[BaseAgent] = [agent1, agent2]

    termination = MaxMessageTermination(max_messages=3)
    orchestrator = RoundRobinOrchestrator(agents, termination)

    items = []
    async for item in orchestrator.run_stream("Test streaming", verbose=True):
        items.append(item)

    # Should have various types of items
    from forla.types import OrchestrationEvent, OrchestrationResponse

    messages = [
        item for item in items if isinstance(item, (UserMessage, AssistantMessage))
    ]
    events = [item for item in items if hasattr(item, "event_type")]
    results = [item for item in items if isinstance(item, OrchestrationResponse)]

    assert len(messages) >= 3  # At least user + 2 assistant messages
    assert len(events) >= 4  # Start + multiple selection/execution events + complete
    assert len(results) == 1  # Final result


@pytest.mark.asyncio
async def test_round_robin_orchestrator_empty_agents():
    """Test orchestrator with empty agent list."""
    with pytest.raises(ValueError, match="At least one agent is required"):
        RoundRobinOrchestrator([], MaxMessageTermination(5))


@pytest.mark.asyncio
async def test_round_robin_orchestrator_duplicate_names():
    """Test orchestrator with duplicate agent names."""
    agent1 = MockAgent("duplicate", "Response 1")
    agent2 = MockAgent("duplicate", "Response 2")

    with pytest.raises(ValueError, match="Agent names must be unique"):
        RoundRobinOrchestrator([agent1, agent2], MaxMessageTermination(5))


@pytest.mark.asyncio
async def test_round_robin_orchestrator_max_iterations():
    """Test orchestrator max iterations safety."""
    agent = MockAgent("agent", "Keep going")
    agents: List[BaseAgent] = [agent]

    # Use a termination that never triggers
    external_flag = [False]

    def never_terminate():
        return external_flag[0]

    from forla.termination import ExternalTermination

    termination = ExternalTermination(never_terminate)

    orchestrator = RoundRobinOrchestrator(agents, termination, max_iterations=3)

    result = await orchestrator.run("Test max iterations")

    # Should stop due to max iterations
    assert "Maximum iterations reached" in result.stop_message.content
    assert result.stop_message.source == "MaxIterations"
    assert result.pattern_metadata["iterations_completed"] == 3


@pytest.mark.asyncio
async def test_round_robin_orchestrator_context_management():
    """Test that agents receive proper context."""

    # Create agent that echoes context information it receives
    class ContextEchoAgent(MockAgent):
        async def run(
            self,
            task: Union[str, UserMessage, List[Message]],
            cancellation_token: Optional[CancellationToken] = None,
        ) -> AgentResponse:
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            await asyncio.sleep(0.01)

            # Create context with messages (matching real Agent behavior)
            context = AgentContext()

            # Analyze the task to count how many turns have occurred
            response_count = 0
            if isinstance(task, str):
                # Count occurrences of "Iteration" in the context (indicating previous agent responses)
                response_count = task.count("Iteration")
                context.add_message(UserMessage(content=task, source="user"))
            elif isinstance(task, list):
                response_count = len(
                    [msg for msg in task if isinstance(msg, AssistantMessage)]
                )
                for msg in task:
                    context.add_message(msg)
            else:
                context.add_message(UserMessage(content=str(task), source="user"))

            assistant_message = AssistantMessage(
                content=f"Iteration {response_count + 1}", source=self.name
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
        ) -> AsyncGenerator[Union[Message, AgentResponse], None]:
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            await asyncio.sleep(0.01)

            # Create context with messages (matching real Agent behavior)
            context = AgentContext()

            # Analyze the task to count how many turns have occurred
            response_count = 0
            if isinstance(task, str):
                # Count occurrences of "Iteration" in the context (indicating previous agent responses)
                response_count = task.count("Iteration")
                msg = UserMessage(content=task, source="user")
                context.add_message(msg)
                yield msg
            elif isinstance(task, list):
                response_count = len(
                    [msg for msg in task if isinstance(msg, AssistantMessage)]
                )
                for msg in task:
                    context.add_message(msg)
                    yield msg
            else:
                msg = UserMessage(content=str(task), source="user")
                context.add_message(msg)
                yield msg

            assistant_message = AssistantMessage(
                content=f"Iteration {response_count + 1}", source=self.name
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

    agent = ContextEchoAgent("context_agent")
    termination = MaxMessageTermination(max_messages=5)
    orchestrator = RoundRobinOrchestrator([agent], termination)

    result = await orchestrator.run("Test context")

    # Check that agent receives increasing context
    assistant_messages = [
        msg for msg in result.messages if isinstance(msg, AssistantMessage)
    ]

    # First agent call should be iteration 1 (no previous context)
    assert "Iteration 1" in assistant_messages[0].content

    # Second agent call should be iteration 2 (with previous context)
    assert "Iteration 2" in assistant_messages[1].content


@pytest.mark.asyncio
async def test_usage_aggregation():
    """Test that orchestrator properly aggregates usage statistics from agents."""
    # Create agents with different usage patterns
    agent1 = MockAgent("agent1", "Response 1")
    agent2 = MockAgent("agent2", "Response 2")
    agents: List[BaseAgent] = [agent1, agent2]

    termination = MaxMessageTermination(max_messages=5)  # User + 4 agent responses
    orchestrator = RoundRobinOrchestrator(agents, termination)

    result = await orchestrator.run("Test usage aggregation")

    # Verify basic functionality
    assert len(result.messages) == 5  # 1 user + 4 assistant messages

    # Verify usage aggregation
    # Each mock agent uses: llm_calls=1, tokens_input=10, tokens_output=5, duration_ms=10
    # With 4 agent executions (2 per agent), we should have aggregated stats
    usage = result.usage

    assert usage.llm_calls == 4  # 4 agent executions
    assert usage.tokens_input == 40  # 4 * 10
    assert usage.tokens_output == 20  # 4 * 5
    assert usage.tool_calls == 0  # Mock agents don't use tools
    assert usage.memory_operations == 0  # Mock agents don't use memory
    assert usage.duration_ms > 0  # Should have some duration from orchestration

    print(f"✅ Usage aggregation test passed:")
    print(f"   LLM calls: {usage.llm_calls}")
    print(f"   Input tokens: {usage.tokens_input}")
    print(f"   Output tokens: {usage.tokens_output}")
    print(f"   Duration: {usage.duration_ms}ms")
