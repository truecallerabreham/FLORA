"""
Tests for the deterministic loop hooks system.

Tests cover:
1. No hooks - identical behavior to before
2. Start hook injection before first LLM call
3. End hook allows stop when all todos complete
4. End hook resumes loop when todos incomplete
5. Max restarts terminates even with incomplete todos
6. Multiple start hooks chained
7. Custom hooks via base class
8. TerminationCondition composition (| and &)
9. Full trajectory message ordering
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Type
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from forla import Agent
from forla._hooks import (
    BaseEndHook,
    BaseStartHook,
    CompletionCheckHook,
    CompositeTermination,
    LoopContext,
    MaxRestartsTermination,
    PlanningHook,
    TerminationCondition,
)
from forla.context import AgentContext
from forla.llm import BaseChatCompletionClient
from forla.messages import (
    AssistantMessage,
    ToolCallRequest,
    ToolMessage,
    UserMessage,
)
from forla.tools import BaseTool
from forla.types import (
    AgentResponse,
    ChatCompletionResult,
    ToolResult,
    Usage,
)

# Patch path for _load_todos - it's imported lazily inside CompletionCheckHook
LOAD_TODOS_PATCH = "forla.tools._context_tools._load_todos"

# === Test Fixtures ===

USAGE = Usage(
    duration_ms=10,
    llm_calls=1,
    tokens_input=50,
    tokens_output=25,
    tool_calls=0,
    memory_operations=0,
)


class MockClient(BaseChatCompletionClient):
    """Mock LLM client that returns pre-programmed responses in sequence."""

    def __init__(self):
        super().__init__(model="test-model")
        self.responses: List[AssistantMessage] = []
        self.call_count = 0
        self.received_messages: List[List] = []  # track what was sent to LLM

    def set_responses(self, responses: List[AssistantMessage]):
        self.responses = list(responses)

    async def create(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> ChatCompletionResult:
        self.call_count += 1
        self.received_messages.append(list(messages))

        if self.responses:
            response = self.responses.pop(0)
        else:
            response = AssistantMessage(content="Done", source="mock")

        return ChatCompletionResult(
            message=response,
            model="test-model",
            finish_reason="stop",
            usage=USAGE,
        )

    async def create_stream(self, messages, tools=None, output_format=None, **kwargs):
        from forla.types import ChatCompletionChunk

        result = await self.create(messages, tools, output_format, **kwargs)
        yield ChatCompletionChunk(
            content=result.message.content or "",
            is_complete=True,
            tool_call_chunk=None,
            usage=USAGE,
        )


class MockTool(BaseTool):
    """Simple mock tool."""

    def __init__(self, name: str = "mock_tool"):
        super().__init__(name=name, description=f"Mock tool: {name}")
        self.call_count = 0

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input"}
            },
            "required": ["input"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        self.call_count += 1
        return ToolResult(
            success=True,
            result=f"result: {parameters.get('input', '')}",
            error=None,
        )


def make_agent(client, tools=None, start_hooks=None, end_hooks=None, max_iterations=10):
    """Helper to create an agent with hooks."""
    return Agent(
        name="test_agent",
        description="Test agent",
        instructions="You are a test assistant.",
        model_client=client,
        tools=tools or [],
        start_hooks=start_hooks or [],
        end_hooks=end_hooks or [],
        max_iterations=max_iterations,
    )


def tool_call_response(tool_name, args, call_id="call_1"):
    """Helper to create an AssistantMessage with a tool call."""
    return AssistantMessage(
        content="",
        source="mock",
        tool_calls=[
            ToolCallRequest(
                tool_name=tool_name,
                parameters=args,
                call_id=call_id,
            )
        ],
    )


# =============================================================================
# Test 1: No hooks - identical behavior to before
# =============================================================================


class TestNoHooks:
    """Verify agents without hooks work identically to before."""

    @pytest.mark.asyncio
    async def test_simple_response_no_hooks(self):
        """Agent without hooks responds normally."""
        client = MockClient()
        client.set_responses([
            AssistantMessage(content="Hello world", source="mock"),
        ])
        agent = make_agent(client)

        result = await agent.run("Hi")

        assert isinstance(result, AgentResponse)
        assert client.call_count == 1

        assistant_msgs = [m for m in result.messages if isinstance(m, AssistantMessage)]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[0].content == "Hello world"

    @pytest.mark.asyncio
    async def test_tool_loop_no_hooks(self):
        """Agent without hooks runs tool loop normally."""
        client = MockClient()
        tool = MockTool()
        client.set_responses([
            tool_call_response("mock_tool", {"input": "test"}),
            AssistantMessage(content="Tool result processed", source="mock"),
        ])
        agent = make_agent(client, tools=[tool])

        result = await agent.run("Use the tool")

        assert client.call_count == 2
        assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_no_hook_messages_in_context(self):
        """Without hooks, no source='hook' messages should exist."""
        client = MockClient()
        client.set_responses([
            AssistantMessage(content="Done", source="mock"),
        ])
        agent = make_agent(client)

        result = await agent.run("Hello")

        hook_msgs = [m for m in result.messages if getattr(m, "source", "") == "hook"]
        assert len(hook_msgs) == 0


# =============================================================================
# Test 2: Start hook injection
# =============================================================================


class TestStartHook:
    """Verify start hooks inject messages before the first LLM call."""

    @pytest.mark.asyncio
    async def test_planning_hook_injects_message(self):
        """PlanningHook should inject a UserMessage before the LLM call."""
        client = MockClient()
        client.set_responses([
            AssistantMessage(content="I'll create a plan", source="mock"),
        ])
        agent = make_agent(client, start_hooks=[PlanningHook()])

        result = await agent.run("Build a website")

        assert client.call_count == 1
        first_call_messages = client.received_messages[0]

        hook_msgs = [m for m in first_call_messages if getattr(m, "source", "") == "hook"]
        assert len(hook_msgs) == 1
        assert "todo" in hook_msgs[0].content.lower()

    @pytest.mark.asyncio
    async def test_custom_start_hook_text(self):
        """PlanningHook with custom instruction uses that text."""
        custom_text = "ALWAYS think step by step."
        client = MockClient()
        client.set_responses([
            AssistantMessage(content="OK", source="mock"),
        ])
        agent = make_agent(client, start_hooks=[PlanningHook(instruction=custom_text)])

        await agent.run("Do something")

        first_call_messages = client.received_messages[0]
        hook_msgs = [m for m in first_call_messages if getattr(m, "source", "") == "hook"]
        assert len(hook_msgs) == 1
        assert hook_msgs[0].content == custom_text

    @pytest.mark.asyncio
    async def test_start_hook_none_does_not_inject(self):
        """A start hook returning None should not inject any message."""

        class NullStartHook(BaseStartHook):
            async def on_start(self, context):
                return None

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="OK", source="mock"),
        ])
        agent = make_agent(client, start_hooks=[NullStartHook()])

        await agent.run("Hello")

        first_call_messages = client.received_messages[0]
        hook_msgs = [m for m in first_call_messages if getattr(m, "source", "") == "hook"]
        assert len(hook_msgs) == 0


# =============================================================================
# Test 3: End hook - all todos complete, should NOT resume
# =============================================================================


class TestEndHookAllComplete:
    """When all todos are complete, CompletionCheckHook should allow stop."""

    @pytest.mark.asyncio
    async def test_all_todos_complete_allows_stop(self):
        """End hook should return None (allow stop) when all todos are done."""
        all_complete = [
            {"content": "Task A", "status": "completed", "activeForm": "Doing A"},
            {"content": "Task B", "status": "completed", "activeForm": "Doing B"},
        ]

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="All done!", source="mock"),
        ])
        agent = make_agent(client, end_hooks=[CompletionCheckHook(max_restarts=2)])

        with patch(LOAD_TODOS_PATCH, return_value=all_complete):
            result = await agent.run("Do tasks")

        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_no_todos_allows_stop(self):
        """End hook should allow stop when there are no todos at all."""
        client = MockClient()
        client.set_responses([
            AssistantMessage(content="Done", source="mock"),
        ])
        agent = make_agent(client, end_hooks=[CompletionCheckHook(max_restarts=2)])

        with patch(LOAD_TODOS_PATCH, return_value=[]):
            result = await agent.run("Hello")

        assert client.call_count == 1


# =============================================================================
# Test 4: End hook with incomplete todos - verify resume
# =============================================================================


class TestEndHookResume:
    """CompletionCheckHook should resume the loop when todos are incomplete."""

    @pytest.mark.asyncio
    async def test_incomplete_todos_resume_loop(self):
        """End hook should inject resume message when todos are incomplete."""
        incomplete_todos = [
            {"content": "Task A", "status": "completed", "activeForm": "Doing A"},
            {"content": "Task B", "status": "pending", "activeForm": "Doing B"},
        ]

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="I think I'm done", source="mock"),
            AssistantMessage(content="Now I'm really done", source="mock"),
            AssistantMessage(content="Final answer", source="mock"),
        ])
        agent = make_agent(client, end_hooks=[CompletionCheckHook(max_restarts=2)])

        with patch(LOAD_TODOS_PATCH, return_value=incomplete_todos):
            result = await agent.run("Do tasks")

        # 1 initial + 2 restarts = 3 total
        assert client.call_count == 3

    @pytest.mark.asyncio
    async def test_resume_message_contains_incomplete_info(self):
        """Resume message should list incomplete tasks."""
        incomplete_todos = [
            {"content": "Write tests", "status": "pending", "activeForm": "Writing tests"},
        ]

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="Stopping", source="mock"),
            AssistantMessage(content="OK continuing", source="mock"),
            AssistantMessage(content="Done now", source="mock"),
        ])
        agent = make_agent(client, end_hooks=[CompletionCheckHook(max_restarts=2)])

        with patch(LOAD_TODOS_PATCH, return_value=incomplete_todos):
            result = await agent.run("Work")

        # Check that second LLM call received a hook resume message
        second_call = client.received_messages[1]
        hook_msgs = [m for m in second_call if getattr(m, "source", "") == "hook"]
        assert len(hook_msgs) >= 1

        resume_text = hook_msgs[-1].content
        assert "Write tests" in resume_text
        assert "incomplete" in resume_text.lower()


# =============================================================================
# Test 5: Max restarts exceeded - agent stops
# =============================================================================


class TestMaxRestarts:
    """Verify max_restarts terminates even with incomplete todos."""

    @pytest.mark.asyncio
    async def test_max_restarts_terminates(self):
        """After max_restarts, agent should stop even if todos are incomplete."""
        always_incomplete = [
            {"content": "Never done", "status": "pending", "activeForm": "Never doing"},
        ]

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="Try 1", source="mock"),
            AssistantMessage(content="Try 2", source="mock"),
        ])
        agent = make_agent(client, end_hooks=[CompletionCheckHook(max_restarts=1)])

        with patch(LOAD_TODOS_PATCH, return_value=always_incomplete):
            result = await agent.run("Do the impossible")

        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_restarts_no_resume(self):
        """max_restarts=0 should never resume."""
        incomplete = [
            {"content": "Task", "status": "pending", "activeForm": "Doing"},
        ]

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="Done", source="mock"),
        ])
        agent = make_agent(client, end_hooks=[CompletionCheckHook(max_restarts=0)])

        with patch(LOAD_TODOS_PATCH, return_value=incomplete):
            result = await agent.run("Work")

        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_restarts_dont_exceed_max_iterations(self):
        """Restarts combined with tool loops shouldn't exceed max_iterations."""
        incomplete = [
            {"content": "Infinite task", "status": "pending", "activeForm": "Doing"},
        ]

        client = MockClient()
        client.set_responses([
            AssistantMessage(content=f"Attempt {i}", source="mock")
            for i in range(20)
        ])
        agent = make_agent(
            client,
            end_hooks=[CompletionCheckHook(max_restarts=100)],
            max_iterations=5,
        )

        with patch(LOAD_TODOS_PATCH, return_value=incomplete):
            result = await agent.run("Work forever")

        assert client.call_count <= 5


# =============================================================================
# Test 6: Multiple start hooks chained
# =============================================================================


class TestMultipleStartHooks:
    """All start hooks should run in order, all injections applied."""

    @pytest.mark.asyncio
    async def test_two_start_hooks_both_inject(self):
        """Both start hooks should inject their messages."""

        class HookA(BaseStartHook):
            async def on_start(self, context):
                return "HOOK_A_INJECTION"

        class HookB(BaseStartHook):
            async def on_start(self, context):
                return "HOOK_B_INJECTION"

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="OK", source="mock"),
        ])
        agent = make_agent(client, start_hooks=[HookA(), HookB()])

        await agent.run("Test")

        first_call = client.received_messages[0]
        hook_msgs = [m for m in first_call if getattr(m, "source", "") == "hook"]
        assert len(hook_msgs) == 2
        assert hook_msgs[0].content == "HOOK_A_INJECTION"
        assert hook_msgs[1].content == "HOOK_B_INJECTION"

    @pytest.mark.asyncio
    async def test_mixed_none_and_injection(self):
        """A None-returning hook should not prevent other hooks from injecting."""

        class NullHook(BaseStartHook):
            async def on_start(self, context):
                return None

        class InjectHook(BaseStartHook):
            async def on_start(self, context):
                return "INJECTED"

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="OK", source="mock"),
        ])
        agent = make_agent(client, start_hooks=[NullHook(), InjectHook()])

        await agent.run("Test")

        first_call = client.received_messages[0]
        hook_msgs = [m for m in first_call if getattr(m, "source", "") == "hook"]
        assert len(hook_msgs) == 1
        assert hook_msgs[0].content == "INJECTED"


# =============================================================================
# Test 7: Custom hooks - base class extensibility
# =============================================================================


class TestCustomHooks:
    """Verify users can create custom hooks via the base classes."""

    @pytest.mark.asyncio
    async def test_custom_start_hook(self):
        """Custom start hook can use loop context metadata."""
        call_log = []

        class LoggingStartHook(BaseStartHook):
            async def on_start(self, context):
                call_log.append(f"start:{context.agent_name}")
                context.metadata["started"] = True
                return "Custom start instruction"

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="OK", source="mock"),
        ])
        agent = make_agent(client, start_hooks=[LoggingStartHook()])

        await agent.run("Test")

        assert call_log == ["start:test_agent"]

    @pytest.mark.asyncio
    async def test_custom_end_hook_resume(self):
        """Custom end hook can decide whether to resume based on context."""
        resume_count = 0

        class CountingEndHook(BaseEndHook):
            async def on_end(self, context):
                nonlocal resume_count
                if context.restart_count < 2:
                    resume_count += 1
                    return f"Keep going (restart {context.restart_count})"
                return None

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="Try 1", source="mock"),
            AssistantMessage(content="Try 2", source="mock"),
            AssistantMessage(content="Try 3", source="mock"),
        ])
        agent = make_agent(client, end_hooks=[CountingEndHook()])

        result = await agent.run("Work")

        assert resume_count == 2
        assert client.call_count == 3

    @pytest.mark.asyncio
    async def test_first_end_hook_wins(self):
        """When multiple end hooks, first to return non-None wins."""
        hook_calls = []

        class HookA(BaseEndHook):
            async def on_end(self, context):
                hook_calls.append("A")
                return "Resume from A"

        class HookB(BaseEndHook):
            async def on_end(self, context):
                hook_calls.append("B")
                return "Resume from B"

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="First", source="mock"),
            AssistantMessage(content="Second", source="mock"),
            AssistantMessage(content="Third", source="mock"),
        ])

        agent = make_agent(client, end_hooks=[HookA(), HookB()], max_iterations=3)

        result = await agent.run("Test")

        # Hook B should never be called because A always returns non-None first
        assert all(h == "A" for h in hook_calls)
        assert "B" not in hook_calls


# =============================================================================
# Test 8: TerminationCondition composition
# =============================================================================


class TestTerminationConditions:
    """Test TerminationCondition base class and composition."""

    def test_max_restarts_basic(self):
        """MaxRestartsTermination terminates at the threshold."""
        ctx = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test",
        )
        cond = MaxRestartsTermination(max_restarts=3)

        ctx.restart_count = 0
        assert cond.should_terminate(ctx) is False

        ctx.restart_count = 2
        assert cond.should_terminate(ctx) is False

        ctx.restart_count = 3
        assert cond.should_terminate(ctx) is True

        ctx.restart_count = 5
        assert cond.should_terminate(ctx) is True

    def test_or_composition(self):
        """A | B should terminate if either is True."""
        ctx = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test",
        )

        class AlwaysTerminate(TerminationCondition):
            def should_terminate(self, context):
                return True

        class NeverTerminate(TerminationCondition):
            def should_terminate(self, context):
                return False

        assert (AlwaysTerminate() | NeverTerminate()).should_terminate(ctx) is True
        assert (NeverTerminate() | NeverTerminate()).should_terminate(ctx) is False
        assert (NeverTerminate() | AlwaysTerminate()).should_terminate(ctx) is True

    def test_and_composition(self):
        """A & B should terminate only if both are True."""
        ctx = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test",
        )

        class AlwaysTerminate(TerminationCondition):
            def should_terminate(self, context):
                return True

        class NeverTerminate(TerminationCondition):
            def should_terminate(self, context):
                return False

        assert (AlwaysTerminate() & NeverTerminate()).should_terminate(ctx) is False
        assert (AlwaysTerminate() & AlwaysTerminate()).should_terminate(ctx) is True

    def test_chained_or(self):
        """A | B | C should work correctly."""
        ctx = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test",
        )

        cond = (
            MaxRestartsTermination(5)
            | MaxRestartsTermination(3)
            | MaxRestartsTermination(10)
        )

        ctx.restart_count = 2
        assert cond.should_terminate(ctx) is False

        ctx.restart_count = 3
        assert cond.should_terminate(ctx) is True

        ctx.restart_count = 5
        assert cond.should_terminate(ctx) is True

    def test_mixed_and_or(self):
        """(A & B) | C should preserve AND grouping."""
        ctx = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test",
        )

        class NeverTerminate(TerminationCondition):
            def should_terminate(self, context):
                return False

        # (Never & MaxRestarts(2)) | MaxRestarts(5)
        # restart_count=3: (False & True) | False = False | False = False
        # restart_count=5: (False & True) | True = False | True = True
        cond = (NeverTerminate() & MaxRestartsTermination(2)) | MaxRestartsTermination(5)

        ctx.restart_count = 3
        assert cond.should_terminate(ctx) is False

        ctx.restart_count = 5
        assert cond.should_terminate(ctx) is True

    def test_composite_repr(self):
        """CompositeTermination repr should be readable."""
        cond = MaxRestartsTermination(3) | MaxRestartsTermination(5)
        r = repr(cond)
        assert "MaxRestartsTermination" in r
        assert "|" in r

    def test_composite_reset(self):
        """Reset should propagate to all child conditions."""
        reset_called = []

        class TrackingCondition(TerminationCondition):
            def __init__(self, name):
                self.name = name

            def should_terminate(self, context):
                return False

            def reset(self):
                reset_called.append(self.name)

        cond = TrackingCondition("a") | TrackingCondition("b")
        cond.reset()

        assert "a" in reset_called
        assert "b" in reset_called


# =============================================================================
# Test 9: Full trajectory inspection - message ordering
# =============================================================================


class TestTrajectory:
    """Inspect the full message trajectory via client.received_messages."""

    @pytest.mark.asyncio
    async def test_start_hook_before_llm_in_messages(self):
        """Hook message should be sent to LLM before the first call."""
        client = MockClient()
        client.set_responses([
            AssistantMessage(content="Response", source="mock"),
        ])
        agent = make_agent(client, start_hooks=[PlanningHook(instruction="PLAN FIRST")])

        await agent.run("Task")

        # The first LLM call should contain: SystemMessage, UserMessage("Task"), UserMessage(hook)
        first_call = client.received_messages[0]

        # Find positions
        user_task_idx = None
        hook_idx = None
        for i, m in enumerate(first_call):
            if isinstance(m, UserMessage) and m.content == "Task":
                user_task_idx = i
            if isinstance(m, UserMessage) and getattr(m, "source", "") == "hook":
                hook_idx = i

        assert user_task_idx is not None, "User task message not found"
        assert hook_idx is not None, "Hook message not found"
        # Hook should come after the user task (appended to end of messages)
        assert hook_idx > user_task_idx

    @pytest.mark.asyncio
    async def test_end_hook_resume_visible_in_subsequent_call(self):
        """After end hook resume, the injected message should be in next LLM call."""
        incomplete = [
            {"content": "Task", "status": "pending", "activeForm": "Doing"},
        ]

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="First attempt", source="mock"),
            AssistantMessage(content="After resume", source="mock"),
        ])
        agent = make_agent(client, end_hooks=[CompletionCheckHook(max_restarts=1)])

        with patch(LOAD_TODOS_PATCH, return_value=incomplete):
            result = await agent.run("Work")

        assert client.call_count == 2

        # Second call should contain the resume hook message
        second_call = client.received_messages[1]
        hook_msgs = [m for m in second_call if getattr(m, "source", "") == "hook"]
        assert len(hook_msgs) >= 1
        assert "Task" in hook_msgs[-1].content

    @pytest.mark.asyncio
    async def test_start_and_end_hooks_together(self):
        """Start hooks and end hooks should both work in the same agent run."""
        incomplete = [
            {"content": "Task A", "status": "pending", "activeForm": "Doing A"},
        ]

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="Starting", source="mock"),
            AssistantMessage(content="Resumed", source="mock"),
            AssistantMessage(content="Final", source="mock"),
        ])
        agent = make_agent(
            client,
            start_hooks=[PlanningHook(instruction="CREATE PLAN")],
            end_hooks=[CompletionCheckHook(max_restarts=2)],
        )

        with patch(LOAD_TODOS_PATCH, return_value=incomplete):
            result = await agent.run("Build something")

        # 1 initial + 2 restarts = 3
        assert client.call_count == 3

        # First call should have the start hook
        first_call = client.received_messages[0]
        start_hooks = [m for m in first_call if getattr(m, "source", "") == "hook"]
        assert len(start_hooks) == 1
        assert start_hooks[0].content == "CREATE PLAN"

        # Subsequent calls should have end hook resume messages
        second_call = client.received_messages[1]
        end_hooks = [m for m in second_call if getattr(m, "source", "") == "hook"]
        # Should have the start hook message + 1 resume message
        assert len(end_hooks) >= 2

    @pytest.mark.asyncio
    async def test_tool_calls_with_end_hook(self):
        """End hooks should only trigger when there are NO tool calls."""
        incomplete = [
            {"content": "Use tool", "status": "pending", "activeForm": "Using tool"},
        ]
        tool = MockTool()

        client = MockClient()
        client.set_responses([
            tool_call_response("mock_tool", {"input": "data"}),
            AssistantMessage(content="Done with tool", source="mock"),
            AssistantMessage(content="Really done", source="mock"),
        ])
        agent = make_agent(
            client,
            tools=[tool],
            end_hooks=[CompletionCheckHook(max_restarts=1)],
        )

        with patch(LOAD_TODOS_PATCH, return_value=incomplete):
            result = await agent.run("Use tool then stop")

        assert tool.call_count == 1
        assert client.call_count == 3

    @pytest.mark.asyncio
    async def test_message_accumulation_across_restarts(self):
        """Messages should accumulate across restarts, not reset."""
        incomplete = [
            {"content": "Work", "status": "pending", "activeForm": "Working"},
        ]

        client = MockClient()
        client.set_responses([
            AssistantMessage(content="First", source="mock"),
            AssistantMessage(content="Second", source="mock"),
            AssistantMessage(content="Third", source="mock"),
        ])
        agent = make_agent(client, end_hooks=[CompletionCheckHook(max_restarts=2)])

        with patch(LOAD_TODOS_PATCH, return_value=incomplete):
            result = await agent.run("Go")

        assert client.call_count == 3

        # Each subsequent call should have MORE messages (accumulated)
        len1 = len(client.received_messages[0])
        len2 = len(client.received_messages[1])
        len3 = len(client.received_messages[2])

        assert len2 > len1, "Second call should have more messages than first"
        assert len3 > len2, "Third call should have more messages than second"


# =============================================================================
# Test: LoopContext unit tests
# =============================================================================


class TestLoopContext:
    """Test LoopContext dataclass behavior."""

    def test_default_values(self):
        ctx = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test",
        )
        assert ctx.iteration == 0
        assert ctx.restart_count == 0
        assert ctx.metadata == {}

    def test_metadata_isolation(self):
        """Each LoopContext should have its own metadata dict."""
        ctx1 = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test1",
        )
        ctx2 = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test2",
        )

        ctx1.metadata["key"] = "value1"
        assert "key" not in ctx2.metadata

    def test_hook_repr(self):
        """Hooks should have readable repr."""
        assert "PlanningHook" in repr(PlanningHook())
        assert "CompletionCheckHook" in repr(CompletionCheckHook(max_restarts=3))
        assert "3" in repr(MaxRestartsTermination(3))


# =============================================================================
# Test: CompletionCheckHook directly (unit test)
# =============================================================================


class TestCompletionCheckHookUnit:
    """Unit test CompletionCheckHook.on_end() directly."""

    @pytest.mark.asyncio
    async def test_returns_none_when_terminated(self):
        """Should return None if termination condition says stop."""
        hook = CompletionCheckHook(max_restarts=0)
        ctx = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test",
            restart_count=0,
        )

        with patch(LOAD_TODOS_PATCH, return_value=[{"status": "pending", "content": "X"}]):
            result = await hook.on_end(ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_resume_message_with_stats(self):
        """Resume message should include completion stats."""
        hook = CompletionCheckHook(max_restarts=5)
        ctx = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test",
            restart_count=0,
        )

        todos = [
            {"content": "A", "status": "completed"},
            {"content": "B", "status": "pending"},
            {"content": "C", "status": "in_progress"},
        ]

        with patch(LOAD_TODOS_PATCH, return_value=todos):
            result = await hook.on_end(ctx)

        assert result is not None
        assert "1/3 completed" in result
        assert "B" in result
        assert "C" in result

    @pytest.mark.asyncio
    async def test_custom_termination_condition(self):
        """CompletionCheckHook should use custom termination condition."""

        class AlwaysTerminate(TerminationCondition):
            def should_terminate(self, context):
                return True

        hook = CompletionCheckHook(termination=AlwaysTerminate())
        ctx = LoopContext(
            agent_context=AgentContext(),
            llm_messages=[],
            agent_name="test",
            restart_count=0,
        )

        with patch(LOAD_TODOS_PATCH, return_value=[{"status": "pending", "content": "X"}]):
            result = await hook.on_end(ctx)

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
