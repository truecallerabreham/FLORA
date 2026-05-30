"""
Tests for tool approval functionality.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

import pytest
from pydantic import BaseModel

from forla.agents import Agent
from forla.context import AgentContext, ToolApprovalRequest, ToolApprovalResponse
from forla.llm import BaseChatCompletionClient
from forla.messages import AssistantMessage, Message, ToolCallRequest, ToolMessage
from forla.tools import ApprovalMode, FunctionTool, tool
from forla.types import (
    AgentEvent,
    AgentResponse,
    ChatCompletionResult,
    ToolApprovalEvent,
    Usage,
)


class MockChatClientWithTools(BaseChatCompletionClient):
    """Mock chat client that returns tool calls."""

    def __init__(self, model: str = "test-model"):
        super().__init__(model=model)
        self.tool_calls_to_return: List[ToolCallRequest] = []
        self.call_count = 0

    def set_tool_calls(self, tool_calls: List[ToolCallRequest]):
        """Set tool calls to return."""
        self.tool_calls_to_return = tool_calls

    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        """Return mock response with tool calls."""
        self.call_count += 1

        # First call returns tool calls, subsequent calls return text
        if self.call_count == 1 and self.tool_calls_to_return:
            response = AssistantMessage(
                content="I'll help with that.",
                tool_calls=self.tool_calls_to_return,
                source="mock",
            )
        else:
            response = AssistantMessage(
                content="Task completed successfully.",
                source="mock",
            )

        return ChatCompletionResult(
            message=response,
            usage=Usage(
                duration_ms=100,
                llm_calls=1,
                tokens_input=50,
                tokens_output=25,
            ),
            model=self.model,
            finish_reason="stop" if not response.tool_calls else "tool_calls",
        )

    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Mock streaming - just yield final result."""
        result = await self.create(messages, tools, output_format, **kwargs)
        from forla.types import ChatCompletionChunk

        yield ChatCompletionChunk(
            content=result.message.content or "",
            is_complete=True,
            tool_call_chunk=None,
        )


@pytest.mark.asyncio
async def test_tool_without_approval():
    """Test that tools without approval execute immediately."""
    # Create a tool without approval
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"

    # Setup mock client
    client = MockChatClientWithTools()
    client.set_tool_calls(
        [
            ToolCallRequest(
                call_id="call_1",
                tool_name="get_weather",
                parameters={"city": "Seattle"},
            )
        ]
    )

    # Create agent with tool
    agent = Agent(
        name="test_agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[get_weather],
    )

    # Run agent
    response = await agent.run("What's the weather in Seattle?")

    # Tool should have executed
    assert len(response.messages) > 0
    assert not response.needs_approval
    assert response.finish_reason != "approval_needed"

    # Check for tool message
    tool_messages = [m for m in response.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Sunny in Seattle" in tool_messages[0].content


@pytest.mark.asyncio
async def test_tool_with_approval_required():
    """Test that tools with approval mode=always_require pause for approval."""
    # Create a tool that requires approval
    @tool(approval_mode="always_require")
    def delete_file(path: str) -> str:
        """Delete a file."""
        return f"Deleted {path}"

    # Setup mock client
    client = MockChatClientWithTools()
    client.set_tool_calls(
        [
            ToolCallRequest(
                call_id="call_1",
                tool_name="delete_file",
                parameters={"path": "/tmp/test.txt"},
            )
        ]
    )

    # Create agent with approval-required tool
    agent = Agent(
        name="test_agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[delete_file],
    )

    # Run agent - should pause for approval
    response = await agent.run("Delete /tmp/test.txt")

    # Should need approval
    assert response.needs_approval
    assert response.finish_reason == "approval_needed"
    assert len(response.approval_requests) == 1

    # Check approval request details
    approval_req = response.approval_requests[0]
    assert approval_req.tool_name == "delete_file"
    assert approval_req.parameters == {"path": "/tmp/test.txt"}
    assert approval_req.tool_call_id == "call_1"

    # Tool should NOT have executed yet
    tool_messages = [m for m in response.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 0


@pytest.mark.asyncio
async def test_approval_flow_approved():
    """Test full approval flow when user approves."""
    # Create tool requiring approval
    @tool(approval_mode="always_require")
    def delete_file(path: str) -> str:
        """Delete a file."""
        return f"Deleted {path}"

    # Setup mock client
    client = MockChatClientWithTools()
    client.set_tool_calls(
        [
            ToolCallRequest(
                call_id="call_1",
                tool_name="delete_file",
                parameters={"path": "/tmp/test.txt"},
            )
        ]
    )

    agent = Agent(
        name="test_agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[delete_file],
    )

    # First run - get approval request
    response = await agent.run("Delete /tmp/test.txt")
    assert response.needs_approval

    # Approve the request
    approval_req = response.approval_requests[0]
    approval_response = approval_req.create_response(approved=True)
    response.context.add_approval_response(approval_response)

    # Continue execution with approval
    response = await agent.run(context=response.context)

    # Should no longer need approval
    assert not response.needs_approval
    assert response.finish_reason != "approval_needed"

    # Tool should have executed
    tool_messages = [m for m in response.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Deleted /tmp/test.txt" in tool_messages[0].content
    assert tool_messages[0].success


@pytest.mark.asyncio
async def test_approval_flow_rejected():
    """Test approval flow when user rejects."""
    # Create tool requiring approval
    @tool(approval_mode="always_require")
    def delete_file(path: str) -> str:
        """Delete a file."""
        return f"Deleted {path}"

    # Setup mock client
    client = MockChatClientWithTools()
    client.set_tool_calls(
        [
            ToolCallRequest(
                call_id="call_1",
                tool_name="delete_file",
                parameters={"path": "/tmp/test.txt"},
            )
        ]
    )

    agent = Agent(
        name="test_agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[delete_file],
    )

    # First run - get approval request
    response = await agent.run("Delete /tmp/test.txt")
    assert response.needs_approval

    # Reject the request
    approval_req = response.approval_requests[0]
    approval_response = approval_req.create_response(approved=False)
    response.context.add_approval_response(approval_response)

    # Continue execution with rejection
    response = await agent.run(context=response.context)

    # Should no longer need approval
    assert not response.needs_approval

    # Tool should NOT have executed successfully
    tool_messages = [m for m in response.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert not tool_messages[0].success
    assert "denied" in tool_messages[0].content.lower() or "rejected" in tool_messages[0].content.lower()


@pytest.mark.asyncio
async def test_mixed_approval_tools():
    """Test agent with both approval-required and no-approval tools."""
    # Create tools with different approval modes
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"

    @tool(approval_mode="always_require")
    def delete_file(path: str) -> str:
        """Delete a file."""
        return f"Deleted {path}"

    # Setup mock client to call both tools
    client = MockChatClientWithTools()
    client.set_tool_calls(
        [
            ToolCallRequest(
                call_id="call_1",
                tool_name="get_weather",
                parameters={"city": "Seattle"},
            ),
            ToolCallRequest(
                call_id="call_2",
                tool_name="delete_file",
                parameters={"path": "/tmp/test.txt"},
            ),
        ]
    )

    agent = Agent(
        name="test_agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[get_weather, delete_file],
    )

    # Run agent
    response = await agent.run("Check weather in Seattle and delete /tmp/test.txt")

    # Should need approval for delete_file but not get_weather
    assert response.needs_approval
    assert len(response.approval_requests) == 1
    assert response.approval_requests[0].tool_name == "delete_file"

    # get_weather should have executed
    tool_messages = [m for m in response.messages if isinstance(m, ToolMessage)]
    weather_msgs = [m for m in tool_messages if "Sunny" in m.content]
    assert len(weather_msgs) == 1

    # delete_file should NOT have executed yet
    delete_msgs = [m for m in tool_messages if "Deleted" in m.content]
    assert len(delete_msgs) == 0


@pytest.mark.asyncio
async def test_approval_event_emission():
    """Test that ToolApprovalEvent is emitted when approval is needed."""
    @tool(approval_mode="always_require")
    def delete_file(path: str) -> str:
        """Delete a file."""
        return f"Deleted {path}"

    client = MockChatClientWithTools()
    client.set_tool_calls(
        [
            ToolCallRequest(
                call_id="call_1",
                tool_name="delete_file",
                parameters={"path": "/tmp/test.txt"},
            )
        ]
    )

    agent = Agent(
        name="test_agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[delete_file],
    )

    # Collect events during streaming
    events = []
    response = None
    async for item in agent.run_stream("Delete /tmp/test.txt"):
        # Check if this is the final response
        if isinstance(item, AgentResponse):
            response = item
        else:
            # This is an event
            events.append(item)

    # Should have emitted ToolApprovalEvent
    approval_events = [e for e in events if isinstance(e, ToolApprovalEvent)]
    assert len(approval_events) == 1
    assert approval_events[0].approval_request.tool_name == "delete_file"


@pytest.mark.asyncio
async def test_context_approval_state_management():
    """Test AgentContext approval state management methods."""
    context = AgentContext()

    # Create a tool call request
    tool_call = ToolCallRequest(
        call_id="call_1",
        tool_name="delete_file",
        parameters={"path": "/tmp/test.txt"},
    )

    # Add approval request
    approval_req = context.add_approval_request(tool_call, "delete_file")
    assert context.waiting_for_approval
    assert len(context.pending_approval_requests) == 1
    assert context.pending_tool_calls["call_1"] == tool_call

    # Add approval response (approved)
    approval_resp = approval_req.create_response(approved=True)
    context.add_approval_response(approval_resp)

    # Should no longer be waiting
    assert not context.waiting_for_approval
    assert len(context.pending_approval_requests) == 0

    # Get approved tool calls
    approved = context.get_approved_tool_calls()
    assert len(approved) == 1
    assert approved[0] == tool_call

    # After getting approved calls, should be cleared
    assert len(context.approval_responses) == 0
    assert len(context.pending_tool_calls) == 0


@pytest.mark.asyncio
async def test_context_rejection_handling():
    """Test context handles rejected approvals correctly."""
    context = AgentContext()

    # Create tool call
    tool_call = ToolCallRequest(
        call_id="call_1",
        tool_name="delete_file",
        parameters={"path": "/tmp/test.txt"},
    )

    # Add approval request
    approval_req = context.add_approval_request(tool_call, "delete_file")

    # Reject it
    approval_resp = approval_req.create_response(approved=False)
    context.add_approval_response(approval_resp)

    # Should have no approved calls
    approved = context.get_approved_tool_calls()
    assert len(approved) == 0

    # Should have rejected calls
    rejected = context.get_rejected_tool_calls()
    assert len(rejected) == 1
    assert rejected[0][0] == "call_1"
    assert rejected[0][1] == tool_call


@pytest.mark.asyncio
async def test_backward_compatibility():
    """Test that AgentResponse.messages still works for backward compatibility."""
    context = AgentContext()
    context.add_message(AssistantMessage(content="Hello", source="test"))

    response = AgentResponse(
        context=context,
        source="test_agent",
        usage=Usage(duration_ms=100),
        finish_reason="stop",
    )

    # Should be able to access messages through response
    assert len(response.messages) == 1
    assert response.messages[0].content == "Hello"
    assert response.final_content == "Hello"


@pytest.mark.asyncio
async def test_approval_with_function_tool():
    """Test approval with FunctionTool class directly."""
    # Create function tool with approval
    def dangerous_op(data: str) -> str:
        """Perform a dangerous operation."""
        return f"Executed: {data}"

    tool = FunctionTool(
        func=dangerous_op,
        name="dangerous_op",
        description="A dangerous operation",
        approval_mode=ApprovalMode.ALWAYS,
    )

    # Verify approval mode is set
    assert tool.approval_mode == ApprovalMode.ALWAYS

    # Setup mock client
    client = MockChatClientWithTools()
    client.set_tool_calls(
        [
            ToolCallRequest(
                call_id="call_1",
                tool_name="dangerous_op",
                parameters={"data": "important"},
            )
        ]
    )

    agent = Agent(
        name="test_agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[tool],
    )

    # Should require approval
    response = await agent.run("Execute dangerous operation")
    assert response.needs_approval
    assert response.approval_requests[0].tool_name == "dangerous_op"