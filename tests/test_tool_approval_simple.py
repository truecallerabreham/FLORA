"""
Simple test for tool approval functionality to verify basic setup.
"""

import pytest

from forla.context import AgentContext, ToolApprovalRequest, ToolApprovalResponse
from forla.messages import ToolCallRequest
from forla.tools import ApprovalMode, FunctionTool, tool
from forla.types import AgentResponse, Usage


def test_imports():
    """Test that all necessary imports work."""
    # This should not raise any import errors
    assert AgentContext is not None
    assert ToolApprovalRequest is not None
    assert ToolApprovalResponse is not None
    assert ApprovalMode is not None
    assert FunctionTool is not None
    assert tool is not None


def test_tool_decorator():
    """Test the tool decorator works with approval mode."""

    @tool
    def simple_tool(x: int) -> int:
        """Simple tool."""
        return x * 2

    assert simple_tool.name == "simple_tool"
    assert simple_tool.approval_mode == ApprovalMode.NEVER

    @tool(approval_mode="always_require")
    def approval_tool(x: int) -> int:
        """Tool that needs approval."""
        return x * 3

    assert approval_tool.name == "approval_tool"
    assert approval_tool.approval_mode == ApprovalMode.ALWAYS


def test_context_approval_methods():
    """Test context approval management."""
    context = AgentContext()

    # Create a tool call
    tool_call = ToolCallRequest(
        call_id="test_1", tool_name="test_tool", parameters={"x": 1}
    )

    # Add approval request
    approval_req = context.add_approval_request(tool_call, "test_tool")
    assert context.waiting_for_approval
    assert len(context.pending_approval_requests) == 1

    # Add approval response
    approval_resp = approval_req.create_response(approved=True)
    context.add_approval_response(approval_resp)
    assert not context.waiting_for_approval

    # Get approved calls
    approved = context.get_approved_tool_calls()
    assert len(approved) == 1
    assert approved[0] == tool_call


def test_agent_response_with_context():
    """Test AgentResponse with context works."""
    context = AgentContext()

    response = AgentResponse(
        context=context,
        source="test",
        usage=Usage(duration_ms=100),
        finish_reason="stop",
    )

    # Test properties work
    assert response.messages == []
    assert not response.needs_approval
    assert response.approval_requests == []


if __name__ == "__main__":
    test_imports()
    test_tool_decorator()
    test_context_approval_methods()
    test_agent_response_with_context()
    print("All tests passed!")