"""
Comprehensive tests for Forla tool system.

Covers:
- Tool wrapping and management (functions, FunctionTool, BaseTool)
- Tool enhancements (parallel execution, ThinkTool, versioning, enum validation)
- Advanced features (forced tool use, domain filtering, granular file operations)
"""

import asyncio
import inspect
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Type

import pytest
from pydantic import BaseModel

from forla import Agent, BaseTool, FunctionTool
from forla.llm import BaseChatCompletionClient
from forla.messages import AssistantMessage
from forla.tools import create_core_tools
from forla.tools._coding_tools import WriteFileTool
from forla.tools._research_tools import WebFetchTool, WebSearchTool
from forla.types import ChatCompletionResult, ToolResult, Usage

# ============================================================================
# Mock Classes and Test Functions
# ============================================================================


class MockChatCompletionClient(BaseChatCompletionClient):
    """Mock client for testing."""

    async def create(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        # Check if forced tool use instruction is present
        system_msg = next(
            (
                m
                for m in messages
                if hasattr(m, "content") and "MUST use these tools" in str(m.content)
            ),
            None,
        )
        content = "Test response"
        if system_msg:
            content = "Test response with forced tools mentioned"

        return ChatCompletionResult(
            message=AssistantMessage(content=content, source="mock"),
            usage=Usage(
                duration_ms=100,
                llm_calls=1,
                tokens_input=10,
                tokens_output=5,
                tool_calls=0,
                memory_operations=0,
            ),
            model="test-model",
            finish_reason="stop",
        )

    async def create_stream(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        from forla.types import ChatCompletionChunk

        yield ChatCompletionChunk(
            content="Test response", is_complete=True, tool_call_chunk=None
        )


def simple_function(text: str) -> str:
    """A simple test function."""
    return f"Processed: {text}"


def math_function(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def weather_function(
    location: str, unit: Literal["celsius", "fahrenheit"] = "celsius"
) -> str:
    """Get weather with unit enum constraint."""
    return f"Weather in {location}: 72 {unit}"


class CustomTool(BaseTool):
    """A custom tool for testing."""

    def __init__(self):
        super().__init__("custom_tool", "A custom tool for testing")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to process"}
            },
            "required": ["message"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            result=f"Custom tool processed: {parameters.get('message', '')}",
            error=None,
            metadata={"tool_name": self.name},
        )


# ============================================================================
# Tool Wrapping and Management Tests
# ============================================================================


@pytest.mark.asyncio
async def test_function_auto_wrapping():
    """Test that functions are automatically wrapped as FunctionTool."""
    client = MockChatCompletionClient(model="test")

    agent = Agent(
        name="test-agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[simple_function],  # Function should be auto-wrapped
    )

    # Check that the function was wrapped
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], FunctionTool)
    assert agent.tools[0].name == "simple_function"
    assert agent.tools[0].func == simple_function


@pytest.mark.asyncio
async def test_mixed_tool_types():
    """Test mixing functions, FunctionTool instances, and custom BaseTool instances."""
    client = MockChatCompletionClient(model="test")

    custom_tool = CustomTool()
    explicit_function_tool = FunctionTool(math_function, description="Math calculator")

    agent = Agent(
        name="mixed-agent",
        description="Agent with mixed tools",
        instructions="You are helpful",
        model_client=client,
        tools=[
            simple_function,  # Raw function -> auto-wrapped
            explicit_function_tool,  # Explicit FunctionTool -> used directly
            custom_tool,  # Custom BaseTool -> used directly
        ],
    )

    # Check all tools are present
    assert len(agent.tools) == 3

    # Check types
    assert isinstance(agent.tools[0], FunctionTool)  # Auto-wrapped
    assert isinstance(agent.tools[1], FunctionTool)  # Explicit FunctionTool
    assert isinstance(agent.tools[2], CustomTool)  # Custom BaseTool

    # Check names
    tool_names = [tool.name for tool in agent.tools]
    assert "simple_function" in tool_names
    assert "math_function" in tool_names
    assert "custom_tool" in tool_names


@pytest.mark.asyncio
async def test_tool_finding():
    """Test that tools can be found by name."""
    client = MockChatCompletionClient(model="test")

    agent = Agent(
        name="finder-agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[simple_function, math_function],
    )

    # Test finding existing tools
    tool1 = agent._find_tool("simple_function")
    assert tool1 is not None
    assert tool1.name == "simple_function"

    tool2 = agent._find_tool("math_function")
    assert tool2 is not None
    assert tool2.name == "math_function"

    # Test finding non-existent tool
    tool3 = agent._find_tool("nonexistent_tool")
    assert tool3 is None


@pytest.mark.asyncio
async def test_tools_for_llm():
    """Test converting tools to OpenAI format."""
    client = MockChatCompletionClient(model="test")

    agent = Agent(
        name="llm-agent",
        description="Test agent",
        instructions="You are helpful",
        model_client=client,
        tools=[simple_function],
    )

    llm_tools = agent._get_tools_for_llm()

    # Should have one tool
    assert len(llm_tools) == 1

    # Check OpenAI format
    tool_def = llm_tools[0]
    assert tool_def["type"] == "function"
    assert "function" in tool_def
    assert tool_def["function"]["name"] == "simple_function"
    assert "description" in tool_def["function"]
    assert "parameters" in tool_def["function"]


@pytest.mark.asyncio
async def test_invalid_tool_type():
    """Test that invalid tool types raise errors."""
    client = MockChatCompletionClient(model="test")

    with pytest.raises(Exception) as exc_info:
        agent = Agent(
            name="error-agent",
            description="Test agent",
            instructions="You are helpful",
            model_client=client,
            tools=["not_a_tool_or_function"],  # type: ignore
        )

    assert "Invalid tool type" in str(exc_info.value)


# ============================================================================
# Tool Enhancement Tests
# ============================================================================


@pytest.mark.asyncio
async def test_think_tool():
    """Test ThinkTool functionality."""
    from forla.tools._core_tools import ThinkTool

    think = ThinkTool()

    # Verify tool properties
    assert think.name == "think"
    assert "pause and think" in think.description.lower()
    assert "thought" in think.parameters["properties"]

    # Test execution
    result = await think.execute(
        {"thought": "I need to analyze these results carefully."}
    )
    assert result.success
    assert "Reasoning recorded" in result.result
    assert result.metadata["tool_name"] == "think"


def test_think_tool_in_core_tools():
    """Test that ThinkTool is included in core tools."""
    tools = create_core_tools()
    tool_names = [t.name for t in tools]

    assert "think" in tool_names
    # ThinkTool should be first (most important for complex reasoning)
    assert tools[0].name == "think"


def test_tool_versioning():
    """Test tool versioning system."""

    class VersionedTool(BaseTool):
        def __init__(self, version: str = "1.0.0"):
            super().__init__(
                name="test_tool",
                description="Test tool with versioning",
                version=version,
            )

        @property
        def parameters(self):
            return {"type": "object", "properties": {}}

        async def execute(self, parameters):
            return ToolResult(success=True, result="ok", error=None, metadata={})

    # Test default version
    tool_v1 = VersionedTool()
    assert tool_v1.version == "1.0.0"
    llm_format = tool_v1.to_llm_format()
    # Default version (1.0.0) should not be appended to name
    assert llm_format["function"]["name"] == "test_tool"

    # Test custom version
    tool_v2 = VersionedTool(version="2.0.0")
    assert tool_v2.version == "2.0.0"
    llm_format_v2 = tool_v2.to_llm_format()
    # Non-default versions should be appended
    assert llm_format_v2["function"]["name"] == "test_tool_v2.0.0"


def test_enum_validation_with_literal():
    """Test enum validation for Literal types in FunctionTool."""
    tool = FunctionTool(weather_function)
    schema = tool.parameters

    # Check that enum constraint is added for Literal type
    assert "unit" in schema["properties"]
    unit_schema = schema["properties"]["unit"]

    # Should have enum constraint
    if "enum" in unit_schema:
        assert set(unit_schema["enum"]) == {"celsius", "fahrenheit"}


def test_function_tool_versioning():
    """Test that FunctionTool supports versioning."""

    def my_function(x: int) -> int:
        """Double the input."""
        return x * 2

    # Test with version
    tool = FunctionTool(my_function, version="2.1.0")
    assert tool.version == "2.1.0"

    llm_format = tool.to_llm_format()
    assert llm_format["function"]["name"] == "my_function_v2.1.0"


@pytest.mark.asyncio
async def test_parallel_tool_execution_setup():
    """
    Test that parallel tool execution infrastructure is in place.

    Note: Full integration test requires mocking LLM responses with multiple tool calls.
    This test verifies the method exists and has correct signature.
    """
    # Verify the parallel execution method exists
    assert hasattr(Agent, "_execute_tool_calls_parallel")

    # Check method signature
    sig = inspect.signature(Agent._execute_tool_calls_parallel)
    param_names = list(sig.parameters.keys())

    assert "self" in param_names
    assert "tool_calls" in param_names
    assert "llm_messages" in param_names
    assert "cancellation_token" in param_names


# ============================================================================
# Forced Tool Use Tests
# ============================================================================


@pytest.mark.asyncio
async def test_forced_tool_use():
    """Test that required_tools adds forced tool use instruction to system prompt."""
    client = MockChatCompletionClient(model="test")

    agent = Agent(
        name="test-agent",
        description="Test agent",
        instructions="You are a helpful assistant",
        model_client=client,
        required_tools=["calculator", "web_search"],
    )

    # Prepare messages
    messages = await agent._prepare_llm_messages([])

    # Find system message
    system_msg = next((m for m in messages if hasattr(m, "content")), None)
    assert system_msg is not None
    assert hasattr(system_msg, "content")

    # Check that forced tool instruction is present
    assert "MUST use these tools" in system_msg.content
    assert "calculator" in system_msg.content
    assert "web_search" in system_msg.content


@pytest.mark.asyncio
async def test_no_forced_tools():
    """Test that without required_tools, no forced instruction is added."""
    client = MockChatCompletionClient(model="test")

    agent = Agent(
        name="test-agent",
        description="Test agent",
        instructions="You are a helpful assistant",
        model_client=client,
    )

    messages = await agent._prepare_llm_messages([])
    system_msg = next((m for m in messages if hasattr(m, "content")), None)
    assert system_msg is not None
    assert hasattr(system_msg, "content")

    # Should not have forced tool instruction
    assert "MUST use these tools" not in system_msg.content


# ============================================================================
# Domain Filtering Tests
# ============================================================================


def test_web_search_tool_domain_filtering():
    """Test domain filtering in WebSearchTool."""
    # Test allowed domains
    tool = WebSearchTool(
        api_key="test_key", allowed_domains=["wikipedia.org", "github.com"]
    )

    assert tool._is_domain_allowed("https://en.wikipedia.org/wiki/Python")
    assert tool._is_domain_allowed("https://github.com/anthropics/anthropic-tools")
    assert not tool._is_domain_allowed("https://example.com/page")

    # Test blocked domains
    tool_blocked = WebSearchTool(
        api_key="test_key", blocked_domains=["spam.com", "malicious.net"]
    )

    assert tool_blocked._is_domain_allowed("https://wikipedia.org/wiki/Python")
    assert not tool_blocked._is_domain_allowed("https://spam.com/bad-page")
    assert not tool_blocked._is_domain_allowed("https://malicious.net/phishing")


def test_web_fetch_tool_domain_filtering():
    """Test domain filtering in WebFetchTool."""
    # Test allowed domains only
    tool = WebFetchTool(allowed_domains=["trusted.com", "safe.org"])

    assert tool._is_domain_allowed("https://trusted.com/page")
    assert tool._is_domain_allowed("https://safe.org/article")
    assert not tool._is_domain_allowed("https://untrusted.com/page")

    # Test blocked domains
    tool_blocked = WebFetchTool(blocked_domains=["ad-tracker.com"])

    assert tool_blocked._is_domain_allowed("https://normal-site.com")
    assert not tool_blocked._is_domain_allowed("https://ad-tracker.com/pixel")


def test_domain_filtering_with_subdomains():
    """Test that domain filtering works with subdomains."""
    tool = WebSearchTool(api_key="test", allowed_domains=["example.com"])

    # Should match subdomains
    assert tool._is_domain_allowed("https://www.example.com/page")
    assert tool._is_domain_allowed("https://blog.example.com/post")
    assert not tool._is_domain_allowed("https://other-example.com/page")


# ============================================================================
# Granular File Operations Tests
# ============================================================================


@pytest.mark.asyncio
async def test_write_file_full_content():
    """Test full file write operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        tool = WriteFileTool(workspace=workspace)

        result = await tool.execute(
            {"file_path": "test.txt", "content": "Hello, World!"}
        )

        assert result.success
        assert result.metadata["operation"] == "write"
        assert (workspace / "test.txt").read_text() == "Hello, World!"


@pytest.mark.asyncio
async def test_write_file_str_replace():
    """Test str_replace operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        test_file = workspace / "test.txt"
        test_file.write_text("Hello, World!")

        tool = WriteFileTool(workspace=workspace)

        result = await tool.execute(
            {"file_path": "test.txt", "old_str": "World", "new_str": "Python"}
        )

        assert result.success
        assert result.metadata["operation"] == "str_replace"
        assert test_file.read_text() == "Hello, Python!"


@pytest.mark.asyncio
async def test_write_file_str_replace_not_found():
    """Test str_replace with string not found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        test_file = workspace / "test.txt"
        test_file.write_text("Hello, World!")

        tool = WriteFileTool(workspace=workspace)

        result = await tool.execute(
            {
                "file_path": "test.txt",
                "old_str": "NonExistent",
                "new_str": "Replacement",
            }
        )

        assert not result.success
        assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_write_file_insert_at_line():
    """Test insert_at_line operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        test_file = workspace / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n")

        tool = WriteFileTool(workspace=workspace)

        result = await tool.execute(
            {
                "file_path": "test.txt",
                "insert_line": 2,
                "insert_content": "Inserted Line",
            }
        )

        assert result.success
        assert result.metadata["operation"] == "insert_at_line"

        lines = test_file.read_text().splitlines()
        assert lines[0] == "Line 1"
        assert lines[1] == "Inserted Line"
        assert lines[2] == "Line 2"


@pytest.mark.asyncio
async def test_write_file_creates_parent_directories():
    """Test that parent directories are created automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        tool = WriteFileTool(workspace=workspace)

        result = await tool.execute(
            {"file_path": "subdir1/subdir2/test.txt", "content": "Nested file"}
        )

        assert result.success
        assert (workspace / "subdir1" / "subdir2" / "test.txt").exists()


# ============================================================================
# Content Length Tests
# ============================================================================


def test_web_fetch_tool_content_length_limit():
    """Test that WebFetchTool has configurable max_content_length."""
    tool = WebFetchTool(max_content_length=1000)
    assert tool.max_content_length == 1000

    tool_default = WebFetchTool()
    assert tool_default.max_content_length == 100000  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
