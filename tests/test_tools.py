import pytest
from forla.tools import BaseTool, ToolResult, FunctionTool, ThinkTool


def test_function_tool_schema_generation():
    """Verify that FunctionTool correctly generates JSON schema from a function."""
    
    def get_weather(location: str, units: str = "celsius") -> str:
        """Get current weather for a city."""
        return f"Weather in {location}: sunny"
    
    tool = FunctionTool(get_weather)
    
    assert tool.name == "get_weather"
    assert "city" in tool.description   # Uses the actual docstring
    assert "Get current weather" in tool.description
    
    schema = tool.parameters
    assert "location" in schema["properties"]
    assert schema["properties"]["location"]["type"] == "string"
    # 'location' has no default, so it should be required
    assert "location" in schema["required"]
    # 'units' has a default, so it should NOT be required
    assert "units" not in schema["required"]


@pytest.mark.asyncio
async def test_function_tool_execution():
    """Verify that FunctionTool correctly executes a function."""
    
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    tool = FunctionTool(add_numbers)
    result = await tool.execute({"a": 3, "b": 7})
    
    assert result.success is True
    assert result.result == 10


@pytest.mark.asyncio
async def test_function_tool_error_handling():
    """Verify that errors are captured gracefully."""
    
    def failing_tool(value: str) -> str:
        """A tool that always fails."""
        raise ValueError("This tool always fails!")
    
    tool = FunctionTool(failing_tool)
    result = await tool.execute({"value": "test"})
    
    assert result.success is False
    assert "always fails" in result.error


@pytest.mark.asyncio
async def test_think_tool():
    """ThinkTool should accept any thought and return success."""
    tool = ThinkTool()
    result = await tool.execute({"thought": "I should check the weather first."})
    
    assert result.success is True
    assert "thought" in tool.to_llm_format()["function"]["name"] or tool.name == "think"


def test_tool_to_llm_format():
    """Verify tools produce the correct JSON format for the LLM API."""
    tool = ThinkTool()
    llm_format = tool.to_llm_format()
    
    assert llm_format["type"] == "function"
    assert "function" in llm_format
    assert "name" in llm_format["function"]
    assert "description" in llm_format["function"]
    assert "parameters" in llm_format["function"]
