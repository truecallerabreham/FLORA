"""Minimal tests for Forla WebUI discovery functionality."""

import types
from typing import Any
from unittest.mock import Mock

import pytest

from forla.webui._discovery import ForlaScanner


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "TestAgent") -> None:
        self.name = name
        self.description = f"Test agent: {name}"
        self.tools = []
        self.model_client = Mock()

    async def run(self, messages: Any) -> Any:
        """Mock run method."""
        return Mock(messages=[])

    async def run_stream(self, messages: Any, verbose: bool = False) -> Any:
        """Mock run_stream method."""
        yield Mock()


class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self, name: str = "TestOrchestrator") -> None:
        self.name = name
        self.description = f"Test orchestrator: {name}"
        self.agents = []
        self.termination = Mock()

    async def run_stream(self, task: str) -> Any:
        """Mock run_stream method."""
        yield Mock()


class MockWorkflow:
    """Mock workflow for testing."""

    def __init__(self, name: str = "TestWorkflow") -> None:
        self.name = name
        self.description = f"Test workflow: {name}"
        self.steps = {}
        self.start_step = None

    async def run_stream(self, input_data: Any) -> Any:
        """Mock run_stream method."""
        yield Mock()


def test_find_agent_in_module_success() -> None:
    """Test finding agent with correct 'agent' variable name."""
    scanner = ForlaScanner("/fake/dir")

    # Create mock module with agent variable
    mock_module = types.ModuleType("test_module")
    setattr(mock_module, "agent", MockAgent("TestAgent"))

    # Should find the agent
    result = scanner._find_entity_in_module(mock_module)

    assert result is not None
    assert result.name == "TestAgent"
    assert hasattr(result, "run_stream")


def test_find_orchestrator_in_module_success() -> None:
    """Test finding orchestrator with correct 'orchestrator' variable name."""
    scanner = ForlaScanner("/fake/dir")

    # Create mock module with orchestrator variable
    mock_module = types.ModuleType("test_module")
    setattr(mock_module, "orchestrator", MockOrchestrator("TestOrchestrator"))

    # Should find the orchestrator
    result = scanner._find_entity_in_module(mock_module)

    assert result is not None
    assert result.name == "TestOrchestrator"
    assert hasattr(result, "run_stream")


def test_find_workflow_in_module_success() -> None:
    """Test finding workflow with correct 'workflow' variable name."""
    scanner = ForlaScanner("/fake/dir")

    # Create mock module with workflow variable
    mock_module = types.ModuleType("test_module")
    setattr(mock_module, "workflow", MockWorkflow("TestWorkflow"))

    # Should find the workflow
    result = scanner._find_entity_in_module(mock_module)

    assert result is not None
    assert result.name == "TestWorkflow"
    assert hasattr(result, "run_stream")


def test_find_entity_wrong_variable_name() -> None:
    """Test that wrong variable names are not found."""
    scanner = ForlaScanner("/fake/dir")

    # Create mock module with wrong variable names
    mock_module = types.ModuleType("test_module")
    setattr(mock_module, "my_agent", MockAgent("TestAgent"))  # Wrong name!
    setattr(
        mock_module, "some_orchestrator", MockOrchestrator("TestOrchestrator")
    )  # Wrong name!
    setattr(mock_module, "some_workflow", MockWorkflow("TestWorkflow"))  # Wrong name!

    # Should not find anything
    result = scanner._find_entity_in_module(mock_module)

    assert result is None


def test_find_entity_missing_methods() -> None:
    """Test that objects without required methods are rejected."""
    scanner = ForlaScanner("/fake/dir")

    # Create mock module with objects missing required methods
    mock_module = types.ModuleType("test_module")

    # Object without run_stream method
    class BadEntity:
        def __init__(self) -> None:
            self.name = "bad_entity"
            # Missing run_stream method!

    setattr(mock_module, "agent", BadEntity())

    # Should not find it
    result = scanner._find_entity_in_module(mock_module)

    assert result is None


def test_extract_agent_tools() -> None:
    """Test tool extraction from agent."""
    scanner = ForlaScanner("/fake/dir")

    # Create agent with mock tools
    agent = MockAgent("TestAgent")

    # Mock tools with names
    tool1 = Mock()
    tool1.name = "get_weather"
    tool2 = Mock()
    tool2.name = "get_forecast"

    agent.tools = [tool1, tool2]

    # Extract tools
    tools = scanner._extract_agent_tools(agent)

    assert len(tools) == 2
    assert "get_weather" in tools
    assert "get_forecast" in tools


def test_get_entity_id() -> None:
    """Test entity ID generation from file path."""
    scanner = ForlaScanner("/base/dir")

    from pathlib import Path

    # Test simple file
    py_file = Path("/base/dir/agent.py")
    entity_id = scanner._get_entity_id(py_file)
    assert entity_id == "agent"

    # Test nested file
    py_file = Path("/base/dir/subdir/workflow.py")
    entity_id = scanner._get_entity_id(py_file)
    assert entity_id == "subdir.workflow"
