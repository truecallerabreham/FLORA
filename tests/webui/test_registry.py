"""Tests for EntityRegistry functionality."""

from unittest.mock import Mock

import pytest

from forla.webui._models import AgentInfo, OrchestratorInfo, WorkflowInfo
from forla.webui._registry import EntityRegistry


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "TestAgent") -> None:
        self.name = name
        self.description = f"Test agent: {name}"
        self.tools = []
        self.model_client = Mock()
        self.model_client.model = "gpt-4.1-mini"  # Set a proper string for model
        self.memory = None

    async def run(self, messages):
        """Mock run method for agent."""
        return Mock()


class MockOrchestrator:
    """Mock orchestrator for testing."""

    def __init__(self, name: str = "TestOrchestrator") -> None:
        self.name = name
        self.description = f"Test orchestrator: {name}"
        self.agents = []
        self.termination = Mock()

    async def run_stream(self, messages):
        """Mock run_stream method for orchestrator."""
        yield Mock()


class MockWorkflow:
    """Mock workflow for testing."""

    def __init__(self, name: str = "TestWorkflow") -> None:
        self.name = name
        self.description = f"Test workflow: {name}"
        self.steps = {"step1": Mock(), "step2": Mock()}
        self.start_step = "step1"
        self.input_schema = {"type": "object"}

    async def run_stream(self, input_data):
        """Mock run_stream method for workflow."""
        yield Mock()


def test_registry_initialization() -> None:
    """Test registry initialization."""
    registry = EntityRegistry()

    assert registry.entities_dir is None
    assert registry.scanner is None
    assert len(registry._entities) == 0
    assert len(registry._in_memory_entities) == 0


def test_registry_with_directory() -> None:
    """Test registry initialization with directory."""
    registry = EntityRegistry("/fake/dir")

    assert registry.entities_dir == "/fake/dir"
    assert registry.scanner is not None


def test_register_agent() -> None:
    """Test registering an in-memory agent."""
    registry = EntityRegistry()
    agent = MockAgent("TestAgent")

    registry.register_entity("test_agent", agent)

    # Should be in both registries
    assert "test_agent" in registry._in_memory_entities
    assert "test_agent" in registry._entities

    # Check entity info
    entity_info = registry.get_entity_info("test_agent")
    assert entity_info is not None
    assert isinstance(entity_info, AgentInfo)
    assert entity_info.id == "test_agent"
    assert entity_info.name == "TestAgent"
    assert entity_info.type == "agent"
    assert entity_info.source == "memory"


def test_register_orchestrator() -> None:
    """Test registering an in-memory orchestrator."""
    registry = EntityRegistry()
    orchestrator = MockOrchestrator("TestOrchestrator")

    registry.register_entity("test_orch", orchestrator)

    entity_info = registry.get_entity_info("test_orch")
    assert entity_info is not None
    assert isinstance(entity_info, OrchestratorInfo)
    assert entity_info.type == "orchestrator"


def test_register_workflow() -> None:
    """Test registering an in-memory workflow."""
    registry = EntityRegistry()
    workflow = MockWorkflow("TestWorkflow")

    registry.register_entity("test_workflow", workflow)

    entity_info = registry.get_entity_info("test_workflow")
    assert entity_info is not None
    assert isinstance(entity_info, WorkflowInfo)
    assert entity_info.type == "workflow"
    assert entity_info.steps == ["step1", "step2"]
    assert entity_info.start_step == "step1"


def test_get_entity_object() -> None:
    """Test retrieving entity objects."""
    registry = EntityRegistry()
    agent = MockAgent("TestAgent")

    registry.register_entity("test_agent", agent)

    retrieved = registry.get_entity_object("test_agent")
    assert retrieved is agent


def test_list_entities() -> None:
    """Test listing all entities."""
    registry = EntityRegistry()

    # Register different types
    registry.register_entity("agent1", MockAgent("Agent1"))
    registry.register_entity("orch1", MockOrchestrator("Orch1"))
    registry.register_entity("workflow1", MockWorkflow("Workflow1"))

    entities = registry.list_entities()
    assert len(entities) == 3

    types = [e.type for e in entities]
    assert "agent" in types
    assert "orchestrator" in types
    assert "workflow" in types


def test_list_by_type() -> None:
    """Test listing entities by type."""
    registry = EntityRegistry()

    # Register multiple of each type
    registry.register_entity("agent1", MockAgent("Agent1"))
    registry.register_entity("agent2", MockAgent("Agent2"))
    registry.register_entity("orch1", MockOrchestrator("Orch1"))
    registry.register_entity("workflow1", MockWorkflow("Workflow1"))

    agents = registry.list_agents()
    orchestrators = registry.list_orchestrators()
    workflows = registry.list_workflows()

    assert len(agents) == 2
    assert len(orchestrators) == 1
    assert len(workflows) == 1

    assert all(e.type == "agent" for e in agents)
    assert all(e.type == "orchestrator" for e in orchestrators)
    assert all(e.type == "workflow" for e in workflows)


def test_get_nonexistent_entity() -> None:
    """Test getting info for nonexistent entity."""
    registry = EntityRegistry()

    entity_info = registry.get_entity_info("nonexistent")
    assert entity_info is None

    entity_obj = registry.get_entity_object("nonexistent")
    assert entity_obj is None


def test_clear_cache() -> None:
    """Test cache clearing functionality."""
    registry = EntityRegistry()

    # This should not crash even without a scanner
    registry.clear_cache()

    # With scanner, should call scanner's clear_cache
    registry.scanner = Mock()
    registry.clear_cache()
    registry.scanner.clear_cache.assert_called_once()
