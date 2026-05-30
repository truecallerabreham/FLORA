"""Tests for SessionManager functionality with new AgentContext-based system."""

import pytest

from forla.context import AgentContext
from forla.messages import AssistantMessage, UserMessage
from forla.webui._sessions import SessionManager


def create_message(role: str, content: str):
    """Helper to create Message with required fields."""
    if role == "user":
        return UserMessage(content=content, source="test_user")
    elif role == "assistant":
        return AssistantMessage(content=content, source="test_assistant")
    else:
        raise ValueError(f"Unsupported role: {role}")


@pytest.mark.asyncio
async def test_session_manager_initialization() -> None:
    """Test session manager initialization."""
    manager = SessionManager()
    assert manager.store is not None


@pytest.mark.asyncio
async def test_get_or_create_new_session() -> None:
    """Test creating a new session."""
    manager = SessionManager()

    context = await manager.get_or_create(
        session_id="test_session", entity_id="test_agent", entity_type="agent"
    )

    assert isinstance(context, AgentContext)
    assert context.session_id == "test_session"
    assert context.metadata["entity_id"] == "test_agent"
    assert context.metadata["entity_type"] == "agent"
    assert len(context.messages) == 0


@pytest.mark.asyncio
async def test_get_or_create_existing_session() -> None:
    """Test retrieving an existing session."""
    manager = SessionManager()

    # Create session
    context1 = await manager.get_or_create("test_session", "test_agent", "agent")
    context1.add_message(create_message("user", "Hello"))
    await manager.update("test_session", context1)

    # Get existing session
    context2 = await manager.get_or_create("test_session", "test_agent", "agent")

    assert context2.session_id == "test_session"
    assert len(context2.messages) == 1
    assert context2.messages[0].content == "Hello"


@pytest.mark.asyncio
async def test_get_nonexistent_session() -> None:
    """Test retrieving nonexistent session."""
    manager = SessionManager()

    context = await manager.get("nonexistent")
    assert context is None


@pytest.mark.asyncio
async def test_update_session() -> None:
    """Test updating session with messages."""
    manager = SessionManager()

    context = await manager.get_or_create("test_session", "test_agent", "agent")
    context.add_message(create_message("user", "Hello"))
    context.add_message(create_message("assistant", "Hi there!"))

    await manager.update("test_session", context)

    # Retrieve and verify
    retrieved = await manager.get("test_session")
    assert retrieved is not None
    assert len(retrieved.messages) == 2
    assert retrieved.messages[0].content == "Hello"
    assert retrieved.messages[1].content == "Hi there!"


@pytest.mark.asyncio
async def test_list_sessions() -> None:
    """Test listing all sessions."""
    manager = SessionManager()

    # Create multiple sessions
    await manager.get_or_create("session1", "agent1", "agent")
    await manager.get_or_create("session2", "agent2", "agent")
    await manager.get_or_create("session3", "orch1", "orchestrator")

    sessions = await manager.list()
    assert len(sessions) == 3

    session_ids = [s["id"] for s in sessions]
    assert "session1" in session_ids
    assert "session2" in session_ids
    assert "session3" in session_ids


@pytest.mark.asyncio
async def test_list_sessions_filtered() -> None:
    """Test listing sessions filtered by entity."""
    manager = SessionManager()

    # Create sessions for different entities
    await manager.get_or_create("session1", "agent1", "agent")
    await manager.get_or_create("session2", "agent1", "agent")
    await manager.get_or_create("session3", "agent2", "agent")

    # Filter by entity
    filtered = await manager.list(entity_id="agent1")
    assert len(filtered) == 2

    entity_ids = [s["entity_id"] for s in filtered]
    assert all(eid == "agent1" for eid in entity_ids)


@pytest.mark.asyncio
async def test_delete_session() -> None:
    """Test deleting a session."""
    manager = SessionManager()

    # Create session
    await manager.get_or_create("test_session", "test_agent", "agent")

    # Verify it exists
    context = await manager.get("test_session")
    assert context is not None

    # Delete session
    success = await manager.delete("test_session")
    assert success is True

    # Should be removed
    context = await manager.get("test_session")
    assert context is None


@pytest.mark.asyncio
async def test_delete_nonexistent_session() -> None:
    """Test deleting nonexistent session."""
    manager = SessionManager()

    success = await manager.delete("nonexistent")
    assert success is False


@pytest.mark.asyncio
async def test_clear_all_sessions() -> None:
    """Test clearing all sessions."""
    manager = SessionManager()

    # Create multiple sessions
    await manager.get_or_create("session1", "agent1", "agent")
    await manager.get_or_create("session2", "agent2", "agent")

    count = await manager.clear_all()
    assert count == 2

    # All sessions should be gone
    sessions = await manager.list()
    assert len(sessions) == 0


@pytest.mark.asyncio
async def test_session_persistence_with_context() -> None:
    """Test that session context is properly persisted."""
    manager = SessionManager()

    # Create session and add messages
    context = await manager.get_or_create("test_session", "test_agent", "agent")
    context.add_message(create_message("user", "First message"))
    context.metadata["custom_key"] = "custom_value"
    context.shared_state["state_key"] = "state_value"
    await manager.update("test_session", context)

    # Retrieve and verify everything is persisted
    retrieved = await manager.get("test_session")
    assert retrieved is not None
    assert len(retrieved.messages) == 1
    assert retrieved.messages[0].content == "First message"
    assert retrieved.metadata["custom_key"] == "custom_value"
    assert retrieved.shared_state["state_key"] == "state_value"


@pytest.mark.asyncio
async def test_create_session_id() -> None:
    """Test session ID generation."""
    manager = SessionManager()

    session_id1 = manager.create_session_id()
    session_id2 = manager.create_session_id()

    assert session_id1 != session_id2
    assert len(session_id1) > 0
    assert len(session_id2) > 0
