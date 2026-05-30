"""
Session management for Forla WebUI.

Manages conversation sessions using AgentContext and pluggable storage.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ..context import AgentContext
from ._session_store import InMemorySessionStore, SessionStore

logger: logging.Logger = logging.getLogger(__name__)


class SessionManager:
    """Manages conversation sessions with pluggable storage backend."""

    def __init__(self, store: Optional[SessionStore] = None) -> None:
        """Initialize session manager.

        Args:
            store: Storage backend (defaults to InMemorySessionStore)
        """
        self.store = store or InMemorySessionStore()

    async def get_or_create(
        self, session_id: str, entity_id: str, entity_type: str = "agent"
    ) -> AgentContext:
        """Get existing or create new session context.

        Args:
            session_id: Unique session identifier
            entity_id: ID of the entity (agent, orchestrator, workflow)
            entity_type: Type of entity

        Returns:
            Session context
        """
        context = await self.store.get(session_id)
        if context is None:
            logger.info(f"Creating new session {session_id} for {entity_type} {entity_id}")
            context = AgentContext(
                session_id=session_id,
                metadata={
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "last_activity": datetime.now(),
                },
            )
            await self.store.save(session_id, context)
        return context

    async def get(self, session_id: str) -> Optional[AgentContext]:
        """Get session context by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session context or None if not found
        """
        return await self.store.get(session_id)

    async def update(self, session_id: str, context: AgentContext) -> None:
        """Update session after agent interaction.

        Args:
            session_id: Session identifier
            context: Updated context
        """
        # Update last activity timestamp
        context.metadata["last_activity"] = datetime.now()
        await self.store.save(session_id, context)
        logger.debug(f"Updated session {session_id} ({len(context.messages)} messages)")

    async def list(self, entity_id: Optional[str] = None) -> List[Dict]:
        """List all sessions with metadata.

        Args:
            entity_id: Optional entity ID to filter by

        Returns:
            List of session metadata dictionaries
        """
        return await self.store.list(entity_id)

    async def delete(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        return await self.store.delete(session_id)

    async def clear_all(self) -> int:
        """Clear all sessions.

        Returns:
            Number of sessions cleared
        """
        return await self.store.clear_all()

    def create_session_id(self) -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())
