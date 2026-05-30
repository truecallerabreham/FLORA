"""
Abstract memory implementations for agent state persistence.

This module defines the memory interfaces and implementations that allow
agents to maintain context and learn from past interactions.

The BaseMemory interface is adapted from AutoGen Core's Memory interface:
https://github.com/microsoft/autogen/blob/main/python/packages/autogen-core/src/autogen_core/memory/_base_memory.py
Licensed under MIT License
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .._component_config import Component, ComponentBase


class MemoryContent(BaseModel):
    """A memory content item."""

    content: Union[str, Dict[str, Any]] = Field(..., description="The memory content")
    mime_type: str = Field(default="text/plain", description="MIME type of the content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional memory metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When memory was stored"
    )

    class Config:
        frozen = True


class MemoryQueryResult(BaseModel):
    """Result of a memory query operation."""

    results: List[MemoryContent] = Field(
        default_factory=list, description="List of retrieved memory contents"
    )

    class Config:
        frozen = True


class BaseMemory(ComponentBase[BaseModel], ABC):
    """
    Abstract base class for agent memory implementations.

    Provides persistent storage and retrieval for agent context,
    enabling agents to maintain continuity across conversations.
    """

    def __init__(self, max_memories: int = 1000):
        """
        Initialize memory with maximum capacity.

        Args:
            max_memories: Maximum number of memories to retain
        """
        self.max_memories = max_memories

    @abstractmethod
    async def add(self, content: MemoryContent) -> None:
        """
        Store new content in memory.

        Args:
            content: MemoryContent object to store
        """
        pass

    @abstractmethod
    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        """
        Retrieve relevant memories based on query.

        Args:
            query: Search query for retrieving memories
            limit: Maximum number of memories to return

        Returns:
            MemoryQueryResult containing relevant memories
        """
        pass

    @abstractmethod
    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        """
        Get recent/relevant context for LLM prompt.

        Args:
            max_items: Maximum number of context items

        Returns:
            MemoryQueryResult containing context memories
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Remove all stored memories."""
        pass

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory usage statistics
        """
        return {
            "max_memories": self.max_memories,
            "implementation": self.__class__.__name__,
        }


class ListMemoryConfig(BaseModel):
    """Configuration for ListMemory serialization."""

    max_memories: int = 1000
    memories: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # Serialized MemoryItem objects


class ListMemory(Component[ListMemoryConfig], BaseMemory):
    """
    Simple in-memory list storage for development and testing.

    Stores memories in a Python list with basic text search capabilities.
    This implementation does not persist between sessions.
    """

    component_config_schema = ListMemoryConfig
    component_type = "memory"
    component_provider_override = "forla.memory.ListMemory"

    def __init__(self, max_memories: int = 1000):
        super().__init__(max_memories)
        self.memories: List[MemoryContent] = []

    async def add(self, content: MemoryContent) -> None:
        """Store new content in memory list."""
        self.memories.append(content)

        # Remove oldest memories if we exceed capacity
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories :]

    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        """Retrieve memories using simple text matching."""
        query_lower = query.lower()
        matching_memories = []

        for memory in reversed(self.memories):  # Most recent first
            content_str = (
                memory.content
                if isinstance(memory.content, str)
                else json.dumps(memory.content)
            )
            if query_lower in content_str.lower():
                matching_memories.append(memory)
                if len(matching_memories) >= limit:
                    break

        return MemoryQueryResult(results=matching_memories)

    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        """Get most recent memories as context."""
        recent_memories = self.memories[-max_items:] if self.memories else []
        return MemoryQueryResult(results=recent_memories)

    async def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        base_stats = await super().get_stats()
        return {
            **base_stats,
            "current_memories": len(self.memories),
            "is_persistent": False,
        }

    def _to_config(self) -> ListMemoryConfig:
        """Convert to configuration for serialization."""
        return ListMemoryConfig(
            max_memories=self.max_memories,
            memories=[memory.model_dump() for memory in self.memories],
        )

    @classmethod
    def _from_config(cls, config: ListMemoryConfig) -> "ListMemory":
        """Create from configuration."""
        instance = cls(max_memories=config.max_memories)
        instance.memories = [
            MemoryContent(**memory_data) for memory_data in config.memories
        ]
        return instance


class FileMemoryConfig(BaseModel):
    """Configuration for FileMemory serialization."""

    file_path: str
    max_memories: int = 1000


class FileMemory(Component[FileMemoryConfig], BaseMemory):
    """
    File-based persistent storage with text search.

    Stores memories in a JSON file for persistence between sessions.
    Provides basic text search capabilities.
    """

    component_config_schema = FileMemoryConfig
    component_type = "memory"
    component_provider_override = "forla.memory.FileMemory"

    def __init__(self, file_path: str, max_memories: int = 1000):
        super().__init__(max_memories)
        self.file_path = file_path
        self.memories: List[MemoryContent] = []
        self._load_memories()

    def _load_memories(self) -> None:
        """Load memories from file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    memories_data = json.load(f)
                    self.memories = [
                        MemoryContent(**memory_data) for memory_data in memories_data
                    ]
            except Exception:
                # If file is corrupted, start fresh
                self.memories = []

    def _save_memories(self) -> None:
        """Save memories to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            with open(self.file_path, "w", encoding="utf-8") as f:
                memories_data = [memory.model_dump() for memory in self.memories]
                json.dump(memories_data, f, indent=2, default=str)
        except Exception:
            # Silently handle save errors to avoid breaking agent execution
            pass

    async def add(self, content: MemoryContent) -> None:
        """Store new content in file memory."""
        self.memories.append(content)

        # Remove oldest memories if we exceed capacity
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories :]

        self._save_memories()

    async def query(self, query: str, limit: int = 10) -> MemoryQueryResult:
        """Retrieve memories using text matching."""
        query_lower = query.lower()
        matching_memories = []

        for memory in reversed(self.memories):  # Most recent first
            content_str = (
                memory.content
                if isinstance(memory.content, str)
                else json.dumps(memory.content)
            )
            if query_lower in content_str.lower():
                matching_memories.append(memory)
                if len(matching_memories) >= limit:
                    break

        return MemoryQueryResult(results=matching_memories)

    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        """Get most recent memories as context."""
        recent_memories = self.memories[-max_items:] if self.memories else []
        return MemoryQueryResult(results=recent_memories)

    async def clear(self) -> None:
        """Clear all memories and remove file."""
        self.memories.clear()
        if os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
            except Exception:
                pass

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        base_stats = await super().get_stats()
        return {
            **base_stats,
            "current_memories": len(self.memories),
            "file_path": self.file_path,
            "is_persistent": True,
        }

    def _to_config(self) -> FileMemoryConfig:
        """Convert to configuration for serialization."""
        return FileMemoryConfig(
            file_path=self.file_path, max_memories=self.max_memories
        )

    @classmethod
    def _from_config(cls, config: FileMemoryConfig) -> "FileMemory":
        """Create from configuration."""
        return cls(file_path=config.file_path, max_memories=config.max_memories)
