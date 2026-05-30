"""
Memory system for forla framework.

Provides persistent storage and retrieval capabilities for agent context,
enabling agents to maintain continuity across conversations.
"""

from ._base import BaseMemory, FileMemory, ListMemory, MemoryContent, MemoryQueryResult

# Optional ChromaDB imports
try:
    from ._chromadb import (
        ChromaDBMemory,
        ChromaDBMemoryConfig,
        HttpChromaDBMemory,
        HttpChromaDBMemoryConfig,
        PersistentChromaDBMemory,
        PersistentChromaDBMemoryConfig,
    )

    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False

__all__ = [
    "BaseMemory",
    "MemoryContent",
    "MemoryQueryResult",
    "ListMemory",
    "FileMemory",
]

# Add ChromaDB exports if available
if _HAS_CHROMADB:
    __all__.extend(
        [
            "ChromaDBMemory",
            "PersistentChromaDBMemory",
            "HttpChromaDBMemory",
            "ChromaDBMemoryConfig",
            "PersistentChromaDBMemoryConfig",
            "HttpChromaDBMemoryConfig",
        ]
    )
