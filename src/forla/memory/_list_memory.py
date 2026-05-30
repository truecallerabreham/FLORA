from __future__ import annotations
from typing import List
from ._base import BaseMemory, MemoryContent


class ListMemory(BaseMemory):
    """Simple in-memory list storage.
    
    WHEN TO USE: Development, testing, prototyping, and simple applications
    where you don't need semantic search.
    
    WHEN TO UPGRADE: Replace with a vector database implementation
    (ChromaDB, Pinecone, Weaviate) when you need semantic similarity search.
    Because both use the same BaseMemory interface, you change one line
    in your agent setup and everything else works identically.
    
    HOW IT WORKS:
    - add(): Appends to a list, evicts oldest when over capacity
    - query(): Simple substring matching (newest first)
    - get_context(): Returns the N most recent memories
    """

    def __init__(self, max_memories: int = 1000):
        self._memories: List[MemoryContent] = []
        self._max_memories = max_memories

    async def add(self, content: MemoryContent) -> None:
        """Store content, evicting the oldest entry when at capacity."""
        self._memories.append(content)
        
        # FIFO eviction: keep the most recent max_memories entries
        if len(self._memories) > self._max_memories:
            self._memories = self._memories[-self._max_memories:]

    async def query(self, query: str, limit: int = 10) -> List[str]:
        """Find memories that contain the query string.
        
        Searches newest-first so the most recent matching memory
        appears first in results.
        """
        query_lower = query.lower()
        matches = []
        
        for memory in reversed(self._memories):
            if query_lower in memory.content.lower():
                matches.append(memory.content)
                if len(matches) >= limit:
                    break
        
        return matches

    async def get_context(self, max_items: int = 10) -> List[str]:
        """Return the most recent memories for prompt augmentation."""
        # Take the last max_items memories (most recent)
        recent = self._memories[-max_items:] if self._memories else []
        return [m.content for m in recent]
