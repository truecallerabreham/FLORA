from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryContent:
    """A single item stored in memory.
    
    'content' is the text to store.
    'metadata' carries extra information like the source agent name,
    the date it was stored, or a category tag.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    """Abstract interface for persistent agent memory.
    
    IMPORTANT DISTINCTION:
    - AgentContext (Part 2) holds the CURRENT SESSION's messages — it resets
    - BaseMemory holds PERSISTENT KNOWLEDGE — it survives session resets
    
    The agent framework calls get_context() automatically during _prepare_llm_messages()
    to inject relevant memories into the system prompt.
    
    The developer (or the agent via the add() tool) calls add() to store new information.
    
    This means agents get relevant past knowledge injected automatically,
    without needing to explicitly ask for it.
    """

    @abstractmethod
    async def add(self, content: MemoryContent) -> None:
        """Store new content. Called by your application code or by an agent tool."""
        pass

    @abstractmethod
    async def query(self, query: str, limit: int = 10) -> List[str]:
        """Retrieve memories relevant to a query.
        
        For simple implementations: substring matching.
        For production: replace with vector similarity search (ChromaDB, Pinecone, etc.)
        using the same interface — no agent code changes needed.
        """
        pass

    @abstractmethod
    async def get_context(self, max_items: int = 10) -> List[str]:
        """Get memory content for injecting into the system prompt.
        
        This is called automatically by the agent before every LLM call.
        The agent receives this context injected into its system instructions.
        
        Simple implementations return the N most recent memories.
        Advanced implementations do semantic similarity search to retrieve
        the most relevant memories for the current task.
        """
        pass
