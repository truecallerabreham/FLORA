"""
ChromaDB-based vector memory implementation for agents.

This implementation is adapted from AutoGen's ChromaDB memory:
https://github.com/microsoft/autogen/blob/main/python/packages/autogen-ext/src/autogen_ext/memory/chromadb/_chromadb.py
Licensed under MIT License
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

try:
    import chromadb

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    chromadb = None

from .._component_config import Component
from ._base import BaseMemory, MemoryContent, MemoryQueryResult

# Configuration Classes


class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""

    COSINE = "cosine"
    L2 = "l2"
    IP = "ip"  # inner product


class ChromaDBMemoryConfig(BaseModel):
    """Configuration for ChromaDB memory."""

    collection_name: str = Field(
        default="agent_memory", description="ChromaDB collection name"
    )
    max_memories: int = Field(default=1000, description="Maximum memories to retain")
    persist_directory: Optional[str] = Field(
        default=None, description="Directory for persistent storage"
    )
    distance_metric: DistanceMetric = Field(
        default=DistanceMetric.COSINE, description="Distance metric for similarity"
    )
    k: int = Field(default=10, description="Number of results to retrieve")
    score_threshold: float = Field(
        default=0.7, description="Similarity threshold (0-1)"
    )


# Main ChromaDB Memory Implementation


class ChromaDBMemory(Component[ChromaDBMemoryConfig], BaseMemory):
    """
    Vector-based memory storage using ChromaDB.

    Provides semantic search capabilities for memory retrieval
    using vector embeddings and similarity search.
    """

    component_config_schema = ChromaDBMemoryConfig
    component_type = "memory"
    component_provider_override = "forla.memory.ChromaDBMemory"

    def __init__(
        self,
        collection_name: str = "agent_memory",
        max_memories: int = 1000,
        persist_directory: Optional[str] = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        k: int = 10,
        score_threshold: float = 0.7,
    ):
        """
        Initialize ChromaDB memory.

        Args:
            collection_name: Name of the ChromaDB collection
            max_memories: Maximum number of memories to retain
            persist_directory: Directory for persistent storage (None for in-memory)
            distance_metric: Distance metric for similarity
            k: Default number of results to retrieve
            score_threshold: Threshold for similarity matching (0-1)
        """
        if not HAS_CHROMADB:
            raise ImportError(
                "ChromaDB is required for ChromaDBMemory. "
                "Install with: pip install chromadb"
            )

        super().__init__(max_memories)

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric
        self.k = k
        self.score_threshold = score_threshold

        # Initialize ChromaDB client
        if persist_directory:
            # chromadb is guaranteed to be available here due to HAS_CHROMADB check above
            self.client = chromadb.PersistentClient(path=persist_directory)  # type: ignore
        else:
            self.client = chromadb.Client()  # type: ignore

        # Create or get collection with default embedding function
        # ChromaDB will use its default SentenceTransformer if no embedding_function specified
        metadata = {"hnsw:space": distance_metric.value}
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata=metadata
        )

    async def add(self, content: MemoryContent) -> None:
        """
        Store new content in vector memory.

        Args:
            content: MemoryContent object to store
        """
        # Convert content to string for embedding
        if isinstance(content.content, str):
            document = content.content
        else:
            document = json.dumps(content.content)

        # Prepare metadata - handle None case
        metadata = {}
        if content.metadata:
            metadata.update(content.metadata)
        metadata.update(
            {"timestamp": content.timestamp.isoformat(), "mime_type": content.mime_type}
        )

        # Add to ChromaDB collection
        memory_id = str(uuid.uuid4())
        self.collection.add(documents=[document], metadatas=[metadata], ids=[memory_id])

        # Enforce max_memories limit
        await self._enforce_memory_limit()

    async def _enforce_memory_limit(self) -> None:
        """Remove oldest memories if over capacity."""
        count = self.collection.count()
        if count > self.max_memories:
            # Get all items
            all_items = self.collection.get()
            if all_items and all_items.get("metadatas") and all_items.get("ids"):
                # Sort by timestamp and remove oldest
                items_with_timestamps = []
                ids = all_items["ids"]
                metadatas = all_items["metadatas"]
                for id_, meta in zip(ids, metadatas or []):
                    timestamp = meta.get("timestamp", "1970-01-01T00:00:00")
                    items_with_timestamps.append((id_, timestamp))

                items_with_timestamps.sort(key=lambda x: x[1])

                # Remove oldest items
                num_to_remove = len(items_with_timestamps) - self.max_memories
                if num_to_remove > 0:
                    ids_to_remove = [
                        item[0] for item in items_with_timestamps[:num_to_remove]
                    ]
                    self.collection.delete(ids=ids_to_remove)

    async def query(self, query: str, limit: Optional[int] = None) -> MemoryQueryResult:
        """
        Retrieve relevant memories using semantic search.

        Args:
            query: Search query for retrieving memories
            limit: Maximum number of memories to return

        Returns:
            MemoryQueryResult containing relevant memories
        """
        if self.collection.count() == 0:
            return MemoryQueryResult(results=[])

        # Use provided limit or default k
        n_results = min(limit or self.k, self.collection.count())

        # Perform similarity search
        results = self.collection.query(query_texts=[query], n_results=n_results)

        if (
            not results
            or not results.get("documents")
            or not (
                results.get("documents")
                and results["documents"]
                and results["documents"][0]
            )
        ):
            return MemoryQueryResult(results=[])

        # Convert results to MemoryContent objects
        memories = []
        documents = (
            results["documents"][0]
            if results.get("documents") and results["documents"]
            else []
        )
        metadatas = (
            results["metadatas"][0]
            if results.get("metadatas") and results["metadatas"]
            else [{}] * len(documents)
        )
        distances = (
            results["distances"][0]
            if results.get("distances") and results["distances"]
            else None
        )

        for i, (doc, meta) in enumerate(zip(documents, metadatas or [])):
            # Check distance threshold if available
            if distances and i < len(distances) and distances[i] > self.score_threshold:
                continue

            # Extract original metadata (remove our added fields)
            original_meta = {}
            if meta:
                original_meta = {
                    k: v for k, v in meta.items() if k not in ["timestamp", "mime_type"]
                }

            # Try to parse JSON content if possible
            try:
                content = json.loads(doc)
            except (json.JSONDecodeError, TypeError):
                content = doc

            # Safe extraction with defaults
            mime_type = meta.get("mime_type", "text/plain") if meta else "text/plain"
            timestamp_str = (
                meta.get("timestamp", datetime.now().isoformat())
                if meta
                else datetime.now().isoformat()
            )

            memory = MemoryContent(
                content=content,
                mime_type=str(mime_type),  # Ensure it's a string
                metadata=original_meta or {},
                timestamp=datetime.fromisoformat(
                    str(timestamp_str)
                ),  # Ensure it's a string
            )
            memories.append(memory)

        return MemoryQueryResult(results=memories)

    async def get_context(self, max_items: int = 10) -> MemoryQueryResult:
        """
        Get recent memories for context.

        Args:
            max_items: Maximum number of context items

        Returns:
            MemoryQueryResult containing context memories
        """
        # Get all items with metadata
        all_items = self.collection.get()

        if not all_items or not all_items.get("documents"):
            return MemoryQueryResult(results=[])

        # Convert to MemoryContent objects with timestamps
        memories = []
        documents = all_items.get("documents", [])
        metadatas = all_items.get("metadatas", [{}] * len(documents or []))

        for doc, meta in zip(documents or [], metadatas or []):
            # Extract original metadata
            original_meta = {}
            if meta:
                original_meta = {
                    k: v for k, v in meta.items() if k not in ["timestamp", "mime_type"]
                }

            # Try to parse JSON content
            try:
                content = json.loads(doc)
            except (json.JSONDecodeError, TypeError):
                content = doc

            # Safe extraction with defaults
            mime_type = meta.get("mime_type", "text/plain") if meta else "text/plain"
            timestamp_str = (
                meta.get("timestamp", datetime.now().isoformat())
                if meta
                else datetime.now().isoformat()
            )

            memory = MemoryContent(
                content=content,
                mime_type=str(mime_type),
                metadata=original_meta or {},
                timestamp=datetime.fromisoformat(str(timestamp_str)),
            )
            memories.append(memory)

        # Sort by timestamp (most recent first) and return top items
        memories.sort(key=lambda x: x.timestamp, reverse=True)
        return MemoryQueryResult(results=memories[:max_items])

    async def clear(self) -> None:
        """Remove all stored memories."""
        # Delete the collection and recreate it
        metadata = {"hnsw:space": self.distance_metric.value}
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=metadata
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        base_stats = await super().get_stats()
        return {
            **base_stats,
            "current_memories": self.collection.count() or 0,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "distance_metric": self.distance_metric.value,
            "k": self.k,
            "score_threshold": self.score_threshold,
            "is_persistent": self.persist_directory is not None,
        }

    def _to_config(self) -> ChromaDBMemoryConfig:
        """Convert to configuration for serialization."""
        return ChromaDBMemoryConfig(
            collection_name=self.collection_name,
            max_memories=self.max_memories,
            persist_directory=self.persist_directory,
            distance_metric=self.distance_metric,
            k=self.k,
            score_threshold=self.score_threshold,
        )

    @classmethod
    def _from_config(cls, config: ChromaDBMemoryConfig) -> "ChromaDBMemory":
        """Create from configuration."""
        return cls(
            collection_name=config.collection_name,
            max_memories=config.max_memories,
            persist_directory=config.persist_directory,
            distance_metric=config.distance_metric,
            k=config.k,
            score_threshold=config.score_threshold,
        )
