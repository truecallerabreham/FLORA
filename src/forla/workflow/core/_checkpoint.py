"""
Checkpoint storage backends for workflow execution state.

This module provides:
- WorkflowCheckpoint data model
- Abstract CheckpointStore base class
- Concrete implementations (file, memory)
"""

import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ._models import StepStatus, WorkflowExecution

logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Models
# ============================================================================


class WorkflowCheckpoint(BaseModel):
    """
    Checkpoint containing workflow execution state.

    This is storage-agnostic - the checkpoint object is the same
    regardless of where it's stored (file, memory, database, etc.)
    """

    # Metadata for compatibility validation
    workflow_id: str = Field(description="Workflow ID this checkpoint belongs to")
    workflow_version: str = Field(
        default="1.0.0", description="Workflow version (from WorkflowMetadata)"
    )
    workflow_structure_hash: str = Field(
        description="Hash of workflow steps+edges for compatibility check"
    )

    # Checkpoint metadata
    checkpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    checkpoint_type: str = Field(
        default="manual", description="manual | auto | on_step | on_error"
    )

    # The actual execution state
    execution: WorkflowExecution = Field(
        description="Complete workflow execution state"
    )

    # Helper metadata (computed from execution)
    completed_step_ids: List[str] = Field(
        default_factory=list, description="Quick lookup for completed steps"
    )
    pending_step_ids: List[str] = Field(
        default_factory=list, description="Steps not yet started"
    )

    model_config = ConfigDict(
        extra="forbid", json_encoders={datetime: lambda v: v.isoformat()}
    )

    @classmethod
    def from_execution(
        cls,
        execution: WorkflowExecution,
        workflow_id: str,
        workflow_version: str,
        workflow_structure_hash: str,
        all_step_ids: List[str],
        checkpoint_type: str = "manual",
    ) -> "WorkflowCheckpoint":
        """
        Create checkpoint from workflow execution state.

        Args:
            execution: Current workflow execution
            workflow_id: Workflow identifier
            workflow_version: Workflow version
            workflow_structure_hash: Structure hash for validation
            all_step_ids: All step IDs in workflow
            checkpoint_type: Type of checkpoint (manual, auto, etc.)

        Returns:
            WorkflowCheckpoint ready to save
        """
        completed_step_ids = [
            step_id
            for step_id, step_exec in execution.step_executions.items()
            if step_exec.status == StepStatus.COMPLETED
        ]

        pending_step_ids = [
            step_id
            for step_id in all_step_ids
            if step_id not in execution.step_executions
            or execution.step_executions[step_id].status == StepStatus.PENDING
        ]

        return cls(
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            workflow_structure_hash=workflow_structure_hash,
            checkpoint_type=checkpoint_type,
            execution=execution,
            completed_step_ids=completed_step_ids,
            pending_step_ids=pending_step_ids,
        )


class CheckpointMetadata(BaseModel):
    """
    Lightweight checkpoint metadata (without full execution state).

    Useful for listing/searching checkpoints without loading full data.
    """

    checkpoint_id: str
    workflow_id: str
    workflow_version: str
    created_at: datetime
    checkpoint_type: str
    completed_steps: int
    pending_steps: int
    total_steps: int
    size_bytes: Optional[int] = None

    @classmethod
    def from_checkpoint(
        cls, checkpoint: WorkflowCheckpoint, size_bytes: Optional[int] = None
    ) -> "CheckpointMetadata":
        """Extract metadata from full checkpoint."""
        return cls(
            checkpoint_id=checkpoint.checkpoint_id,
            workflow_id=checkpoint.workflow_id,
            workflow_version=checkpoint.workflow_version,
            created_at=checkpoint.created_at,
            checkpoint_type=checkpoint.checkpoint_type,
            completed_steps=len(checkpoint.completed_step_ids),
            pending_steps=len(checkpoint.pending_step_ids),
            total_steps=len(checkpoint.completed_step_ids)
            + len(checkpoint.pending_step_ids),
            size_bytes=size_bytes,
        )


class CheckpointValidationResult(BaseModel):
    """Result of checkpoint validation."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    can_resume: bool = Field(default=False)
    checkpoint_info: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Abstract Base Class: CheckpointStore
# ============================================================================

CheckpointT = TypeVar("CheckpointT", bound=WorkflowCheckpoint)


class CheckpointStore(ABC, Generic[CheckpointT]):
    """
    Abstract base class for checkpoint storage backends.

    Subclasses implement specific storage mechanisms:
    - FileCheckpointStore: Local filesystem
    - InMemoryCheckpointStore: In-memory (testing)
    """

    @abstractmethod
    async def save(self, checkpoint: CheckpointT) -> None:
        """
        Save checkpoint to storage.

        Args:
            checkpoint: Checkpoint to save

        Raises:
            IOError: If save fails
        """
        pass

    @abstractmethod
    async def load(self, checkpoint_id: str) -> Optional[CheckpointT]:
        """
        Load checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Checkpoint if found, None otherwise
        """
        pass

    @abstractmethod
    async def load_latest(self, workflow_id: str) -> Optional[CheckpointT]:
        """
        Load most recent checkpoint for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Latest checkpoint if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(self, checkpoint_id: str) -> bool:
        """
        Delete checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_metadata(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CheckpointMetadata]:
        """
        List checkpoint metadata without loading full data.

        Args:
            workflow_id: Filter by workflow ID (None = all workflows)
            limit: Maximum number of results

        Returns:
            List of checkpoint metadata, sorted by created_at desc
        """
        pass

    @abstractmethod
    async def cleanup_old(
        self,
        workflow_id: str,
        keep_last_n: int = 5,
    ) -> int:
        """
        Remove old checkpoints, keeping only the N most recent.

        Args:
            workflow_id: Workflow to cleanup
            keep_last_n: Number of recent checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        pass


# ============================================================================
# Concrete Implementation: FileCheckpointStore
# ============================================================================


class FileCheckpointStore(CheckpointStore[WorkflowCheckpoint]):
    """
    File-based checkpoint storage.

    Storage layout:
        {base_path}/
            {workflow_id}/
                {checkpoint_id}.json

    Features:
    - Human-readable JSON files
    - Organized by workflow ID
    - Works with network drives
    """

    def __init__(self, base_path: Path):
        """
        Initialize file-based checkpoint store.

        Args:
            base_path: Root directory for checkpoint files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_workflow_dir(self, workflow_id: str) -> Path:
        """Get directory for workflow's checkpoints."""
        workflow_dir = self.base_path / workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=True)
        return workflow_dir

    def _get_checkpoint_path(self, workflow_id: str, checkpoint_id: str) -> Path:
        """Get file path for checkpoint."""
        return self._get_workflow_dir(workflow_id) / f"{checkpoint_id}.json"

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint to file."""
        checkpoint_path = self._get_checkpoint_path(
            checkpoint.workflow_id, checkpoint.checkpoint_id
        )

        # Serialize to JSON
        json_data = checkpoint.model_dump_json(indent=2)

        # Write to file
        checkpoint_path.write_text(json_data)
        logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id} to {checkpoint_path}")

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load checkpoint by ID (searches all workflow directories)."""
        # Search all workflow directories for checkpoint
        for workflow_dir in self.base_path.iterdir():
            if workflow_dir.is_dir():
                checkpoint_path = workflow_dir / f"{checkpoint_id}.json"
                if checkpoint_path.exists():
                    json_data = checkpoint_path.read_text()
                    return WorkflowCheckpoint.model_validate_json(json_data)

        return None

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Load most recent checkpoint for workflow."""
        workflow_dir = self._get_workflow_dir(workflow_id)

        # Find most recent checkpoint file
        checkpoint_files = list(workflow_dir.glob("*.json"))
        if not checkpoint_files:
            return None

        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Load most recent
        json_data = checkpoint_files[0].read_text()
        return WorkflowCheckpoint.model_validate_json(json_data)

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint file."""
        # Search all workflow directories
        for workflow_dir in self.base_path.iterdir():
            if workflow_dir.is_dir():
                checkpoint_path = workflow_dir / f"{checkpoint_id}.json"
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.debug(f"Deleted checkpoint {checkpoint_id}")
                    return True

        return False

    async def list_metadata(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CheckpointMetadata]:
        """List checkpoint metadata."""
        metadata_list: List[CheckpointMetadata] = []

        # Determine which directories to search
        if workflow_id:
            search_dirs = [self._get_workflow_dir(workflow_id)]
        else:
            search_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]

        # Scan checkpoint files
        for workflow_dir in search_dirs:
            for checkpoint_file in workflow_dir.glob("*.json"):
                # Load just enough to get metadata
                json_data = checkpoint_file.read_text()
                checkpoint = WorkflowCheckpoint.model_validate_json(json_data)
                metadata = CheckpointMetadata.from_checkpoint(
                    checkpoint, size_bytes=checkpoint_file.stat().st_size
                )
                metadata_list.append(metadata)

        # Sort by created_at desc, limit
        metadata_list.sort(key=lambda m: m.created_at, reverse=True)
        return metadata_list[:limit]

    async def cleanup_old(
        self,
        workflow_id: str,
        keep_last_n: int = 5,
    ) -> int:
        """Remove old checkpoint files."""
        workflow_dir = self._get_workflow_dir(workflow_id)

        # Get all checkpoint files sorted by modification time
        checkpoint_files = list(workflow_dir.glob("*.json"))
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Keep only the N most recent
        files_to_delete = checkpoint_files[keep_last_n:]

        for file_path in files_to_delete:
            file_path.unlink()

        logger.debug(
            f"Cleaned up {len(files_to_delete)} old checkpoints for {workflow_id}"
        )
        return len(files_to_delete)


# ============================================================================
# Concrete Implementation: InMemoryCheckpointStore
# ============================================================================


class InMemoryCheckpointStore(CheckpointStore[WorkflowCheckpoint]):
    """
    In-memory checkpoint storage.

    Features:
    - Fast (no I/O)
    - Useful for testing
    - Ephemeral (lost on restart)
    """

    def __init__(self):
        """Initialize in-memory store."""
        self._checkpoints: Dict[str, WorkflowCheckpoint] = {}
        # Index by workflow_id for fast lookup
        self._by_workflow: Dict[str, List[str]] = {}

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint to memory."""
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint

        # Update workflow index
        if checkpoint.workflow_id not in self._by_workflow:
            self._by_workflow[checkpoint.workflow_id] = []
        self._by_workflow[checkpoint.workflow_id].append(checkpoint.checkpoint_id)

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load checkpoint from memory."""
        return self._checkpoints.get(checkpoint_id)

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Load most recent checkpoint for workflow."""
        checkpoint_ids = self._by_workflow.get(workflow_id, [])
        if not checkpoint_ids:
            return None

        # Get all checkpoints for this workflow
        checkpoints = [
            self._checkpoints[cid]
            for cid in checkpoint_ids
            if cid in self._checkpoints
        ]

        if not checkpoints:
            return None

        # Sort by created_at and return most recent
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        return checkpoints[0]

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory."""
        if checkpoint_id in self._checkpoints:
            checkpoint = self._checkpoints.pop(checkpoint_id)

            # Remove from workflow index
            if checkpoint.workflow_id in self._by_workflow:
                self._by_workflow[checkpoint.workflow_id].remove(checkpoint_id)

            return True

        return False

    async def list_metadata(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CheckpointMetadata]:
        """List checkpoint metadata."""
        # Filter checkpoints
        if workflow_id:
            checkpoint_ids = self._by_workflow.get(workflow_id, [])
            checkpoints = [
                self._checkpoints[cid]
                for cid in checkpoint_ids
                if cid in self._checkpoints
            ]
        else:
            checkpoints = list(self._checkpoints.values())

        # Convert to metadata
        metadata_list = [
            CheckpointMetadata.from_checkpoint(cp) for cp in checkpoints
        ]

        # Sort and limit
        metadata_list.sort(key=lambda m: m.created_at, reverse=True)
        return metadata_list[:limit]

    async def cleanup_old(
        self,
        workflow_id: str,
        keep_last_n: int = 5,
    ) -> int:
        """Remove old checkpoints from memory."""
        checkpoint_ids = self._by_workflow.get(workflow_id, [])

        # Get all checkpoints for this workflow
        checkpoints = [
            (cid, self._checkpoints[cid])
            for cid in checkpoint_ids
            if cid in self._checkpoints
        ]

        # Sort by created_at
        checkpoints.sort(key=lambda x: x[1].created_at, reverse=True)

        # Delete old ones
        to_delete = checkpoints[keep_last_n:]
        for checkpoint_id, _ in to_delete:
            await self.delete(checkpoint_id)

        return len(to_delete)

    def clear(self) -> None:
        """Clear all checkpoints (useful for testing)."""
        self._checkpoints.clear()
        self._by_workflow.clear()


# ============================================================================
# Checkpoint Configuration
# ============================================================================


class CheckpointConfig(BaseModel):
    """
    Configuration for checkpoint behavior with reasonable defaults.

    If not provided, workflow runner will use InMemoryCheckpointStore
    with auto-save enabled.
    """

    # Storage backend (defaults to in-memory)
    store: CheckpointStore[WorkflowCheckpoint] = Field(
        default_factory=InMemoryCheckpointStore,
        description="Checkpoint storage backend",
    )

    # Auto-save settings
    auto_save: bool = Field(
        default=True, description="Automatically save checkpoint after each step"
    )
    save_interval_steps: int = Field(
        default=1, description="Save checkpoint every N steps (if auto_save=True)"
    )

    # Cleanup settings
    auto_cleanup: bool = Field(
        default=False, description="Automatically cleanup old checkpoints"
    )
    keep_last_n: int = Field(
        default=5, description="Number of recent checkpoints to keep (if auto_cleanup)"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Helper: Compute Workflow Structure Hash
# ============================================================================


def compute_workflow_structure_hash(
    steps: Dict[str, Any],
    edges: List[Any],
    start_step_id: Optional[str],
    end_step_ids: List[str],
) -> str:
    """
    Compute hash of workflow structure for checkpoint compatibility.

    Hash includes:
    - Step IDs and their types
    - Edge connections (from_step, to_step, condition types)
    - Start/end step IDs

    Does NOT include:
    - Step metadata (name, description, tags)
    - Workflow metadata (author, created_at)

    Args:
        steps: Workflow steps dict
        edges: Workflow edges list
        start_step_id: Start step ID
        end_step_ids: End step IDs

    Returns:
        16-character hex hash string
    """
    structure = {
        "steps": {
            step_id: {
                "type": step.__class__.__name__,
                "input_type": step.input_type.__name__,
                "output_type": step.output_type.__name__,
            }
            for step_id, step in sorted(steps.items())
        },
        "edges": [
            {
                "from": edge.from_step,
                "to": edge.to_step,
                "condition_type": edge.condition.type,
            }
            for edge in sorted(edges, key=lambda e: (e.from_step, e.to_step))
        ],
        "start_step": start_step_id,
        "end_steps": sorted(end_step_ids),
    }

    # Create deterministic JSON and hash it
    json_str = json.dumps(structure, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]
