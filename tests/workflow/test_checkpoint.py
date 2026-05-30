"""
Tests for workflow checkpoint functionality.

Minimal high-coverage tests for checkpoint save/resume.
"""

import pytest
from pathlib import Path
from pydantic import BaseModel

from forla.workflow import (
    Workflow,
    WorkflowRunner,
    WorkflowMetadata,
    StepMetadata,
    WorkflowCheckpoint,
    CheckpointConfig,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    FunctionStep,
)


# Test data models
class NumberInput(BaseModel):
    value: int


class NumberOutput(BaseModel):
    result: int


# Test step functions
async def double_number(input_data: NumberInput, context) -> NumberOutput:
    return NumberOutput(result=input_data.value * 2)


async def add_ten(input_data: NumberOutput, context) -> NumberOutput:
    return NumberOutput(result=input_data.result + 10)


async def square(input_data: NumberOutput, context) -> NumberOutput:
    return NumberOutput(result=input_data.result ** 2)


@pytest.mark.asyncio
async def test_checkpoint_save_and_resume():
    """Test basic checkpoint save and resume functionality."""
    # Create workflow
    double_step = FunctionStep(
        step_id="double",
        metadata=StepMetadata(name="Double"),
        input_type=NumberInput,
        output_type=NumberOutput,
        func=double_number,
    )

    add_step = FunctionStep(
        step_id="add_ten",
        metadata=StepMetadata(name="Add Ten"),
        input_type=NumberOutput,
        output_type=NumberOutput,
        func=add_ten,
    )

    square_step = FunctionStep(
        step_id="square",
        metadata=StepMetadata(name="Square"),
        input_type=NumberOutput,
        output_type=NumberOutput,
        func=square,
    )

    workflow = Workflow(metadata=WorkflowMetadata(name="Math Pipeline")).chain(
        double_step, add_step, square_step
    )

    # Run workflow with in-memory checkpointing
    store = InMemoryCheckpointStore()
    config = CheckpointConfig(store=store, auto_save=True, save_interval_steps=1)

    runner = WorkflowRunner()

    # Execute workflow
    events = []
    async for event in runner.run_stream(
        workflow=workflow,
        initial_input={"value": 3},
        checkpoint_config=config,
    ):
        events.append(event)

    # Verify execution completed
    final_event = events[-1]
    assert final_event.event_type == "workflow_completed"

    # Verify checkpoints were saved (3 steps = 3 checkpoints)
    metadata = await store.list_metadata(workflow_id=workflow.id)
    assert len(metadata) == 3

    # Verify final result: (3 * 2 + 10) ** 2 = 256
    execution = final_event.execution
    assert execution.state["square_output"]["result"] == 256

    # Load latest checkpoint
    checkpoint = await store.load_latest(workflow.id)
    assert checkpoint is not None
    assert len(checkpoint.completed_step_ids) == 3
    assert len(checkpoint.pending_step_ids) == 0


@pytest.mark.asyncio
async def test_checkpoint_resume_from_failure():
    """Test resuming workflow from checkpoint after simulated failure."""
    # Create workflow
    double_step = FunctionStep(
        step_id="double",
        metadata=StepMetadata(name="Double"),
        input_type=NumberInput,
        output_type=NumberOutput,
        func=double_number,
    )

    add_step = FunctionStep(
        step_id="add_ten",
        metadata=StepMetadata(name="Add Ten"),
        input_type=NumberOutput,
        output_type=NumberOutput,
        func=add_ten,
    )

    workflow = Workflow(metadata=WorkflowMetadata(name="Resume Test")).chain(
        double_step, add_step
    )

    runner = WorkflowRunner()
    store = InMemoryCheckpointStore()

    # Run first step only (simulate partial execution)
    config = CheckpointConfig(store=store, auto_save=True, save_interval_steps=1)

    # Collect events until first checkpoint is saved
    checkpoint_saved = False
    async for event in runner.run_stream(
        workflow=workflow,
        initial_input={"value": 5},
        checkpoint_config=config,
    ):
        if event.event_type == "checkpoint_saved":
            checkpoint_saved = True
            # Simulate failure after first checkpoint
            break

    # Load checkpoint after "failure"
    checkpoint = await store.load_latest(workflow.id)
    assert checkpoint is not None
    assert len(checkpoint.completed_step_ids) == 1
    assert "double" in checkpoint.completed_step_ids

    # Resume from checkpoint
    resume_events = []
    async for event in runner.run_stream(
        workflow=workflow,
        initial_input={"value": 5},
        checkpoint=checkpoint,
        checkpoint_config=config,
    ):
        resume_events.append(event)

    # Verify resume event was emitted
    assert any(e.event_type == "workflow_resumed" for e in resume_events)

    # Verify only one step executed (add_ten, since double was already done)
    step_completed_events = [
        e for e in resume_events if e.event_type == "step_completed"
    ]
    assert len(step_completed_events) == 1
    assert step_completed_events[0].step_id == "add_ten"

    # Verify final result: 5 * 2 + 10 = 20
    final_event = resume_events[-1]
    assert final_event.event_type == "workflow_completed"
    assert final_event.execution.state["add_ten_output"]["result"] == 20


@pytest.mark.asyncio
async def test_file_checkpoint_store(tmp_path):
    """Test file-based checkpoint storage."""
    checkpoint_dir = tmp_path / "checkpoints"
    store = FileCheckpointStore(base_path=checkpoint_dir)

    # Create workflow
    double_step = FunctionStep(
        step_id="double",
        metadata=StepMetadata(name="Double"),
        input_type=NumberInput,
        output_type=NumberOutput,
        func=double_number,
    )

    workflow = Workflow(metadata=WorkflowMetadata(name="File Test")).add_step(
        double_step
    ).set_start_step("double").add_end_step("double")

    runner = WorkflowRunner()
    config = CheckpointConfig(store=store, auto_save=True)

    # Run workflow
    async for event in runner.run_stream(
        workflow=workflow,
        initial_input={"value": 7},
        checkpoint_config=config,
    ):
        pass

    # Verify checkpoint file was created
    workflow_dir = checkpoint_dir / workflow.id
    assert workflow_dir.exists()

    checkpoint_files = list(workflow_dir.glob("*.json"))
    assert len(checkpoint_files) == 1

    # Load checkpoint from file
    checkpoint = await store.load_latest(workflow.id)
    assert checkpoint is not None
    assert checkpoint.workflow_id == workflow.id
    assert len(checkpoint.completed_step_ids) == 1


@pytest.mark.asyncio
async def test_checkpoint_validation():
    """Test checkpoint validation detects incompatible workflows."""
    # Create original workflow
    double_step = FunctionStep(
        step_id="double",
        metadata=StepMetadata(name="Double"),
        input_type=NumberInput,
        output_type=NumberOutput,
        func=double_number,
    )

    workflow_v1 = Workflow(metadata=WorkflowMetadata(name="V1")).add_step(
        double_step
    ).set_start_step("double").add_end_step("double")

    runner = WorkflowRunner()
    store = InMemoryCheckpointStore()
    config = CheckpointConfig(store=store, auto_save=True)

    # Run workflow and create checkpoint
    async for event in runner.run_stream(
        workflow=workflow_v1,
        initial_input={"value": 3},
        checkpoint_config=config,
    ):
        pass

    checkpoint = await store.load_latest(workflow_v1.id)
    assert checkpoint is not None

    # Create modified workflow (different structure)
    add_step = FunctionStep(
        step_id="add_ten",
        metadata=StepMetadata(name="Add Ten"),
        input_type=NumberOutput,
        output_type=NumberOutput,
        func=add_ten,
    )

    workflow_v2 = Workflow(metadata=WorkflowMetadata(name="V2")).chain(
        double_step, add_step
    )

    # Try to resume with incompatible workflow
    validation = runner.validate_checkpoint(workflow_v2, checkpoint)

    # Should fail validation due to structure change
    assert not validation.can_resume
    assert len(validation.errors) > 0
    assert "structure" in validation.errors[0].lower()


@pytest.mark.asyncio
async def test_checkpoint_cleanup():
    """Test automatic checkpoint cleanup."""
    store = InMemoryCheckpointStore()

    double_step = FunctionStep(
        step_id="double",
        metadata=StepMetadata(name="Double"),
        input_type=NumberInput,
        output_type=NumberOutput,
        func=double_number,
    )

    workflow = Workflow(metadata=WorkflowMetadata(name="Cleanup Test")).add_step(
        double_step
    ).set_start_step("double").add_end_step("double")

    runner = WorkflowRunner()

    # Run workflow 10 times to create 10 checkpoints
    for i in range(10):
        config = CheckpointConfig(store=store, auto_save=True)
        async for event in runner.run_stream(
            workflow=workflow,
            initial_input={"value": i},
            checkpoint_config=config,
        ):
            pass

    # Verify 10 checkpoints exist
    metadata_before = await store.list_metadata(workflow_id=workflow.id)
    assert len(metadata_before) == 10

    # Run cleanup keeping only last 3
    deleted = await store.cleanup_old(workflow_id=workflow.id, keep_last_n=3)

    assert deleted == 7

    # Verify only 3 remain
    metadata_after = await store.list_metadata(workflow_id=workflow.id)
    assert len(metadata_after) == 3


@pytest.mark.asyncio
async def test_checkpoint_metadata_list():
    """Test listing checkpoint metadata without loading full checkpoints."""
    store = InMemoryCheckpointStore()

    double_step = FunctionStep(
        step_id="double",
        metadata=StepMetadata(name="Double"),
        input_type=NumberInput,
        output_type=NumberOutput,
        func=double_number,
    )

    add_step = FunctionStep(
        step_id="add_ten",
        metadata=StepMetadata(name="Add Ten"),
        input_type=NumberOutput,
        output_type=NumberOutput,
        func=add_ten,
    )

    workflow = Workflow(metadata=WorkflowMetadata(name="Metadata Test")).chain(
        double_step, add_step
    )

    runner = WorkflowRunner()
    config = CheckpointConfig(store=store, auto_save=True, save_interval_steps=1)

    # Run workflow
    async for event in runner.run_stream(
        workflow=workflow,
        initial_input={"value": 5},
        checkpoint_config=config,
    ):
        pass

    # List metadata
    metadata_list = await store.list_metadata(workflow_id=workflow.id)

    assert len(metadata_list) == 2  # 2 steps = 2 checkpoints
    assert all(m.workflow_id == workflow.id for m in metadata_list)
    assert all(m.total_steps == 2 for m in metadata_list)

    # First checkpoint: 1 completed, 1 pending
    first_meta = metadata_list[1]  # List is sorted desc by created_at
    assert first_meta.completed_steps == 1
    assert first_meta.pending_steps == 1

    # Last checkpoint: 2 completed, 0 pending
    last_meta = metadata_list[0]
    assert last_meta.completed_steps == 2
    assert last_meta.pending_steps == 0


@pytest.mark.asyncio
async def test_workflow_structure_hash_stable():
    """Test that workflow structure hash is deterministic."""
    # Create workflow twice with same structure
    def create_workflow():
        double_step = FunctionStep(
            step_id="double",
            metadata=StepMetadata(name="Double"),
            input_type=NumberInput,
            output_type=NumberOutput,
            func=double_number,
        )

        add_step = FunctionStep(
            step_id="add_ten",
            metadata=StepMetadata(name="Add Ten"),
            input_type=NumberOutput,
            output_type=NumberOutput,
            func=add_ten,
        )

        return Workflow(metadata=WorkflowMetadata(name="Hash Test")).chain(
            double_step, add_step
        )

    workflow1 = create_workflow()
    workflow2 = create_workflow()

    hash1 = workflow1.compute_structure_hash()
    hash2 = workflow2.compute_structure_hash()

    # Hashes should be identical for same structure
    assert hash1 == hash2
    assert len(hash1) == 16  # 16-character hash


@pytest.mark.asyncio
async def test_checkpoint_with_default_config():
    """Test that checkpoint works with default configuration (in-memory)."""
    double_step = FunctionStep(
        step_id="double",
        metadata=StepMetadata(name="Double"),
        input_type=NumberInput,
        output_type=NumberOutput,
        func=double_number,
    )

    workflow = Workflow(metadata=WorkflowMetadata(name="Default Config")).add_step(
        double_step
    ).set_start_step("double").add_end_step("double")

    runner = WorkflowRunner()

    # Use default config (should create in-memory store automatically)
    config = CheckpointConfig()  # Uses defaults: InMemoryCheckpointStore, auto_save=True

    checkpoint_events = []
    async for event in runner.run_stream(
        workflow=workflow,
        initial_input={"value": 4},
        checkpoint_config=config,
    ):
        if event.event_type == "checkpoint_saved":
            checkpoint_events.append(event)

    # Should have saved checkpoint automatically
    assert len(checkpoint_events) == 1
    assert checkpoint_events[0].completed_steps == 1
