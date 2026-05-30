"""
Tests for workflow step progress events.
"""

import asyncio
from typing import Any, Dict

import pytest
from pydantic import BaseModel

from forla.workflow import Workflow, WorkflowRunner
from forla.workflow.core._models import (
    Context,
    StepMetadata,
    StepProgressEvent,
    WorkflowEventType,
    WorkflowMetadata,
)
from forla.workflow.steps import BaseStep


class ProgressTestInput(BaseModel):
    """Test input model."""

    count: int = 10


class ProgressTestOutput(BaseModel):
    """Test output model."""

    result: str


class ProgressEmittingStep(BaseStep[ProgressTestInput, ProgressTestOutput]):
    """Step that emits progress updates."""

    async def run(
        self, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute step with progress updates."""
        # Access the typed context that was created by the runner
        # The runner should have already set _progress_callback on it
        typed_context = Context.from_state_ref(context.get("workflow_state", {}))

        # The callback should be set directly on the context object, not in the dict
        # But since we're getting a new Context here, we need to manually transfer it
        # This is a test-specific workaround - in real code, steps wouldn't recreate Context
        if "_context_obj" in context:
            typed_context = context["_context_obj"]
        else:
            # Fallback: try to get callback from dict
            callback = context.get("_progress_callback")
            if callback:
                typed_context._progress_callback = callback

        # Support both count (initial input) and result (from previous step)
        count = input_data.get("count", 10)
        if "result" in input_data and isinstance(input_data["result"], str):
            # Parse count from result string like "Processed 3 items"
            import re

            match = re.search(r"(\d+) items", input_data["result"])
            if match:
                count = int(match.group(1))

        # Emit progress updates during execution
        for i in range(count):
            typed_context.emit_progress(
                message=f"Processing item {i + 1}",
                completed=i + 1,
                total=count,
                metadata={"item_id": i},
            )
            await asyncio.sleep(0.01)  # Simulate work

        # Final result
        return {"result": f"Processed {count} items"}


@pytest.mark.asyncio
async def test_context_emit_progress():
    """Test that Context.emit_progress() calls callback correctly."""
    progress_updates = []

    def callback(data: Dict[str, Any]) -> None:
        progress_updates.append(data)

    context = Context(state={})
    context._progress_callback = callback

    # Emit progress
    context.emit_progress(
        message="Test progress", completed=5, total=10, metadata={"foo": "bar"}
    )

    # Verify callback was called
    assert len(progress_updates) == 1
    assert progress_updates[0]["message"] == "Test progress"
    assert progress_updates[0]["completed"] == 5
    assert progress_updates[0]["total"] == 10
    assert progress_updates[0]["metadata"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_context_emit_progress_no_callback():
    """Test that emit_progress() handles missing callback gracefully."""
    context = Context(state={})
    # Should not raise error when callback is None
    context.emit_progress(message="Test", completed=1, total=10)


@pytest.mark.asyncio
async def test_workflow_step_progress_events():
    """Test that progress events are yielded during workflow execution."""
    # Create workflow with progress-emitting step
    metadata = WorkflowMetadata(name="test_progress")
    workflow = Workflow(metadata=metadata)

    step = ProgressEmittingStep(
        step_id="progress_step",
        metadata=StepMetadata(name="Progress Test Step"),
        input_type=ProgressTestInput,
        output_type=ProgressTestOutput,
    )
    workflow.add_step(step)
    workflow.set_start_step("progress_step")
    workflow.add_end_step("progress_step")

    # Run workflow and collect events
    runner = WorkflowRunner()
    events = []

    async for event in runner.run_stream(workflow, {"count": 5}):
        events.append(event)
        print(f"Event: {type(event).__name__} - {event}")

    # Verify we got progress events
    progress_events = [e for e in events if isinstance(e, StepProgressEvent)]
    print(f"\nTotal events: {len(events)}, Progress events: {len(progress_events)}")
    for e in events:
        print(f"  - {type(e).__name__}")
    assert len(progress_events) == 5, f"Should have 5 progress events (one per item), got {len(progress_events)}"

    # Verify first progress event
    first_progress = progress_events[0]
    assert first_progress.event_type == WorkflowEventType.STEP_PROGRESS
    assert first_progress.step_id == "progress_step"
    assert first_progress.message == "Processing item 1"
    assert first_progress.completed == 1
    assert first_progress.total == 5
    assert first_progress.metadata["item_id"] == 0

    # Verify last progress event
    last_progress = progress_events[-1]
    assert last_progress.message == "Processing item 5"
    assert last_progress.completed == 5
    assert last_progress.total == 5


@pytest.mark.asyncio
async def test_step_progress_event_str():
    """Test StepProgressEvent string representation."""
    from datetime import datetime

    from forla.workflow.core._models import StepProgressEvent

    event = StepProgressEvent(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        workflow_id="test_workflow",
        step_id="test_step",
        message="Processing data",
        completed=7,
        total=10,
        metadata={"foo": "bar"},
    )

    str_repr = str(event)
    assert "test_step" in str_repr
    assert "Processing data" in str_repr
    assert "7/10" in str_repr
    assert "70%" in str_repr


@pytest.mark.asyncio
async def test_step_progress_event_str_no_counts():
    """Test StepProgressEvent string representation without counts."""
    from datetime import datetime

    from forla.workflow.core._models import StepProgressEvent

    event = StepProgressEvent(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        workflow_id="test_workflow",
        step_id="test_step",
        message="Starting process",
        completed=None,
        total=None,
    )

    str_repr = str(event)
    assert "test_step" in str_repr
    assert "Starting process" in str_repr
    # Should not have counts in string
    assert "/" not in str_repr
    assert "%" not in str_repr


@pytest.mark.asyncio
async def test_workflow_with_multiple_progress_steps():
    """Test progress events from multiple steps in a workflow."""
    # Create workflow with two progress-emitting steps
    metadata = WorkflowMetadata(name="test_multi_progress")
    workflow = Workflow(metadata=metadata)

    step1 = ProgressEmittingStep(
        step_id="step1",
        metadata=StepMetadata(name="Step 1"),
        input_type=ProgressTestInput,
        output_type=ProgressTestOutput,
    )
    step2 = ProgressEmittingStep(
        step_id="step2",
        metadata=StepMetadata(name="Step 2"),
        input_type=ProgressTestOutput,  # Changed to accept output from step1
        output_type=ProgressTestOutput,
    )

    workflow.add_step(step1).add_step(step2)
    workflow.add_edge("step1", "step2")
    workflow.set_start_step("step1")
    workflow.add_end_step("step2")

    # Run workflow
    runner = WorkflowRunner()
    events = []

    async for event in runner.run_stream(workflow, {"count": 3}):
        events.append(event)

    # Verify we got progress events from both steps
    progress_events = [e for e in events if isinstance(e, StepProgressEvent)]
    assert len(progress_events) == 6, "Should have 6 progress events (3 per step)"

    # Verify step IDs
    step1_progress = [e for e in progress_events if e.step_id == "step1"]
    step2_progress = [e for e in progress_events if e.step_id == "step2"]

    assert len(step1_progress) == 3
    assert len(step2_progress) == 3
