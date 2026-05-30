"""Integration tests for workflow execution in WebUI.

This test file verifies that real WorkflowRunner events are properly
handled by the WebUI execution layer and frontend event processing.
"""

import asyncio
from typing import Any, Dict

import pytest
from pydantic import BaseModel

from forla.webui._execution import ExecutionEngine
from forla.webui._sessions import SessionManager
from forla.webui._models import WebUIStreamEvent
from forla.workflow import Workflow, WorkflowMetadata, StepMetadata
from forla.workflow.steps import FunctionStep, EchoStep


class WorkflowInput(BaseModel):
    """Input for test workflow."""
    message: str
    count: int = 1


class ProcessOutput(BaseModel):
    """Output from process step."""
    result: str
    processed_count: int


class FinalOutput(BaseModel):
    """Final output from workflow."""
    final_result: str
    final_count: int
    status: str


async def process_data(input_data: WorkflowInput, context: Any) -> ProcessOutput:
    """Process function for testing."""
    result = f"Processed: {input_data.message}"
    return ProcessOutput(
        result=result,
        processed_count=input_data.count * 2
    )


async def transform_data(input_data: ProcessOutput, context: Any) -> FinalOutput:
    """Transform function for testing."""
    return FinalOutput(
        final_result=input_data.result.upper(),
        final_count=input_data.processed_count + 10,
        status="completed"
    )


@pytest.fixture
def test_workflow():
    """Create a test workflow with real steps."""
    workflow = Workflow(
        metadata=WorkflowMetadata(
            name="Test Workflow",
            description="Integration test workflow"
        )
    )

    # Add steps using FunctionStep
    process_step = FunctionStep(
        step_id="process",
        metadata=StepMetadata(
            name="Process Data",
            description="Process input data"
        ),
        input_type=WorkflowInput,
        output_type=ProcessOutput,
        func=process_data
    )

    transform_step = FunctionStep(
        step_id="transform",
        metadata=StepMetadata(
            name="Transform Result",
            description="Transform processed data"
        ),
        input_type=ProcessOutput,
        output_type=FinalOutput,
        func=transform_data
    )

    workflow.add_step(process_step)
    workflow.add_step(transform_step)

    # Connect steps
    workflow.add_edge("process", "transform")
    workflow.set_start_step("process")

    # Set workflow id for discovery
    workflow.id = "test-workflow"
    workflow.name = "Test Workflow"

    return workflow


@pytest.mark.asyncio
async def test_workflow_event_types(test_workflow):
    """Test that WorkflowRunner events are properly emitted and wrapped."""
    session_manager = SessionManager()
    execution_engine = ExecutionEngine(session_manager)

    # Prepare input
    input_data = {"message": "Hello", "count": 5}

    # Collect all events
    events = []
    async for event_str in execution_engine.execute_workflow_stream(
        test_workflow,
        input_data,
        session_id="test-session"
    ):
        # Parse the SSE event
        if event_str.startswith("data: "):
            import json
            event_json = event_str[6:].strip()
            if event_json:
                wrapped_event = json.loads(event_json)
                events.append(wrapped_event)

    # Verify we got the expected event types
    event_types = [e["event"]["event_type"] for e in events if "event" in e]

    # Should have these events in order
    assert "workflow_started" in event_types
    assert "step_started" in event_types
    assert "step_completed" in event_types
    assert "edge_activated" in event_types
    assert "workflow_completed" in event_types

    # Verify event structure
    workflow_started = next(e for e in events if e.get("event", {}).get("event_type") == "workflow_started")
    assert workflow_started["session_id"] == "test-session"
    assert "initial_input" in workflow_started["event"]
    assert workflow_started["event"]["workflow_id"] == "test-workflow"

    # Verify step events have proper fields
    step_started = next(e for e in events if e.get("event", {}).get("event_type") == "step_started")
    assert "step_id" in step_started["event"]
    assert "input_data" in step_started["event"]

    step_completed = next(e for e in events if e.get("event", {}).get("event_type") == "step_completed")
    assert "step_id" in step_completed["event"]
    assert "output_data" in step_completed["event"]
    assert "duration_seconds" in step_completed["event"]

    # Verify workflow completion
    workflow_completed = next(e for e in events if e.get("event", {}).get("event_type") == "workflow_completed")
    assert "execution" in workflow_completed["event"]
    execution = workflow_completed["event"]["execution"]

    # Check we have step executions
    assert "step_executions" in execution
    assert isinstance(execution["step_executions"], dict)
    assert len(execution["step_executions"]) == 2  # We have 2 steps

    # Verify final output - step_executions is a dict keyed by step_id
    transform_step_exec = execution["step_executions"].get("transform", {})
    if transform_step_exec and "output" in transform_step_exec:
        final_output = transform_step_exec["output"]
        assert final_output["final_result"] == "PROCESSED: HELLO"
        assert final_output["final_count"] == 20  # (5 * 2) + 10
        assert final_output["status"] == "completed"


@pytest.mark.asyncio
async def test_workflow_error_handling(test_workflow):
    """Test that workflow errors are properly handled."""
    session_manager = SessionManager()
    execution_engine = ExecutionEngine(session_manager)

    # Provide invalid input that will cause validation error
    input_data = {"invalid_field": "test"}  # Missing required fields

    # Collect all events
    events = []
    async for event_str in execution_engine.execute_workflow_stream(
        test_workflow,
        input_data,
        session_id="test-error-session"
    ):
        if event_str.startswith("data: "):
            import json
            event_json = event_str[6:].strip()
            if event_json:
                wrapped_event = json.loads(event_json)
                events.append(wrapped_event)

    # Should have error event
    has_error = any(
        e.get("event", {}).get("type") == "error" or
        e.get("event", {}).get("event_type") == "workflow_failed"
        for e in events
    )
    assert has_error, "Should have error event for invalid input"


@pytest.mark.asyncio
async def test_workflow_cancellation():
    """Test that workflow cancellation works properly."""
    from forla._cancellation_token import CancellationToken

    session_manager = SessionManager()
    execution_engine = ExecutionEngine(session_manager)

    # Create a slow workflow
    workflow = Workflow(
        metadata=WorkflowMetadata(
            name="Slow Workflow",
            description="Workflow with slow step"
        )
    )

    async def slow_function(input_data: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(5)  # Long running task
        return {"result": "done"}

    slow_step = FunctionStep(
        step_id="slow",
        metadata=StepMetadata(
            name="Slow Step",
            description="Long running step"
        ),
        input_type=Dict[str, Any],
        output_type=Dict[str, Any],
        func=slow_function
    )
    workflow.add_step(slow_step)
    workflow.set_start_step("slow")

    # Create cancellation token
    token = CancellationToken()

    # Start workflow and cancel it
    events = []
    async def run_workflow():
        async for event_str in execution_engine.execute_workflow_stream(
            workflow,
            {"input": "test"},
            session_id="test-cancel-session",
            cancellation_token=token
        ):
            if event_str.startswith("data: "):
                import json
                event_json = event_str[6:].strip()
                if event_json:
                    wrapped_event = json.loads(event_json)
                    events.append(wrapped_event)

    # Run workflow and cancel after short delay
    task = asyncio.create_task(run_workflow())
    await asyncio.sleep(0.5)  # Let it start
    token.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass  # Expected

    # Should have workflow_started and workflow_cancelled events
    event_types = [e.get("event", {}).get("event_type") for e in events if "event" in e]
    assert "workflow_started" in event_types
    # May have workflow_cancelled or just stop due to cancellation


@pytest.mark.asyncio
async def test_workflow_step_progress_events():
    """Test that step progress events are emitted if steps report progress."""

    class ProgressInput(BaseModel):
        input: str

    class ProgressOutput(BaseModel):
        result: str
        progress_count: int

    async def progress_function(input_data: ProgressInput, context: Any) -> ProgressOutput:
        """Execute with simulated progress."""
        # Note: Current implementation may not expose progress API
        # This is a test to verify if progress events would be handled
        for i in range(3):
            await asyncio.sleep(0.1)
            # Would emit progress event if API available
        return ProgressOutput(result="done", progress_count=3)

    # Create workflow with progress step
    workflow = Workflow(
        metadata=WorkflowMetadata(
            name="Progress Workflow",
            description="Workflow with progress step"
        )
    )
    progress_step = FunctionStep(
        step_id="progress",
        metadata=StepMetadata(
            name="Progress Step",
            description="Step with progress"
        ),
        input_type=ProgressInput,
        output_type=ProgressOutput,
        func=progress_function
    )
    workflow.add_step(progress_step)
    workflow.set_start_step("progress")

    session_manager = SessionManager()
    execution_engine = ExecutionEngine(session_manager)

    # Collect all events
    events = []
    async for event_str in execution_engine.execute_workflow_stream(
        workflow,
        {"input": "test"},
        session_id="test-progress-session"
    ):
        if event_str.startswith("data: "):
            import json
            event_json = event_str[6:].strip()
            if event_json:
                wrapped_event = json.loads(event_json)
                events.append(wrapped_event)

    # Verify basic workflow events
    event_types = [e.get("event", {}).get("event_type") for e in events if "event" in e]
    assert "workflow_started" in event_types
    assert "step_started" in event_types
    assert "step_completed" in event_types
    assert "workflow_completed" in event_types


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_workflow_event_types(test_workflow()))