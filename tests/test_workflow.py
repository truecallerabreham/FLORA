import pytest
from pydantic import BaseModel
from forla.workflow import (
    FunctionStep, StepMetadata, Context,
    Workflow, WorkflowMetadata,
    WorkflowRunner, WorkflowCompleteEvent,
)


# Simple data models for testing
class NumberInput(BaseModel):
    value: int

class NumberOutput(BaseModel):
    result: int


# Test step functions
async def double_number(input_data: NumberInput, ctx: Context) -> NumberOutput:
    return NumberOutput(result=input_data.value * 2)

async def add_ten(input_data: NumberOutput, ctx: Context) -> NumberOutput:
    return NumberOutput(result=input_data.result + 10)


@pytest.mark.asyncio
async def test_simple_chain():
    """Test a simple A → B sequential workflow.
    
    Example from the book: 3 → double → 6 → add_ten → 16
    """
    double_step = FunctionStep(
        step_id="double",
        metadata=StepMetadata(name="Double the number"),
        input_type=NumberInput,
        output_type=NumberOutput,
        func=double_number,
    )
    
    add_ten_step = FunctionStep(
        step_id="add_ten",
        metadata=StepMetadata(name="Add ten"),
        input_type=NumberOutput,
        output_type=NumberOutput,
        func=add_ten,
    )
    
    workflow = Workflow(
        metadata=WorkflowMetadata(name="Simple Math Pipeline")
    ).chain(double_step, add_ten_step)
    
    runner = WorkflowRunner()
    result = await runner.run(workflow, {"value": 3})
    
    assert isinstance(result, WorkflowCompleteEvent)
    assert result.final_output["result"] == 16   # 3 * 2 + 10 = 16
    assert result.steps_completed == 2
    assert result.steps_failed == 0


@pytest.mark.asyncio
async def test_workflow_streaming():
    """Test that run_stream yields events in the correct order."""
    double_step = FunctionStep(
        step_id="double",
        metadata=StepMetadata(name="Double"),
        input_type=NumberInput,
        output_type=NumberOutput,
        func=double_number,
    )
    
    add_ten_step = FunctionStep(
        step_id="add_ten",
        metadata=StepMetadata(name="Add Ten"),
        input_type=NumberOutput,
        output_type=NumberOutput,
        func=add_ten,
    )
    
    workflow = Workflow(metadata=WorkflowMetadata(name="Test")).chain(
        double_step, add_ten_step
    )
    
    from forla.workflow import WorkflowStepStartEvent, WorkflowStepCompleteEvent
    
    events = []
    async for event in WorkflowRunner().run_stream(workflow, {"value": 5}):
        events.append(event)
    
    # Should have: Start(double), Complete(double), Start(add_ten), Complete(add_ten), Complete(workflow)
    event_types = [type(e).__name__ for e in events]
    assert "WorkflowStepStartEvent" in event_types
    assert "WorkflowStepCompleteEvent" in event_types
    assert "WorkflowCompleteEvent" in event_types
    
    # Final result: 5 * 2 + 10 = 20
    final = events[-1]
    assert final.final_output["result"] == 20
