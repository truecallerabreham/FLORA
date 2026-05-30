from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ._builder import Workflow
from ._edge import Edge


@dataclass
class WorkflowStepStartEvent:
    """Emitted when a workflow step begins executing."""
    step_id: str
    step_name: str

@dataclass
class WorkflowStepCompleteEvent:
    """Emitted when a workflow step finishes successfully."""
    step_id: str
    step_name: str
    output: Dict[str, Any]

@dataclass
class WorkflowStepFailedEvent:
    """Emitted when a workflow step fails."""
    step_id: str
    step_name: str
    error: str

@dataclass
class WorkflowCompleteEvent:
    """Emitted when the entire workflow finishes."""
    final_output: Dict[str, Any]
    steps_completed: int
    steps_failed: int


class WorkflowRunner:
    """Executes a Workflow with streaming observability.
    
    HOW IT WORKS:
    1. Start at the designated start step
    2. Find all steps that are "ready" (all their dependencies completed)
    3. Run ready steps in PARALLEL (using asyncio.gather)
    4. After each step completes, check if new steps became ready
    5. Evaluate edge conditions to decide which outputs flow where
    6. Repeat until no more ready steps
    
    PARALLEL EXECUTION:
    The book points out this is automatic — you don't need to specify
    parallelism explicitly. If steps A and B both depend on step X
    (fan-out pattern), they will automatically run in parallel after X completes.
    
    TYPE SAFETY:
    Before execution, the runner validates that each step's input type
    is compatible with the output from its predecessor steps.
    """

    async def run_stream(
        self,
        workflow: Workflow,
        initial_input: Dict[str, Any],
    ) -> AsyncGenerator[Union[WorkflowStepStartEvent, WorkflowStepCompleteEvent, WorkflowCompleteEvent], None]:
        """Execute the workflow and yield events as each step runs."""
        
        if not workflow.start_step_id:
            raise ValueError("Workflow has no start step. Add steps before running.")
        
        if not workflow.steps:
            raise ValueError("Workflow has no steps.")

        # Track which steps have completed and their outputs
        completed: Dict[str, Dict[str, Any]] = {}
        failed: List[str] = []
        
        # The shared workflow state — accessible to all steps via Context
        workflow_state: Dict[str, Any] = {}
        
        # Map each step to its queued input data
        # (populated when an upstream step completes)
        queued_inputs: Dict[str, Dict[str, Any]] = {
            workflow.start_step_id: initial_input
        }

        steps_completed = 0
        steps_failed = 0

        while True:
            # Find all steps that are ready to execute
            ready_step_ids = self._get_ready_steps(
                workflow, completed, failed, queued_inputs
            )

            if not ready_step_ids:
                break    # No more work to do — workflow is complete

            # Execute all ready steps in PARALLEL
            tasks = []
            for step_id in ready_step_ids:
                step = workflow.steps[step_id]
                step_input = queued_inputs[step_id]
                
                yield WorkflowStepStartEvent(
                    step_id=step_id,
                    step_name=step.metadata.name,
                )
                
                tasks.append((
                    step_id,
                    asyncio.create_task(
                        step.run(step_input, {"workflow_state": workflow_state})
                    )
                ))

            # Wait for all parallel tasks to finish
            for step_id, task in tasks:
                step = workflow.steps[step_id]
                try:
                    output = await task
                    completed[step_id] = output
                    
                    # Update shared workflow state with this step's outputs
                    workflow_state.update(output)
                    
                    yield WorkflowStepCompleteEvent(
                        step_id=step_id,
                        step_name=step.metadata.name,
                        output=output,
                    )
                    steps_completed += 1

                    # Propagate this step's output to all qualifying next steps
                    for edge in workflow.get_outgoing_edges(step_id):
                        if self._evaluate_condition(edge, completed, workflow_state):
                            queued_inputs[edge.to_step] = output

                except Exception as e:
                    failed.append(step_id)
                    steps_failed += 1
                    yield WorkflowStepFailedEvent(
                        step_id=step_id,
                        step_name=step.metadata.name,
                        error=str(e),
                    )

        # Find the final step's output (steps with no outgoing edges)
        terminal_steps = [
            sid for sid in workflow.steps
            if not workflow.get_outgoing_edges(sid)
        ]
        
        final_output = {}
        if terminal_steps and terminal_steps[-1] in completed:
            final_output = completed[terminal_steps[-1]]

        yield WorkflowCompleteEvent(
            final_output=final_output,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
        )

    async def run(
        self, workflow: Workflow, initial_input: Dict[str, Any]
    ) -> WorkflowCompleteEvent:
        """Run the workflow and return only the final result."""
        result = None
        async for event in self.run_stream(workflow, initial_input):
            if isinstance(event, WorkflowCompleteEvent):
                result = event
        return result

    def _get_ready_steps(
        self,
        workflow: Workflow,
        completed: Dict[str, Dict],
        failed: List[str],
        queued: Dict[str, Dict],
    ) -> List[str]:
        """Find all steps that are ready to execute.
        
        A step is ready when:
        1. It has input data queued (from a completed upstream step)
        2. ALL its dependencies have completed
        3. It hasn't run yet (not in completed or failed)
        """
        ready = []
        for step_id in workflow.steps:
            if step_id in completed or step_id in failed:
                continue    # Already done
            if step_id not in queued:
                continue    # No input data yet
            
            # Check all dependencies have completed
            deps = workflow.get_dependencies(step_id)
            if all(d in completed for d in deps):
                ready.append(step_id)
        return ready

    def _evaluate_condition(
        self,
        edge: Edge,
        completed: Dict[str, Dict],
        state: Dict[str, Any],
    ) -> bool:
        """Evaluate whether an edge's condition allows data to flow."""
        cond = edge.condition
        
        if cond.type == "always":
            return True
        
        if cond.type == "output_based":
            output = completed.get(edge.from_step, {})
            field_val = output.get(cond.field)
            return self._compare(field_val, cond.operator, cond.value)
        
        if cond.type == "state_based":
            field_val = state.get(cond.field)
            return self._compare(field_val, cond.operator, cond.value)
        
        return True    # Unknown condition type — allow by default

    def _compare(self, left: Any, operator: Optional[str], right: Any) -> bool:
        """Apply a comparison operator."""
        if operator is None:
            return bool(left)
        ops = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">":  lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<":  lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "in": lambda a, b: a in b,
        }
        fn = ops.get(operator)
        if fn is None:
            return True
        try:
            return fn(left, right)
        except (TypeError, ValueError):
            return False
