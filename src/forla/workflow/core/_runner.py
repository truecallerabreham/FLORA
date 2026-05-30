"""
Workflow runner implementation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

from forla import CancellationToken

from ..steps._step import BaseStep
from ._checkpoint import (
    CheckpointConfig,
    CheckpointValidationResult,
    WorkflowCheckpoint,
)
from ._models import (
    CheckpointSavedEvent,
    EdgeActivatedEvent,
    StepCompletedEvent,
    StepExecution,
    StepFailedEvent,
    StepProgressEvent,
    StepStartedEvent,
    StepStatus,
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowEvent,
    WorkflowExecution,
    WorkflowFailedEvent,
    WorkflowResumedEvent,
    WorkflowStartedEvent,
    WorkflowStatus,
)
from ._workflow import Workflow

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Executes workflows with support for parallel execution."""

    def __init__(self, max_concurrent_steps: int = 5):
        """Initialize the runner.

        Args:
            max_concurrent_steps: Maximum number of steps to run concurrently
        """
        self.max_concurrent_steps = max_concurrent_steps
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_steps)
        # Add cancellation token support
        self._cancellation_tokens: Dict[str, CancellationToken] = {}

    async def run(
        self,
        workflow: Workflow,
        initial_input: Optional[Dict[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> WorkflowExecution:
        """Run a complete workflow and return the final result.

        This is a convenience method that consumes the stream and returns
        only the final WorkflowExecution result.

        Args:
            workflow: Workflow to execute
            initial_input: Initial input data for the start step
            cancellation_token: Optional cancellation token for stopping execution

        Returns:
            Final workflow execution result
        """
        final_execution = None
        async for event in self.run_stream(workflow, initial_input, cancellation_token):
            if event.event_type == "workflow_completed":
                final_execution = getattr(event, "execution", None)
            elif event.event_type == "workflow_failed":
                execution = getattr(event, "execution", None)
                if execution:
                    final_execution = execution
                # Re-raise the error
                error = getattr(event, "error", "Unknown workflow error")
                raise RuntimeError(error)
            elif event.event_type == "workflow_cancelled":
                execution = getattr(event, "execution", None)
                if execution:
                    final_execution = execution
                # Re-raise cancellation
                reason = getattr(event, "reason", "Workflow cancelled")
                raise RuntimeError(reason)

        if final_execution is None:
            raise RuntimeError("Workflow completed but no final execution received")

        return final_execution

    async def run_stream(
        self,
        workflow: Workflow,
        initial_input: Optional[Dict[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        checkpoint: Optional[WorkflowCheckpoint] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run a workflow and yield real-time events.

        Args:
            workflow: Workflow to execute
            initial_input: Initial input data for the start step
            cancellation_token: Optional cancellation token for stopping execution
            checkpoint: Optional checkpoint to resume from
            checkpoint_config: Optional config for checkpoint behavior (defaults to
                in-memory storage with auto-save enabled)

        Yields:
            WorkflowEvent: Real-time workflow events

        Raises:
            Exception: If workflow validation fails or execution encounters errors
        """
        logger.info(f"Starting workflow execution: {workflow.id}")

        # Store cancellation token for this execution
        if cancellation_token:
            self._cancellation_tokens[workflow.id] = cancellation_token

        # Handle checkpoint resume
        if checkpoint:
            # Validate checkpoint compatibility
            validation = self.validate_checkpoint(workflow, checkpoint)

            if not validation.can_resume:
                error_msg = f"Checkpoint validation failed: {validation.errors}"
                logger.error(error_msg)
                yield WorkflowFailedEvent(
                    timestamp=datetime.now(), workflow_id=workflow.id, error=error_msg
                )
                return

            # Emit warnings if any
            if validation.warnings:
                logger.warning(f"Checkpoint warnings: {validation.warnings}")

            # Use checkpoint's execution state
            execution = checkpoint.execution
            logger.info(
                f"Resuming from checkpoint: "
                f"{len(checkpoint.completed_step_ids)} steps completed, "
                f"{len(checkpoint.pending_step_ids)} pending"
            )

            # Emit resume event
            yield WorkflowResumedEvent(
                timestamp=datetime.now(),
                workflow_id=workflow.id,
                checkpoint_id=checkpoint.checkpoint_id,
                completed_steps=checkpoint.completed_step_ids,
                pending_steps=checkpoint.pending_step_ids,
            )
        else:
            # Fresh start - emit started event
            yield WorkflowStartedEvent(
                timestamp=datetime.now(),
                workflow_id=workflow.id,
                initial_input=initial_input or {},
            )

            # Validate workflow
            validation = workflow.validate_workflow()
            if not validation.is_valid:
                error_msg = f"Workflow validation failed: {validation.errors}"
                logger.error(error_msg)
                yield WorkflowFailedEvent(
                    timestamp=datetime.now(), workflow_id=workflow.id, error=error_msg
                )
                return

            # Validate initial input matches start step's input type
            if initial_input and workflow.start_step_id:
                start_step = workflow.steps.get(workflow.start_step_id)
                if start_step:
                    try:
                        # Try to validate initial input against start step's input type
                        start_step.input_type(**initial_input)
                    except Exception as e:
                        error_msg = (
                            f"Initial input validation failed: Input does not match start step '{workflow.start_step_id}' "
                            f"input type {start_step.input_type.__name__}: {str(e)}"
                        )
                        logger.error(error_msg)
                        yield WorkflowFailedEvent(
                            timestamp=datetime.now(),
                            workflow_id=workflow.id,
                            error=error_msg,
                        )
                        return

            # Create fresh execution record
            execution = WorkflowExecution(
                workflow_id=workflow.id,
                status=WorkflowStatus.RUNNING,
                start_time=datetime.now(),
                state=workflow.initial_state.copy(),
            )

            # Add initial input to state
            if initial_input:
                execution.state.update(initial_input)

        try:
            # Execute the workflow with streaming events (with checkpoint support)
            async for event in self._execute_workflow_stream(
                workflow=workflow,
                execution=execution,
                initial_input=initial_input or {},
                checkpoint_config=checkpoint_config,
            ):
                yield event

            # Check final status and emit completion event
            if all(
                step_exec.status == StepStatus.COMPLETED
                for step_exec in execution.step_executions.values()
            ):
                execution.status = WorkflowStatus.COMPLETED
                execution.end_time = datetime.now()
                logger.info(f"Workflow {workflow.id} completed successfully")

                yield WorkflowCompletedEvent(
                    timestamp=datetime.now(),
                    workflow_id=workflow.id,
                    execution=execution,
                )
            else:
                execution.status = WorkflowStatus.FAILED
                execution.end_time = datetime.now()
                error_msg = f"Workflow {workflow.id} failed"
                logger.error(error_msg)

                yield WorkflowFailedEvent(
                    timestamp=datetime.now(),
                    workflow_id=workflow.id,
                    error=error_msg,
                    execution=execution,
                )

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            logger.error(f"Workflow {workflow.id} failed with error: {e}")

            yield WorkflowFailedEvent(
                timestamp=datetime.now(),
                workflow_id=workflow.id,
                error=str(e),
                execution=execution,
            )
        finally:
            # Clean up cancellation token
            self._cancellation_tokens.pop(workflow.id, None)

    async def _execute_workflow_stream(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        initial_input: Dict[str, Any],
        checkpoint_config: Optional[CheckpointConfig] = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Execute the workflow steps and yield events.

        Args:
            workflow: Workflow to execute
            execution: Execution context
            initial_input: Initial input data
            checkpoint_config: Optional checkpoint configuration

        Yields:
            WorkflowEvent: Step execution events
        """
        # Identify already-completed steps from checkpoint
        completed_steps: set[str] = {
            step_id
            for step_id, step_exec in execution.step_executions.items()
            if step_exec.status == StepStatus.COMPLETED
        }

        # Log resume info
        if completed_steps:
            logger.info(
                f"Skipping {len(completed_steps)} completed steps: {completed_steps}"
            )

        running_tasks: Dict[str, asyncio.Task[Dict[str, Any]]] = {}
        steps_since_last_checkpoint = 0

        # Create a shared queue for progress events from all steps
        progress_queue: asyncio.Queue[tuple[str, Dict[str, Any]]] = asyncio.Queue()

        while len(completed_steps) < len(workflow.steps):
            # Check for cancellation before starting any new steps
            cancellation_token = self._cancellation_tokens.get(workflow.id)
            if cancellation_token and cancellation_token.is_cancelled():
                # Mark all running steps as cancelled (graceful)
                for step_id, task in running_tasks.items():
                    step_execution = execution.step_executions.get(step_id)
                    if step_execution and step_execution.status == StepStatus.RUNNING:
                        step_execution.status = StepStatus.CANCELLED
                        step_execution.end_time = datetime.now()
                        step_execution.error = (
                            "Step cancelled due to workflow cancellation"
                        )
                        yield StepFailedEvent(
                            timestamp=datetime.now(),
                            workflow_id=workflow.id,
                            step_id=step_id,
                            error="Step cancelled due to workflow cancellation",
                            duration_seconds=0.0,
                        )
                if not running_tasks:
                    execution.status = WorkflowStatus.CANCELLED
                    execution.end_time = datetime.now()
                    yield WorkflowCancelledEvent(
                        timestamp=datetime.now(),
                        workflow_id=workflow.id,
                        execution=execution,
                        reason="Cancelled by user",
                    )
                    return

            # Get steps ready to run
            ready_steps = workflow.get_ready_steps(execution)
            ready_steps = [
                s
                for s in ready_steps
                if s not in completed_steps and s not in running_tasks
            ]

            if not ready_steps and not running_tasks:
                # No ready steps and nothing running - check if we're stuck
                remaining_steps = set(workflow.steps.keys()) - completed_steps
                if remaining_steps:
                    error_msg = f"Workflow stuck: remaining steps {remaining_steps} cannot be executed"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                break

            # Start new tasks for ready steps (only if not cancelled)
            if not (cancellation_token and cancellation_token.is_cancelled()):
                for step_id in ready_steps:
                    if len(running_tasks) >= self.max_concurrent_steps:
                        break

                    step: BaseStep[Any, Any] = workflow.steps[step_id]
                    input_data = self._prepare_step_input(
                        step_id, workflow, execution, initial_input
                    )

                    # Create step execution record
                    step_execution = StepExecution(
                        step_id=step_id,
                        status=StepStatus.RUNNING,
                        start_time=datetime.now(),
                        input_data=input_data,
                    )
                    execution.step_executions[step_id] = step_execution

                    # Emit step started event
                    yield StepStartedEvent(
                        timestamp=datetime.now(),
                        workflow_id=workflow.id,
                        step_id=step_id,
                        input_data=input_data,
                    )

                    # Start the step task
                    task = asyncio.create_task(
                        self._run_step_with_semaphore(
                            step,
                            step_id,
                            input_data,
                            execution.state,
                            progress_queue,
                            workflow.id,
                            cancellation_token,
                        )
                    )
                    # Link task to cancellation token for immediate cancellation
                    if cancellation_token:
                        cancellation_token.link_future(task)
                    running_tasks[step_id] = task

                    logger.info(f"Started step {step_id} in workflow {workflow.id}")

            # Wait for at least one task to complete (or very short timeout to check progress)
            if running_tasks:
                try:
                    done, pending = await asyncio.wait(
                        running_tasks.values(),
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=0.01,  # Check progress queue frequently
                    )
                except asyncio.TimeoutError:
                    done = set()

                # Drain progress queue and yield progress events
                while not progress_queue.empty():
                    try:
                        step_id_prog, progress_data = progress_queue.get_nowait()
                        yield StepProgressEvent(
                            timestamp=datetime.now(),
                            workflow_id=workflow.id,
                            step_id=step_id_prog,
                            message=progress_data["message"],
                            completed=progress_data.get("completed"),
                            total=progress_data.get("total"),
                            metadata=progress_data.get("metadata", {}),
                        )
                    except asyncio.QueueEmpty:
                        break

                # If no tasks completed, continue to next loop iteration
                if not done:
                    continue

                # Process completed tasks
                for task in done:
                    step_id = None
                    for sid, t in running_tasks.items():
                        if t == task:
                            step_id = sid
                            break

                    if step_id:
                        step_execution = execution.step_executions[step_id]

                        try:
                            result = await task
                            step_execution.status = StepStatus.COMPLETED
                            step_execution.output_data = result
                            step_execution.end_time = datetime.now()

                            # Calculate duration
                            duration = 0.0
                            if step_execution.end_time and step_execution.start_time:
                                duration = (
                                    step_execution.end_time - step_execution.start_time
                                ).total_seconds()

                            # Update workflow state with step output
                            execution.state[f"{step_id}_output"] = result

                            completed_steps.add(step_id)
                            logger.info(f"Step {step_id} completed successfully")

                            # Emit step completed event
                            yield StepCompletedEvent(
                                timestamp=datetime.now(),
                                workflow_id=workflow.id,
                                step_id=step_id,
                                output_data=result,
                                duration_seconds=duration,
                            )

                            # Emit edge activation events for next steps
                            for edge in workflow.edges:
                                if edge.from_step == step_id:
                                    yield EdgeActivatedEvent(
                                        timestamp=datetime.now(),
                                        workflow_id=workflow.id,
                                        from_step=step_id,
                                        to_step=edge.to_step,
                                        data=result,
                                    )

                            # Auto-checkpoint logic
                            steps_since_last_checkpoint += 1
                            if checkpoint_config and checkpoint_config.auto_save:
                                if (
                                    steps_since_last_checkpoint
                                    >= checkpoint_config.save_interval_steps
                                ):
                                    # Create checkpoint
                                    checkpoint = self._create_checkpoint(
                                        workflow=workflow,
                                        execution=execution,
                                        checkpoint_type="auto",
                                    )

                                    # Save using configured store
                                    await checkpoint_config.store.save(checkpoint)

                                    # Emit checkpoint event
                                    yield CheckpointSavedEvent(
                                        timestamp=datetime.now(),
                                        workflow_id=workflow.id,
                                        checkpoint_id=checkpoint.checkpoint_id,
                                        completed_steps=len(completed_steps),
                                        total_steps=len(workflow.steps),
                                    )

                                    # Auto-cleanup if enabled
                                    if checkpoint_config.auto_cleanup:
                                        await checkpoint_config.store.cleanup_old(
                                            workflow_id=workflow.id,
                                            keep_last_n=checkpoint_config.keep_last_n,
                                        )

                                    steps_since_last_checkpoint = 0

                        except asyncio.CancelledError:
                            step_execution.status = StepStatus.FAILED
                            step_execution.error = "Step was cancelled"
                            step_execution.end_time = datetime.now()

                            # Calculate duration
                            duration = 0.0
                            if step_execution.end_time and step_execution.start_time:
                                duration = (
                                    step_execution.end_time - step_execution.start_time
                                ).total_seconds()

                            logger.info(f"Step {step_id} was cancelled")

                            # Emit step failed event for cancellation
                            yield StepFailedEvent(
                                timestamp=datetime.now(),
                                workflow_id=workflow.id,
                                step_id=step_id,
                                error="Step was cancelled",
                                duration_seconds=duration,
                            )

                            # Don't re-raise CancelledError, just continue

                        except Exception as e:
                            step_execution.status = StepStatus.FAILED
                            step_execution.error = str(e)
                            step_execution.end_time = datetime.now()

                            # Calculate duration
                            duration = 0.0
                            if step_execution.end_time and step_execution.start_time:
                                duration = (
                                    step_execution.end_time - step_execution.start_time
                                ).total_seconds()

                            logger.error(f"Step {step_id} failed: {e}")

                            # Emit step failed event
                            yield StepFailedEvent(
                                timestamp=datetime.now(),
                                workflow_id=workflow.id,
                                step_id=step_id,
                                error=str(e),
                                duration_seconds=duration,
                            )

                            # For now, fail the entire workflow if any step fails
                            # In the future, we could add error handling strategies
                            raise

                        finally:
                            del running_tasks[step_id]

            # Check if we've reached an end step
            if any(step_id in completed_steps for step_id in workflow.end_step_ids):
                logger.info(f"Reached end step in workflow {workflow.id}")
                break

    async def _run_step_with_semaphore(
        self,
        step: BaseStep[Any, Any],
        step_id: str,
        input_data: Dict[str, Any],
        workflow_state: Dict[str, Any],
        progress_queue: asyncio.Queue[tuple[str, Dict[str, Any]]],
        workflow_id: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Dict[str, Any]:
        """Run a step with concurrency control and progress tracking.

        Args:
            step: Step to execute
            step_id: ID of the step being executed
            input_data: Input data for the step
            workflow_state: Current workflow state
            progress_queue: Queue for progress events
            workflow_id: Workflow ID for logging
            cancellation_token: Optional cancellation token

        Returns:
            Step output data
        """
        async with self._execution_semaphore:
            from ._models import Context

            # Create typed context that directly references workflow_state
            # This ensures modifications are persistent across steps
            typed_context = Context.from_state_ref(workflow_state)

            # Set up progress callback
            def progress_callback(progress_data: Dict[str, Any]) -> None:
                """Callback for progress updates from step execution."""
                try:
                    progress_queue.put_nowait((step_id, progress_data))
                except asyncio.QueueFull:
                    logger.warning(
                        f"Progress queue full, dropping progress update for step {step_id}"
                    )

            typed_context._progress_callback = progress_callback

            # Convert to dict for step.run(), but context modifications
            # will still affect the original workflow_state since it's the same dict reference
            context = typed_context.to_dict()

            # Add the typed context object itself so steps can access methods like emit_progress()
            context["_context_obj"] = typed_context

            # Add cancellation token to context so steps can check it
            if cancellation_token:
                context["cancellation_token"] = cancellation_token

            return await step.run(input_data, context)

    def _prepare_step_input(
        self,
        step_id: str,
        workflow: Workflow,
        execution: WorkflowExecution,
        initial_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare input data for a step using direct type forwarding.

        Args:
            step_id: Step to prepare input for
            workflow: Workflow being executed
            execution: Current execution state
            initial_input: Initial workflow input

        Returns:
            Input data for the step
        """
        # Start with initial input for the start step
        if step_id == workflow.start_step_id:
            return initial_input.copy()

        # For other steps, use direct output forwarding from dependencies
        dependencies = workflow.get_step_dependencies(step_id)

        if not dependencies:
            # No dependencies, use initial input
            return initial_input.copy()

        # For conditional workflows: find the dependency that actually executed and has output
        # Check all dependencies and find the one that completed successfully
        completed_dependency = None
        for dep_id in dependencies:
            dep_execution = execution.step_executions.get(dep_id)
            if (
                dep_execution
                and dep_execution.output_data
                and dep_execution.status.value == "completed"
            ):
                # For conditional workflows, we should use the most recent dependency that completed
                # The workflow's get_ready_steps already handles edge condition evaluation
                completed_dependency = dep_id
                break

        if completed_dependency:
            dep_execution = execution.step_executions.get(completed_dependency)
            if dep_execution and dep_execution.output_data:
                logger.info(
                    f"Using output from completed dependency {completed_dependency} for step {step_id}"
                )
                return dep_execution.output_data.copy()

        # Fallback to initial input if no valid dependency output is available
        logger.warning(
            f"No valid completed dependency found for step {step_id}, using initial input"
        )
        return initial_input.copy()

    async def run_step(
        self,
        step: BaseStep[Any, Any],
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single step independently.

        Args:
            step: Step to execute
            input_data: Input data
            context: Additional context

        Returns:
            Step output data
        """
        context = context or {}
        return await step.run(input_data, context)

    def get_execution_status(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Get detailed status of a workflow execution.

        Args:
            execution: Workflow execution to analyze

        Returns:
            Status information
        """
        total_steps = len(execution.step_executions)
        completed_steps = sum(
            1
            for step_exec in execution.step_executions.values()
            if step_exec.status == StepStatus.COMPLETED
        )
        failed_steps = sum(
            1
            for step_exec in execution.step_executions.values()
            if step_exec.status == StepStatus.FAILED
        )
        running_steps = sum(
            1
            for step_exec in execution.step_executions.values()
            if step_exec.status == StepStatus.RUNNING
        )

        duration = None
        if execution.start_time and execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()

        return {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "progress": {
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "running_steps": running_steps,
                "percentage": (completed_steps / total_steps * 100)
                if total_steps > 0
                else 0,
            },
            "timing": {
                "start_time": execution.start_time,
                "end_time": execution.end_time,
                "duration_seconds": duration,
            },
            "error": execution.error,
        }

    def validate_checkpoint(
        self, workflow: Workflow, checkpoint: WorkflowCheckpoint
    ) -> CheckpointValidationResult:
        """
        Validate if checkpoint is compatible with workflow.

        Checks:
        1. Workflow ID matches (warning if different)
        2. Structure hash matches (error if different)
        3. All completed steps still exist in workflow

        Args:
            workflow: Workflow to validate against
            checkpoint: Checkpoint to validate

        Returns:
            Validation result with can_resume flag
        """
        result = CheckpointValidationResult(is_valid=True, can_resume=True)

        # Check 1: Workflow ID match (warning only)
        if checkpoint.workflow_id != workflow.id:
            result.warnings.append(
                f"Checkpoint workflow_id '{checkpoint.workflow_id}' differs from "
                f"current workflow '{workflow.id}'. This is OK if you renamed the workflow."
            )

        # Check 2: Structure hash match (CRITICAL)
        current_hash = workflow.compute_structure_hash()
        if checkpoint.workflow_structure_hash != current_hash:
            result.errors.append(
                f"Workflow structure has changed since checkpoint was created. "
                f"Cannot safely resume. "
                f"Checkpoint hash: {checkpoint.workflow_structure_hash}, "
                f"Current hash: {current_hash}"
            )
            result.is_valid = False
            result.can_resume = False
            return result

        # Check 3: Completed steps still exist
        for step_id in checkpoint.completed_step_ids:
            if step_id not in workflow.steps:
                result.errors.append(
                    f"Checkpoint references completed step '{step_id}' "
                    f"that no longer exists in workflow"
                )
                result.is_valid = False
                result.can_resume = False

        # Add helpful info
        result.checkpoint_info = {
            "created_at": checkpoint.created_at.isoformat(),
            "completed_steps": len(checkpoint.completed_step_ids),
            "pending_steps": len(checkpoint.pending_step_ids),
            "checkpoint_type": checkpoint.checkpoint_type,
        }

        return result

    def _create_checkpoint(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        checkpoint_type: str = "manual",
    ) -> WorkflowCheckpoint:
        """
        Create a checkpoint from current execution state.

        Args:
            workflow: Workflow being executed
            execution: Current execution state
            checkpoint_type: Type of checkpoint (manual, auto, etc.)

        Returns:
            WorkflowCheckpoint ready to save
        """
        return WorkflowCheckpoint.from_execution(
            execution=execution,
            workflow_id=workflow.id,
            workflow_version=workflow.metadata.version,
            workflow_structure_hash=workflow.compute_structure_hash(),
            all_step_ids=list(workflow.steps.keys()),
            checkpoint_type=checkpoint_type,
        )

    async def cancel_workflow(
        self, workflow_id: str, reason: str = "Cancelled by user"
    ) -> bool:
        """Cancel a running workflow.

        Args:
            workflow_id: ID of the workflow to cancel
            reason: Reason for cancellation

        Returns:
            True if workflow was cancelled, False if not found or already completed
        """
        cancellation_token = self._cancellation_tokens.get(workflow_id)
        if cancellation_token:
            cancellation_token.cancel()
            logger.info(f"Workflow {workflow_id} cancellation requested: {reason}")
            return True
        else:
            logger.warning(f"Workflow {workflow_id} not found or already completed")
            return False
