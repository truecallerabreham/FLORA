from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel

# Generic type variables for type-safe step inputs and outputs
InputType = TypeVar("InputType", bound=BaseModel)
OutputType = TypeVar("OutputType", bound=BaseModel)


class StepStatus(str, Enum):
    """The lifecycle states of a workflow step."""
    PENDING = "pending"     # Waiting to be executed
    RUNNING = "running"     # Currently executing
    COMPLETED = "completed" # Successfully finished
    FAILED = "failed"       # Finished with an error
    SKIPPED = "skipped"     # Skipped due to a false condition


class StepMetadata(BaseModel):
    """Configuration for how a step should execute."""
    name: str
    description: str = ""
    max_retries: int = 0              # How many times to retry on failure
    timeout_seconds: Optional[int] = None  # None means no timeout


class Context(BaseModel):
    """Shared state accessible to all steps in a workflow.
    
    WHY is this separate from step input/output?
    Sometimes steps need to communicate beyond their direct data flow.
    For example:
    - An authentication step puts a token in context
    - All subsequent API-calling steps retrieve that token
    - Without context, you'd have to pass the token through every step explicitly
    
    Context is mutable — any step can read and write to it.
    Input/output types enforce a specific data contract between steps.
    """
    state: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.state


class BaseStep(ABC, Generic[InputType, OutputType]):
    """A single unit of computation in a workflow.
    
    GENERIC TYPES:
    BaseStep[InputType, OutputType] means this step takes InputType in
    and produces OutputType out.
    
    Example:
        class NumberInput(BaseModel): value: int
        class NumberOutput(BaseModel): result: int
        
        class DoubleStep(BaseStep[NumberInput, NumberOutput]):
            async def execute(self, input, ctx):
                return NumberOutput(result=input.value * 2)
    
    The generic types enable:
    1. Type checking at development time (your IDE can catch type errors)
    2. Runtime validation (Pydantic validates the data before passing to execute)
    3. Workflow validation (you can check A.OutputType == B.InputType before running)
    """

    def __init__(
        self,
        step_id: str,
        metadata: StepMetadata,
        input_type: Type[InputType],
        output_type: Type[OutputType],
    ):
        self.step_id = step_id
        self.metadata = metadata
        self.input_type = input_type
        self.output_type = output_type

        # Status tracking for observability and debugging
        self._status = StepStatus.PENDING
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        self._error: Optional[str] = None

    @abstractmethod
    async def execute(self, input_data: InputType, context: Context) -> OutputType:
        """The core computation. Implement this in subclasses."""
        pass

    async def run(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute this step with full lifecycle management.
        
        This wrapper handles:
        1. Input validation (via Pydantic)
        2. Status tracking
        3. Timeout management
        4. Retry logic
        5. Error capture
        
        Note: It takes and returns Dict (not typed models)
        because the workflow runner works with raw data between steps.
        """
        self._status = StepStatus.RUNNING
        self._started_at = datetime.now()

        last_exception = None
        
        for attempt in range(self.metadata.max_retries + 1):
            try:
                # Step 1: Validate and convert the raw dict to the typed input model
                validated_input = self.input_type(**input_data)
                
                # Step 2: Build the typed context
                workflow_state = context.get("workflow_state", {})
                typed_context = Context(state=dict(workflow_state))

                # Step 3: Execute with optional timeout
                if self.metadata.timeout_seconds:
                    output = await asyncio.wait_for(
                        self.execute(validated_input, typed_context),
                        timeout=self.metadata.timeout_seconds,
                    )
                else:
                    output = await self.execute(validated_input, typed_context)

                # Step 4: Convert the typed output back to a dict
                self._status = StepStatus.COMPLETED
                self._completed_at = datetime.now()
                
                # output may be a Pydantic model or a dict
                if hasattr(output, "model_dump"):
                    return output.model_dump()
                return dict(output) if output else {}

            except asyncio.TimeoutError as e:
                last_exception = e
                if attempt < self.metadata.max_retries:
                    await asyncio.sleep(2 ** attempt)   # Exponential backoff

            except Exception as e:
                last_exception = e
                if attempt < self.metadata.max_retries:
                    await asyncio.sleep(2 ** attempt)

        # All retries exhausted
        self._status = StepStatus.FAILED
        self._error = str(last_exception)
        raise last_exception


class FunctionStep(BaseStep[InputType, OutputType]):
    """Wraps a Python function as a workflow step.
    
    This is the most common step type for simple transformations.
    The function can be synchronous or asynchronous.
    
    FLEXIBLE OUTPUT HANDLING:
    The function can return:
    - A Pydantic model of output_type: used directly
    - A dict: converted to output_type(**dict)
    - A simple value: wrapped as output_type(result=value) if possible
    """

    def __init__(
        self,
        step_id: str,
        metadata: StepMetadata,
        input_type: Type[InputType],
        output_type: Type[OutputType],
        func: Callable,
    ):
        super().__init__(step_id, metadata, input_type, output_type)
        self._func = func

    async def execute(self, input_data: InputType, context: Context) -> OutputType:
        """Execute the wrapped function."""
        if asyncio.iscoroutinefunction(self._func):
            result = await self._func(input_data, context)
        else:
            result = self._func(input_data, context)

        # Flexible coercion to the expected output type
        if isinstance(result, self.output_type):
            return result
        elif isinstance(result, dict):
            return self.output_type(**result)
        else:
            # Try wrapping a simple value in a 'result' field
            try:
                return self.output_type(result=result)
            except Exception:
                raise TypeError(
                    f"FunctionStep '{self.step_id}': cannot convert return value "
                    f"of type {type(result)} to {self.output_type}"
                )
