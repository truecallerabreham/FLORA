import asyncio
from typing import Any, Callable, Optional, Type

from typing_extensions import Self

from ..._component_config import Component
from ..core._models import Context, InputType, OutputType, StepMetadata
from ._step import BaseStep, BaseStepConfig


class FunctionStepConfig(BaseStepConfig):
    """Configuration for FunctionStep serialization."""

    # Base fields inherited: step_id, metadata, input_type_name, output_type_name, input_schema, output_schema
    # Note: We can't easily serialize functions, so we'll store a reference
    function_name: Optional[str] = None
    function_module: Optional[str] = None


class FunctionStep(Component[FunctionStepConfig], BaseStep[InputType, OutputType]):
    """A step that executes a function as its core operation."""

    component_config_schema = FunctionStepConfig
    component_type = "step"
    component_provider_override = "forla.workflow.FunctionStep"

    def __init__(
        self,
        step_id: str,
        metadata: StepMetadata,
        input_type: Type[InputType],
        output_type: Type[OutputType],
        func: Callable[..., Any],
    ):
        """Initialize with a function to execute.

        Args:
            step_id: Unique identifier for this step
            metadata: Step metadata
            input_type: Input validation model
            output_type: Output validation model
            func: Function to execute (can be sync or async)
        """
        super().__init__(step_id, metadata, input_type, output_type)
        self.func = func

    async def execute(self, input_data: InputType, context: Context) -> OutputType:
        """Execute the wrapped function.

        Args:
            input_data: Validated input data
            context: Additional context

        Returns:
            Function output
        """
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(input_data, context)
        else:
            result = self.func(input_data, context)

        if isinstance(result, dict):
            return self.output_type(**result)
        elif hasattr(result, "dict"):
            return result
        else:
            # Assume it's a simple value that can be wrapped
            return self.output_type(result=result)

    def _to_config(self) -> FunctionStepConfig:
        """Convert step to configuration for serialization."""
        func_name = None
        func_module = None

        if hasattr(self.func, "__name__"):
            func_name = self.func.__name__
        if hasattr(self.func, "__module__"):
            func_module = self.func.__module__

        # Get base type serialization data
        base_data = self._serialize_types()

        return FunctionStepConfig(
            **base_data, function_name=func_name, function_module=func_module
        )

    @classmethod
    def _from_config(cls, config: FunctionStepConfig) -> Self:
        """Create step from configuration.

        Args:
            config: Step configuration

        Note:
            This basic implementation cannot recreate the function.
            In practice, you'd need a function registry or other mechanism
            to deserialize callable functions.
        """
        raise NotImplementedError(
            "FunctionStep deserialization is not fully supported as functions "
            "cannot be easily serialized. Consider using a function registry "
            "or other mechanism for this use case."
        )
