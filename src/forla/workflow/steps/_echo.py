import asyncio
from typing import Type

from typing_extensions import Self

from ..._component_config import Component
from ..core._models import Context, InputType, OutputType, StepMetadata
from ._step import BaseStep, BaseStepConfig


class EchoStepConfig(BaseStepConfig):
    """Configuration for EchoStep serialization."""

    # Base fields inherited: step_id, metadata, input_type_name, output_type_name, input_schema, output_schema
    prefix: str = "Echo: "
    suffix: str = ""
    delay_seconds: float = 3  # Optional delay for testing/demo


class EchoStep(Component[EchoStepConfig], BaseStep[InputType, OutputType]):
    """A simple step that echoes input with prefix/suffix - fully serializable."""

    component_config_schema = EchoStepConfig
    component_type = "step"
    component_provider_override = "forla.workflow.EchoStep"

    def __init__(
        self,
        step_id: str,
        metadata: StepMetadata,
        input_type: Type[InputType],
        output_type: Type[OutputType],
        prefix: str = "Echo: ",
        suffix: str = "",
        delay_seconds: float = 0.0,
    ):
        """Initialize the echo step.

        Args:
            step_id: Unique identifier for this step
            metadata: Step metadata
            input_type: Pydantic model class for input validation
            output_type: Pydantic model class for output validation
            prefix: String to prepend to input
            suffix: String to append to input
            delay_seconds: Optional delay for testing/demo
        """
        super().__init__(step_id, metadata, input_type, output_type)
        self.prefix = prefix
        self.suffix = suffix
        self.delay_seconds = delay_seconds

    async def execute(self, input_data: InputType, context: Context) -> OutputType:
        """Execute the echo operation, with optional delay for testing/demo."""
        # Optional delay for testing/demo
        if self.delay_seconds and self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        # Try to get the message from different possible field names
        message = None

        # Try common field names
        for field_name in ["message", "result", "text", "content", "data"]:
            if hasattr(input_data, field_name):
                message = getattr(input_data, field_name)
                break

        # If no common field found, try the first field
        if message is None:
            field_names = list(input_data.model_fields.keys())
            if field_names:
                message = getattr(input_data, field_names[0])
            else:
                # Fall back to string representation
                message = str(input_data)

        result = f"{self.prefix}{message}{self.suffix}"

        # Store echo operation in context
        context.set(
            f"{self.step_id}_echo_info",
            {
                "original": message,
                "prefix": self.prefix,
                "suffix": self.suffix,
                "result": result,
            },
        )

        # Create output - try different field names
        output_fields = list(self.output_type.model_fields.keys())
        if "result" in output_fields:
            return self.output_type(result=result)
        elif "message" in output_fields:
            return self.output_type(message=result)
        elif "text" in output_fields:
            return self.output_type(text=result)
        elif "response" in output_fields:
            return self.output_type(response=result)
        elif "content" in output_fields:
            return self.output_type(content=result)
        else:
            # Fall back to first field
            field_name = output_fields[0]
            return self.output_type(**{field_name: result})

    def _to_config(self) -> EchoStepConfig:
        """Convert step to configuration for serialization."""
        # Get base type serialization data
        base_data = self._serialize_types()
        return EchoStepConfig(
            **base_data,
            prefix=self.prefix,
            suffix=self.suffix,
            delay_seconds=self.delay_seconds,
        )

    @classmethod
    def _from_config(cls, config: EchoStepConfig) -> Self:
        """Create step from configuration using shared schema-based deserialization.
        Args:
            config: Step configuration with embedded schemas
        Returns:
            Recreated EchoStep instance with dynamically created types
        """
        # Use shared type deserialization
        input_type, output_type = cls._deserialize_types(config)
        return cls(
            step_id=config.step_id,
            metadata=config.metadata,
            input_type=input_type,
            output_type=output_type,
            prefix=config.prefix,
            suffix=config.suffix,
            delay_seconds=getattr(config, "delay_seconds", 0.0),
        )
