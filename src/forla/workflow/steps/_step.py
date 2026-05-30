import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type

from pydantic import BaseModel, create_model

from ..._component_config import ComponentBase
from ..core._models import Context, InputType, OutputType, StepMetadata, StepStatus

logger = logging.getLogger(__name__)


class BaseStepConfig(BaseModel):
    """Base configuration that all step configs must inherit from.

    Ensures UI compatibility by requiring type schema information.
    """

    step_id: str
    metadata: StepMetadata
    input_type_name: str
    output_type_name: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class BaseStep(ComponentBase[BaseStepConfig], Generic[InputType, OutputType]):
    """Base class for all workflow steps with automatic type serialization."""

    def __init__(
        self,
        step_id: str,
        metadata: StepMetadata,
        input_type: Type[InputType],
        output_type: Type[OutputType],
    ):
        """Initialize the step.

        Args:
            step_id: Unique identifier for this step
            metadata: Step metadata including name, description, etc.
            input_type: Pydantic model class for input validation
            output_type: Pydantic model class for output validation
        """
        self.step_id = step_id
        self.metadata = metadata
        self.input_type = input_type
        self.output_type = output_type
        self._status = StepStatus.PENDING
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._error: Optional[str] = None

    def _serialize_types(self) -> Dict[str, Any]:
        """Serialize input/output types to config data.

        Returns:
            Dictionary containing type names and schemas for serialization
        """
        return {
            "step_id": self.step_id,
            "metadata": self.metadata,
            "input_type_name": self.input_type.__name__,
            "output_type_name": self.output_type.__name__,
            "input_schema": self.input_type.model_json_schema(),
            "output_schema": self.output_type.model_json_schema(),
        }

    @classmethod
    def _deserialize_types(
        cls, config: BaseStepConfig
    ) -> Tuple[Type[InputType], Type[OutputType]]:
        """Deserialize input/output types from config data using Pydantic's create_model.

        Args:
            config: Step configuration with embedded schemas

        Returns:
            Tuple of (input_type, output_type) recreated from schemas
        """
        from typing import Any as AnyType
        from typing import Dict, List, Optional, Union

        def extract_type_from_schema_improved(field_schema: Dict[str, Any]) -> Any:
            """
            Extract the proper Python type from a JSON schema field definition.

            This improved version properly handles Optional types and preserves type information.
            """
            # Handle direct type
            if "type" in field_schema:
                return _json_type_to_python_type(field_schema["type"], field_schema)

            # Handle anyOf schemas (common for Optional and Union types)
            if "anyOf" in field_schema:
                any_of_types = field_schema["anyOf"]

                # Extract all type options
                type_options = []
                has_null = False

                for type_option in any_of_types:
                    if isinstance(type_option, dict):
                        option_type = type_option.get("type")
                        if option_type == "null":
                            has_null = True
                        elif option_type:
                            python_type = _json_type_to_python_type(
                                option_type, type_option
                            )
                            type_options.append(python_type)

                # If we have multiple non-null types, it's a Union
                if len(type_options) > 1:
                    if has_null:
                        # Optional[Union[...]]
                        return Optional[Union[tuple(type_options)]]
                    else:
                        # Union[...]
                        return Union[tuple(type_options)]
                elif len(type_options) == 1:
                    if has_null:
                        # Optional[Type]
                        return Optional[type_options[0]]
                    else:
                        # Just the single type
                        return type_options[0]

            # Fallback to Any
            return AnyType

        def _json_type_to_python_type(
            json_type: str, schema_details: Optional[Dict[str, Any]] = None
        ) -> Any:
            """Convert JSON schema type to Python type with enhanced type preservation."""
            if schema_details is None:
                schema_details = {}

            if json_type == "string":
                return str
            elif json_type == "integer":
                return int
            elif json_type == "number":
                return float
            elif json_type == "boolean":
                return bool
            elif json_type == "array":
                # Try to preserve item type information
                items_schema = schema_details.get("items", {})
                if items_schema:
                    if isinstance(items_schema, dict) and "type" in items_schema:
                        item_type = _json_type_to_python_type(
                            items_schema["type"], items_schema
                        )
                        return List[item_type]
                    else:
                        # Complex items schema, fallback to Any
                        return List[AnyType]
                return List[AnyType]
            elif json_type == "object":
                # Try to preserve value type information for Dict
                additional_props = schema_details.get("additionalProperties", {})
                if additional_props and isinstance(additional_props, dict):
                    if "type" in additional_props:
                        value_type = _json_type_to_python_type(
                            additional_props["type"], additional_props
                        )
                        return Dict[str, value_type]
                    elif "items" in additional_props:
                        # Dict[str, List[...]]
                        items_type = _json_type_to_python_type(
                            additional_props["items"].get("type", "string"),
                            additional_props["items"],
                        )
                        return Dict[str, List[items_type]]
                return Dict[str, AnyType]
            else:
                return AnyType

        def schema_to_field_definitions(schema: Dict[str, Any]) -> Dict[str, Any]:
            """Convert JSON schema to create_model field definitions with proper Optional handling."""
            properties = schema.get("properties", {})
            required_fields = set(schema.get("required", []))
            field_definitions = {}

            for field_name, field_schema in properties.items():
                # Use improved type extraction
                python_type = extract_type_from_schema_improved(field_schema)

                if field_name in required_fields:
                    # For required fields, use (type, ...) format
                    field_definitions[field_name] = (python_type, ...)
                else:
                    # For optional fields, use the explicit default or None
                    default_value = field_schema.get("default", None)
                    field_definitions[field_name] = (python_type, default_value)

            return field_definitions

        # Extract field definitions from schemas
        input_fields = schema_to_field_definitions(config.input_schema)
        output_fields = schema_to_field_definitions(config.output_schema)

        # Use create_model directly with the field definitions
        input_type = create_model(config.input_type_name, **input_fields)
        output_type = create_model(config.output_type_name, **output_fields)

        return input_type, output_type  # type: ignore

    @property
    def status(self) -> StepStatus:
        """Get current step status."""
        return self._status

    @property
    def start_time(self) -> Optional[datetime]:
        """Get step start time."""
        return self._start_time

    @property
    def end_time(self) -> Optional[datetime]:
        """Get step end time."""
        return self._end_time

    @property
    def error(self) -> Optional[str]:
        """Get step error if any."""
        return self._error

    @property
    def duration(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self._start_time and self._end_time:
            return (self._end_time - self._start_time).total_seconds()
        return None

    @abstractmethod
    async def execute(self, input_data: InputType, context: Context) -> OutputType:
        """Execute the step logic.

        Args:
            input_data: Validated input data
            context: Additional context including workflow state

        Returns:
            Validated output data

        Raises:
            Exception: If step execution fails
        """
        pass

    async def run(
        self, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the step with input validation and error handling.

        Args:
            input_data: Raw input data to validate
            context: Additional context including workflow state

        Returns:
            Dictionary containing output data

        Raises:
            Exception: If step execution fails after retries
        """
        logger.info(f"Starting step {self.step_id} ({self.metadata.name})")

        self._status = StepStatus.RUNNING
        self._start_time = datetime.now()
        self._error = None

        retry_count = 0
        max_retries = self.metadata.max_retries

        while retry_count <= max_retries:
            try:
                # Validate input
                validated_input = self.input_type(**input_data)

                # Create typed context from dict
                if isinstance(context, dict):
                    workflow_state = context.get("workflow_state", {})
                    # Use from_state_ref to avoid copying the state dict
                    typed_context = Context.from_state_ref(workflow_state)
                else:
                    typed_context = context

                # Execute with timeout if specified
                if self.metadata.timeout_seconds:
                    output = await asyncio.wait_for(
                        self.execute(validated_input, typed_context),
                        timeout=self.metadata.timeout_seconds,
                    )
                else:
                    output = await self.execute(validated_input, typed_context)

                # Validate output
                if not isinstance(output, self.output_type):
                    if hasattr(output, "model_dump"):
                        output = self.output_type(**output.model_dump())
                    elif isinstance(output, dict):
                        output = self.output_type(**output)
                    else:
                        # Try to convert to dict if possible
                        output = self.output_type(result=output)

                self._status = StepStatus.COMPLETED
                self._end_time = datetime.now()

                logger.info(
                    f"Step {self.step_id} completed successfully in {self.duration:.2f}s"
                )
                return output.model_dump()

            except asyncio.TimeoutError:
                error_msg = f"Step {self.step_id} timed out after {self.metadata.timeout_seconds}s"
                logger.error(error_msg)
                self._error = error_msg
                self._status = StepStatus.FAILED
                self._end_time = datetime.now()
                raise Exception(error_msg)

            except Exception as e:
                retry_count += 1
                error_msg = f"Step {self.step_id} failed (attempt {retry_count}/{max_retries + 1}): {str(e)}"
                logger.error(error_msg)

                if retry_count <= max_retries:
                    logger.info(f"Retrying step {self.step_id} in 1 second...")
                    await asyncio.sleep(1)
                    continue
                else:
                    self._error = str(e)
                    self._status = StepStatus.FAILED
                    self._end_time = datetime.now()
                    raise

        # Should never reach here
        raise Exception(f"Unexpected error in step {self.step_id}")

    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data against the input schema.

        Args:
            data: Input data to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            self.input_type(**data)
            return True
        except Exception:
            return False

    def validate_output(self, data: Dict[str, Any]) -> bool:
        """Validate output data against the output schema.

        Args:
            data: Output data to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            self.output_type(**data)
            return True
        except Exception:
            return False

    def get_schema(self) -> Dict[str, Any]:
        """Get the input/output schema for this step.

        Returns:
            Dictionary containing input and output schemas
        """
        return {
            "step_id": self.step_id,
            "metadata": self.metadata.model_dump(),
            "input_type": self.input_type.__name__,
            "output_type": self.output_type.__name__,
            "input_schema": self.input_type.model_json_schema(),
            "output_schema": self.output_type.model_json_schema(),
        }
