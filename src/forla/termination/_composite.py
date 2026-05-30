"""
Composite termination condition for combining multiple conditions.
"""

from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

from .._component_config import Component, ComponentModel
from ..messages import Message
from ..types import StopMessage
from ._base import BaseTermination


class CompositeTerminationConfig(BaseModel):
    """Configuration for CompositeTermination serialization."""

    conditions: List[ComponentModel] = Field(default_factory=list)
    mode: str = "any"


class CompositeTermination(Component[CompositeTerminationConfig], BaseTermination):
    """Combines multiple termination conditions with logical operators."""

    component_config_schema = CompositeTerminationConfig
    component_type = "termination"
    component_provider_override = "forla.termination.CompositeTermination"

    def __init__(self, conditions: List[BaseTermination], mode: str = "any"):
        super().__init__()
        if mode not in ("any", "all"):
            raise ValueError("Mode must be 'any' or 'all'")

        self.conditions = conditions
        self.mode = mode

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """Check all conditions based on mode."""
        results = []
        for condition in self.conditions:
            result = condition.check(new_messages)
            if result:
                results.append(result)

        if self.mode == "any" and results:
            # Return first termination result
            first_result = results[0]
            return self._set_termination(
                f"Composite (any): {first_result.content}",
                {"mode": "any", "triggered_conditions": [r.source for r in results]},
            )
        elif self.mode == "all" and len(results) == len(self.conditions):
            # All conditions met
            reasons = [r.content for r in results]
            return self._set_termination(
                f"Composite (all): {'; '.join(reasons)}",
                {"mode": "all", "triggered_conditions": [r.source for r in results]},
            )

        return None

    def reset(self) -> None:
        """Reset all contained conditions."""
        super().reset()
        for condition in self.conditions:
            condition.reset()

    def is_met(self) -> bool:
        """Check if composite condition is met."""
        met_conditions = [c.is_met() for c in self.conditions]

        if self.mode == "any":
            return any(met_conditions)
        else:  # mode == "all"
            return all(met_conditions)

    def __or__(self, other: BaseTermination) -> "CompositeTermination":
        """Extend OR composition."""
        if isinstance(other, CompositeTermination) and other.mode == "any":
            return CompositeTermination(self.conditions + other.conditions, mode="any")
        else:
            return CompositeTermination(self.conditions + [other], mode="any")

    def __and__(self, other: BaseTermination) -> "CompositeTermination":
        """Extend AND composition."""
        if isinstance(other, CompositeTermination) and other.mode == "all":
            return CompositeTermination(self.conditions + other.conditions, mode="all")
        else:
            return CompositeTermination(self.conditions + [other], mode="all")

    def _to_config(self) -> CompositeTerminationConfig:
        """Convert to configuration for serialization."""
        condition_configs = []
        for condition in self.conditions:
            try:
                condition_configs.append(condition.dump_component())
            except NotImplementedError:
                # Skip conditions that don't support serialization
                continue

        return CompositeTerminationConfig(conditions=condition_configs, mode=self.mode)

    @classmethod
    def _from_config(cls, config: CompositeTerminationConfig) -> "CompositeTermination":
        """Create from configuration."""
        conditions = []
        for condition_config in config.conditions:
            try:
                condition = BaseTermination.load_component(condition_config)
                conditions.append(condition)
            except Exception:
                # Skip conditions that fail to deserialize
                continue

        return cls(conditions=conditions, mode=config.mode)
