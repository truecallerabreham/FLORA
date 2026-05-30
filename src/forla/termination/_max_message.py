"""
Maximum message termination condition.
"""

from typing import Optional, Sequence

from pydantic import BaseModel

from .._component_config import Component
from ..messages import Message
from ..types import StopMessage
from ._base import BaseTermination


class MaxMessageTerminationConfig(BaseModel):
    """Configuration for MaxMessageTermination serialization."""

    max_messages: int


class MaxMessageTermination(Component[MaxMessageTerminationConfig], BaseTermination):
    """Terminates when maximum messages is reached."""

    component_config_schema = MaxMessageTerminationConfig
    component_type = "termination"
    component_provider_override = "forla.termination.MaxMessageTermination"

    def __init__(self, max_messages: int):
        super().__init__()
        self.max_messages = max_messages
        self.message_count = 0  # Runtime state - resets on deserialization

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """Check if message limit is exceeded."""
        self.message_count += len(new_messages)

        if self.message_count >= self.max_messages:
            return self._set_termination(
                f"Maximum messages reached ({self.message_count}/{self.max_messages})",
                {
                    "message_count": self.message_count,
                    "max_messages": self.max_messages,
                },
            )

        return None

    def reset(self) -> None:
        """Reset message counter."""
        super().reset()
        self.message_count = 0

    def _to_config(self) -> MaxMessageTerminationConfig:
        """Convert to configuration for serialization."""
        return MaxMessageTerminationConfig(max_messages=self.max_messages)

    @classmethod
    def _from_config(
        cls, config: MaxMessageTerminationConfig
    ) -> "MaxMessageTermination":
        """Create from configuration."""
        return cls(max_messages=config.max_messages)
