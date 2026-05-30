"""
Text mention termination condition.
"""

from typing import Optional, Sequence

from pydantic import BaseModel

from .._component_config import Component
from ..messages import Message
from ..types import StopMessage
from ._base import BaseTermination


class TextMentionTerminationConfig(BaseModel):
    """Configuration for TextMentionTermination serialization."""

    text: str
    case_sensitive: bool = False


class TextMentionTermination(Component[TextMentionTerminationConfig], BaseTermination):
    """Terminates when specific text is mentioned."""

    component_config_schema = TextMentionTerminationConfig
    component_type = "termination"
    component_provider_override = "forla.termination.TextMentionTermination"

    def __init__(self, text: str, case_sensitive: bool = False):
        super().__init__()
        self.text = text
        self.case_sensitive = case_sensitive
        self.search_text = text if case_sensitive else text.lower()

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """Check if termination text is found in any new message."""
        for message in new_messages:
            content = (
                message.content if self.case_sensitive else message.content.lower()
            )

            if self.search_text in content:
                return self._set_termination(
                    f"Text mention found: '{self.text}'",
                    {
                        "text": self.text,
                        "case_sensitive": self.case_sensitive,
                        "found_in": type(message).__name__,
                    },
                )

        return None

    def _to_config(self) -> TextMentionTerminationConfig:
        """Convert to configuration for serialization."""
        return TextMentionTerminationConfig(
            text=self.text, case_sensitive=self.case_sensitive
        )

    @classmethod
    def _from_config(
        cls, config: TextMentionTerminationConfig
    ) -> "TextMentionTermination":
        """Create from configuration."""
        return cls(text=config.text, case_sensitive=config.case_sensitive)
