"""
Handoff termination condition.
"""

from typing import Optional, Sequence

from ..messages import AssistantMessage, Message
from ..types import StopMessage
from ._base import BaseTermination


class HandoffTermination(BaseTermination):
    """Terminates when agent requests handoff to specific target."""

    def __init__(self, target: str):
        super().__init__()
        self.target = target

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """Check for handoff requests in assistant messages."""
        for message in new_messages:
            if isinstance(message, AssistantMessage):
                # Look for handoff patterns in content
                content_lower = message.content.lower()
                handoff_patterns = [
                    f"handoff to {self.target.lower()}",
                    f"transfer to {self.target.lower()}",
                    f"pass to {self.target.lower()}",
                    f"delegate to {self.target.lower()}",
                ]

                for pattern in handoff_patterns:
                    if pattern in content_lower:
                        return self._set_termination(
                            f"Handoff requested to '{self.target}'",
                            {"target": self.target, "pattern": pattern},
                        )

        return None
