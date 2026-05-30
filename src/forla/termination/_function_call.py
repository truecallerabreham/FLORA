"""
Function call termination condition.
"""

from typing import Optional, Sequence

from ..messages import Message, ToolMessage
from ..types import StopMessage
from ._base import BaseTermination


class FunctionCallTermination(BaseTermination):
    """Terminates when specific function is called."""

    def __init__(self, function_name: str):
        super().__init__()
        self.function_name = function_name

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """Check for specific function calls in tool messages."""
        for message in new_messages:
            if (
                isinstance(message, ToolMessage)
                and message.tool_name == self.function_name
            ):
                return self._set_termination(
                    f"Function '{self.function_name}' was called",
                    {"function_name": self.function_name, "success": message.success},
                )

        return None
