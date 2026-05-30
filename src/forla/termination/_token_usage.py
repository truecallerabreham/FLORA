"""
Token usage termination condition.
"""

from typing import Optional, Sequence

from ..messages import Message
from ..types import StopMessage
from ._base import BaseTermination


class TokenUsageTermination(BaseTermination):
    """Terminates when token usage exceeds limit."""

    def __init__(self, max_tokens: int):
        super().__init__()
        self.max_tokens = max_tokens
        self.total_tokens = 0

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """Check token usage (approximate based on content length)."""
        # Simple token estimation: ~4 characters per token
        new_tokens = sum(len(msg.content) // 4 for msg in new_messages)
        self.total_tokens += new_tokens

        if self.total_tokens >= self.max_tokens:
            return self._set_termination(
                f"Token limit exceeded ({self.total_tokens}/{self.max_tokens})",
                {"total_tokens": self.total_tokens, "max_tokens": self.max_tokens},
            )

        return None

    def reset(self) -> None:
        """Reset token counter."""
        super().reset()
        self.total_tokens = 0
