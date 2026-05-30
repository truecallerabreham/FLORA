"""
Cancellation termination condition.
"""

from typing import Optional, Sequence

from .._cancellation_token import CancellationToken
from ..messages import Message
from ..types import StopMessage
from ._base import BaseTermination


class CancellationTermination(BaseTermination):
    """Terminates when cancellation token is triggered."""

    def __init__(self, cancellation_token: CancellationToken):
        super().__init__()
        self.cancellation_token = cancellation_token

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """Check if cancellation token is triggered."""
        if self.cancellation_token.is_cancelled():
            return self._set_termination(
                "Cancellation token triggered", {"source": "cancellation_token"}
            )

        return None
