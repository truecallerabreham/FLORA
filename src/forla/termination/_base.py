"""
Base termination condition.

This module provides the foundational BaseTermination class for all
termination conditions, following the PRD specification.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from pydantic import BaseModel

from .._component_config import ComponentBase
from ..messages import Message
from ..types import StopMessage

if TYPE_CHECKING:
    from ._composite import CompositeTermination


class BaseTermination(ComponentBase[BaseModel], ABC):
    """Abstract base class for all termination conditions."""

    def __init__(self) -> None:
        self._met = False
        self._reason = ""
        self._metadata: Dict[str, Any] = {}

    @abstractmethod
    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """
        Check termination on delta messages.

        Args:
            new_messages: New messages since last check

        Returns:
            StopMessage if termination is met, None otherwise
        """
        pass

    def is_met(self) -> bool:
        """Current termination state."""
        return self._met

    def reset(self) -> None:
        """Reset for next orchestration run."""
        self._met = False
        self._reason = ""
        self._metadata = {}

    def get_reason(self) -> str:
        """Why termination occurred."""
        return self._reason

    def get_metadata(self) -> Dict[str, Any]:
        """Additional termination metadata."""
        return self._metadata.copy()

    def _set_termination(
        self, reason: str, metadata: Optional[Dict[str, Any]] = None
    ) -> StopMessage:
        """Helper to set termination state and return StopMessage."""
        self._met = True
        self._reason = reason
        self._metadata = metadata or {}

        return StopMessage(
            content=reason, source=self.__class__.__name__, metadata=self._metadata
        )

    def __or__(self, other: "BaseTermination") -> "CompositeTermination":
        """Implement OR logic with | operator."""
        from ._composite import CompositeTermination

        return CompositeTermination([self, other], mode="any")

    def __and__(self, other: "BaseTermination") -> "CompositeTermination":
        """Implement AND logic with & operator."""
        from ._composite import CompositeTermination

        return CompositeTermination([self, other], mode="all")
