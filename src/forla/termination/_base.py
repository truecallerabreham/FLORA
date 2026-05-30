from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Sequence
from ..messages import Message, StopMessage


class BaseTermination(ABC):
    """Abstract base for all termination conditions.
    
    KEY DESIGN: check() receives ONLY NEW messages since the last check (delta).
    
    WHY delta instead of full history?
    1. Efficiency: no need to re-scan hundreds of messages every iteration
    2. Reactivity: conditions respond to recent changes, not the whole past
    3. Composability: conditions can be stateful without replaying history
    
    The __or__ and __and__ operators enable Pythonic combination:
    MaxMessageTermination(10) | TextMentionTermination("DONE")
    
    This means: "stop when EITHER 10 messages are reached OR 'DONE' is mentioned"
    """

    def __init__(self):
        self._met = False

    @abstractmethod
    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """Check if we should stop.
        
        Returns a StopMessage explaining WHY we stopped,
        or None if we should continue.
        """
        pass

    def is_met(self) -> bool:
        return self._met

    def reset(self) -> None:
        """Called at the start of each new orchestrator run.
        
        IMPORTANT: Always call reset() before reusing a termination condition
        in a new run. Otherwise, accumulated state from the previous run
        will incorrectly trigger early termination.
        """
        self._met = False

    def __or__(self, other: "BaseTermination") -> "CompositeTermination":
        """Enable: cond_a | cond_b  →  stop when EITHER condition is met."""
        return CompositeTermination([self, other], mode="any")

    def __and__(self, other: "BaseTermination") -> "CompositeTermination":
        """Enable: cond_a & cond_b  →  stop when BOTH conditions are met."""
        return CompositeTermination([self, other], mode="all")


class CompositeTermination(BaseTermination):
    """Combines multiple conditions with AND or OR logic."""

    def __init__(self, conditions: list, mode: str = "any"):
        super().__init__()
        self._conditions = conditions
        self._mode = mode    # "any" = OR, "all" = AND

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        results = [c.check(new_messages) for c in self._conditions]

        if self._mode == "any":
            # Stop if ANY condition triggered
            for r in results:
                if r is not None:
                    self._met = True
                    return r
        elif self._mode == "all":
            # Stop only if ALL conditions triggered
            if all(r is not None for r in results):
                self._met = True
                combined_content = " AND ".join(
                    r.content for r in results if r is not None
                )
                return StopMessage(content=combined_content, source="CompositeTermination")

        return None

    def reset(self):
        super().reset()
        for c in self._conditions:
            c.reset()
