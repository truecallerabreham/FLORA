from __future__ import annotations
from typing import Optional, Sequence
from ._base import BaseTermination
from ..messages import Message, AssistantMessage, StopMessage


class MaxMessageTermination(BaseTermination):
    """Stop after a total of N messages have been exchanged.
    
    The simplest and most reliable safeguard. Always include this
    in any orchestration as a final backstop against infinite loops.
    
    Example: MaxMessageTermination(20) will stop after 20 messages total,
    regardless of content. Combine with other conditions:
    
    termination = MaxMessageTermination(20) | TextMentionTermination("DONE")
    """

    def __init__(self, max_messages: int):
        super().__init__()
        self._max = max_messages
        self._count = 0

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        self._count += len(new_messages)
        if self._count >= self._max:
            self._met = True
            return StopMessage(
                content=f"Maximum message limit reached: {self._count}/{self._max}",
                source="MaxMessageTermination",
            )
        return None

    def reset(self):
        super().reset()
        self._count = 0


class TextMentionTermination(BaseTermination):
    """Stop when specific text appears in any message.
    
    This is how the book's poet-critic example works:
    - Critic says "APPROVED" when satisfied with the poem
    - TextMentionTermination("APPROVED") catches this and stops
    
    The text matching is case-insensitive by default.
    
    Common patterns:
    - TextMentionTermination("TERMINATE") — explicit stop signal
    - TextMentionTermination("APPROVED") — quality gate
    - TextMentionTermination("COMPLETE") — task completion marker
    - TextMentionTermination("FINAL VERSION") — iteration done
    """

    def __init__(self, text: str, case_sensitive: bool = False):
        super().__init__()
        self._text = text
        self._case_sensitive = case_sensitive

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        search = self._text if self._case_sensitive else self._text.lower()

        for msg in new_messages:
            content = str(getattr(msg, "content", "") or "")
            haystack = content if self._case_sensitive else content.lower()

            if search in haystack:
                self._met = True
                return StopMessage(
                    content=f"Text mention found: '{self._text}'",
                    source="TextMentionTermination",
                )
        return None


class TokenBudgetTermination(BaseTermination):
    """Stop when a token budget is exceeded.
    
    WHY is this important? A multi-agent system with 5 agents,
    each doing 10 iterations, can consume tens of thousands of tokens.
    At API pricing, this can become expensive quickly.
    
    Use this as a cost control measure in production systems.
    Call add_usage() after each agent response to track consumption.
    
    Example:
    termination = MaxMessageTermination(50) | TokenBudgetTermination(10000)
    """

    def __init__(self, max_tokens: int):
        super().__init__()
        self._max_tokens = max_tokens
        self._used_tokens = 0

    def add_usage(self, tokens: int) -> None:
        """Call this after each agent response to track token consumption."""
        self._used_tokens += tokens

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        if self._used_tokens >= self._max_tokens:
            self._met = True
            return StopMessage(
                content=f"Token budget exhausted: {self._used_tokens}/{self._max_tokens} tokens used",
                source="TokenBudgetTermination",
            )
        return None

    def reset(self):
        super().reset()
        self._used_tokens = 0
