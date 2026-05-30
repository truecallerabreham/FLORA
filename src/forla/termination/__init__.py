"""
Termination conditions for orchestration patterns.

This package provides various termination conditions that determine when
orchestration should stop, following the PRD specification.
"""

from ._base import BaseTermination
from ._cancellation import CancellationTermination
from ._composite import CompositeTermination
from ._external import ExternalTermination
from ._function_call import FunctionCallTermination
from ._handoff import HandoffTermination
from ._max_message import MaxMessageTermination
from ._text_mention import TextMentionTermination
from ._timeout import TimeoutTermination
from ._token_usage import TokenUsageTermination

__all__ = [
    # Base
    "BaseTermination",
    # Individual conditions
    "MaxMessageTermination",
    "TextMentionTermination",
    "TokenUsageTermination",
    "TimeoutTermination",
    "HandoffTermination",
    "ExternalTermination",
    "CancellationTermination",
    "FunctionCallTermination",
    # Composite
    "CompositeTermination",
]
