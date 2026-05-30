"""Context compaction strategies for Forla.

This module provides strategies for managing context/message history during
long-running agent sessions. Strategies are called BEFORE each LLM call in
the tool loop, and the returned (potentially compacted) message list continues
to the next iteration.

The key insight is that compaction must happen INSIDE the tool loop with
reassignment: `messages = strategy.compact(messages)`. This ensures
the compacted list is used for subsequent iterations, actually reducing
cumulative token usage.

Example:
    from forla import Agent
    from forla.compaction import HeadTailCompaction

    # Create compaction strategy with token budget
    compaction = HeadTailCompaction(token_budget=100_000, head_ratio=0.2)

    # Agent uses compaction for context management
    agent = Agent(
        name="assistant",
        ...,
        compaction=compaction,
    )
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Set, Tuple, runtime_checkable

from .messages import AssistantMessage, Message, ToolMessage


@runtime_checkable
class CompactionStrategy(Protocol):
    """Protocol for context compaction strategies.

    Called BEFORE each LLM call in the tool loop, allowing
    the strategy to compact the message list. The returned list
    REPLACES the working message list for subsequent iterations.

    Important: Implementations must preserve "atomic groups" - assistant
    messages with tool_calls must stay together with their corresponding
    ToolMessage results. Splitting these causes API errors.
    """

    @abstractmethod
    def compact(self, messages: List[Message]) -> List[Message]:
        """Compact messages for the next LLM call.

        Args:
            messages: Current message list

        Returns:
            Messages to use (may be compacted). This list REPLACES
            the working list for subsequent iterations.
        """
        ...


class NoCompaction:
    """Baseline: no compaction, context grows unbounded.

    Use this for benchmarking to see how context grows without management,
    or for short tasks where context limits won't be hit.

    Example:
        compaction = NoCompaction()
        agent = Agent(..., compaction=compaction)
    """

    def compact(self, messages: List[Message]) -> List[Message]:
        """Return messages unchanged."""
        return messages


@dataclass
class HeadTailCompaction:
    """Token-aware head+tail compaction strategy.

    Preserves:
    - Head: System prompt, initial user message (critical context)
    - Tail: Recent tool calls and results (working memory)

    Drops middle messages when over budget, respecting atomic groups
    (tool calls and their results must stay together).

    This is the recommended strategy for most use cases - it preserves
    the task context while keeping recent work visible.

    Args:
        token_budget: Maximum tokens for context (default: 100,000)
        head_ratio: Fraction of budget for head messages (default: 0.2 = 20%)
        model: Model name for tiktoken encoding (default: "gpt-4o")

    Example:
        compaction = HeadTailCompaction(
            token_budget=50_000,
            head_ratio=0.3,  # 30% head, 70% tail
        )
        agent = Agent(..., compaction=compaction)

    Statistics:
        After running, check `compaction.compaction_count` and
        `compaction.total_tokens_saved` for compaction metrics.
    """

    token_budget: int = 100_000
    head_ratio: float = 0.2
    model: str = "gpt-4o"

    # Statistics (not included in repr)
    compaction_count: int = field(default=0, repr=False)
    total_tokens_saved: int = field(default=0, repr=False)

    # Encoder cached on first use
    _encoder: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize tiktoken encoder."""
        try:
            import tiktoken

            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback for unknown models
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            # tiktoken not installed - will use character estimation
            self._encoder = None

    def _count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages.

        Uses tiktoken if available, falls back to character estimation.
        """
        if self._encoder is None:
            # Rough estimate: ~4 chars per token + overhead
            return sum(len(str(getattr(m, "content", "") or "")) // 4 + 10 for m in messages)

        total = 0
        for msg in messages:
            # Role overhead (approximately 4 tokens per message)
            total += 4

            content = getattr(msg, "content", None)
            if content:
                total += len(self._encoder.encode(content))  # type: ignore

            # Count tool calls if present
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    # Tool call overhead
                    total += 4
                    total += len(self._encoder.encode(tc.tool_name))  # type: ignore
                    # Parameters as JSON string
                    import json

                    params_str = json.dumps(tc.parameters)
                    total += len(self._encoder.encode(params_str))  # type: ignore

        return total

    def _find_atomic_groups(self, messages: List[Message]) -> List[Tuple[int, ...]]:
        """Group tool_call messages with their results.

        OpenAI/Anthropic require every tool_call to have a corresponding result.
        This ensures we never split a tool call from its results.

        Returns:
            List of tuples, where each tuple contains indices that must stay together.
        """
        groups: List[Tuple[int, ...]] = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                # This message has tool calls - find all results
                call_ids: Set[str] = {tc.call_id for tc in msg.tool_calls}
                group_indices = [i]

                # Look ahead for results
                j = i + 1
                while j < len(messages) and call_ids:
                    if isinstance(messages[j], ToolMessage):
                        tool_msg = messages[j]
                        if tool_msg.tool_call_id in call_ids:
                            group_indices.append(j)
                            call_ids.remove(tool_msg.tool_call_id)
                    j += 1

                groups.append(tuple(group_indices))
                i = max(group_indices) + 1 if group_indices else i + 1
            else:
                groups.append((i,))
                i += 1

        return groups

    def compact(self, messages: List[Message]) -> List[Message]:
        """Compact messages if over budget.

        Preserves head (system prompt, initial context) and tail (recent work),
        dropping middle messages when necessary.
        """
        if not messages:
            return messages

        current_tokens = self._count_tokens(messages)

        if current_tokens <= self.token_budget:
            return messages

        # COMPACTION NEEDED
        self.compaction_count += 1

        groups = self._find_atomic_groups(messages)
        head_budget = int(self.token_budget * self.head_ratio)
        tail_budget = self.token_budget - head_budget

        # Fill head from start
        head_groups: List[Tuple[int, ...]] = []
        head_tokens = 0

        for group in groups:
            group_msgs = [messages[i] for i in group]
            group_tokens = self._count_tokens(group_msgs)

            if head_tokens + group_tokens <= head_budget:
                head_groups.append(group)
                head_tokens += group_tokens
            else:
                break

        # Fill tail from end (skip head groups)
        remaining_groups = groups[len(head_groups) :]
        tail_groups: List[Tuple[int, ...]] = []
        tail_tokens = 0

        for group in reversed(remaining_groups):
            group_msgs = [messages[i] for i in group]
            group_tokens = self._count_tokens(group_msgs)

            if tail_tokens + group_tokens <= tail_budget:
                tail_groups.insert(0, group)
                tail_tokens += group_tokens
            else:
                break

        # Build compacted list
        kept_indices: Set[int] = set()
        for group in head_groups + tail_groups:
            kept_indices.update(group)

        compacted = [messages[i] for i in sorted(kept_indices)]

        # Track savings
        compacted_tokens = self._count_tokens(compacted)
        self.total_tokens_saved += current_tokens - compacted_tokens

        return compacted


@dataclass
class SlidingWindowCompaction:
    """Keep only recent messages within budget.

    Always preserves the system message (if present) plus the most
    recent messages that fit in the budget. Respects atomic groups
    (tool calls and their results must stay together).

    Simpler than HeadTailCompaction but may lose important early context.
    Best for conversational agents where recent context matters most.

    Args:
        token_budget: Maximum tokens for context (default: 100,000)
        model: Model name for tiktoken encoding (default: "gpt-4o")

    Example:
        compaction = SlidingWindowCompaction(token_budget=50_000)
        agent = Agent(..., compaction=compaction)
    """

    token_budget: int = 100_000
    model: str = "gpt-4o"

    # Statistics
    compaction_count: int = field(default=0, repr=False)
    total_tokens_saved: int = field(default=0, repr=False)

    # Encoder cached on first use
    _encoder: Optional[object] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize tiktoken encoder."""
        try:
            import tiktoken

            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self._encoder = None

    def _count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages."""
        if self._encoder is None:
            return sum(len(str(getattr(m, "content", "") or "")) // 4 + 10 for m in messages)

        total = 0
        for msg in messages:
            total += 4
            content = getattr(msg, "content", None)
            if content:
                total += len(self._encoder.encode(content))  # type: ignore
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    total += 4 + len(self._encoder.encode(tc.tool_name))  # type: ignore
                    import json

                    total += len(self._encoder.encode(json.dumps(tc.parameters)))  # type: ignore
        return total

    def _find_atomic_groups(self, messages: List[Message]) -> List[Tuple[int, ...]]:
        """Group tool_call messages with their results."""
        groups: List[Tuple[int, ...]] = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                call_ids: Set[str] = {tc.call_id for tc in msg.tool_calls}
                group_indices = [i]

                j = i + 1
                while j < len(messages) and call_ids:
                    if isinstance(messages[j], ToolMessage):
                        tool_msg = messages[j]
                        if tool_msg.tool_call_id in call_ids:
                            group_indices.append(j)
                            call_ids.remove(tool_msg.tool_call_id)
                    j += 1

                groups.append(tuple(group_indices))
                i = max(group_indices) + 1 if group_indices else i + 1
            else:
                groups.append((i,))
                i += 1

        return groups

    def compact(self, messages: List[Message]) -> List[Message]:
        """Keep system message + most recent messages within budget."""
        if not messages:
            return messages

        current_tokens = self._count_tokens(messages)

        if current_tokens <= self.token_budget:
            return messages

        # COMPACTION NEEDED
        self.compaction_count += 1

        groups = self._find_atomic_groups(messages)

        # Always keep system message if present (first message, first group)
        system_groups: List[Tuple[int, ...]] = []
        system_tokens = 0

        if groups and messages[groups[0][0]].role == "system":
            system_groups.append(groups[0])
            system_tokens = self._count_tokens([messages[i] for i in groups[0]])
            groups = groups[1:]

        # Fill from end with remaining budget
        remaining_budget = self.token_budget - system_tokens
        kept_groups: List[Tuple[int, ...]] = []
        kept_tokens = 0

        for group in reversed(groups):
            group_msgs = [messages[i] for i in group]
            group_tokens = self._count_tokens(group_msgs)

            if kept_tokens + group_tokens <= remaining_budget:
                kept_groups.insert(0, group)
                kept_tokens += group_tokens
            else:
                break

        # Build compacted list
        kept_indices: Set[int] = set()
        for group in system_groups + kept_groups:
            kept_indices.update(group)

        compacted = [messages[i] for i in sorted(kept_indices)]

        # Track savings
        compacted_tokens = self._count_tokens(compacted)
        self.total_tokens_saved += current_tokens - compacted_tokens

        return compacted
