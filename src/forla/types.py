from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class Usage:
    """Tracks resource consumption for every operation.
    
    WHY track this from the very start? 
    Because LLM API calls cost money. A multi-agent system with 5 agents,
    each making 10 iterations, each with 3 tool calls, can rack up
    thousands of tokens very quickly. You need visibility into this.
    
    The __add__ method lets you aggregate usage across multiple agents:
        total = agent1_usage + agent2_usage + agent3_usage
    """
    tokens_input: int = 0
    tokens_output: int = 0
    duration_ms: int = 0
    num_calls: int = 0      # Number of LLM API calls made in this run

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            tokens_input=self.tokens_input + other.tokens_input,
            tokens_output=self.tokens_output + other.tokens_output,
            duration_ms=self.duration_ms + other.duration_ms,
            num_calls=self.num_calls + other.num_calls,
        )

    def __str__(self) -> str:
        return (
            f"duration: {self.duration_ms}ms, "
            f"tokens: in:{self.tokens_input}, out:{self.tokens_output}, "
            f"calls: {self.num_calls}"
        )

import asyncio
import threading


class CancellationToken:
    """A thread-safe signal for stopping a long-running operation.
    
    WHY do we need this?
    Agent tasks can run for minutes. Users will start a task,
    then realize they made a mistake or the agent is stuck in a loop,
    and they want to stop it.
    
    HOW it works:
    1. You create a token: token = CancellationToken()
    2. You pass it into the agent: agent.run_stream(task, cancellation_token=token)
    3. At any point from any thread, you call: token.cancel()
    4. The agent loop checks token.is_cancelled() at checkpoints and stops cleanly
    
    WHY thread-safe? Because the UI might be on a different thread than the agent.
    The Lock prevents race conditions.
    """
    
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
        # asyncio.Event lets async code 'await' until cancellation happens
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Signal that we want to stop. Can be called from any thread."""
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                # Schedule the event set on the running event loop, if there is one
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.call_soon_threadsafe(self._event.set)
                    else:
                        self._event.set()
                except RuntimeError:
                    pass

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested. Call this at checkpoints."""
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        """Convenience: raise asyncio.CancelledError if already cancelled."""
        if self._cancelled:
            raise asyncio.CancelledError("Operation was cancelled")

from pydantic import BaseModel
from .messages import AssistantMessage


class ChatCompletionResult(BaseModel):
    """The raw result from a single call to the LLM API.
    
    This is what the model client produces.
    The agent then decides what to do with it:
    - If message.tool_calls is set: execute the tools and call the model again
    - If message.content is set: this is the final answer
    """
    message: AssistantMessage
    usage: Usage
    model: str = ""
    finish_reason: str = "stop"   # "stop", "tool_calls", "length", "cancelled"
    structured_output: Optional[Any] = None   # Set when output_format was used

    class Config:
        arbitrary_types_allowed = True

class AgentResponse(BaseModel):
    """The final result of an agent.run() or agent.run_stream() call.
    
    WHY separate from ChatCompletionResult?
    An agent might make 5 LLM calls (one per tool use) before producing 
    a final response. AgentResponse aggregates ALL of those into one result
    that includes the full conversation context and total resource usage.
    """
    content: str                   # The final text answer
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"    # Why the agent stopped

    class Config:
        arbitrary_types_allowed = True
