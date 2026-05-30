from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .messages import Message, UserMessage, AssistantMessage


class AgentContext(BaseModel):
    """The single source of truth for an agent's current session state.
    
    WHY is this a Pydantic BaseModel?
    Serialization. This means you can:
    1. Save a mid-conversation state to JSON
    2. Pass it to a new agent.run() call to continue the conversation
    3. Share it between agents in an orchestration (Section 4.12.3 in the book)
    4. Restore state after a server restart
    
    This is how the "stateless context" pattern works in Chapter 8:
    any server can resume any conversation because the full state is in the context.
    
    HOW it relates to the agent loop:
    Every time the agent calls the LLM, it passes:
        [SystemMessage(instructions)] + context.messages
    This gives the model the full conversation history.
    """
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    max_messages: int = 100    # Prevent unbounded memory growth

    def add_message(self, message: Message) -> None:
        """Append a message to the conversation history.
        
        We never trim SystemMessages — they define agent behavior.
        When over capacity, we trim the oldest non-system messages.
        """
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            # Keep system messages, drop oldest others
            system_msgs = [m for m in self.messages if getattr(m, 'role', '') == 'system']
            other_msgs = [m for m in self.messages if getattr(m, 'role', '') != 'system']
            # Keep the most recent messages
            other_msgs = other_msgs[-(self.max_messages - len(system_msgs)):]
            self.messages = system_msgs + other_msgs

    def get_messages(self) -> List[Message]:
        """Return all messages for building the LLM call context."""
        return list(self.messages)

    def clear(self) -> None:
        """Reset context — start a new session while keeping agent config."""
        self.messages = []

    @property
    def message_count(self) -> int:
        return len(self.messages)

    class Config:
        arbitrary_types_allowed = True
