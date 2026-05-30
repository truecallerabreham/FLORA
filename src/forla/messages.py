from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import uuid

class Message(BaseModel, ABC):
    """The base class for all messages in the system.
    
    Every message must have a 'source' that identifies who sent it.
    The 'metadata' field carries extra information (timestamps, session IDs, etc.)
    without polluting the core message structure.
    """
    source: str = Field(description="Who sent this message: 'user', 'assistant', an agent name, or a tool name")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class SystemMessage(Message):
    """Developer instructions to the model.
    
    IMPORTANT: System messages are never shown to users.
    They are internal configuration for the LLM.
    Examples: "You are a haiku poet.", "You are a code reviewer."
    """
    role: str = "system"
    content: str

class UserMessage(Message):
    """Input from a human user OR from an orchestrator.
    
    'content' can be a string for text-only input,
    or a List for multimodal input (text + images, text + audio, etc.).
    In practice, you start with strings and add multimodal later.
    """
    role: str = "user"
    content: Union[str, List[Any]]

class AssistantMessage(Message):
    """The model's response.
    
    There are TWO possible states for an AssistantMessage:
    
    State 1 — Text response: 'content' is a string, 'tool_calls' is None.
    This means the model has a final answer to give back.
    
    State 2 — Tool call request: 'content' is None (or empty), 'tool_calls' is a list.
    This means the model wants to DO something before answering.
    It is your framework's job to execute those tool calls and send back the results.
    
    Your agent execution loop must handle BOTH states.
    State 2 is what makes agents powerful — the model can request information or actions.
    """
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List["ToolCallRequest"]] = None

class ToolCallRequest(BaseModel):
    """A structured request from the model to execute a tool.
    
    'call_id' is a unique identifier. When you send back the result,
    it must include this same ID so the model knows which call produced which result.
    
    'tool_name' is the name of the function the model wants to call.
    'parameters' are the arguments it wants to pass.
    
    This is the bridge between "the model wants to do something"
    and "the framework actually does it."
    """
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ToolMessage(Message):
    """The result of executing a tool.
    
    'tool_call_id' MUST match the 'call_id' from the ToolCallRequest
    that triggered this execution. This is how the model knows
    "the result of my weather call was: Paris is sunny."
    
    'success' and 'error' let you handle failures gracefully.
    The model can see that a tool failed and try a different approach.
    """
    role: str = "tool"
    content: str
    tool_call_id: str   # Must match ToolCallRequest.call_id
    tool_name: str = ""
    success: bool = True
    error: Optional[str] = None


class StopMessage(BaseModel):
    """Signals that an orchestration should stop.
    
    'content' is a human-readable explanation of why we stopped.
    'source' identifies which termination condition triggered the stop.
    """
    content: str
    source: str
