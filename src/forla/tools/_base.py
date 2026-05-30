from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """The result of executing a tool.
    
    WHY always use this instead of returning a plain string?
    Because tools can fail. When a tool fails, you want:
    1. A clear 'success=False' signal
    2. The actual error message
    3. The agent to see this failure and potentially try something different
    
    A plain string can't carry all this information cleanly.
    """
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            return str(self.result) if self.result is not None else ""
        return f"Error: {self.error}"


class BaseTool(ABC):
    """Abstract interface for all tools.
    
    Every tool must have:
    - name: What the model calls it by (used in tool call requests)
    - description: What it does (the model reads this to decide when to use it)
    - parameters: JSON Schema describing its inputs
    - execute(): The actual implementation (async)
    - to_llm_format(): Converts to the JSON format the LLM API expects
    
    WHY abstract? So that simple function tools, REST API tools,
    database tools, and MCP tools all look the same to the agent.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for this tool's input parameters.
        
        Example for a weather tool:
        {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'Paris, France'"
                }
            },
            "required": ["location"]
        }
        
        The model uses this schema to know what parameters to send.
        Write clear descriptions — they directly influence when and how
        the model uses the tool.
        """
        pass

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with the given parameters.
        
        MUST be async — tools may call external APIs, read files, etc.
        Always return a ToolResult, never raise an exception to the agent.
        Wrap exceptions in ToolResult(success=False, error=str(e)).
        """
        pass

    def to_llm_format(self) -> Dict[str, Any]:
        """Convert this tool to the JSON format the LLM API expects.
        
        This is the OpenAI function-calling format, which has become
        the de facto standard — Anthropic, Google, and others all accept
        a similar format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
