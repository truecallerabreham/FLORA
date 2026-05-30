from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Type
from pydantic import BaseModel
from ..messages import Message
from ..types import ChatCompletionResult


class BaseChatCompletionClient(ABC):
    """Abstract interface all LLM providers must implement.
    
    WHY abstract? Because agents should not know WHICH LLM they are using.
    An agent built for OpenAI's GPT-4 should work unchanged with Anthropic's Claude
    or a local model running on your laptop. The abstract interface enforces this.
    
    The two core methods:
    - create(): Make a single blocking LLM call, get back the complete response
    - create_stream(): Make a streaming LLM call, get back chunks as they arrive
    
    Both methods accept 'tools' (function schemas for tool calling)
    and 'output_format' (a Pydantic model for structured output).
    """

    @abstractmethod
    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> ChatCompletionResult:
        """Make a single complete LLM call.
        
        Args:
            messages: The full conversation history. Always starts with SystemMessage,
                      followed by the alternating UserMessage/AssistantMessage history.
            tools: List of JSON schemas describing available tools (function calling format).
                   When provided, the model may respond with tool_calls instead of text.
            output_format: A Pydantic model class. When provided, constrains the model's
                           response to match this JSON schema exactly.
        """
        pass

    @abstractmethod
    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncGenerator:
        """Make a streaming LLM call.
        
        Yields chunks of the response as they arrive from the API.
        This is what enables the "typing" effect — you see each word appear
        rather than waiting for the complete response.
        """
        pass
