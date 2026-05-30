"""
Abstract base class for BaseChatCompletionClient implementations.

This module defines the interface that all LLM providers must implement,
providing a unified way to interact with different language models.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from pydantic import BaseModel

from .._component_config import ComponentBase
from ..messages import Message
from ..types import ChatCompletionChunk, ChatCompletionResult


class BaseChatCompletionClient(ComponentBase[BaseModel], ABC):
    """
    Abstract base class for LLM provider implementations.

    Provides unified interface for different LLM providers (OpenAI, Anthropic, etc.)
    while abstracting away provider-specific API differences.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        """
        Initialize the chat completion client.

        Args:
            model: The model identifier (e.g., "gpt-4", "claude-3")
            api_key: API key for the provider
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        """
        Make a single LLM API call.

        Args:
            messages: List of messages to send to the LLM
            tools: Optional list of tools available to the LLM
            output_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Standardized chat completion result

        Raises:
            BaseChatCompletionError: If the API call fails
        """
        pass

    @abstractmethod
    def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Make a streaming LLM API call.

        Args:
            messages: List of messages to send to the LLM
            tools: Optional list of tools available to the LLM
            output_format: Optional Pydantic model for structured output
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Yields:
            ChatCompletionChunk objects with incremental response data

        Raises:
            BaseChatCompletionError: If the API call fails
        """
        pass

    def _convert_messages_to_api_format(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """
        Convert internal Message objects to provider-specific API format.

        This is a helper method that subclasses can override to handle
        provider-specific message formatting requirements.

        Args:
            messages: List of internal Message objects

        Returns:
            List of messages in provider API format
        """
        from ..messages import AssistantMessage, MultiModalMessage, ToolMessage

        api_messages = []
        for msg in messages:
            # Handle MultiModalMessage with special formatting
            if isinstance(msg, MultiModalMessage):
                api_msg: Dict[str, Any] = {"role": msg.role}

                # For text content, use simple content field
                if msg.is_text():
                    api_msg["content"] = msg.content
                else:
                    # For non-text content, use the content array format
                    content_parts = []

                    # Add text part if there's text content
                    if msg.content and msg.content.strip():
                        content_parts.append({"type": "text", "text": msg.content})

                    # Add media part
                    if msg.is_image():
                        if msg.data:
                            # Use base64 data URL format (to_base64 handles both str and bytes)
                            base64_data = msg.to_base64()
                            data_url = f"data:{msg.mime_type};base64,{base64_data}"
                            content_parts.append(
                                {"type": "image_url", "image_url": {"url": data_url}}
                            )
                        elif msg.media_url:
                            # Use direct URL
                            content_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": msg.media_url},
                                }
                            )

                    # For now, only images are fully supported by OpenAI vision models
                    # Audio/video would need different handling or transcription

                    api_msg["content"] = content_parts

            else:
                # Handle regular messages
                api_msg: Dict[str, Any] = {"role": msg.role, "content": msg.content}

                # Handle assistant messages with tool calls
                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                    api_msg["tool_calls"] = [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.parameters)
                                if isinstance(tc.parameters, dict)
                                else tc.parameters,
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                # Handle tool messages
                if isinstance(msg, ToolMessage):
                    api_msg["tool_call_id"] = msg.tool_call_id

            api_messages.append(api_msg)

        return api_messages


class BaseChatCompletionError(Exception):
    """Raised when a chat completion API call fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(message)


class RateLimitError(BaseChatCompletionError):
    """Raised when API rate limits are exceeded."""

    pass


class AuthenticationError(BaseChatCompletionError):
    """Raised when API authentication fails."""

    pass


class InvalidRequestError(BaseChatCompletionError):
    """Raised when the API request is invalid."""

    pass
