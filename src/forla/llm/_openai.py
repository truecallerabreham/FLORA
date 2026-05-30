from __future__ import annotations
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Type
from pydantic import BaseModel
from openai import AsyncOpenAI
from ._base import BaseChatCompletionClient
from ..messages import (
    Message, UserMessage, AssistantMessage, SystemMessage,
    ToolMessage, ToolCallRequest
)
from ..types import ChatCompletionResult, Usage

class OpenAIChatCompletionClient(BaseChatCompletionClient):
    """OpenAI API implementation.
    
    Also works with:
    - Azure OpenAI: pass base_url="https://your-resource.openai.azure.com/"
    - Local models (vLLM, Ollama): pass base_url="http://localhost:11434/v1"
    - Any OpenAI-compatible API (Together AI, Groq, etc.)
    
    This is why the book standardizes on the OpenAI API format —
    it has become the de facto standard across the industry.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.model = model
        # AsyncOpenAI is the async version — required because all our methods are async
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _convert_messages_to_api_format(self, messages: List[Message]) -> List[Dict]:
        """Convert our typed message objects to the raw dict format OpenAI expects.
        
        This is STEP 1 of the three-step conversion pattern.
        
        Notice that we handle each message type differently:
        - SystemMessage → simple {"role": "system", "content": "..."}
        - UserMessage → same, but content can be a list for multimodal
        - AssistantMessage → must include tool_calls if present
        - ToolMessage → must include tool_call_id to link back to the request
        """
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})

            elif isinstance(msg, UserMessage):
                # Handle both simple strings and multimodal content lists
                result.append({"role": "user", "content": msg.content})

            elif isinstance(msg, AssistantMessage):
                # Build the assistant dict carefully
                api_msg: Dict[str, Any] = {"role": "assistant"}
                
                # Only add content if it exists (it may be None when tool_calls are present)
                if msg.content:
                    api_msg["content"] = msg.content
                
                # Convert our ToolCallRequest objects to OpenAI's format
                if msg.tool_calls:
                    api_msg["tool_calls"] = [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                # OpenAI expects arguments as a JSON string, not a dict
                                "arguments": json.dumps(tc.parameters),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(api_msg)

            elif isinstance(msg, ToolMessage):
                # Tool results must include tool_call_id
                result.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })

        return result

    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> ChatCompletionResult:
        """The core LLM call. Three steps: convert → call → convert back."""
        
        # === STEP 1: Convert our types to OpenAI format ===
        api_messages = self._convert_messages_to_api_format(messages)
        
        # Build the request parameters
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }
        
        # If tools are provided, add them and set tool_choice to "auto"
        # (the model decides whether to use tools or respond with text)
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        
        # If structured output is requested, use OpenAI's json_schema response format
        if output_format:
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_format.__name__,
                    "schema": output_format.model_json_schema(),
                    "strict": True,
                },
            }

        # === STEP 2: Make the actual API call ===
        response = await self.client.chat.completions.create(**request_kwargs)

        # === STEP 3: Convert OpenAI's response back to our types ===
        return self._convert_response_to_our_types(response, output_format)

    def _convert_response_to_our_types(self, response, output_format) -> ChatCompletionResult:
        """Convert OpenAI's response object into our standardized ChatCompletionResult."""
        choice = response.choices[0]
        raw_message = choice.message

        # Convert tool_calls if the model requested any
        tool_calls = None
        if raw_message.tool_calls:
            tool_calls = [
                ToolCallRequest(
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    # OpenAI returns arguments as a JSON string — parse it back to a dict
                    parameters=json.loads(tc.function.arguments),
                )
                for tc in raw_message.tool_calls
            ]

        # Parse structured output if we requested it
        structured_output = None
        if output_format and raw_message.content:
            try:
                data = json.loads(raw_message.content)
                structured_output = output_format(**data)
            except Exception:
                pass  # Fall back gracefully — structured_output will be None

        # Build the AssistantMessage from the raw response
        assistant_message = AssistantMessage(
            source="assistant",
            content=raw_message.content,
            tool_calls=tool_calls,
        )

        # Build the Usage tracking object
        usage = Usage(
            tokens_input=response.usage.prompt_tokens if response.usage else 0,
            tokens_output=response.usage.completion_tokens if response.usage else 0,
            num_calls=1,
        )

        return ChatCompletionResult(
            message=assistant_message,
            usage=usage,
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            structured_output=structured_output,
        )

    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncGenerator:
        """Streaming version — yields chunks as they arrive from the API."""
        api_messages = self._convert_messages_to_api_format(messages)
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            tools=tools if tools else None,
            stream=True,
        )
        async for chunk in stream:
            yield chunk
