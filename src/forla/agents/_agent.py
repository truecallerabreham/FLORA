from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union
from pydantic import BaseModel as PydanticBaseModel

from ._base import BaseAgent
from ..messages import (
    Message, UserMessage, AssistantMessage, SystemMessage, ToolMessage, ToolCallRequest
)
from ..context import AgentContext
from ..types import AgentResponse, CancellationToken, Usage, ChatCompletionResult
from ..tools._base import BaseTool
from ..middleware._base import MiddlewareContext


# ── Events emitted during streaming ──────────────────────────────────────────
# These are yielded by run_stream() alongside messages.
# A user interface consuming run_stream() can use these to show
# progress indicators, tool call badges, error banners, etc.

@dataclass
class TaskStartEvent:
    """Emitted at the very beginning of an agent run."""
    agent_name: str
    task_preview: str    # First 100 chars of the task

@dataclass
class ModelCallEvent:
    """Emitted just before calling the LLM API."""
    agent_name: str
    message_count: int   # How many messages are in the context

@dataclass
class ModelResponseEvent:
    """Emitted immediately after the LLM responds."""
    agent_name: str
    finish_reason: str   # "stop" or "tool_calls"
    has_tool_calls: bool

@dataclass
class ToolCallEvent:
    """Emitted when the agent is about to execute a tool."""
    agent_name: str
    tool_name: str
    parameters: Dict[str, Any]

@dataclass
class ToolCallResponseEvent:
    """Emitted after a tool finishes executing."""
    agent_name: str
    tool_name: str
    result_preview: str   # First 200 chars of the result
    success: bool

@dataclass
class TaskCompleteEvent:
    """Emitted when the agent finishes."""
    agent_name: str
    finish_reason: str
    usage: Usage

class Agent(BaseAgent):
    """The concrete agent implementation.
    
    This class implements the agent execution loop described in Chapter 4:
    1. Prepare context
    2. Call model (through middleware)
    3. If tool_calls: execute tools, add results, repeat
    4. If text: yield final response
    
    IMPORTANT: run_stream() is the primary method.
    run() is implemented as 'run_stream() filtered to AgentResponse'.
    """

    async def run(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AgentResponse:
        """Execute the agent and wait for the complete final result.
        
        Internally calls run_stream() and extracts only the final AgentResponse.
        This is the pattern used throughout picoagents: streaming is the primitive.
        """
        final_response: Optional[AgentResponse] = None
        
        async for item in self.run_stream(task, cancellation_token):
            if isinstance(item, AgentResponse):
                final_response = item
        
        return final_response or AgentResponse(
            content="",
            usage=Usage(),
            finish_reason="error",
        )

    async def run_stream(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[Message, Any, AgentResponse], None]:
        """The real implementation: execute the agent, yielding events as they happen.
        
        Everything this generator yields can be handled by the caller:
        - Messages: show them in a chat UI
        - Events (TaskStartEvent, ToolCallEvent, etc.): show progress indicators
        - AgentResponse: the final result with usage statistics
        """
        start_time = time.time()
        total_usage = Usage()

        # ── Step 1: Normalize the task input ───────────────────────────────
        # The task can be a plain string, a UserMessage, or a list of messages.
        # We always want a list of Message objects.
        task_messages = self._normalize_task(task)
        task_preview = str(task)[:100]
        
        yield TaskStartEvent(agent_name=self.name, task_preview=task_preview)

        # Add task messages to conversation context
        for msg in task_messages:
            self.context.add_message(msg)
            yield msg    # Stream to caller so UI can show the user's input

        # ── Main execution loop ─────────────────────────────────────────────
        iteration = 0
        
        while iteration < self.max_iterations:

            # ── Check for cancellation ──────────────────────────────────────
            if cancellation_token and cancellation_token.is_cancelled():
                yield AgentResponse(
                    content="Task was cancelled.",
                    usage=total_usage,
                    finish_reason="cancelled",
                )
                return

            # ── Step 2: Prepare the full message context for the LLM ───────
            # This combines: system instructions + memory context + conversation history
            llm_messages = await self._prepare_llm_messages()
            
            # Get tool schemas if we have tools
            tool_schemas = self._get_tools_for_llm() if self.tools else None
            
            yield ModelCallEvent(
                agent_name=self.name,
                message_count=len(llm_messages),
            )

            # ── Step 3: Call the model (through middleware) ─────────────────
            # We route through middleware so interceptors can log, block, etc.
            completion_result: Optional[ChatCompletionResult] = None
            
            try:
                async for item in self.middleware_chain.execute_stream(
                    operation="model_call",
                    agent_name=self.name,
                    data=llm_messages,
                    func=lambda msgs: self.model_client.create(
                        msgs,
                        tools=tool_schemas,
                        output_format=self.output_format,
                    ),
                ):
                    if isinstance(item, ChatCompletionResult):
                        completion_result = item
                    else:
                        yield item    # Forward middleware events to caller
                        
            except asyncio.CancelledError:
                raise
            except Exception as e:
                error_msg = AssistantMessage(
                    source=self.name,
                    content=f"Model call failed: {e}",
                )
                self.context.add_message(error_msg)
                yield error_msg
                yield AgentResponse(content=str(e), usage=total_usage, finish_reason="error")
                return

            if completion_result is None:
                yield AgentResponse(
                    content="No response from model.",
                    usage=total_usage,
                    finish_reason="error",
                )
                return

            total_usage = total_usage + completion_result.usage
            assistant_msg = completion_result.message
            assistant_msg = AssistantMessage(
                source=self.name,
                content=assistant_msg.content,
                tool_calls=assistant_msg.tool_calls,
                metadata=assistant_msg.metadata,
            )

            yield ModelResponseEvent(
                agent_name=self.name,
                finish_reason=completion_result.finish_reason,
                has_tool_calls=bool(assistant_msg.tool_calls),
            )

            # ── Step 4: Handle the response ─────────────────────────────────
            
            if assistant_msg.tool_calls:
                # The model wants to use tools before answering.
                # Add the assistant message to context (it contains the tool requests).
                self.context.add_message(assistant_msg)
                
                # Execute each tool call
                for tool_call in assistant_msg.tool_calls:
                    yield ToolCallEvent(
                        agent_name=self.name,
                        tool_name=tool_call.tool_name,
                        parameters=tool_call.parameters,
                    )
                    
                    # Execute the tool (through middleware for logging/approval)
                    tool_msg = await self._execute_tool_call(tool_call)
                    self.context.add_message(tool_msg)
                    
                    yield ToolCallResponseEvent(
                        agent_name=self.name,
                        tool_name=tool_call.tool_name,
                        result_preview=tool_msg.content[:200],
                        success=tool_msg.success,
                    )
                    yield tool_msg    # Stream the actual ToolMessage to caller
                
                # Loop again — the model will now see the tool results
                # and can decide to call more tools or give a final answer
                iteration += 1
                continue

            else:
                # No tool calls — this IS the final answer.
                self.context.add_message(assistant_msg)
                yield assistant_msg    # Stream the final text response

                # ── Step 5: Update long-term memory ─────────────────────────
                if self.memory and assistant_msg.content:
                    from ..memory._base import MemoryContent
                    last_user = self._get_last_user_message()
                    await self.memory.add(MemoryContent(
                        content=f"Q: {last_user}\nA: {assistant_msg.content}",
                        metadata={"agent": self.name},
                    ))

                # ── Step 6: Yield the final AgentResponse ───────────────────
                total_usage.duration_ms = int((time.time() - start_time) * 1000)
                
                yield TaskCompleteEvent(
                    agent_name=self.name,
                    finish_reason=completion_result.finish_reason,
                    usage=total_usage,
                )
                
                yield AgentResponse(
                    content=assistant_msg.content or "",
                    usage=total_usage,
                    finish_reason=completion_result.finish_reason,
                )
                return    # Done!

        # If we exit the loop without returning, we hit max_iterations
        total_usage.duration_ms = int((time.time() - start_time) * 1000)
        yield AgentResponse(
            content="Maximum iterations reached without completing task.",
            usage=total_usage,
            finish_reason="max_iterations",
        )

    # ── Helper Methods ────────────────────────────────────────────────────

    def _normalize_task(
        self, task: Union[str, UserMessage, List[Message]]
    ) -> List[Message]:
        """Convert any task input format into a list of Messages."""
        if isinstance(task, str):
            return [UserMessage(content=task, source="user")]
        elif isinstance(task, UserMessage):
            return [task]
        elif isinstance(task, list):
            return task
        else:
            return [UserMessage(content=str(task), source="user")]

    async def _prepare_llm_messages(self) -> List[Message]:
        """Build the complete message list for the LLM call.
        
        The order matters for LLMs:
        [SystemMessage(instructions + memory)] + [conversation history]
        
        Memory is injected into the system message to provide context
        without polluting the conversation history with extra messages.
        """
        system_content = self.instructions

        # Augment the system message with long-term memory context
        if self.memory:
            context_items = await self.memory.get_context(max_items=5)
            if context_items:
                memory_text = "\n".join(f"- {item}" for item in context_items)
                system_content += f"\n\nRelevant context from past interactions:\n{memory_text}"

        return [
            SystemMessage(content=system_content, source="system"),
            *self.context.get_messages(),
        ]

    async def _execute_tool_call(self, tool_call: ToolCallRequest) -> ToolMessage:
        """Find and execute a tool, returning the result as a ToolMessage.
        
        The ToolMessage is what gets added to the conversation history.
        The model reads it to understand what the tool returned.
        
        WHY always return ToolMessage even on error?
        Because the model needs to SEE the error to decide what to do next.
        If we raise an exception, the agent crashes.
        If we return a ToolMessage with success=False, the model can
        try a different approach or explain the failure to the user.
        """
        tool = self._find_tool(tool_call.tool_name)

        if tool is None:
            available = [t.name for t in self.tools]
            return ToolMessage(
                source=self.name,
                content=(
                    f"Tool '{tool_call.tool_name}' was not found. "
                    f"Available tools: {available}"
                ),
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=f"Tool not found: {tool_call.tool_name}",
            )

        # Execute through middleware (enables logging, approval flows, etc.)
        tool_result = None
        try:
            async for item in self.middleware_chain.execute_stream(
                operation="tool_call",
                agent_name=self.name,
                data=tool_call.parameters,
                func=lambda params: tool.execute(params),
                tool_name=tool_call.tool_name,
            ):
                if hasattr(item, "success"):   # It's a ToolResult
                    tool_result = item
        except Exception as e:
            return ToolMessage(
                source=self.name,
                content=f"Tool execution failed: {e}",
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=str(e),
            )

        if tool_result is None:
            # Middleware chain produced no result — shouldn't happen normally
            return ToolMessage(
                source=self.name,
                content="Tool returned no result.",
                tool_call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error="No result from tool execution",
            )

        # Format the content for the conversation
        if tool_result.success:
            content = f"✓ {tool_result.result}"
        else:
            content = f"✗ Error: {tool_result.error}"

        return ToolMessage(
            source=self.name,
            content=content,
            tool_call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            success=tool_result.success,
            error=tool_result.error,
        )

    def _get_last_user_message(self) -> str:
        """Find the most recent UserMessage in the conversation context."""
        for msg in reversed(self.context.messages):
            if isinstance(msg, UserMessage):
                return str(msg.content)
        return ""
