"""
Base orchestrator implementation.

This module provides the foundational BaseOrchestrator class following
the PRD specification.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Sequence, Union, cast

from pydantic import BaseModel

from .._cancellation_token import CancellationToken
from .._component_config import ComponentBase
from ..agents import BaseAgent
from ..messages import Message, UserMessage
from ..termination import BaseTermination
from ..types import (
    AgentExecutionCompleteEvent,
    AgentExecutionStartEvent,
    AgentResponse,
    AgentSelectionEvent,
    OrchestrationCompleteEvent,
    OrchestrationEvent,
    OrchestrationResponse,
    OrchestrationStartEvent,
    StopMessage,
    Usage,
)


class BaseOrchestrator(ComponentBase[BaseModel], ABC):
    """
    Abstract base class for all orchestration patterns.

    Defines the core orchestration interface and universal orchestration loop
    as specified in the PRD.
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        termination: BaseTermination,
        max_iterations: int = 50,
        name: Optional[str] = None,
        description: Optional[str] = None,
        example_tasks: Optional[List[str]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            agents: List of agents available for orchestration
            termination: Termination condition
            max_iterations: Safety fallback for iterations
            name: Optional name for the orchestrator
            description: Optional description of the orchestrator's purpose
            example_tasks: Optional list of example tasks to help users discover orchestrator capabilities
        """
        if not agents:
            raise ValueError("At least one agent is required")

        self.agents = list(agents)  # Convert to list for internal use
        self.termination = termination
        self.max_iterations = max_iterations
        self.name = name or self.__class__.__name__
        self.description = description
        self.example_tasks = example_tasks or []

        # Runtime state
        self.shared_messages: List[Message] = []
        self.iteration_count = 0
        self.start_time: Optional[float] = None

        # Validate agent names are unique
        names = [agent.name for agent in agents]
        if len(names) != len(set(names)):
            raise ValueError("Agent names must be unique")

    async def run(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
        persist: bool = False,
    ) -> OrchestrationResponse:
        """
        Execute the orchestration pattern.

        Args:
            task: The task to orchestrate (same type as agent.run())
            cancellation_token: Optional cancellation token
            persist: If True, save the run to ~/.forla/ (DB
                index + JSON file with full response data)

        Returns:
            OrchestrationResponse with messages, usage, and metadata
        """
        # Reset state for new run
        self._reset_for_run()

        final_result = None

        try:
            async for item in self.run_stream(task, cancellation_token):
                if isinstance(item, OrchestrationResponse):
                    final_result = item

            result = final_result or self._create_fallback_result("No result produced")

        except asyncio.CancelledError:
            # Re-raise cancellation for caller to handle
            raise
        except Exception as e:
            # Handle errors gracefully
            elapsed_time = int((time.time() - (self.start_time or time.time())) * 1000)
            result = OrchestrationResponse(
                messages=self.shared_messages,
                final_result=f"Orchestration failed: {str(e)}",
                usage=Usage(duration_ms=elapsed_time),
                stop_message=StopMessage(
                    content=f"Error: {str(e)}", source="Exception"
                ),
                pattern_metadata=self._get_pattern_metadata(),
            )

        if persist:
            try:
                from ..store import get_default_store

                store = get_default_store()
                await store.save_orchestrator_run(self, result)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to persist orchestrator run: {e}"
                )

        return result

    async def run_stream(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
        verbose: bool = False,
    ) -> AsyncGenerator[
        Union[Message, OrchestrationEvent, OrchestrationResponse], None
    ]:
        """
        Execute orchestration with streaming output.

        Args:
            task: The task to orchestrate (same type as agent.run())
            cancellation_token: Optional cancellation token
            verbose: If True, emit orchestration events; if False, only emit messages and results

        Yields:
            Messages, events (if verbose=True), and final OrchestrationResponse
        """
        # Reset state for new run
        self._reset_for_run()
        self.start_time = time.time()

        # Initialize stop_message
        stop_message: Optional[StopMessage] = None
        # Track all messages streamed to user for accurate termination counting
        streamed_messages: List[Message] = []
        # Track position of last termination check for delta calculation
        self._last_termination_check_count = 0
        # Track agent usage statistics for aggregation
        agent_usage_stats: List[Usage] = []

        try:
            # Emit orchestration start event
            if verbose:
                yield OrchestrationStartEvent(
                    source="orchestrator",
                    task=str(task),
                    pattern=self.__class__.__name__,
                )

            # Normalize task to initial messages
            initial_messages = self._normalize_task_to_messages(task)
            self.shared_messages.extend(initial_messages)

            # Yield initial messages
            for message in initial_messages:
                yield message
                streamed_messages.append(message)

            # Initialize termination with initial messages
            self.termination.check(initial_messages)
            # Update termination check counter after initial check
            self._last_termination_check_count = len(streamed_messages)

            # Universal orchestration loop
            while self.iteration_count < self.max_iterations:
                # Check for cancellation at the start of each iteration
                if cancellation_token and cancellation_token.is_cancelled():
                    raise asyncio.CancelledError()

                # Check termination BEFORE processing next agent (but not on first iteration)
                if self.iteration_count > 0 and self.termination.is_met():
                    stop_message = StopMessage(
                        content=self.termination.get_reason(),
                        source=self.termination.__class__.__name__,
                        metadata=self.termination.get_metadata(),
                    )
                    break

                # 1. Select next agent (pattern-specific logic)
                next_agent = await self.select_next_agent()

                if verbose:
                    yield AgentSelectionEvent(
                        source="orchestrator",
                        selected_agent=next_agent.name,
                        selection_reason=f"Iteration {self.iteration_count + 1}",
                    )

                # 2. Prepare context for agent (pattern-specific)
                context = await self.prepare_context_for_agent(next_agent)

                context_size = len(context) if isinstance(context, list) else 1
                if verbose:
                    yield AgentExecutionStartEvent(
                        source="orchestrator",
                        executing_agent=next_agent.name,
                        context_size=context_size,
                    )

                # 3. Execute agent with streaming support and cancellation
                agent_messages = []
                result: Optional[AgentResponse] = None

                try:
                    async for item in next_agent.run_stream(
                        context, cancellation_token=cancellation_token, verbose=verbose
                    ):
                        # Check for cancellation during agent execution
                        if cancellation_token and cancellation_token.is_cancelled():
                            raise asyncio.CancelledError()

                        # Type guard for Message (has content and role attributes)
                        if hasattr(item, "content") and hasattr(item, "role"):
                            # This is a Message - collect it but only forward non-user messages
                            message_item = cast(Message, item)
                            agent_messages.append(message_item)

                            # Don't stream UserMessages - they're just context we sent to agents
                            if not isinstance(message_item, UserMessage):
                                yield message_item
                                streamed_messages.append(message_item)
                        elif hasattr(item, "messages") and hasattr(item, "usage"):
                            # This is an AgentResponse - store it but don't forward
                            result = cast(AgentResponse, item)
                        # Note: Other agent events (AgentEvent) are not forwarded to maintain type safety

                except asyncio.CancelledError:
                    # Handle cancellation gracefully
                    if verbose:
                        yield AgentExecutionCompleteEvent(
                            source="orchestrator",
                            executing_agent=next_agent.name,
                            success=False,
                            message_count=len(agent_messages),
                        )
                    raise

                # Check for cancellation again after agent execution
                if cancellation_token and cancellation_token.is_cancelled():
                    raise asyncio.CancelledError()

                # Ensure we have a result
                if result is None:
                    # Create fallback result from collected messages
                    result = AgentResponse(
                        source=next_agent.name,
                        messages=agent_messages,
                        usage=Usage(duration_ms=0, llm_calls=0),
                        finish_reason="completed_without_response",
                    )

                # Collect agent usage statistics for aggregation
                agent_usage_stats.append(result.usage)

                # Type guard: ensure result is AgentResponse
                assert hasattr(
                    result, "messages"
                ), "Result must be AgentResponse with messages"

                if verbose:
                    yield AgentExecutionCompleteEvent(
                        source="orchestrator",
                        executing_agent=next_agent.name,
                        success=True,
                        message_count=len(result.messages),
                    )

                # 4. Update shared state (pattern-specific)
                # result is guaranteed to be AgentResponse due to the assertion above
                await self.update_shared_state(result)

                # 5. Check termination with new messages streamed in this iteration
                # Calculate messages streamed since last termination check
                new_streamed_messages = streamed_messages[
                    self._last_termination_check_count :
                ]
                self._last_termination_check_count = len(streamed_messages)

                stop_message = self.termination.check(new_streamed_messages)
                if stop_message:
                    break

                self.iteration_count += 1

            # Handle max iterations reached
            if self.iteration_count >= self.max_iterations and stop_message is None:
                stop_message = StopMessage(
                    content=f"Maximum iterations reached ({self.max_iterations})",
                    source="MaxIterations",
                )

            # Ensure we always have a stop_message
            if stop_message is None:
                stop_message = StopMessage(
                    content="Orchestration completed normally", source="Completion"
                )

            # Emit orchestration complete event
            final_result = self._generate_final_result()
            if verbose:
                yield OrchestrationCompleteEvent(
                    source="orchestrator",
                    result=final_result,
                    stop_reason=stop_message.content,
                )

            # Calculate usage statistics
            elapsed_time = int((time.time() - self.start_time) * 1000)

            # Aggregate usage from all agent executions
            total_usage = Usage(duration_ms=elapsed_time)
            for agent_usage in agent_usage_stats:
                total_usage = total_usage + agent_usage

            # Yield final OrchestrationResponse
            orchestration_result = OrchestrationResponse(
                messages=self.shared_messages,
                final_result=final_result,
                usage=total_usage,
                stop_message=stop_message,
                pattern_metadata=self._get_pattern_metadata(),
            )
            yield orchestration_result

        except asyncio.CancelledError:
            # Handle cancellation at orchestration level
            elapsed_time = int((time.time() - (self.start_time or time.time())) * 1000)

            if verbose:
                yield OrchestrationCompleteEvent(
                    source="orchestrator",
                    result="Orchestration cancelled",
                    stop_reason="Cancellation",
                )

            # Aggregate usage from all agent executions before cancellation
            total_usage = Usage(duration_ms=elapsed_time)
            for agent_usage in agent_usage_stats:
                total_usage = total_usage + agent_usage

            cancellation_result = OrchestrationResponse(
                messages=self.shared_messages,
                final_result="Orchestration was cancelled",
                usage=total_usage,
                stop_message=StopMessage(
                    content="Orchestration cancelled", source="CancellationToken"
                ),
                pattern_metadata=self._get_pattern_metadata(),
            )
            yield cancellation_result

            # Re-raise for proper cancellation handling
            raise

    @abstractmethod
    async def select_next_agent(self) -> BaseAgent:
        """Pattern-specific agent selection logic."""
        pass

    @abstractmethod
    async def prepare_context_for_agent(
        self, agent: BaseAgent
    ) -> Union[str, UserMessage, List[Message]]:
        """Pattern-specific context preparation."""
        pass

    @abstractmethod
    async def update_shared_state(self, result: AgentResponse) -> None:
        """Pattern-specific state update after agent execution."""
        pass

    def _normalize_task_to_messages(
        self, task: Union[str, UserMessage, List[Message]]
    ) -> List[Message]:
        """Convert task input to list of messages."""
        if isinstance(task, str):
            return [UserMessage(content=task, source="user")]
        elif isinstance(task, UserMessage):
            return [task]
        elif isinstance(task, list):
            return task
        else:
            # Fallback for any other message type
            return (
                [task]
                if hasattr(task, "content")
                else [UserMessage(content=str(task), source="user")]
            )

    def _extract_new_messages(
        self,
        agent_messages: List[Message],
        sent_context: Union[str, UserMessage, List[Message]],
    ) -> List[Message]:
        """Extract only new messages from agent response, excluding the context we sent."""
        # If we sent a list context, agent should return context + new messages
        if isinstance(sent_context, list):
            context_len = len(sent_context)
            return (
                agent_messages[context_len:]
                if len(agent_messages) > context_len
                else []
            )
        else:
            # If we sent str/UserMessage, agent returns [UserMessage, ...new messages]
            return agent_messages[1:] if len(agent_messages) > 1 else []

    def _reset_for_run(self) -> None:
        """Reset orchestrator state for new run."""
        self.shared_messages = []
        self.iteration_count = 0
        self.start_time = None
        self.termination.reset()
        self._last_termination_check_count = 0

    def _generate_final_result(self) -> str:
        """Generate final result summary."""
        if not self.shared_messages:
            return "No messages generated"

        # Find the last assistant message
        for message in reversed(self.shared_messages):
            if hasattr(message, "role") and message.role == "assistant":
                return message.content

        return "Task completed"

    def get_agent_capabilities_summary(self) -> str:
        """Get agent capabilities summary for LLM consumption."""
        summary_lines = []
        for agent in self.agents:
            line = f"- {agent.name}: {agent.description}"

            # Add tool information if available
            if hasattr(agent, "tools") and agent.tools:
                tool_names = []
                for tool in agent.tools:
                    if hasattr(tool, "name"):
                        tool_names.append(tool.name)
                    elif hasattr(tool, "__name__"):
                        tool_names.append(tool.__name__)
                    else:
                        tool_names.append(str(tool)[:20])

                if tool_names:
                    line += f" | Tools: {', '.join(tool_names)}"

            summary_lines.append(line)

        return "\n".join(summary_lines)

    def _get_pattern_metadata(self) -> Dict[str, Any]:
        """Get pattern-specific metadata for result."""
        return {
            "pattern": self.__class__.__name__,
            "iterations_completed": self.iteration_count,
            "agents_count": len(self.agents),
            "message_count": len(self.shared_messages),
        }

    def _create_fallback_result(self, reason: str) -> OrchestrationResponse:
        """Create fallback result for error cases."""
        elapsed_time = int((time.time() - (self.start_time or time.time())) * 1000)

        return OrchestrationResponse(
            messages=self.shared_messages,
            final_result=reason,
            usage=Usage(duration_ms=elapsed_time),
            stop_message=StopMessage(content=reason, source="Fallback"),
            pattern_metadata=self._get_pattern_metadata(),
        )
