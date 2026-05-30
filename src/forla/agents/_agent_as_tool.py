"""
AgentAsTool wrapper - allows any agent to be used as a tool by other agents.

This module provides the AgentAsTool class that wraps BaseAgent instances,
exposing them as BaseTool instances for composition patterns.
"""

from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .._cancellation_token import CancellationToken
from ..messages import Message
from ..tools import BaseTool
from ..types import AgentEvent, AgentResponse, ToolResult

if TYPE_CHECKING:
    from ._base import BaseAgent

ResultStrategy = Union[str, Callable[[List[Message]], str]]


class AgentAsTool(BaseTool):
    """
    Wraps any BaseAgent to expose it as a tool that other agents can use.

    This enables hierarchical composition where specialized agents can be
    used as tools by higher-level coordinating agents.

    The result_strategy parameter controls how agent messages are summarized:
    - "last" (default): Return only the last message
    - "last:N": Return the last N messages concatenated
    - "all": Return all messages concatenated
    - Callable: Custom function that takes messages and returns a string
    """

    def __init__(
        self,
        agent: "BaseAgent",
        task_parameter_name: str = "task",
        result_strategy: ResultStrategy = "last",
    ):
        """
        Initialize the agent-as-tool wrapper.

        Args:
            agent: The agent to wrap as a tool
            task_parameter_name: Parameter name for the task input
            result_strategy: Strategy for extracting result from messages.
                Can be "last", "last:N", "all", or a callable that takes
                a list of messages and returns a string.

        Examples:
            >>> # Use last message only (default)
            >>> tool = AgentAsTool(agent)
            >>>
            >>> # Use last 3 messages
            >>> tool = AgentAsTool(agent, result_strategy="last:3")
            >>>
            >>> # Use all messages
            >>> tool = AgentAsTool(agent, result_strategy="all")
            >>>
            >>> # Custom extraction logic
            >>> def extract_assistant_messages(messages):
            ...     return "\\n".join(m.content for m in messages if m.role == "assistant")
            >>> tool = AgentAsTool(agent, result_strategy=extract_assistant_messages)
        """
        from ._base import BaseAgent

        if not isinstance(agent, BaseAgent):
            raise TypeError("agent must be a BaseAgent instance")

        super().__init__(name=agent.name, description=agent.description)

        self.agent = agent
        self.task_parameter_name = task_parameter_name
        self.result_strategy = result_strategy
        self._validate_result_strategy()

    def _validate_result_strategy(self) -> None:
        """Validate that the result_strategy is properly formatted."""
        if callable(self.result_strategy):
            return

        if not isinstance(self.result_strategy, str):
            raise TypeError("result_strategy must be a string or callable")

        # Validate string strategies
        if self.result_strategy == "all" or self.result_strategy == "last":
            return

        # Validate "last:N" format
        if self.result_strategy.startswith("last:"):
            try:
                n = int(self.result_strategy.split(":")[1])
                if n <= 0:
                    raise ValueError("N must be positive in 'last:N' strategy")
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Invalid result_strategy format: {self.result_strategy}. "
                    "Expected 'last:N' where N is a positive integer"
                ) from e
        else:
            raise ValueError(
                f"Unknown result_strategy: {self.result_strategy}. "
                "Expected 'last', 'last:N', 'all', or a callable"
            )

    def _extract_result(self, messages: List[Message]) -> str:
        """
        Extract result from messages based on the configured strategy.

        Args:
            messages: List of messages from the agent

        Returns:
            Extracted result string
        """
        if not messages:
            return ""

        # Custom callable strategy
        if callable(self.result_strategy):
            return self.result_strategy(messages)

        # String-based strategies
        if self.result_strategy == "last":
            return messages[-1].content

        if self.result_strategy == "all":
            return "\n".join(msg.content for msg in messages)

        # "last:N" strategy
        if self.result_strategy.startswith("last:"):
            n = int(self.result_strategy.split(":")[1])
            selected = messages[-n:]
            return "\n".join(msg.content for msg in selected)

        # Should never reach here due to validation
        return messages[-1].content

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Define the tool's parameter schema.

        Returns:
            JSON schema for tool parameters
        """
        return {
            "type": "object",
            "properties": {
                self.task_parameter_name: {
                    "type": "string",
                    "description": f"Task for {self.agent.name} to complete",
                }
            },
            "required": [self.task_parameter_name],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the wrapped agent and return final result.

        Args:
            parameters: Tool parameters containing the task

        Returns:
            ToolResult with agent's final response
        """
        task = parameters.get(self.task_parameter_name, "")

        try:
            response = await self.agent.run(task=task, context=None, cancellation_token=None)

            # Extract content using configured strategy
            final_content = self._extract_result(response.messages)

            return ToolResult(
                success=True,
                result=final_content,
                error=None,
                metadata={
                    "agent_name": self.agent.name,
                    "message_count": len(response.messages),
                    "usage": response.usage.model_dump() if response.usage else None,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result="",
                error=f"Agent execution failed: {str(e)}",
                metadata={"agent_name": self.agent.name},
            )

    async def execute_stream(
        self,
        parameters: Dict[str, Any],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[Message, "AgentEvent", ToolResult], None]:
        """
        Execute the wrapped agent with streaming output.

        Args:
            parameters: Tool parameters containing the task
            cancellation_token: Optional cancellation token

        Yields:
            Agent messages/events, followed by final ToolResult
        """
        task = parameters.get(self.task_parameter_name, "")

        final_response = None
        error_occurred = False
        error_message = ""

        try:
            # Stream all agent output
            async for item in self.agent.run_stream(
                task=task,
                context=None,
                cancellation_token=cancellation_token,
                verbose=False,
                stream_tokens=False,
            ):
                if isinstance(item, AgentResponse):
                    final_response = item
                else:
                    # Forward agent messages and events
                    yield item

        except Exception as e:
            error_occurred = True
            error_message = str(e)

        # Emit final ToolResult
        if error_occurred:
            yield ToolResult(
                success=False,
                result="",
                error=f"Agent execution failed: {error_message}",
                metadata={"agent_name": self.agent.name},
            )
        else:
            # Extract content using configured strategy
            final_content = ""
            if final_response and final_response.messages:
                final_content = self._extract_result(final_response.messages)

            yield ToolResult(
                success=True,
                result=final_content,
                error=None,
                metadata={
                    "agent_name": self.agent.name,
                    "message_count": len(final_response.messages)
                    if final_response
                    else 0,
                    "usage": final_response.usage.model_dump()
                    if final_response and final_response.usage
                    else None,
                },
            )

    def model_dump(self) -> Dict[str, Any]:
        """Serialize the agent-as-tool wrapper for persistence."""
        return {
            "type": "agent_as_tool",
            "agent": {"name": self.agent.name},
            "task_parameter_name": self.task_parameter_name,
        }
