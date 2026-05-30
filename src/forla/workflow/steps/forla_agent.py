"""
ForlaAgent step implementation for workflows.

This step wraps a forla BaseAgent to work within the workflow system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field

from forla import Component

from ...agents import BaseAgent
from ...messages import AssistantMessage, Message, UserMessage
from ...types import AgentResponse, Usage
from ..core._models import Context, StepMetadata
from ._step import BaseStep, BaseStepConfig

logger = logging.getLogger(__name__)


class ForlaAgentInput(BaseModel):
    """Input schema for ForlaAgent workflow step."""

    task: str = Field(..., description="The task or question to send to the agent")
    additional_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context or parameters"
    )


class ForlaAgentOutput(BaseModel):
    """Output schema for ForlaAgent workflow step."""

    response: str = Field(..., description="The agent's final response")
    messages: list = Field(
        default_factory=list, description="Complete conversation messages"
    )
    usage: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )


class ForlaAgentStepConfig(BaseStepConfig):
    """Configuration for ForlaAgent step serialization."""

    agent_name: str = Field(..., description="Name of the agent")
    agent_description: str = Field(..., description="Description of the agent")
    agent_instructions: str = Field(..., description="Agent instructions/system prompt")
    agent_model: str = Field(..., description="LLM model used by agent")
    agent_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional agent configuration"
    )


class ForlaAgentStep(
    Component[ForlaAgentStepConfig], BaseStep[ForlaAgentInput, ForlaAgentOutput]
):
    """Workflow step that executes a Forla BaseAgent."""

    component_config_schema = ForlaAgentStepConfig
    component_type = "forla_agent_step"
    component_provider_override = "forla.workflow.steps.ForlaAgentStep"

    def __init__(self, step_id: str, metadata: StepMetadata, agent: BaseAgent):
        """Initialize ForlaAgent step.

        Args:
            step_id: Unique identifier for this step
            metadata: Step metadata
            agent: Forla BaseAgent instance to wrap
        """
        BaseStep.__init__(
            self,
            step_id=step_id,
            metadata=metadata,
            input_type=ForlaAgentInput,
            output_type=ForlaAgentOutput,
        )

        self.agent = agent
        logger.debug(f"Created ForlaAgentStep '{step_id}' with agent '{agent.name}'")

    async def execute(
        self, input_data: ForlaAgentInput, context: Context
    ) -> ForlaAgentOutput:
        """Execute the ForlaAgent.

        Args:
            input_data: Validated input containing the task
            context: Workflow context for state sharing

        Returns:
            ForlaAgentOutput containing response and metadata
        """
        logger.info(
            f"Executing ForlaAgent step '{self.step_id}' with task: {input_data.task[:50]}..."
        )

        # Store agent request info in context for debugging
        context.set(
            f"{self.step_id}_request_info",
            {
                "agent_name": self.agent.name,
                "task": input_data.task,
                "timestamp": datetime.now().isoformat(),
                "additional_context": input_data.additional_context,
            },
        )

        try:
            # Execute agent with the task
            agent_result: AgentResponse = await self.agent.run(input_data.task)

            # Extract final response from messages
            final_response = ""
            if agent_result.messages:
                # Get the last assistant message as the response
                for message in reversed(agent_result.messages):
                    if isinstance(message, AssistantMessage):
                        final_response = message.content
                        break

                # Fallback to last message if no assistant message found
                if not final_response and agent_result.messages:
                    final_response = agent_result.messages[-1].content

            if not final_response:
                final_response = "No response generated"

            # Convert messages to serializable format
            serializable_messages = []
            for msg in agent_result.messages:
                msg_dict = {
                    "role": msg.role,
                    "content": msg.content,
                    "source": msg.source,
                    "timestamp": msg.timestamp.isoformat()
                    if hasattr(msg, "timestamp") and msg.timestamp
                    else None,
                }
                serializable_messages.append(msg_dict)

            # Convert usage to dict
            usage_dict = {
                "duration_ms": agent_result.usage.duration_ms,
                "llm_calls": agent_result.usage.llm_calls,
                "tokens_input": agent_result.usage.tokens_input,
                "tokens_output": agent_result.usage.tokens_output,
                "tool_calls": agent_result.usage.tool_calls,
                "memory_operations": agent_result.usage.memory_operations,
                "cost_estimate": agent_result.usage.cost_estimate,
            }

            # Build metadata
            execution_metadata = {
                "agent_name": self.agent.name,
                "message_count": len(agent_result.messages),
                "elapsed_time": agent_result.usage.duration_ms / 1000.0,
                "llm_calls": agent_result.usage.llm_calls,
                "tokens_total": agent_result.usage.tokens_input
                + agent_result.usage.tokens_output,
                "execution_timestamp": datetime.now().isoformat(),
            }

            # Add additional context if provided
            if input_data.additional_context:
                execution_metadata["additional_context"] = input_data.additional_context

            # Store agent output in context
            context.set(
                f"{self.step_id}_output",
                {
                    "response": final_response,
                    "messages": serializable_messages,
                    "usage": usage_dict,
                    "metadata": execution_metadata,
                },
            )

            return ForlaAgentOutput(
                response=final_response,
                messages=serializable_messages,
                usage=usage_dict,
                metadata=execution_metadata,
            )

        except Exception as e:
            error_msg = f"ForlaAgent execution failed: {str(e)}"
            logger.error(f"Step '{self.step_id}' failed: {error_msg}")

            # Store error info in context
            context.set(
                f"{self.step_id}_error",
                {"error": error_msg, "timestamp": datetime.now().isoformat()},
            )

            # Return error response
            return ForlaAgentOutput(
                response=f"Error: {error_msg}",
                messages=[],
                usage={},
                metadata={
                    "agent_name": self.agent.name,
                    "error": error_msg,
                    "execution_timestamp": datetime.now().isoformat(),
                },
            )

    def _to_config(self) -> ForlaAgentStepConfig:
        """Convert to configuration for serialization."""
        return ForlaAgentStepConfig(
            **self._serialize_types(),
            agent_name=self.agent.name,
            agent_description=getattr(self.agent, "description", "ForlaAgent"),
            agent_instructions=getattr(self.agent, "instructions", ""),
            agent_model=getattr(self.agent, "model", "unknown"),
            agent_config={},
        )

    @classmethod
    def _from_config(cls, config: ForlaAgentStepConfig) -> "ForlaAgentStep":
        """Create step from configuration.

        Note: This is a placeholder implementation. In a real scenario, you would need
        a way to reconstruct the BaseAgent from the configuration.
        """
        # This is where you'd need to recreate the agent from config
        # For now, we'll raise an error indicating this needs implementation
        raise NotImplementedError(
            "ForlaAgentStep deserialization requires agent reconstruction logic. "
            "You need to implement agent factory/registry to recreate agents from config."
        )
