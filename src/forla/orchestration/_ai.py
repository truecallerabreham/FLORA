"""
AI-driven conversation orchestration pattern.

This module provides the AISelectorOrchestrator that uses LLM reasoning
with structured output to select the most appropriate next agent.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .._component_config import Component, ComponentModel
from ..agents import BaseAgent
from ..llm import BaseChatCompletionClient
from ..messages import Message, UserMessage
from ..termination import BaseTermination
from ..types import AgentResponse
from ._base import BaseOrchestrator


class AgentSelection(BaseModel):
    """Structured output for agent selection decision."""

    selected_agent: str = Field(..., description="Name of the selected agent")
    reasoning: str = Field(..., description="Explanation for why this agent was chosen")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in selection (0.0-1.0)"
    )


class AIOrchestratorConfig(BaseModel):
    """Configuration for AIOrchestrator serialization."""

    agents: List[ComponentModel] = Field(default_factory=list)
    termination: ComponentModel
    model_client: ComponentModel
    max_iterations: int = 50


class AIOrchestrator(Component[AIOrchestratorConfig], BaseOrchestrator):
    """
    AI-driven conversation orchestration pattern.

    Uses LLM reasoning with structured output to select the most appropriate
    next agent based on conversation context and agent capabilities.
    """

    component_config_schema = AIOrchestratorConfig
    component_type = "orchestrator"
    component_provider_override = "forla.orchestration.AIOrchestrator"

    def __init__(
        self,
        agents: List[BaseAgent],
        termination: BaseTermination,
        model_client: BaseChatCompletionClient,
        max_iterations: int = 50,
    ):
        super().__init__(agents, termination, max_iterations)
        self.model_client = model_client
        self.selection_history: List[Dict[str, Any]] = []  # Track decisions
        self.agent_capabilities_cache: Optional[str] = None  # Performance optimization

    async def select_next_agent(self) -> BaseAgent:
        """Use LLM reasoning with structured output to select most appropriate next agent."""

        # Get cached agent capabilities summary
        capabilities = self.get_agent_capabilities_summary()

        # Prepare conversation context for selection
        conversation_context = self._format_conversation_for_selection()

        # Build selection prompt
        selection_prompt = f"""You are coordinating a team of AI agents working collaboratively on a task.

Available agents and their capabilities:
{capabilities}

Recent conversation history:
{conversation_context}

Based on the conversation context and each agent's specific capabilities, choose which agent should respond next to move the conversation or task forward. Consider:
- What type of response is needed right now?
- Which agent's skills/tools best match the current need?
- Natural flow of the conversation
- Avoiding repetitive selections unless justified

Select the most appropriate agent and explain your reasoning. Your reason should be a single clean and clear line."""

        # Create messages for LLM call
        messages: List[Message] = [
            UserMessage(content=selection_prompt, source="orchestrator")
        ]

        try:
            # Make structured LLM call for agent selection - now properly async!
            result = await self.model_client.create(
                messages=messages, output_format=AgentSelection
            )

            # Extract structured selection
            if result.structured_output:
                selection = result.structured_output
                if isinstance(selection, AgentSelection):
                    selected_name = selection.selected_agent
                    reasoning = selection.reasoning
                    confidence = selection.confidence
                else:
                    # Fallback if wrong type
                    selected_name = self._get_fallback_agent_name()
                    reasoning = "Fallback due to unexpected structured output type"
                    confidence = 0.3
            else:
                # Fallback if structured output fails
                selected_name = self._extract_agent_name_from_text(
                    result.message.content
                )
                reasoning = "Fallback selection due to parsing error"
                confidence = 0.5

        except Exception as e:
            # Graceful fallback on LLM call failure
            print(f"Warning: Agent selection LLM call failed: {e}")
            selected_name = self._get_fallback_agent_name()
            reasoning = f"Fallback due to LLM error: {str(e)}"
            confidence = 0.1

        # Find selected agent
        selected_agent = self._find_agent_by_name(selected_name)

        # Track selection history for analysis
        selection_choice = {
            "selected_agent": selected_agent.name,
            "iteration": self.iteration_count,
            "reasoning": reasoning,
            "confidence": confidence,
            "conversation_length": len(self.shared_messages),
        }

        self.selection_history.append(selection_choice)

        return selected_agent

    async def prepare_context_for_agent(self, agent: BaseAgent) -> str:
        """Format full shared conversation history as a single context string."""
        if not self.shared_messages:
            return "You are part of a team taking turns to collaboratively addressing a task. It is now your turn. "

        context = "You are part of a team taking turns to collaboratively addressing a task. Here's the progress/history so far:\n\n"
        for msg in self.shared_messages:
            context += f"{msg}\n"
        context += "\nIt is now your turn."
        return context

    async def update_shared_state(self, result: AgentResponse) -> None:
        """Add new messages to shared conversation."""
        # Skip the context message we sent, add agent's actual responses
        new_messages = result.messages[1:] if len(result.messages) > 1 else []
        self.shared_messages.extend(new_messages)

    def get_agent_capabilities_summary(self) -> str:
        """
        Construct description from agent.name, agent.description, and available tools.

        Cached for performance since agent capabilities don't change during orchestration.
        """
        if self.agent_capabilities_cache is None:
            summary_lines = []
            for agent in self.agents:
                line = f"• {agent.name}: {agent.description}"

                # Add tool information if available
                if hasattr(agent, "tools") and agent.tools:
                    tool_names = []
                    for tool in agent.tools:
                        if hasattr(tool, "name"):
                            tool_names.append(tool.name)
                        elif hasattr(tool, "__name__"):
                            tool_names.append(tool.__name__)
                        else:
                            tool_names.append(
                                str(tool)[:20]
                            )  # Truncate long tool representations

                    if tool_names:
                        line += f" | Tools: {', '.join(tool_names)}"

                summary_lines.append(line)

            self.agent_capabilities_cache = "\n".join(summary_lines)

        return self.agent_capabilities_cache

    def _get_pattern_metadata(self) -> Dict[str, Any]:
        """Get AI selector specific metadata."""
        base_metadata = super()._get_pattern_metadata()

        # Add AI selector specific metrics
        unique_agents = set(sel["selected_agent"] for sel in self.selection_history)
        recent_selections = (
            self.selection_history[-5:] if self.selection_history else []
        )
        avg_confidence = (
            sum(sel["confidence"] for sel in self.selection_history)
            / len(self.selection_history)
            if self.selection_history
            else 0.0
        )

        base_metadata.update(
            {
                "selection_history": [
                    {
                        "agent": sel["selected_agent"],
                        "iteration": sel["iteration"],
                        "confidence": sel["confidence"],
                    }
                    for sel in self.selection_history
                ],
                "unique_agents_selected": len(unique_agents),
                "agent_diversity": len(unique_agents) / len(self.agents)
                if self.agents
                else 0.0,
                "average_confidence": round(avg_confidence, 3),
                "recent_reasoning": [sel["reasoning"] for sel in recent_selections],
                "model_used": self.model_client.model,
            }
        )
        return base_metadata

    def _reset_for_run(self) -> None:
        """Reset AI selector state."""
        super()._reset_for_run()
        self.selection_history = []
        self.agent_capabilities_cache = None  # Clear cache for fresh run

    # Helper methods
    def _find_agent_by_name(self, name: str) -> BaseAgent:
        """Find agent by name with fuzzy matching."""
        name_lower = name.lower().strip()

        # Exact match first
        for agent in self.agents:
            if agent.name.lower() == name_lower:
                return agent

        # Partial match fallback
        for agent in self.agents:
            if name_lower in agent.name.lower() or agent.name.lower() in name_lower:
                return agent

        # No match found - return first agent as fallback
        print(
            f"Warning: Agent '{name}' not found, using fallback: {self.agents[0].name}"
        )
        return self.agents[0]

    def _format_conversation_for_selection(self) -> str:
        """Format recent conversation for selection context."""

        if not self.shared_messages:
            return "No conversation yet."

        context = "History so far:\n\n"
        for msg in self.shared_messages:
            context += f"{msg}\n"

        return context

    def _extract_agent_name_from_text(self, text: str) -> str:
        """Extract agent name from text response as fallback."""
        text_lower = text.lower()

        # Look for agent names in the response
        for agent in self.agents:
            if agent.name.lower() in text_lower:
                return agent.name

        # Ultimate fallback
        return self.agents[0].name

    def _get_fallback_agent_name(self) -> str:
        """Get fallback agent name when selection fails."""
        # Simple round-robin fallback
        if self.selection_history:
            last_agent_name = self.selection_history[-1]["selected_agent"]
            agent_names = [a.name for a in self.agents]
            try:
                last_index = agent_names.index(last_agent_name)
                return agent_names[(last_index + 1) % len(agent_names)]
            except ValueError:
                pass

        return self.agents[0].name

    def _to_config(self) -> AIOrchestratorConfig:
        """Convert to configuration for serialization."""
        # Serialize agents
        agent_configs = []
        for agent in self.agents:
            try:
                agent_configs.append(agent.dump_component())
            except NotImplementedError:
                # Skip agents that don't support serialization
                continue

        # Serialize termination condition
        termination_config = self.termination.dump_component()

        # Serialize model client
        model_client_config = self.model_client.dump_component()

        return AIOrchestratorConfig(
            agents=agent_configs,
            termination=termination_config,
            model_client=model_client_config,
            max_iterations=self.max_iterations,
        )

    @classmethod
    def _from_config(cls, config: AIOrchestratorConfig) -> "AIOrchestrator":
        """Create from configuration."""
        from ..agents import BaseAgent

        # Deserialize agents
        agents = []
        for agent_config in config.agents:
            try:
                agent = BaseAgent.load_component(agent_config)
                agents.append(agent)
            except Exception:
                # Skip agents that fail to deserialize
                continue

        # Deserialize termination condition
        termination = BaseTermination.load_component(config.termination)

        # Deserialize model client
        model_client = BaseChatCompletionClient.load_component(config.model_client)

        return cls(
            agents=agents,
            termination=termination,
            model_client=model_client,
            max_iterations=config.max_iterations,
        )
