"""
Round-robin orchestration pattern.

This module provides the RoundRobinOrchestrator that cycles through agents 
in fixed order, giving each agent access to the complete shared conversation history.
"""

from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

from .._component_config import Component, ComponentModel
from ..agents import BaseAgent
from ..messages import Message, UserMessage
from ..termination import BaseTermination
from ..types import AgentResponse
from ._base import BaseOrchestrator


class RoundRobinOrchestratorConfig(BaseModel):
    """Configuration for RoundRobinOrchestrator serialization."""

    agents: List[ComponentModel] = Field(default_factory=list)
    termination: ComponentModel
    max_iterations: int = 50


class RoundRobinOrchestrator(Component[RoundRobinOrchestratorConfig], BaseOrchestrator):
    """
    Round-robin orchestration pattern.

    Cycles through agents in fixed order, giving each agent access to
    the complete shared conversation history.
    """

    component_config_schema = RoundRobinOrchestratorConfig
    component_type = "orchestrator"
    component_provider_override = "forla.orchestration.RoundRobinOrchestrator"

    def __init__(
        self,
        agents: List[BaseAgent],
        termination: BaseTermination,
        max_iterations: int = 50,
    ):
        super().__init__(agents, termination, max_iterations)
        self.current_agent_index = 0

    async def select_next_agent(self) -> BaseAgent:
        """Select next agent in round-robin order."""
        agent = self.agents[self.current_agent_index]
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)
        return agent

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
        new_messages = result.messages[1:] if len(result.messages) > 1 else []
        self.shared_messages.extend(new_messages)

    def _get_pattern_metadata(self) -> Dict[str, Any]:
        """Get round-robin specific metadata."""
        base_metadata = super()._get_pattern_metadata()
        base_metadata.update(
            {
                "cycles_completed": self.iteration_count // len(self.agents),
                "current_agent_index": self.current_agent_index,
                "agents_order": [agent.name for agent in self.agents],
            }
        )
        return base_metadata

    def _reset_for_run(self) -> None:
        """Reset round-robin state."""
        super()._reset_for_run()
        self.current_agent_index = 0

    def _to_config(self) -> RoundRobinOrchestratorConfig:
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

        return RoundRobinOrchestratorConfig(
            agents=agent_configs,
            termination=termination_config,
            max_iterations=self.max_iterations,
        )

    @classmethod
    def _from_config(
        cls, config: RoundRobinOrchestratorConfig
    ) -> "RoundRobinOrchestrator":
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

        return cls(
            agents=agents, termination=termination, max_iterations=config.max_iterations
        )
