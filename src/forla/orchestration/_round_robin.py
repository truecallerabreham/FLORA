from __future__ import annotations
from typing import List, Sequence
from ._base import BaseOrchestrator
from ..agents._base import BaseAgent
from ..messages import Message
from ..termination._base import BaseTermination
from ..types import AgentResponse


class RoundRobinOrchestrator(BaseOrchestrator):
    """Fixed sequential turn-taking between agents.
    
    HOW IT WORKS:
    If you have agents [poet, critic, editor], the order is:
    poet → critic → editor → poet → critic → editor → (repeat until termination)
    
    WHEN TO USE:
    - When all agents are equal partners (writer and reviewer)
    - When you want predictable, deterministic turn order
    - When prototyping a multi-agent system for the first time
    - When the book's "poet and critic" example applies to your use case
    
    WHEN NOT TO USE:
    - When some agents should speak more than others based on task needs
    - When context determines who is most relevant next
    
    IMPLEMENTATION:
    This class only needs to implement the three abstract methods.
    All the complex infrastructure (streaming, cancellation, error handling,
    usage tracking, termination checking) is inherited from BaseOrchestrator.
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        termination: BaseTermination,
        max_iterations: int = 50,
    ):
        super().__init__(agents, termination, max_iterations)
        self._current_index: int = 0   # Tracks which agent's turn it is

    async def select_next_agent(self) -> BaseAgent:
        """Return the next agent in the round-robin sequence."""
        agent = self.agents[self._current_index]
        # Advance to the next agent, wrapping around at the end
        self._current_index = (self._current_index + 1) % len(self.agents)
        return agent

    async def prepare_context_for_agent(self, agent: BaseAgent) -> List[Message]:
        """Give the agent the full shared conversation history.
        
        In round-robin, every agent sees everything — the complete conversation
        history including all previous agents' contributions.
        
        NOTE: The book points out (Section 7.3.1) that in practice you may want
        to customize this — e.g., limit to the last N messages for token efficiency,
        or format as JSON vs plain text. This is a deliberate starting point.
        """
        return list(self.shared_messages)

    async def update_shared_state(
        self, agent: BaseAgent, response: "AgentResponse"
    ) -> None:
        """Add any new messages from this agent to the shared conversation."""
        # The agent's context has all messages — find the ones we don't have yet
        for msg in agent.context.get_messages():
            if msg not in self.shared_messages:
                self.shared_messages.append(msg)

    def _reset_for_run(self) -> None:
        super()._reset_for_run()
        self._current_index = 0    # Reset the turn order too
