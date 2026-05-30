from __future__ import annotations
from typing import List, Optional, Sequence
from pydantic import BaseModel, Field
from ._base import BaseOrchestrator
from ..agents._base import BaseAgent
from ..llm._base import BaseChatCompletionClient
from ..messages import Message, UserMessage, SystemMessage, AssistantMessage
from ..termination._base import BaseTermination
from ..types import AgentResponse


class AgentSelection(BaseModel):
    """Structured output for AI-driven agent selection.
    
    WHY Pydantic? Because we need a guaranteed boolean decision
    and a clear agent name from the LLM's response.
    Free-form text like "I think the researcher should go next"
    would be unreliable to parse.
    """
    selected_agent: str = Field(description="Name of the agent who should speak next")
    reasoning: str = Field(description="Brief explanation of why this agent was selected")


class AIOrchestrator(BaseOrchestrator):
    """An LLM decides which agent speaks next.
    
    HOW IT WORKS:
    Before each turn, we send the conversation history + agent descriptions
    to a selector LLM. It returns structured output (AgentSelection) naming
    which agent should go next and why.
    
    WHEN TO USE:
    - When tasks require context-aware routing
    - When some questions belong to one specialist and others to another
    - When the conversation needs to adapt dynamically
    
    COST WARNING: This adds one extra LLM call per iteration just for
    agent selection. This can significantly increase costs.
    Use RoundRobinOrchestrator for simple use cases.
    
    IMPLEMENTATION NOTE (from the book Section 7.4):
    "The key insight is that we can reuse all the infrastructure from
    round-robin orchestration and only change the select_next_agent() method."
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        termination: BaseTermination,
        selector_model_client: BaseChatCompletionClient,
        max_iterations: int = 50,
    ):
        super().__init__(agents, termination, max_iterations)
        self._selector = selector_model_client
        self._last_selected_name: Optional[str] = None

    async def select_next_agent(self) -> BaseAgent:
        """Ask an LLM which agent should speak next."""
        # Build agent capability descriptions
        agent_descriptions = "\n".join(
            f"- {a.name}: {a.description}"
            for a in self.agents
        )

        # Show the last 5 messages as recent conversation context
        # (Showing all would waste tokens and may confuse the selector)
        recent = self.shared_messages[-5:]
        context_lines = []
        for msg in recent:
            content = str(getattr(msg, "content", ""))[:300]
            source = getattr(msg, "source", "unknown")
            context_lines.append(f"{source}: {content}")
        context_str = "\n".join(context_lines)

        # Ask the selector LLM with structured output
        selection_messages = [
            SystemMessage(
                content=(
                    "You are coordinating a team of AI agents. "
                    "Choose the best agent for the next turn based on the conversation state."
                ),
                source="system",
            ),
            UserMessage(
                content=(
                    f"Available agents:\n{agent_descriptions}\n\n"
                    f"Recent conversation:\n{context_str}\n\n"
                    f"Which agent should speak next? "
                    f"Return JSON with selected_agent (exact name) and reasoning."
                ),
                source="user",
            ),
        ]

        try:
            result = await self._selector.create(
                messages=selection_messages,
                output_format=AgentSelection,
            )

            if result.structured_output:
                selected_name = result.structured_output.selected_agent
                agent = next(
                    (a for a in self.agents if a.name == selected_name), None
                )
                if agent:
                    self._last_selected_name = agent.name
                    return agent

        except Exception:
            pass    # Fall back to round-robin if selection fails

        # Fallback: simple round-robin
        if self._last_selected_name is None:
            return self.agents[0]
        last_idx = next(
            (i for i, a in enumerate(self.agents) if a.name == self._last_selected_name), 0
        )
        return self.agents[(last_idx + 1) % len(self.agents)]

    async def prepare_context_for_agent(self, agent: BaseAgent) -> List[Message]:
        """Same as round-robin — the full shared conversation."""
        return list(self.shared_messages)

    async def update_shared_state(self, agent: BaseAgent, response: "AgentResponse") -> None:
        """Same as round-robin — append new messages."""
        for msg in agent.context.get_messages():
            if msg not in self.shared_messages:
                self.shared_messages.append(msg)
