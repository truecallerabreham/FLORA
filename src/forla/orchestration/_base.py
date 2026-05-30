from __future__ import annotations
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Sequence, Union

from ..agents._base import BaseAgent
from ..messages import Message, UserMessage, AssistantMessage
from ..termination._base import BaseTermination
from ..types import CancellationToken, Usage
from ..messages import StopMessage


@dataclass
class OrchestrationStartEvent:
    """Emitted when an orchestration run begins."""
    agent_names: List[str]
    task_preview: str

@dataclass
class AgentTurnStartEvent:
    """Emitted when an agent's turn begins."""
    agent_name: str
    iteration: int

@dataclass
class AgentTurnCompleteEvent:
    """Emitted when an agent's turn completes."""
    agent_name: str
    iteration: int
    message_count: int


class OrchestrationResponse:
    """The complete result of an orchestration run."""
    
    def __init__(
        self,
        messages: List[Message],
        stop_message: StopMessage,
        usage: Usage,
    ):
        self.messages = messages
        self.stop_message = stop_message
        self.usage = usage
        
        # Extract the final result: the last assistant message with content
        self.final_result = ""
        for msg in reversed(messages):
            if isinstance(msg, AssistantMessage) and msg.content:
                self.final_result = msg.content
                break

    def __str__(self) -> str:
        return (
            f"OrchestrationResponse(\n"
            f"  final_result='{self.final_result[:100]}...'\n"
            f"  stop_reason='{self.stop_message.content}'\n"
            f"  usage={self.usage}\n"
            f")"
        )


class BaseOrchestrator(ABC):
    """The base class for all multi-agent coordination patterns.
    
    WHAT IT PROVIDES (so subclasses don't have to):
    - The complete orchestration loop (run_stream, run)
    - Cancellation token support
    - Delta-based termination checking (efficient)
    - Usage aggregation across all agents
    - Streaming of messages and events
    - Error handling and graceful shutdown
    
    WHAT SUBCLASSES IMPLEMENT (the pattern-specific logic):
    - select_next_agent(): Who speaks next?
    - prepare_context_for_agent(): What does that agent see?
    - update_shared_state(): How does the response update shared state?
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        termination: BaseTermination,
        max_iterations: int = 50,
    ):
        if not agents:
            raise ValueError("At least one agent is required for orchestration")
        
        # Validate that agent names are unique
        names = [a.name for a in agents]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Agent names must be unique. Got: {names}. "
                f"Duplicates: {[n for n in names if names.count(n) > 1]}"
            )
        
        self.agents = list(agents)
        self.termination = termination
        self.max_iterations = max_iterations
        
        # Runtime state — reset before each run
        self.shared_messages: List[Message] = []
        self.iteration_count: int = 0
        self._agent_usage: Dict[str, Usage] = {}

    def _reset_for_run(self) -> None:
        """Clear all state before starting a new orchestration run."""
        self.shared_messages = []
        self.iteration_count = 0
        self._agent_usage = {}
        self.termination.reset()

    def _normalize_task(self, task) -> List[Message]:
        """Convert any task format to a list of Messages."""
        if isinstance(task, str):
            return [UserMessage(content=task, source="user")]
        elif isinstance(task, UserMessage):
            return [task]
        elif isinstance(task, list):
            return task
        return [UserMessage(content=str(task), source="user")]

    @abstractmethod
    async def select_next_agent(self) -> BaseAgent:
        """Choose which agent runs next. Pattern-specific logic."""
        pass

    @abstractmethod
    async def prepare_context_for_agent(self, agent: BaseAgent) -> List[Message]:
        """Build the message list this agent will receive. Pattern-specific."""
        pass

    @abstractmethod
    async def update_shared_state(
        self, agent: BaseAgent, response: "AgentResponse"
    ) -> None:
        """Update shared state after an agent's turn. Pattern-specific."""
        pass

    async def run_stream(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator:
        """The universal orchestration loop. All patterns share this."""
        self._reset_for_run()
        start_time = time.time()
        stop_message: Optional[StopMessage] = None
        streamed_messages: List[Message] = []

        # Emit start event
        yield OrchestrationStartEvent(
            agent_names=[a.name for a in self.agents],
            task_preview=str(task)[:100],
        )

        # ── Initialize with the task ──────────────────────────────────────
        initial_messages = self._normalize_task(task)
        self.shared_messages.extend(initial_messages)

        for msg in initial_messages:
            yield msg
            streamed_messages.append(msg)

        # Check termination on initial messages (e.g., task already contains "DONE")
        stop_message = self.termination.check(initial_messages)

        # ── Main orchestration loop ───────────────────────────────────────
        while self.iteration_count < self.max_iterations and not stop_message:

            # Cancel check
            if cancellation_token and cancellation_token.is_cancelled():
                stop_message = StopMessage(
                    content="Orchestration cancelled by user",
                    source="CancellationToken",
                )
                break

            # 1. Select the next agent (pattern-specific)
            next_agent = await self.select_next_agent()

            yield AgentTurnStartEvent(
                agent_name=next_agent.name,
                iteration=self.iteration_count,
            )

            # 2. Prepare context for this agent (pattern-specific)
            context_messages = await self.prepare_context_for_agent(next_agent)

            # 3. Execute the agent with the prepared context
            from ..types import AgentResponse
            agent_new_messages = []
            agent_response: Optional[AgentResponse] = None

            try:
                async for item in next_agent.run_stream(context_messages, cancellation_token):
                    if isinstance(item, AgentResponse):
                        agent_response = item
                    elif isinstance(item, Message):
                        # Only stream non-UserMessage messages (avoid echoing the context)
                        if not isinstance(item, UserMessage):
                            agent_new_messages.append(item)
                            streamed_messages.append(item)
                            yield item
            except asyncio.CancelledError:
                stop_message = StopMessage(
                    content="Agent execution cancelled",
                    source="CancellationToken",
                )
                break
            except Exception as e:
                stop_message = StopMessage(
                    content=f"Agent '{next_agent.name}' raised an error: {e}",
                    source="AgentError",
                )
                break

            if agent_response is None:
                break

            # 4. Update shared state (pattern-specific)
            await self.update_shared_state(next_agent, agent_response)

            # Track usage per agent
            if next_agent.name not in self._agent_usage:
                self._agent_usage[next_agent.name] = Usage()
            self._agent_usage[next_agent.name] = (
                self._agent_usage[next_agent.name] + agent_response.usage
            )

            yield AgentTurnCompleteEvent(
                agent_name=next_agent.name,
                iteration=self.iteration_count,
                message_count=len(agent_new_messages),
            )

            # 5. Check termination on the delta (only new messages since last check)
            # This is more efficient than checking the full history each time
            checkpoint = len(streamed_messages) - len(agent_new_messages)
            delta_messages = streamed_messages[checkpoint:]
            stop_message = self.termination.check(delta_messages)

            self.iteration_count += 1

        # ── Handle reaching max_iterations ──────────────────────────────
        if not stop_message:
            stop_message = StopMessage(
                content=f"Maximum iterations reached ({self.max_iterations})",
                source="MaxIterations",
            )

        # ── Aggregate total usage ────────────────────────────────────────
        total_usage = Usage(
            duration_ms=int((time.time() - start_time) * 1000)
        )
        for agent_usage in self._agent_usage.values():
            total_usage = total_usage + agent_usage

        yield OrchestrationResponse(
            messages=self.shared_messages,
            stop_message=stop_message,
            usage=total_usage,
        )

    async def run(
        self,
        task,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> OrchestrationResponse:
        """Run orchestration and return only the final result."""
        result = None
        async for item in self.run_stream(task, cancellation_token):
            if isinstance(item, OrchestrationResponse):
                result = item
        return result
