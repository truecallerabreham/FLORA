"""
Evaluation targets - what we run tasks against.

This module provides concrete Target implementations for running tasks
against different systems: Forla, direct model calls, orchestrators,
Claude Code SDK, and arbitrary callables.
"""

import time
from typing import Any, Dict, List, Optional

from .._cancellation_token import CancellationToken
from ..agents import BaseAgent
from ..llm import BaseChatCompletionClient
from ..messages import SystemMessage, UserMessage
from ..orchestration import BaseOrchestrator
from ..types import EvalScore, RunTrajectory, Task, Usage
from ._base import Target
from ._config import AgentConfig


class AgentEvalTarget(Target):
    """Target that wraps a forla Agent.

    Safe for concurrent use: Agent.run() and run_stream() use local
    working_context variables internally, so parallel task execution
    (parallel_tasks=True) does not cause shared-state races.
    """

    def __init__(self, agent: BaseAgent, name: Optional[str] = None):
        super().__init__(name or getattr(agent, "name", "Agent"))
        self.agent = agent

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        start_time = time.time()

        try:
            response = await self.agent.run(
                task.input, cancellation_token=cancellation_token
            )

            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=response.messages,
                success=True,
                error=None,
                usage=response.usage,
                metadata={
                    "target_type": "agent",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )

        except Exception as e:
            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=int((end_time - start_time) * 1000),
                    llm_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                ),
                metadata={
                    "target_type": "agent",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )


class ModelEvalTarget(Target):
    """Target for direct LLM model calls."""

    def __init__(
        self,
        client: BaseChatCompletionClient,
        system_message: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or getattr(client, "model", "Model"))
        self.client = client
        self.system_message = system_message

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        start_time = time.time()

        try:
            messages = []
            if self.system_message:
                messages.append(
                    SystemMessage(content=self.system_message, source="system")
                )
            messages.append(UserMessage(content=task.input, source="user"))

            result = await self.client.create(messages)

            end_time = time.time()

            response_messages = messages + [result.message]

            return RunTrajectory(
                task=task,
                messages=response_messages,
                success=True,
                error=None,
                usage=result.usage,
                metadata={
                    "target_type": "model",
                    "target_name": self.name,
                    "model": result.model,
                    "finish_reason": result.finish_reason,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )

        except Exception as e:
            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=int((end_time - start_time) * 1000),
                    llm_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                ),
                metadata={
                    "target_type": "model",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )


class OrchestratorEvalTarget(Target):
    """Target for forla orchestrators."""

    def __init__(self, orchestrator: BaseOrchestrator, name: Optional[str] = None):
        super().__init__(name or f"{orchestrator.__class__.__name__}")
        self.orchestrator = orchestrator

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        start_time = time.time()

        try:
            response = await self.orchestrator.run(
                task.input, cancellation_token=cancellation_token
            )

            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=response.messages,
                success=True,
                error=None,
                usage=response.usage,
                metadata={
                    "target_type": "orchestrator",
                    "target_name": self.name,
                    "pattern": response.pattern_metadata.get("pattern", "unknown"),
                    "iterations": response.pattern_metadata.get(
                        "iterations_completed", 0
                    ),
                    "stop_reason": response.stop_message.source,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )

        except Exception as e:
            end_time = time.time()

            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(
                    duration_ms=int((end_time - start_time) * 1000),
                    llm_calls=0,
                    tokens_input=0,
                    tokens_output=0,
                ),
                metadata={
                    "target_type": "orchestrator",
                    "target_name": self.name,
                    "execution_time_ms": int((end_time - start_time) * 1000),
                },
            )


class ForlaAgentTarget(Target):
    """Target that creates an agent from an AgentConfig and runs tasks.

    Uses run_stream to capture the full message and event trace.
    """

    def __init__(
        self,
        config: AgentConfig,
        middlewares: Optional[List] = None,
    ):
        super().__init__(config.name)
        self.config = config
        self.middlewares = middlewares or []

    def _get_agent(self, extra_middlewares: Optional[List] = None):
        """Create agent with combined middleware."""
        all_middlewares = self.middlewares + (extra_middlewares or [])
        return self.config.to_agent(middlewares=all_middlewares)

    async def run(
        self,
        task: Task,
        cancellation_token: Optional[CancellationToken] = None,
        *,
        middlewares: Optional[List] = None,
    ) -> RunTrajectory:
        """Execute task with Forla using run_stream to capture full trace.

        Args:
            task: Task to run
            cancellation_token: Optional cancellation
            middlewares: Additional middleware (e.g., RunMiddleware injected by runner)

        Returns:
            RunTrajectory with execution data including full message/event history
        """
        from ..messages import AssistantMessage, Message, ToolMessage
        from ..types import AgentEvent, AgentResponse

        agent = self._get_agent(middlewares)

        try:
            all_messages: List[Any] = []
            all_events: List[Any] = []
            response = None
            output = ""

            async for item in agent.run_stream(
                task.input,
                cancellation_token=cancellation_token,
                verbose=True,
            ):
                if isinstance(item, AgentResponse):
                    response = item
                elif isinstance(item, Message):
                    all_messages.append(item)
                    if isinstance(item, AssistantMessage) and item.content:
                        output = item.content
                elif isinstance(item, AgentEvent):
                    all_events.append(item)

            if response is None:
                return RunTrajectory(
                    task=task,
                    messages=all_messages,
                    success=False,
                    error="No response from agent",
                    usage=Usage(duration_ms=0, llm_calls=0, tokens_input=0, tokens_output=0),
                    metadata={"exception_type": "NoResponse", "events": all_events},
                )

            # Get messages from context if available (more complete)
            context_messages = list(response.context.messages) if response.context else []

            # Build events metadata
            metadata: Dict[str, Any] = {
                "finish_reason": response.finish_reason,
                "tool_calls": response.usage.tool_calls,
            }
            if all_events:
                metadata["events"] = [
                    {
                        "type": type(e).__name__,
                        "source": getattr(e, "source", None),
                        **{k: v for k, v in vars(e).items() if k != "source" and not k.startswith("_")}
                    }
                    for e in all_events
                ]
                metadata["event_count"] = len(all_events)

            return RunTrajectory(
                task=task,
                messages=context_messages if context_messages else all_messages,
                success=response.finish_reason == "stop",
                error=None if response.finish_reason == "stop" else response.finish_reason,
                usage=Usage(
                    duration_ms=response.usage.duration_ms,
                    llm_calls=response.usage.llm_calls,
                    tokens_input=response.usage.tokens_input,
                    tokens_output=response.usage.tokens_output,
                    tool_calls=response.usage.tool_calls,
                ),
                metadata=metadata,
            )

        except Exception as e:
            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(duration_ms=0, llm_calls=0, tokens_input=0, tokens_output=0),
                metadata={"exception_type": type(e).__name__},
            )


class ClaudeCodeTarget(Target):
    """Target that runs tasks with Claude Code SDK.

    Captures the full tool trace (ToolUseBlock, ToolResultBlock) so that
    evaluation can inspect file access patterns, tool call counts, and
    redundancy — not just the final text output.

    Requires ``claude-code-sdk`` package: ``pip install claude-code-sdk``
    """

    def __init__(
        self,
        name: str = "claude_code",
        max_turns: int = 30,
        allowed_tools: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        permission_mode: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(name)
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools or ["Read", "Bash", "Glob", "Grep"]
        self.cwd = cwd
        self.permission_mode = permission_mode
        self.model = model

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        import os
        if os.environ.get("CLAUDECODE"):
            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=(
                    "Cannot run ClaudeCodeTarget inside a Claude Code session "
                    "(CLAUDECODE env var is set). Run from Jupyter or a plain terminal."
                ),
                usage=Usage(duration_ms=0, llm_calls=0, tokens_input=0, tokens_output=0),
            )

        try:
            from claude_code_sdk import (
                AssistantMessage as CCAssistantMessage,
                ClaudeCodeOptions,
                ResultMessage,
                TextBlock,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage as CCUserMessage,
                query,
            )
        except ImportError:
            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error="claude-code-sdk not installed. Install with: pip install claude-code-sdk",
                usage=Usage(duration_ms=0, llm_calls=0, tokens_input=0, tokens_output=0),
            )

        from ..messages import AssistantMessage, ToolMessage, UserMessage
        from ..messages import ToolCallRequest

        options = ClaudeCodeOptions(
            allowed_tools=self.allowed_tools,
            max_turns=self.max_turns,
        )
        if self.cwd:
            options.cwd = self.cwd
        if self.permission_mode:
            options.permission_mode = self.permission_mode  # type: ignore[assignment]
        if self.model:
            options.model = self.model

        # Collect full message trace
        all_messages: list = [
            UserMessage(content=task.input, source="user"),
        ]
        # Map tool_use_id -> tool_name for resolving names in results
        tool_use_id_to_name: Dict[str, str] = {}
        iterations = 0
        tool_call_count = 0
        input_tokens = 0
        output_tokens = 0
        duration_ms = 0
        total_cost_usd: Optional[float] = None
        usage_breakdown: Dict[str, Any] = {}
        success = False
        error = None

        try:
            async for message in query(prompt=task.input, options=options):
                if isinstance(message, CCAssistantMessage):
                    iterations += 1

                    # AssistantMessage contains TextBlock and ToolUseBlock
                    text_parts: list[str] = []
                    tool_calls: list = []

                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(ToolCallRequest(
                                tool_name=block.name,
                                parameters=block.input,
                                call_id=block.id,
                            ))
                            tool_use_id_to_name[block.id] = block.name
                            tool_call_count += 1

                    asst_content = "\n".join(text_parts) if text_parts else ""
                    all_messages.append(AssistantMessage(
                        content=asst_content,
                        source="assistant",
                        tool_calls=tool_calls if tool_calls else None,
                    ))

                elif isinstance(message, CCUserMessage):
                    # UserMessage contains ToolResultBlocks (tool execution results)
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                content = ""
                                if isinstance(block.content, str):
                                    content = block.content
                                elif isinstance(block.content, list):
                                    parts = []
                                    for item in block.content:
                                        if isinstance(item, dict):
                                            parts.append(
                                                item.get("text", str(item))
                                            )
                                        else:
                                            parts.append(str(item))
                                    content = "\n".join(parts)

                                tool_name = tool_use_id_to_name.get(
                                    block.tool_use_id, ""
                                )
                                all_messages.append(ToolMessage(
                                    content=content,
                                    source=tool_name or block.tool_use_id,
                                    tool_call_id=block.tool_use_id,
                                    tool_name=tool_name,
                                    success=not (block.is_error or False),
                                ))

                elif isinstance(message, ResultMessage):
                    success = not message.is_error
                    duration_ms = message.duration_ms
                    total_cost_usd = message.total_cost_usd
                    if message.usage:
                        # SDK reports cache tokens separately
                        input_tokens = (
                            message.usage.get("input_tokens", 0)
                            + message.usage.get("cache_creation_input_tokens", 0)
                            + message.usage.get("cache_read_input_tokens", 0)
                        )
                        output_tokens = message.usage.get("output_tokens", 0)
                        # Preserve the full breakdown for cost analysis
                        usage_breakdown = {
                            k: v for k, v in message.usage.items()
                            if isinstance(v, (int, float))
                        }
                    if message.is_error:
                        error = message.result or "Claude Code returned error"

        except Exception as e:
            error = str(e)

        metadata: Dict[str, Any] = {
            "target_type": "claude_code",
            "target_name": self.name,
        }
        if total_cost_usd is not None:
            metadata["total_cost_usd"] = total_cost_usd
        if usage_breakdown:
            metadata["usage_breakdown"] = usage_breakdown

        return RunTrajectory(
            task=task,
            messages=all_messages,
            success=success,
            error=error,
            usage=Usage(
                duration_ms=duration_ms,
                llm_calls=iterations,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                tool_calls=tool_call_count,
            ),
            metadata=metadata,
        )


class CopilotTarget(Target):
    """Target that runs tasks with GitHub Copilot SDK.

    Captures the full event trace (tool calls, token usage, assistant messages)
    so that evaluation can inspect file access patterns, tool call counts, and
    redundancy.

    Requires ``github-copilot-sdk`` package (Python 3.11+):
    ``pip install github-copilot-sdk``
    """

    def __init__(
        self,
        name: str = "copilot",
        model: Optional[str] = None,
        available_tools: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        timeout: float = 600.0,
        cli_path: Optional[str] = None,
    ):
        super().__init__(name)
        self.model = model
        self.available_tools = available_tools
        self.cwd = cwd
        self.timeout = timeout
        self.cli_path = cli_path

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        try:
            from copilot import CopilotClient, PermissionHandler
            from copilot.generated.session_events import SessionEventType
        except ImportError:
            return RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=(
                    "github-copilot-sdk not installed (requires Python 3.11+). "
                    "Install with: pip install github-copilot-sdk"
                ),
                usage=Usage(
                    duration_ms=0, llm_calls=0,
                    tokens_input=0, tokens_output=0,
                ),
            )

        import asyncio

        from ..messages import AssistantMessage, ToolMessage, UserMessage
        from ..messages import ToolCallRequest

        # Build session config
        config: Dict[str, Any] = {
            "on_permission_request": PermissionHandler.approve_all,
        }
        if self.model:
            config["model"] = self.model
        if self.available_tools:
            config["available_tools"] = self.available_tools
        if self.cwd:
            config["working_directory"] = self.cwd

        # Collect events
        all_events: list = []
        idle_event = asyncio.Event()
        error_msg: Optional[str] = None

        def on_event(event: Any) -> None:
            nonlocal error_msg
            all_events.append(event)
            if event.type == SessionEventType.SESSION_IDLE:
                idle_event.set()
            elif event.type == SessionEventType.SESSION_ERROR:
                error_msg = getattr(event.data, "content", None) or "Session error"
                idle_event.set()
            elif event.type == SessionEventType.SESSION_SHUTDOWN:
                idle_event.set()

        client_opts: Dict[str, Any] = {}
        if self.cli_path:
            client_opts["cli_path"] = self.cli_path
        client = CopilotClient(client_opts if client_opts else None)
        success = False
        all_messages: list = [
            UserMessage(content=task.input, source="user"),
        ]
        tool_call_id_to_name: Dict[str, str] = {}
        iterations = 0
        tool_call_count = 0
        input_tokens = 0
        output_tokens = 0
        duration_ms = 0
        total_cost: Optional[float] = None

        try:
            await client.start()
            session = await client.create_session(config)
            session.on(on_event)

            await session.send({"prompt": task.input})
            await asyncio.wait_for(
                idle_event.wait(), timeout=self.timeout
            )

            # Process collected events into forla messages
            for event in all_events:
                etype = event.type

                if etype == SessionEventType.ASSISTANT_MESSAGE:
                    iterations += 1
                    content = getattr(event.data, "content", "") or ""

                    # Extract tool requests from this message
                    tool_calls: list = []
                    tool_requests = getattr(
                        event.data, "tool_requests", None
                    )
                    if tool_requests:
                        for tr in tool_requests:
                            tool_calls.append(ToolCallRequest(
                                tool_name=tr.name,
                                parameters=(
                                    tr.arguments
                                    if isinstance(tr.arguments, dict)
                                    else {}
                                ),
                                call_id=tr.tool_call_id,
                            ))
                            tool_call_id_to_name[tr.tool_call_id] = tr.name
                            tool_call_count += 1

                    all_messages.append(AssistantMessage(
                        content=content,
                        source="assistant",
                        tool_calls=tool_calls if tool_calls else None,
                    ))

                elif etype == SessionEventType.TOOL_EXECUTION_COMPLETE:
                    tool_name = (
                        getattr(event.data, "tool_name", "")
                        or tool_call_id_to_name.get(
                            getattr(event.data, "tool_call_id", ""), ""
                        )
                    )
                    tool_call_id = (
                        getattr(event.data, "tool_call_id", "") or ""
                    )
                    result = getattr(event.data, "result", None)
                    content = ""
                    if result:
                        content = getattr(result, "content", "") or ""

                    all_messages.append(ToolMessage(
                        content=content,
                        source=tool_name or tool_call_id,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        success=True,
                    ))

                elif etype == SessionEventType.ASSISTANT_USAGE:
                    d = event.data
                    input_tokens += int(
                        getattr(d, "input_tokens", 0) or 0
                    )
                    input_tokens += int(
                        getattr(d, "cache_read_tokens", 0) or 0
                    )
                    input_tokens += int(
                        getattr(d, "cache_write_tokens", 0) or 0
                    )
                    output_tokens += int(
                        getattr(d, "output_tokens", 0) or 0
                    )
                    cost_val = getattr(d, "cost", None)
                    if cost_val is not None:
                        total_cost = (total_cost or 0) + float(cost_val)
                    dur_val = getattr(d, "duration", None)
                    if dur_val is not None:
                        duration_ms += int(dur_val)

                elif etype == SessionEventType.SESSION_SHUTDOWN:
                    # Use aggregated metrics from shutdown if available
                    metrics = getattr(event.data, "model_metrics", None)
                    if metrics and isinstance(metrics, dict):
                        agg_in = 0
                        agg_out = 0
                        agg_cost = 0.0
                        for mm in metrics.values():
                            u = getattr(mm, "usage", None)
                            if u:
                                agg_in += int(
                                    getattr(u, "input_tokens", 0) or 0
                                )
                                agg_in += int(
                                    getattr(u, "cache_read_tokens", 0) or 0
                                )
                                agg_in += int(
                                    getattr(u, "cache_write_tokens", 0) or 0
                                )
                                agg_out += int(
                                    getattr(u, "output_tokens", 0) or 0
                                )
                            r = getattr(mm, "requests", None)
                            if r:
                                agg_cost += float(
                                    getattr(r, "cost", 0) or 0
                                )
                        # Prefer shutdown aggregates over per-call sums
                        if agg_in > 0:
                            input_tokens = agg_in
                        if agg_out > 0:
                            output_tokens = agg_out
                        if agg_cost > 0:
                            total_cost = agg_cost

                    api_dur = getattr(
                        event.data, "total_api_duration_ms", None
                    )
                    if api_dur:
                        duration_ms = int(api_dur)

            success = error_msg is None

            await session.destroy()

        except asyncio.TimeoutError:
            error_msg = f"Timed out after {self.timeout}s"
        except Exception as e:
            error_msg = str(e)
        finally:
            try:
                await client.stop()
            except Exception:
                pass

        metadata: Dict[str, Any] = {
            "target_type": "copilot",
            "target_name": self.name,
            "event_count": len(all_events),
        }
        if total_cost is not None:
            # Note: Copilot SDK reports cost in premium request
            # units, not USD. Store raw value for reference.
            metadata["total_cost_raw"] = total_cost

        return RunTrajectory(
            task=task,
            messages=all_messages,
            success=success,
            error=error_msg,
            usage=Usage(
                duration_ms=duration_ms,
                llm_calls=iterations,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                tool_calls=tool_call_count,
            ),
            metadata=metadata,
        )


class CallableTarget(Target):
    """Wrap any async callable as a target.

    Useful for custom agent implementations or quick testing.
    The callable receives a Task and returns a RunTrajectory.
    """

    def __init__(self, name: str, func):
        super().__init__(name)
        self.func = func

    async def run(
        self, task: Task, cancellation_token: Optional[CancellationToken] = None
    ) -> RunTrajectory:
        """Execute the wrapped callable."""
        return await self.func(task)
