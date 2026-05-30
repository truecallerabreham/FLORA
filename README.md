# Forla

Forla is a compact, async-first Python framework for building single-agent, multi-agent, and workflow-based AI systems from first principles.

It is intentionally small enough to understand, but complete enough to demonstrate the core architecture behind modern agent products: typed messages, model clients, tool execution, memory, middleware, orchestration, termination, workflows, evaluation, and a web UI surface.

## Why Forla

Most agent frameworks hide the loop. Forla exposes it.

An agent in Forla is the composition of three things:

```text
Agent = model reasoning + tool execution + memory
```

That simple contract makes it easier to inspect behavior, test components independently, and teach or extend the system without fighting a large abstraction stack.

## Highlights

- Async-first execution for LLM calls, tool calls, workflows, and orchestration.
- Streaming-first agents: `run_stream()` yields messages and events; `run()` is the convenience wrapper.
- Typed message protocol for system, user, assistant, tool, and stop messages.
- Tool abstraction with JSON-schema conversion for function calling.
- Agent-managed memory through a sandboxed file memory tool.
- Application-managed memory through pluggable `BaseMemory` implementations.
- Middleware hooks for logging, security checks, policy gates, and observability.
- Multi-agent orchestration with round-robin and AI-driven routing patterns.
- Deterministic workflow engine for graph-like business logic.
- Evaluation runner and LLM judge primitives.
- Example applications, including an autonomous software-engineering agent team.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For OpenAI-backed examples, set:

```bash
set OPENAI_API_KEY=your_api_key_here
```

On macOS or Linux:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Quick Start

```python
import asyncio
import os

from forla import Agent, OpenAIChatCompletionClient
from forla.tools import ThinkTool


async def main():
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = Agent(
        name="assistant",
        description="A concise technical assistant",
        instructions="Answer clearly. Use tools when useful.",
        model_client=client,
        tools=[ThinkTool()],
    )

    response = await agent.run("Explain async agents in one paragraph.")
    print(response.content)
    print(response.usage)


if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

| Concept | Purpose | Key classes |
|---|---|---|
| Messages | Shared protocol for all model, tool, and orchestration communication | `UserMessage`, `AssistantMessage`, `ToolMessage`, `StopMessage` |
| Model clients | Provider abstraction for LLM calls | `BaseChatCompletionClient`, `OpenAIChatCompletionClient` |
| Tools | Structured actions an agent can call | `BaseTool`, `FunctionTool`, `ThinkTool`, `MemoryTool` |
| Agent memory | Persistent agent-managed knowledge | `MemoryTool` |
| App memory | Developer-managed context injection | `BaseMemory`, `ListMemory` |
| Middleware | Intercept model and tool calls | `BaseMiddleware`, `LoggingMiddleware`, `SecurityMiddleware` |
| Orchestration | Coordinate multiple agents | `RoundRobinOrchestrator`, `AIOrchestrator` |
| Termination | Stop runs safely and predictably | `MaxMessageTermination`, `TextMentionTermination`, `TokenBudgetTermination` |
| Workflow | Deterministic graph execution | `Workflow`, `FunctionStep`, `WorkflowRunner` |
| Evaluation | Score agent outputs | `EvaluationRunner`, `LLMJudge` |

## Architecture

```text
User task
   |
   v
Agent.run_stream()
   |
   +-- prepare context
   |     +-- system instructions
   |     +-- application memory
   |     +-- conversation history
   |
   +-- call model through middleware
   |
   +-- if model requests tools
   |     +-- execute tools through middleware
   |     +-- append ToolMessage results
   |     +-- loop
   |
   +-- if model returns text
         +-- update memory
         +-- emit AgentResponse
```

## Project Layout

```text
src/forla/
  agents/          Agent loop and base abstractions
  llm/             Model client interfaces and OpenAI-compatible client
  tools/           Tool interfaces, function tools, memory, status, thinking
  memory/          Application-managed memory
  middleware/      Request/response interceptors
  orchestration/   Multi-agent coordination
  termination/     Stop conditions
  workflow/        Deterministic workflow graph engine
  eval/            Evaluation runner and judges
  webui/           FastAPI/SSE web UI
examples/          Runnable demos
tests/             Unit and integration tests
```

## Examples

| File | What it demonstrates |
|---|---|
| `examples/01_basic_agent.py` | A basic single agent with an OpenAI-compatible model client |
| `examples/02_poet_critic.py` | Multi-agent writer and critic collaboration |
| `examples/03_web_app.py` | FastAPI web app integration |
| `examples/04_full_system.py` | Workflow, orchestration, memory, and middleware together |
| `examples/05_autonomous_coding_agent.py` | A software-engineering agent team with tools, memory, planning, execution, and evaluation |

Run the coding-agent showcase:

```bash
python examples/05_autonomous_coding_agent.py
```

Or point it at a disposable workspace:

```bash
python examples/05_autonomous_coding_agent.py ^
  --workspace .\scratch\agent-workspace ^
  "Add input validation, update tests, run verification, and summarize the diff."
```

## Autonomous Coding Agent Showcase

The coding-agent example models the pattern used by modern AI coding tools: a model-driven agent loop combined with precise tools and persistent memory.

It includes:

- Sandboxed file operations: `tree`, `view`, `create`, and `str_replace`.
- Command execution with a conservative allowlist for tests and inspections.
- Persistent markdown memory under `/memories`.
- `ThinkTool` for explicit metacognition.
- `TaskStatusTool` for explicit completion checks.
- A five-phase engineering workflow:
  1. Memory check
  2. Planning
  3. Execution
  4. Learning
  5. Completion
- Markdown task tracking in `/memories/current_task.md`.
- Multi-agent roles: architect, implementer, and reviewer.
- Termination that requires an explicit `SHIP_READY` signal from review.

This is the example to show when you want to demonstrate that Forla can model real software-engineering agent behavior, not just toy chat loops.

## Positioning

Forla is not trying to replace every production agent platform. It is designed to be readable, hackable, and educational while still covering the essential runtime patterns.

| Project | Primary orientation | Best fit | Forla difference |
|---|---|---|---|
| Forla | Small async framework with transparent internals | Learning, prototypes, interviews, custom agent runtimes | Minimal codebase, explicit loop, easy to inspect and extend |
| [LangGraph](https://www.langchain.com/langgraph) | Low-level orchestration/runtime for stateful agents | Durable graph-based agent applications | Forla favors simpler primitives before adding graph/runtime complexity |
| [Microsoft AutoGen / Agent Framework](https://learn.microsoft.com/agent-framework/overview/agent-framework-overview) | Multi-agent applications and enterprise agent workflows | Conversational multi-agent systems and Microsoft ecosystem work | Forla is smaller and easier to reason about end to end |
| [CrewAI](https://docs.crewai.com/en/index) | Crews, flows, and multi-agent automation | Role-based automation and fast team-style agent setup | Forla exposes the lower-level building blocks behind those patterns |
| Direct model SDK | Raw LLM calls | Simple generation or tool-calling tasks | Forla adds memory, middleware, orchestration, termination, and workflows |

## Quality Bar

Run tests:

```bash
pytest tests -q
```

Compile-check the package and examples:

```bash
python -m compileall -q src tests examples
```

Current local verification:

```text
12 passed, 7 skipped
```

The skipped tests require `OPENAI_API_KEY`.

## Security Notes

Agents that can read files or run commands need boundaries.

Forla demonstrates the following controls:

- Tool interfaces return structured success/error results instead of crashing the agent loop.
- `MemoryTool` validates paths to prevent directory traversal.
- The coding-agent example keeps file operations inside a configured workspace.
- Command execution in the coding-agent example uses an allowlist and rejects shell metacharacters.
- `SecurityMiddleware` shows how model-call inputs can be blocked before reaching the LLM.

Treat the examples as engineering patterns, not permission to run agents against sensitive systems without review.

## Roadmap Ideas

- Pydantic v2 `ConfigDict` cleanup.
- More model providers.
- Human approval middleware.
- Richer streaming UI.
- Persistent workflow checkpoints.
- MCP tool integration.
- First-class tracing and OpenTelemetry examples.

## Contributing

Keep contributions small, typed, and easy to test.

Good pull requests should include:

- A clear problem statement.
- Focused implementation.
- Tests or an example.
- No unrelated rewrites.

## License

Add a `LICENSE` file before publishing or distributing this project.
