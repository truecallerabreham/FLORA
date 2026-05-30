"""
EXAMPLE 5: Autonomous Software-Engineering Agent Team

This example demonstrates a coding-agent pattern built from Forla primitives:

- Agent + tools + memory
- Explicit planning and metacognition
- Sandboxed file operations with surgical str_replace edits
- Command execution with a conservative allowlist
- Markdown task tracking in /memories/current_task.md
- Multi-agent review with an explicit SHIP_READY termination signal

Run:
    python examples/05_autonomous_coding_agent.py

Or point the team at a disposable workspace:
    python examples/05_autonomous_coding_agent.py --workspace ./scratch/coding-demo "Fix the failing tests"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence

from forla import Agent, OpenAIChatCompletionClient
from forla.messages import AssistantMessage, ToolMessage
from forla.orchestration import OrchestrationResponse, RoundRobinOrchestrator
from forla.termination import MaxMessageTermination, TextMentionTermination
from forla.tools import BaseTool, MemoryTool, TaskStatusTool, ThinkTool
from forla.types import ToolResult
from forla import GuardrailMiddleware, LoggingMiddleware


class WorkspaceTool(BaseTool):
    """Sandboxed file operations for a coding agent workspace."""

    def __init__(self, workspace: Path):
        super().__init__(
            name="workspace",
            description=(
                "Inspect and edit files inside the project workspace. "
                "Use str_replace for precise surgical edits."
            ),
        )
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["tree", "view", "create", "str_replace"],
                    "description": "Workspace operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Relative path inside the workspace.",
                },
                "file_text": {
                    "type": "string",
                    "description": "Text for create.",
                },
                "old_str": {
                    "type": "string",
                    "description": "Unique text to replace.",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["command"],
        }

    def _resolve(self, path: str = ".") -> Path:
        clean = path.strip().lstrip("/\\") or "."
        target = (self.workspace / clean).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {path}") from exc
        return target

    def _tree(self, path: str = ".", limit: int = 120) -> str:
        root = self._resolve(path)
        if not root.exists():
            return f"Path does not exist: {path}"

        rows: List[str] = []
        for item in sorted(root.rglob("*")):
            if len(rows) >= limit:
                rows.append(f"... truncated at {limit} entries")
                break
            if any(part in {".git", ".venv", "__pycache__", "node_modules"} for part in item.parts):
                continue
            relative = item.relative_to(self.workspace)
            suffix = "/" if item.is_dir() else ""
            rows.append(f"{relative.as_posix()}{suffix}")
        return "\n".join(rows) if rows else "Workspace is empty."

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        command = parameters.get("command")
        path = parameters.get("path", ".")

        try:
            if command == "tree":
                return ToolResult(success=True, result=self._tree(path))

            target = self._resolve(path)

            if command == "view":
                if not target.exists():
                    return ToolResult(success=False, error=f"Path does not exist: {path}")
                if target.is_dir():
                    return ToolResult(success=True, result=self._tree(path))
                return ToolResult(success=True, result=target.read_text(encoding="utf-8"))

            if command == "create":
                if target.exists():
                    return ToolResult(success=False, error=f"File already exists: {path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                text = parameters.get("file_text", "")
                target.write_text(text, encoding="utf-8")
                return ToolResult(success=True, result=f"Created {path} ({len(text)} chars).")

            if command == "str_replace":
                if not target.is_file():
                    return ToolResult(success=False, error=f"Not a file: {path}")
                old = parameters.get("old_str", "")
                new = parameters.get("new_str", "")
                content = target.read_text(encoding="utf-8")
                count = content.count(old)
                if count == 0:
                    return ToolResult(success=False, error="old_str was not found.")
                if count > 1:
                    return ToolResult(success=False, error=f"old_str appears {count} times.")
                target.write_text(content.replace(old, new, 1), encoding="utf-8")
                return ToolResult(success=True, result=f"Updated {path}.")

            return ToolResult(success=False, error=f"Unknown workspace command: {command}")

        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class CommandTool(BaseTool):
    """Runs a small allowlist of inspection and test commands."""

    FORBIDDEN_TOKENS = (";", "&&", "||", "|", ">", "<", "`", "$(", "\n", "\r")
    ALLOWED_PREFIXES: Sequence[Sequence[str]] = (
        ("python", "-m", "pytest"),
        ("python", "-m", "compileall"),
        ("pytest",),
        ("ruff",),
        ("mypy",),
        ("git", "status"),
        ("git", "diff"),
    )

    def __init__(self, workspace: Path):
        super().__init__(
            name="run_command",
            description=(
                "Run allowed verification commands in the workspace. "
                "Use this for tests, compile checks, lint, and git diff/status."
            ),
        )
        self.workspace = workspace.resolve()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to run, for example: python -m pytest -q",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum runtime in seconds.",
                },
            },
            "required": ["command"],
        }

    def _is_allowed(self, args: Sequence[str]) -> bool:
        normalized = [arg.lower() for arg in args]
        for prefix in self.ALLOWED_PREFIXES:
            if normalized[: len(prefix)] == list(prefix):
                return True
        return False

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        command = parameters.get("command", "")
        timeout = int(parameters.get("timeout_seconds", 30))

        if any(token in command for token in self.FORBIDDEN_TOKENS):
            return ToolResult(success=False, error="Command contains blocked shell syntax.")

        try:
            args = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            return ToolResult(success=False, error=f"Could not parse command: {exc}")

        if not args or not self._is_allowed(args):
            allowed = [" ".join(prefix) for prefix in self.ALLOWED_PREFIXES]
            return ToolResult(success=False, error=f"Command is not allowlisted. Allowed prefixes: {allowed}")

        try:
            completed = subprocess.run(
                args,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        output = "\n".join(
            part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
        )
        if not output:
            output = "(no output)"

        result = f"exit_code={completed.returncode}\n{output[-6000:]}"
        return ToolResult(success=completed.returncode == 0, result=result, error=None if completed.returncode == 0 else result)


TEAM_PROMPT = """
You are part of an autonomous software-engineering team.

Core pattern:
- Tools give you controlled action.
- Prompts encode engineering workflow.
- Memory preserves knowledge across sessions.

You must follow this five-phase workflow:
1. Memory check: inspect /memories/current_task.md and relevant memory before planning.
2. Planning: use think before editing. Make the plan concrete.
3. Execution: use workspace.str_replace for surgical edits. Avoid broad rewrites.
4. Learning: update /memories/current_task.md checkboxes and store important project facts.
5. Completion: use task_status before claiming completion.

Rules:
- Do not claim success until verification ran or you clearly explain why it could not run.
- Prefer minimal diffs.
- Keep /memories/current_task.md current with markdown checkboxes.
- The reviewer is the only role that may write SHIP_READY, and only after evaluation passes.
"""


def build_agent(
    *,
    name: str,
    description: str,
    role_instructions: str,
    client: OpenAIChatCompletionClient,
    tools: List[BaseTool],
) -> Agent:
    return Agent(
        name=name,
        description=description,
        instructions=f"{TEAM_PROMPT}\n\nRole:\n{role_instructions}",
        model_client=client,
        tools=tools,
        middlewares=[GuardrailMiddleware(), LoggingMiddleware()],
        max_iterations=6,
    )


def seed_demo_workspace(workspace: Path) -> None:
    """Create a tiny failing Python project if the workspace is empty."""
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        return

    package = workspace / "src" / "demo_math"
    tests = workspace / "tests"
    package.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)

    (package / "__init__.py").write_text(
        "from .calculator import add, safe_divide\n\n__all__ = ['add', 'safe_divide']\n",
        encoding="utf-8",
    )
    (package / "calculator.py").write_text(
        textwrap.dedent(
            """
            def add(a: int, b: int) -> int:
                return a + b


            def safe_divide(a: float, b: float) -> float:
                # TODO: implement zero-safe division.
                return a / b
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (tests / "test_calculator.py").write_text(
        textwrap.dedent(
            """
            import pytest

            from demo_math import add, safe_divide


            def test_add():
                assert add(2, 3) == 5


            def test_safe_divide_regular_numbers():
                assert safe_divide(10, 2) == 5


            def test_safe_divide_by_zero_returns_none():
                assert safe_divide(10, 0) is None
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (workspace / "pytest.ini").write_text(
        "[pytest]\npythonpath = src\n",
        encoding="utf-8",
    )


def seed_memory(memory_root: Path, task: str) -> None:
    memories = memory_root / "memories"
    memories.mkdir(parents=True, exist_ok=True)
    current_task = memories / "current_task.md"

    if not current_task.exists():
        current_task.write_text(
            textwrap.dedent(
                f"""
                # Current Task

                {task}

                ## Checklist

                - [ ] Memory check complete
                - [ ] Plan written
                - [ ] Surgical edits applied
                - [ ] Tests or verification run
                - [ ] Learning captured
                - [ ] Reviewer approved
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

    notes = memories / "engineering_notes.md"
    if not notes.exists():
        notes.write_text(
            textwrap.dedent(
                """
                # Engineering Notes

                - Prefer small, testable diffs.
                - Read before editing.
                - Use str_replace with unique old_str values.
                - Run the narrowest meaningful verification command.
                - Capture reusable lessons in memory.
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )


def build_coding_team(workspace: Path, memory_root: Path, model: str) -> RoundRobinOrchestrator:
    client = OpenAIChatCompletionClient(
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    workspace_tool = WorkspaceTool(workspace)
    command_tool = CommandTool(workspace)
    memory_tool = MemoryTool(base_path=str(memory_root))
    shared_tools: List[BaseTool] = [
        ThinkTool(),
        memory_tool,
        workspace_tool,
        TaskStatusTool(),
    ]

    architect = build_agent(
        name="architect",
        description="Plans safe implementation steps for software-engineering tasks.",
        role_instructions=(
            "Inspect memory and the workspace, then produce a concise implementation plan. "
            "Do not edit files unless the plan requires a tiny preparatory note."
        ),
        client=client,
        tools=shared_tools,
    )

    implementer = build_agent(
        name="implementer",
        description="Applies precise code changes and runs focused verification.",
        role_instructions=(
            "Implement the plan with workspace.str_replace or create. "
            "Run verification with run_command when changes are ready."
        ),
        client=client,
        tools=[*shared_tools, command_tool],
    )

    reviewer = build_agent(
        name="reviewer",
        description="Reviews diffs, evaluates tests, updates memory, and approves shipping.",
        role_instructions=(
            "Review the final workspace state and command output. "
            "If the task is complete and verified, say SHIP_READY. "
            "If not, give specific corrective instructions."
        ),
        client=client,
        tools=[*shared_tools, command_tool],
    )

    termination = MaxMessageTermination(24) | TextMentionTermination("SHIP_READY")
    return RoundRobinOrchestrator(
        agents=[architect, implementer, reviewer],
        termination=termination,
        max_iterations=9,
    )


async def run(task: str, workspace: Path, memory_root: Path, model: str) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to run this example.")

    seed_demo_workspace(workspace)
    seed_memory(memory_root, task)

    orchestrator = build_coding_team(workspace, memory_root, model)

    print(f"Workspace: {workspace.resolve()}")
    print(f"Memory:    {memory_root.resolve()}")
    print(f"Task:      {task}\n")

    async for event in orchestrator.run_stream(task):
        if isinstance(event, AssistantMessage) and event.content:
            print(f"\n[{event.source}]\n{event.content}\n")
        elif isinstance(event, ToolMessage):
            status = "ok" if event.success else "error"
            print(f"[tool:{event.tool_name}:{status}] {event.content[:500]}\n")
        elif isinstance(event, OrchestrationResponse):
            print("\n=== FINAL ===")
            print(event.final_result)
            print(f"\nStop: {event.stop_message.content}")
            print(f"Usage: {event.usage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Forla autonomous coding-agent team.")
    parser.add_argument(
        "task",
        nargs="*",
        help="Task for the agent team. Defaults to fixing the seeded demo project.",
    )
    parser.add_argument(
        "--workspace",
        default="examples/coding_agent_workspace",
        help="Disposable project workspace the agents may edit.",
    )
    parser.add_argument(
        "--memory",
        default="examples/coding_agent_memory",
        help="Persistent memory directory for the agent team.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI-compatible model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = " ".join(args.task).strip() or (
        "Fix the failing tests in the workspace. Implement safe_divide so normal "
        "division still works and division by zero returns None. Update "
        "/memories/current_task.md, run pytest, and only approve when verified."
    )

    asyncio.run(
        run(
            task=task,
            workspace=Path(args.workspace),
            memory_root=Path(args.memory),
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
