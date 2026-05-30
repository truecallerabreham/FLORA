from __future__ import annotations
from typing import Any, Dict
from ._base import BaseTool, ToolResult


class ThinkTool(BaseTool):
    """Forces the agent to reason explicitly before taking action.
    
    WHY does this help? Without it, agents sometimes jump straight to
    tool calls without fully thinking through the problem.
    By adding ThinkTool, you give the agent a 'thinking' step where it
    reasons through what to do — similar to chain-of-thought prompting.
    
    Research by Anthropic found this improves performance by 54% on
    complex domains. The book mentions this in Section 4.6.2.
    
    Usage in agent instructions:
    "Always use the think tool to plan before calling other tools."
    """

    def __init__(self):
        super().__init__(
            name="think",
            description=(
                "Use this tool to reason through a problem step by step "
                "before taking action. Record your thought process clearly."
            ),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your step-by-step reasoning about what to do next",
                }
            },
            "required": ["thought"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        # The value is the thinking itself — just return acknowledgment
        thought = parameters.get("thought", "")
        return ToolResult(success=True, result=f"Thought recorded: {thought}")


class TaskStatusTool(BaseTool):
    """Lets the agent explicitly signal task completion or failure.
    
    WHY is this needed? Without it, agents sometimes keep running even
    when the task is complete — wasting tokens and API calls.
    
    The agent calls this tool when it believes the task is done.
    The framework checks result.parameters['is_complete'] to decide
    whether to continue the execution loop.
    
    Also useful for debugging: the 'rationale' explains why the agent
    thinks it's done, helping you identify premature terminations.
    """

    def __init__(self):
        super().__init__(
            name="task_status",
            description=(
                "Report whether the current task is complete. "
                "Call this when you have finished the task or cannot proceed further."
            ),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "is_complete": {
                    "type": "boolean",
                    "description": "True if the task is fully complete, False if you need to stop for another reason",
                },
                "rationale": {
                    "type": "string",
                    "description": "Explain why the task is complete or why you are stopping",
                },
                "result_summary": {
                    "type": "string",
                    "description": "A brief summary of what was accomplished",
                },
            },
            "required": ["is_complete", "rationale"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, result=parameters)
