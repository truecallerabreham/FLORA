"""
Structured output models for computer use planning and observation.

This module provides Pydantic models for all LLM interactions in the
computer use system, ensuring reliable structured output.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ._interface_clients import ActionType


class InterfaceRepresentation(str, Enum):
    """How to represent the interface to the LLM."""

    TEXT = "text"  # DOM text content only
    HTML = "html"  # Clean HTML structure
    VISUAL = "visual"  # Screenshot description
    HYBRID = "hybrid"  # Text + visual description


class PlanningStrategy(str, Enum):
    """Planning strategy for the computer use tool."""

    IMPLICIT = "implicit"  # Step-by-step, one action at a time
    EXPLICIT = "explicit"  # Plan multiple steps ahead
    AUTO = "auto"  # Let LLM decide based on task complexity


class PageObservation(BaseModel):
    """Structured observation of the current page state."""

    url: str = Field(description="Current page URL")
    title: str = Field(description="Page title")
    summary: str = Field(description="Brief summary of what's visible on the page")
    key_elements: List[str] = Field(
        description="Important interactive elements or content"
    )
    task_relevance: str = Field(description="How this page relates to the current task")
    is_task_complete: bool = Field(
        description="Whether the task appears to be complete"
    )
    confidence: float = Field(
        description="Confidence in this observation (0-1)", ge=0, le=1
    )


class NextActionPlan(BaseModel):
    """Structured plan for the next action to take."""

    action_type: ActionType = Field(description="Type of action to perform")
    selector: Optional[str] = Field(
        default=None, description="CSS selector for the target element"
    )
    value: Optional[str] = Field(
        default=None,
        description="Value to input (for type/select actions) or URL (for navigate)",
    )
    coordinates: Optional[Dict[str, int]] = Field(
        default=None, description="Coordinates for click if selector fails"
    )
    reasoning: str = Field(description="Why this action is being taken")
    expected_outcome: str = Field(description="What should happen after this action")
    confidence: float = Field(
        description="Confidence in this action plan (0-1)", ge=0, le=1
    )


class MultiStepPlan(BaseModel):
    """Structured plan with multiple steps (for explicit planning)."""

    steps: List[NextActionPlan] = Field(description="Sequence of actions to take")
    overall_strategy: str = Field(
        description="High-level strategy for completing the task"
    )
    estimated_complexity: Literal["simple", "moderate", "complex"] = Field(
        description="Estimated complexity of the task"
    )
    requires_exploration: bool = Field(
        description="Whether this task requires exploration/discovery"
    )


class PlanningDecision(BaseModel):
    """Decision on which planning strategy to use."""

    chosen_strategy: PlanningStrategy = Field(
        description="Which planning approach to use"
    )
    reasoning: str = Field(description="Why this strategy was chosen")
    task_complexity: Literal["simple", "moderate", "complex"] = Field(
        description="Assessed complexity of the task"
    )


class TaskCompletion(BaseModel):
    """Assessment of whether the task is complete."""

    is_complete: bool = Field(description="Whether the task has been completed")
    completion_confidence: float = Field(
        description="Confidence in completion assessment (0-1)", ge=0, le=1
    )
    summary: str = Field(description="Summary of what was accomplished")
    remaining_work: Optional[str] = Field(
        default=None, description="What still needs to be done if incomplete"
    )


class DOMFilter(BaseModel):
    """Configuration for filtering DOM content."""

    max_text_length: int = Field(2000, description="Maximum text content length")
    include_hidden: bool = Field(False, description="Include hidden elements")
    interactive_only: bool = Field(
        False, description="Only include interactive elements"
    )
    exclude_tags: List[str] = Field(
        default_factory=lambda: ["script", "style", "meta", "link"],
        description="Tags to exclude from DOM",
    )


class InterfaceConfig(BaseModel):
    """Configuration for interface representation."""

    representation: InterfaceRepresentation = Field(
        InterfaceRepresentation.HYBRID, description="How to represent the interface"
    )
    dom_filter: DOMFilter = Field(
        default_factory=lambda: DOMFilter(
            max_text_length=2000, include_hidden=False, interactive_only=False
        ),
        description="How to filter DOM content",
    )
    include_screenshot: bool = Field(True, description="Whether to include screenshots")
    screenshot_description: bool = Field(
        True, description="Whether to describe screenshots with LLM"
    )
