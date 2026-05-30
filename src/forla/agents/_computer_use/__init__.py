"""
Computer Use components for forla - Interface automation capabilities.

This package provides interface clients, agents, and tools that enable
computer use automation for web browsers, desktop applications, and mobile apps.
"""
from ._computer_use import ComputerUseAgent
from ._interface_clients import (
    Action,
    ActionResult,
    ActionType,
    BaseInterfaceClient,
    InterfaceState,
    PlaywrightWebClient,
)
from ._planning_models import (
    DOMFilter,
    InterfaceConfig,
    InterfaceRepresentation,
    PlanningStrategy,
)

# Import playwright tools
from ._playwright_tools import (
    ClickTool,
    NavigateTool,
    ObservePageTool,
    ScrollTool,
    TypeTool,
    create_playwright_tools,
)

__all__ = [
    # Interface clients
    "BaseInterfaceClient",
    "PlaywrightWebClient",
    "InterfaceState",
    "Action",
    "ActionResult",
    "ActionType",
    # Planning models (legacy)
    "InterfaceConfig",
    "PlanningStrategy",
    "InterfaceRepresentation",
    "DOMFilter",
    # Tools
    "NavigateTool",
    "ClickTool",
    "TypeTool",
    "ScrollTool",
    "ObservePageTool",
    "create_playwright_tools",
]
