"""
Playwright actions as tools for computer use agents.

This module defines Playwright actions as standard tools that can be used
with LLM tool calling, simplifying the computer use agent implementation.
"""

from typing import Any, Dict

from ...tools import BaseTool
from ...types import ToolResult
from ._interface_clients import Action, ActionType


class NavigateTool(BaseTool):
    """Navigate to a URL."""

    def __init__(self, interface_client):
        super().__init__(name="navigate", description="Navigate to a specific URL")
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to"}
            },
            "required": ["url"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(action_type=ActionType.NAVIGATE, value=parameters["url"])
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class ClickTool(BaseTool):
    """Click on an element."""

    def __init__(self, interface_client):
        super().__init__(
            name="click",
            description="Click on an element using a CSS selector or text content",
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector or text content to click",
                }
            },
            "required": ["selector"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(action_type=ActionType.CLICK, selector=parameters["selector"])
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class TypeTool(BaseTool):
    """Type text into an input."""

    def __init__(self, interface_client):
        super().__init__(name="type", description="Type text into an input element")
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the input element",
                },
                "text": {"type": "string", "description": "Text to type"},
            },
            "required": ["selector", "text"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(
            action_type=ActionType.TYPE,
            selector=parameters["selector"],
            value=parameters["text"],
        )
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class SelectTool(BaseTool):
    """Select an option from a dropdown."""

    def __init__(self, interface_client):
        super().__init__(
            name="select", description="Select an option from a dropdown element"
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the dropdown element",
                },
                "value": {"type": "string", "description": "Option value to select"},
            },
            "required": ["selector", "value"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(
            action_type=ActionType.SELECT,
            selector=parameters["selector"],
            value=parameters["value"],
        )
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class PressTool(BaseTool):
    """Press a key or key combination."""

    def __init__(self, interface_client):
        super().__init__(
            name="press", description="Press a key or key combination on an element"
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to focus",
                },
                "key": {
                    "type": "string",
                    "description": "Key or key combination (e.g., 'Enter', 'Control+a')",
                },
            },
            "required": ["selector", "key"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(
            action_type=ActionType.PRESS,
            selector=parameters["selector"],
            value=parameters["key"],
        )
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class HoverTool(BaseTool):
    """Hover over an element."""

    def __init__(self, interface_client):
        super().__init__(
            name="hover", description="Hover over an element to trigger hover effects"
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to hover over",
                }
            },
            "required": ["selector"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(action_type=ActionType.HOVER, selector=parameters["selector"])
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class ScrollTool(BaseTool):
    """Scroll the page."""

    def __init__(self, interface_client):
        super().__init__(name="scroll", description="Scroll the page or an element")
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Direction to scroll",
                },
                "amount": {
                    "type": "integer",
                    "description": "Pixels to scroll (default: 500)",
                },
                "selector": {
                    "type": "string",
                    "description": "Optional element selector to scroll (default: page)",
                },
            },
            "required": ["direction"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        # Use the SCROLL action type
        action = Action(
            action_type=ActionType.SCROLL,
            value=parameters.get("direction", "down"),
            selector=parameters.get("selector"),
        )
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success,
            result=f"Scrolled {parameters.get('direction', 'down')} {parameters.get('amount', 500)} pixels"
            if result.success
            else result.description,
            error=result.error,
        )


class ObservePageTool(BaseTool):
    """Observe the current page state."""

    def __init__(self, interface_client):
        super().__init__(
            name="observe_page",
            description="Get information about the current page state",
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        state = await self.interface_client.get_state("hybrid")

        # Build a concise description of the page
        description = f"URL: {state.url}\n"
        description += f"Title: {state.title}\n"

        # Extract visible text content with semantic extraction
        page = self.interface_client.page
        if page:
            try:
                # Get semantic content (headings, articles, main content)
                content_data = await page.evaluate(
                    """
                    () => {
                        // Extract headings
                        const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
                            .slice(0, 10)
                            .map(h => `${h.tagName}: ${h.innerText.trim()}`)
                            .filter(h => h.length > 5);

                        // Extract article content or main content
                        let mainText = '';
                        const article = document.querySelector('article, main, [role="main"]');
                        if (article) {
                            mainText = article.innerText.substring(0, 5000);
                        } else {
                            mainText = document.body.innerText.substring(0, 5000);
                        }

                        return {
                            headings: headings,
                            text: mainText
                        };
                    }
                """
                )

                # Format the content
                if content_data.get("headings"):
                    description += "Key headings:\n"
                    for heading in content_data["headings"][:8]:
                        description += f"  {heading}\n"
                    description += "\n"

                if content_data.get("text"):
                    description += f"Page content:\n{content_data['text']}\n"

            except Exception as e:
                # Fallback to basic text extraction
                try:
                    visible_text = await page.evaluate(
                        "() => document.body.innerText.substring(0, 5000)"
                    )
                    description += f"Page content: {visible_text}\n"
                except:
                    # Last resort fallback
                    description += f"Content preview: {state.content[:500]}...\n"

        description += (
            f"Interactive elements: {len(state.interactive_elements)} found\n"
        )

        # Show more interactive elements, especially links and headings
        if state.interactive_elements:
            description += "Key elements:\n"
            # Filter for links with meaningful text
            meaningful_elements = [
                elem
                for elem in state.interactive_elements[:15]
                if elem.get("text", "").strip()
                and len(elem.get("text", "").strip()) > 2
            ]
            for elem in meaningful_elements[:10]:
                text = elem.get("text", "").strip()[:50]
                tag = elem.get("tag", "")
                if text:
                    description += f"  - {tag}: {text}\n"

        return ToolResult(success=True, result=description, error=None)


def create_playwright_tools(interface_client) -> list[BaseTool]:
    """
    Create a list of Playwright tools for computer use.

    Args:
        interface_client: The interface client to use for actions

    Returns:
        List of tool instances
    """
    return [
        NavigateTool(interface_client),
        ClickTool(interface_client),
        TypeTool(interface_client),
        SelectTool(interface_client),
        PressTool(interface_client),
        HoverTool(interface_client),
        ScrollTool(interface_client),
        ObservePageTool(interface_client),
    ]
