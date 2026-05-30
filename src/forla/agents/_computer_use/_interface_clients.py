"""
Interface client abstractions for computer use tools.

This module provides base classes and implementations for different
interface automation backends (Playwright for web, PyAutoGUI for desktop, etc).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Supported action types for interface automation."""

    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    NAVIGATE = "navigate"
    SCREENSHOT = "screenshot"
    SCROLL = "scroll"
    PRESS = "press"
    HOVER = "hover"


class Action(BaseModel):
    """Represents an action to be executed on an interface."""

    action_type: ActionType
    selector: Optional[str] = None
    value: Optional[str] = None
    coordinates: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """Result of executing an action on an interface."""

    success: bool
    description: str
    error: Optional[str] = None
    screenshot: Optional[bytes] = None
    task_complete: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InterfaceState(BaseModel):
    """Represents the current state of an interface."""

    url: Optional[str] = None
    title: Optional[str] = None
    content: str = ""
    interactive_elements: List[Dict[str, Any]] = Field(default_factory=list)
    screenshot: Optional[bytes] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseInterfaceClient(ABC):
    """
    Abstract base class for interface automation clients.

    Provides a unified interface for different automation backends
    (Playwright for web, PyAutoGUI for desktop, etc).
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the interface client."""
        pass

    @abstractmethod
    async def get_state(self, format: str = "hybrid") -> InterfaceState:
        """
        Get current state of the interface.

        Args:
            format: State format - 'text', 'visual', 'hybrid'

        Returns:
            InterfaceState containing current interface information
        """
        pass

    @abstractmethod
    async def execute_action(self, action: Action) -> ActionResult:
        """
        Execute an action on the interface.

        Args:
            action: Action to execute

        Returns:
            ActionResult containing execution outcome
        """
        pass

    @abstractmethod
    async def get_screenshot(self) -> bytes:
        """
        Get current screenshot of the interface.

        Returns:
            Screenshot as bytes (PNG format)
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up and close the interface client."""
        pass


class PlaywrightWebClient(BaseInterfaceClient):
    """
    Web interface automation using Playwright.

    Provides browser automation capabilities for web applications.
    """

    def __init__(
        self, start_url: str = "https://www.google.com", headless: bool = True
    ):
        """
        Initialize Playwright web client.

        Args:
            start_url: Initial URL to navigate to
            headless: Whether to run browser in headless mode
        """
        self.start_url = start_url
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.action_history = []

    async def initialize(self) -> None:
        """Initialize Playwright browser session."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is not installed. Install with: pip install 'forla[computer-use]'"
            )

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        await self.page.goto(self.start_url)

    async def get_state(self, format: str = "hybrid") -> InterfaceState:
        """Get current state of the web page."""
        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        state = InterfaceState(url=self.page.url, title=await self.page.title())

        if format in ["text", "hybrid"]:
            # Get text content
            state.content = await self.page.content()

            # Get interactive elements
            elements = await self._get_interactive_elements()
            state.interactive_elements = elements

        if format in ["visual", "hybrid"]:
            # Get screenshot
            state.screenshot = await self.get_screenshot()

        return state

    async def _get_interactive_elements(self) -> List[Dict[str, Any]]:
        """Extract interactive elements from the page."""
        if not self.page:
            return []
        elements = await self.page.evaluate(
            """
            () => {
                const interactiveSelectors = [
                    'button', 'a', 'input', 'select', 'textarea',
                    '[role="button"]', '[role="link"]', '[onclick]'
                ];

                const elements = [];
                interactiveSelectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => {
                        if (el.offsetParent !== null) {  // Is visible
                            elements.push({
                                tag: el.tagName.toLowerCase(),
                                text: el.innerText || el.value || '',
                                type: el.type || '',
                                placeholder: el.placeholder || '',
                                href: el.href || '',
                                selector: el.id ? `#${el.id}` :
                                         el.className ? `.${el.className.split(' ')[0]}` :
                                         el.tagName.toLowerCase()
                            });
                        }
                    });
                });
                return elements;
            }
        """
        )
        return elements

    async def execute_action(self, action: Action) -> ActionResult:
        """Execute an action on the web page."""
        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        try:
            if action.action_type == ActionType.NAVIGATE:
                if not action.value:
                    raise ValueError("Navigate action requires a URL value")

                # Try navigation with robust fallback strategy
                try:
                    # First attempt: wait for networkidle (ideal but may timeout on slow sites)
                    await self.page.goto(
                        action.value, wait_until="networkidle", timeout=5000
                    )
                    result = ActionResult(
                        success=True, description=f"Navigated to {action.value}"
                    )
                except Exception as e:
                    # Fallback: if networkidle times out, check if page loaded at all
                    current_url = self.page.url
                    if current_url and (
                        action.value in current_url or current_url != "about:blank"
                    ):
                        # Page loaded even if not fully idle - consider it a success
                        try:
                            # Wait a bit for DOM to be ready
                            await self.page.wait_for_load_state(
                                "domcontentloaded", timeout=5000
                            )
                        except:
                            pass
                        result = ActionResult(
                            success=True,
                            description=f"Navigated to {action.value} (page loaded but not fully idle)",
                        )
                    else:
                        # Navigation truly failed
                        raise e

            elif action.action_type == ActionType.CLICK:
                if not action.selector:
                    raise ValueError("Click action requires a selector")

                # Try different selector strategies
                clicked = False
                selector = action.selector

                try:
                    # First try as direct CSS selector
                    await self.page.click(selector, timeout=2000)
                    clicked = True
                except Exception as e:
                    # If selector contains :contains(), suggest alternative
                    if ":contains(" in selector:
                        # Extract text and try text-based click
                        import re

                        match = re.search(r":contains\(['\"]?(.*?)['\"]?\)", selector)
                        if match:
                            text = match.group(1)
                            # Try clicking by text content
                            try:
                                await self.page.click(f"text={text}", timeout=2000)
                                clicked = True
                                selector = f"text={text}"
                            except:
                                # Try partial text match with href
                                try:
                                    await self.page.click(
                                        f"a[href*='{text.lower()}']", timeout=2000
                                    )
                                    clicked = True
                                    selector = f"a[href*='{text.lower()}']"
                                except:
                                    pass

                    # If still not clicked, try as text selector directly
                    if (
                        not clicked
                        and not selector.startswith("#")
                        and not selector.startswith(".")
                    ):
                        try:
                            await self.page.click(f"text={selector}", timeout=2000)
                            clicked = True
                            selector = f"text={selector}"
                        except:
                            pass

                    if not clicked:
                        raise e

                result = ActionResult(
                    success=True, description=f"Clicked on {selector}"
                )

            elif action.action_type == ActionType.TYPE:
                if not action.selector or not action.value:
                    raise ValueError("Type action requires both selector and value")
                await self.page.fill(action.selector, action.value)
                result = ActionResult(
                    success=True,
                    description=f"Typed '{action.value}' into {action.selector}",
                )

            elif action.action_type == ActionType.SELECT:
                if not action.selector or not action.value:
                    raise ValueError("Select action requires both selector and value")
                await self.page.select_option(action.selector, action.value)
                result = ActionResult(
                    success=True,
                    description=f"Selected '{action.value}' in {action.selector}",
                )

            elif action.action_type == ActionType.PRESS:
                if not action.selector or not action.value:
                    raise ValueError("Press action requires both selector and value")
                await self.page.press(action.selector, action.value)
                result = ActionResult(
                    success=True,
                    description=f"Pressed '{action.value}' on {action.selector}",
                )

            elif action.action_type == ActionType.SCROLL:
                # Map direction to scroll values
                direction = action.value or "down"
                scroll_x, scroll_y = 0, 0

                if direction == "down":
                    scroll_y = 500
                elif direction == "up":
                    scroll_y = -500
                elif direction == "right":
                    scroll_x = 500
                elif direction == "left":
                    scroll_x = -500
                else:
                    # If it's a number, use it directly for vertical scrolling
                    try:
                        scroll_y = int(direction)
                    except ValueError:
                        scroll_y = 500  # Default to scrolling down

                await self.page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")
                result = ActionResult(
                    success=True,
                    description=f"Scrolled {direction} by {abs(scroll_y or scroll_x)} pixels",
                )

            elif action.action_type == ActionType.HOVER:
                if not action.selector:
                    raise ValueError("Hover action requires a selector")
                await self.page.hover(action.selector)
                result = ActionResult(
                    success=True, description=f"Hovered over {action.selector}"
                )

            else:
                result = ActionResult(
                    success=False,
                    description="",
                    error=f"Unsupported action type: {action.action_type}",
                )

            # Record action in history
            self.action_history.append(action)

            # Add screenshot if requested
            if action.metadata.get("capture_screenshot", False):
                result.screenshot = await self.get_screenshot()

            return result

        except Exception as e:
            return ActionResult(success=False, description="", error=str(e))

    async def get_screenshot(self) -> bytes:
        """Get screenshot of current page."""
        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        return await self.page.screenshot(type="png")

    async def close(self) -> None:
        """Close browser and clean up."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
