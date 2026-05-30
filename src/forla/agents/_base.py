from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union
from pydantic import BaseModel as PydanticBaseModel
from ..messages import Message, UserMessage
from ..types import AgentResponse, CancellationToken
from ..context import AgentContext
from ..middleware._chain import MiddlewareChain


class BaseAgent(ABC):
    """Abstract interface every agent must implement.
    
    KEY DESIGN DECISIONS encoded here:
    
    1. run_stream() is the PRIMITIVE. run() is just run_stream() filtered.
       Build streaming first. This is exactly how picoagents implements it.
    
    2. Both accept cancellation_token for clean shutdown when users want to stop.
    
    3. Context can be injected — this enables:
       - Pre-populating with conversation history
       - Sharing context between agents in an orchestration
       - Restoring state after a server restart (Chapter 8 pattern)
       - Testing with controlled initial state
    
    4. 'description' is separate from 'instructions':
       - instructions = internal system prompt (private to the agent)
       - description = what the agent does (shown to orchestrators and other agents)
       This separation is needed for the agent-as-tool pattern (Section 4.11).
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        model_client: "BaseChatCompletionClient",
        tools: Optional[List] = None,
        memory: Optional["BaseMemory"] = None,
        context: Optional[AgentContext] = None,
        middlewares: Optional[List] = None,
        max_iterations: int = 10,
        output_format: Optional[Type[PydanticBaseModel]] = None,
    ):
        if not name:
            raise ValueError("Agent 'name' cannot be empty")
        if not description:
            raise ValueError("Agent 'description' cannot be empty")
        if not instructions:
            raise ValueError("Agent 'instructions' cannot be empty")
        if model_client is None:
            raise ValueError("Agent 'model_client' is required")

        self.name = name
        self.description = description
        self.instructions = instructions
        self.model_client = model_client
        self.memory = memory
        self.context = context or AgentContext()
        self.middleware_chain = MiddlewareChain(middlewares or [])
        self.max_iterations = max_iterations
        self.output_format = output_format

        # Process tools: convert plain functions to FunctionTool instances
        self.tools = self._process_tools(tools or [])

    def _process_tools(self, tools: list) -> list:
        """Convert plain Python functions into FunctionTool instances.
        
        This is the convenience layer that lets developers write:
            Agent(tools=[my_function])
        instead of:
            Agent(tools=[FunctionTool(my_function)])
        """
        from ..tools._function_tool import FunctionTool
        from ..tools._base import BaseTool

        processed = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                processed.append(tool)     # Already a proper tool
            elif callable(tool):
                processed.append(FunctionTool(tool))  # Wrap the function
            else:
                raise ValueError(
                    f"Tool must be a BaseTool instance or callable function, "
                    f"got: {type(tool)}"
                )
        return processed

    def _get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Convert our tools to the JSON schemas the LLM API expects."""
        return [tool.to_llm_format() for tool in self.tools]

    def _find_tool(self, name: str):
        """Find a tool by name. Returns None if not found."""
        return next((t for t in self.tools if t.name == name), None)

    def reset(self) -> None:
        """Clear conversation history while keeping agent configuration.
        
        Use this to start a new session without creating a new agent.
        The agent keeps its name, instructions, tools, and memory —
        only the conversation messages are cleared.
        """
        self.context.clear()

    def get_info(self) -> Dict[str, Any]:
        """Return metadata about this agent for debugging and orchestration."""
        return {
            "name": self.name,
            "description": self.description,
            "tools": [t.name for t in self.tools],
            "max_iterations": self.max_iterations,
            "has_memory": self.memory is not None,
        }

    @abstractmethod
    async def run(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AgentResponse:
        """Execute the agent and return only the final result."""
        pass

    @abstractmethod
    def run_stream(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[Message, Any, AgentResponse], None]:
        """Execute the agent, yielding events and messages in real-time."""
        pass

    def as_tool(self, result_strategy: str = "last"):
        """Convert this agent into a tool that other agents can use.
        
        This enables the 'agents as tools' composition pattern (Section 4.11).
        A coordinator agent can use a specialist agent as if it were a function.
        
        'result_strategy' controls what gets returned to the calling agent:
        - "last": only the final message (maximum context isolation)
        - "last:N": last N messages
        - "all": all messages (full transparency, risks context explosion)
        """
        from ..tools._base import BaseTool, ToolResult

        parent_agent = self

        class AgentTool(BaseTool):
            def __init__(self):
                super().__init__(
                    name=parent_agent.name,
                    description=parent_agent.description,
                )

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task to delegate to this agent",
                        }
                    },
                    "required": ["task"],
                }

            async def execute(self, parameters):
                task = parameters.get("task", "")
                # Run the agent in its own context
                agent_copy = parent_agent.__class__(
                    name=parent_agent.name,
                    description=parent_agent.description,
                    instructions=parent_agent.instructions,
                    model_client=parent_agent.model_client,
                    tools=parent_agent.tools,
                    memory=parent_agent.memory,
                    max_iterations=parent_agent.max_iterations,
                )
                response = await agent_copy.run(task)
                return ToolResult(success=True, result=response.content)

        return AgentTool()
