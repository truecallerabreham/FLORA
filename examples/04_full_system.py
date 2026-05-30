"""
EXAMPLE 4: The Complete System
This demonstrates everything working together.
"""
import asyncio
import os
from forla import Agent, OpenAIChatCompletionClient
from forla.orchestration import RoundRobinOrchestrator, OrchestrationResponse
from forla.termination import MaxMessageTermination, TextMentionTermination
from forla.middleware import LoggingMiddleware
from forla.memory import ListMemory, MemoryContent
from forla.tools import ThinkTool
from forla.workflow import (
    Workflow, WorkflowMetadata, FunctionStep, StepMetadata,
    WorkflowRunner, Context,
)
from pydantic import BaseModel


# ── Workflow Example ──────────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str

class TextOutput(BaseModel):
    result: str


async def to_uppercase(input_data: TextInput, ctx: Context) -> TextOutput:
    return TextOutput(result=input_data.text.upper())

async def add_exclamation(input_data: TextOutput, ctx: Context) -> TextOutput:
    return TextOutput(result=input_data.result + "!")


async def demo_workflow():
    print("\n=== WORKFLOW DEMO ===")
    step1 = FunctionStep("upper", StepMetadata(name="Uppercase"), TextInput, TextOutput, to_uppercase)
    step2 = FunctionStep("exclaim", StepMetadata(name="Exclaim"), TextOutput, TextOutput, add_exclamation)
    
    workflow = Workflow(metadata=WorkflowMetadata(name="Text Transformer")).chain(step1, step2)
    result = await WorkflowRunner().run(workflow, {"text": "hello world"})
    print(f"Workflow result: {result.final_output}")   # HELLO WORLD!


# ── Multi-Agent Orchestration Example ────────────────────────────────────

async def demo_orchestration():
    print("\n=== ORCHESTRATION DEMO ===")
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    researcher = Agent(
        name="researcher",
        description="Researches topics and provides factual information",
        instructions="You are a researcher. Provide concise, factual information.",
        model_client=client,
        tools=[ThinkTool()],
        middlewares=[LoggingMiddleware()],
    )

    writer = Agent(
        name="writer",
        description="Writes clear, engaging content",
        instructions=(
            "You are a writer. Given research, write a clear 2-sentence summary. "
            "When done, end with 'TERMINATE'."
        ),
        model_client=client,
    )

    termination = (
        MaxMessageTermination(6) |
        TextMentionTermination("TERMINATE")
    )

    orchestrator = RoundRobinOrchestrator(
        agents=[researcher, writer],
        termination=termination,
    )

    from forla.messages import AssistantMessage
    
    async for event in orchestrator.run_stream("Explain what a multi-agent system is"):
        if isinstance(event, AssistantMessage) and event.content:
            print(f"\n[{event.source}]: {event.content[:200]}")
        elif isinstance(event, OrchestrationResponse):
            print(f"\n--- Done: {event.stop_message.content} ---")
            print(f"Usage: {event.usage}")


# ── Memory Example ─────────────────────────────────────────────────────────

async def demo_memory():
    print("\n=== MEMORY DEMO ===")
    
    memory = ListMemory()
    await memory.add(MemoryContent("User prefers concise responses"))
    await memory.add(MemoryContent("User's name is Alice"))
    
    context_items = await memory.get_context(max_items=5)
    print(f"Memory context: {context_items}")
    
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = Agent(
        name="personalized_assistant",
        description="A personalized assistant",
        instructions="You are a helpful assistant.",
        model_client=client,
        memory=memory,
    )
    
    response = await agent.run("Who am I?")
    print(f"Response: {response.content}")   # Should mention Alice


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    await demo_workflow()
    
    if os.getenv("OPENAI_API_KEY"):
        await demo_orchestration()
        await demo_memory()
    else:
        print("\nSet OPENAI_API_KEY to run agent demos.")


if __name__ == "__main__":
    asyncio.run(main())
