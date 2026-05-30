"""
EXAMPLE 2: The Poet-Critic Multi-Agent System
This replicates the example from Chapter 1 of the book.
Run: python examples/02_poet_critic.py
"""
import asyncio
import os
from forla import Agent, OpenAIChatCompletionClient
from forla.orchestration import RoundRobinOrchestrator, OrchestrationResponse
from forla.termination import MaxMessageTermination, TextMentionTermination
from forla.messages import AssistantMessage


async def main():
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    poet = Agent(
        name="poet",
        description="A haiku poet that writes 5-7-5 syllable poems",
        instructions="You are a haiku poet. Write haiku poems with exactly 5-7-5 syllables.",
        model_client=client,
    )

    critic = Agent(
        name="critic",
        description="A poetry critic that gives feedback on haiku poems",
        instructions=(
            "You are a haiku critic. Give 2-3 specific actionable suggestions for improvement. "
            "Be constructive and brief. "
            "If you are satisfied with the haiku, say 'APPROVED' followed by brief praise."
        ),
        model_client=client,
    )

    termination = MaxMessageTermination(8) | TextMentionTermination("APPROVED")

    orchestrator = RoundRobinOrchestrator(
        agents=[poet, critic],
        termination=termination,
    )

    print("🎯 Task: Write a haiku about cherry blossoms in spring")
    print("=" * 60)

    async for event in orchestrator.run_stream(
        "Write a haiku about cherry blossoms in spring"
    ):
        if isinstance(event, AssistantMessage) and event.content:
            print(f"\n[{event.source}]:")
            print(event.content)
        elif isinstance(event, OrchestrationResponse):
            print("\n" + "=" * 60)
            print(f"🏁 Stop reason: {event.stop_message.content}")
            print(f"📊 Usage: {event.usage}")


if __name__ == "__main__":
    asyncio.run(main())
