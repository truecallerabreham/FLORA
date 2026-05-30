import os
import pytest
from forla import Agent, OpenAIChatCompletionClient
from forla.orchestration import RoundRobinOrchestrator, OrchestrationResponse
from forla.termination import MaxMessageTermination, TextMentionTermination

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.mark.asyncio
async def test_round_robin_orchestration():
    """Test the classic poet-critic example from Chapter 1."""
    client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    poet = Agent(
        name="poet",
        description="A haiku poet",
        instructions="You are a haiku poet. Write haiku poems when asked.",
        model_client=client,
    )

    critic = Agent(
        name="critic",
        description="A haiku critic",
        instructions=(
            "You are a haiku critic. Provide brief feedback on haiku poems. "
            "If the haiku is good, respond with 'APPROVED' and brief praise."
        ),
        model_client=client,
    )

    # Stop after 6 messages OR when the critic says APPROVED
    termination = MaxMessageTermination(6) | TextMentionTermination("APPROVED")

    orchestrator = RoundRobinOrchestrator(
        agents=[poet, critic],
        termination=termination,
    )

    result = await orchestrator.run("Write a haiku about cherry blossoms in spring")

    assert isinstance(result, OrchestrationResponse)
    assert result.stop_message is not None
    # Either approved or hit message limit
    assert ("APPROVED" in result.stop_message.content or
            "Maximum" in result.stop_message.content or
            "Text mention" in result.stop_message.content)


@pytest.mark.asyncio
async def test_termination_composability():
    """Test that | and & operators work correctly."""
    from forla.termination import MaxMessageTermination, TextMentionTermination
    from forla.messages import AssistantMessage

    # Test OR: stops when first condition is met
    or_term = MaxMessageTermination(3) | TextMentionTermination("STOP")
    or_term.reset()

    messages = [
        AssistantMessage(content="Message 1", source="agent"),
        AssistantMessage(content="Message 2 contains STOP", source="agent"),
    ]

    result = or_term.check(messages)
    assert result is not None
    assert "STOP" in result.content
