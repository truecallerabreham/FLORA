"""
Plan-based orchestration pattern.

This module provides the PlanBasedOrchestrator that uses LLM-based planning
to create step-by-step execution plans with agent assignments and retry logic.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

from .._component_config import Component, ComponentModel
from ..agents import BaseAgent
from ..llm import BaseChatCompletionClient
from ..messages import Message, UserMessage
from ..termination import BaseTermination
from ..types import AgentResponse
from ._base import BaseOrchestrator


class StepProgressEvaluation(BaseModel):
    """Structured evaluation of step completion using LLM reasoning."""

    model_config = {"extra": "forbid"}  # This generates additionalProperties: false

    step_completed: bool = Field(
        description="Whether the step was successfully completed"
    )
    failure_reason: str = Field(
        description="Brief explanation if step failed, use 'None' if successful"
    )
    confidence_score: float = Field(
        description="Confidence in the evaluation (0.0 to 1.0)", ge=0.0, le=1.0
    )
    suggested_improvements: List[str] = Field(
        description="Specific suggestions for retry if step failed"
    )


class PlanStep(BaseModel):
    """Simple plan step for LLM generation - only essential planning data."""

    model_config = {"extra": "forbid"}  # This generates additionalProperties: false

    task: str = Field(description="Clear, actionable task description")
    agent_name: str = Field(
        description="Name of the agent that should handle this step"
    )
    reasoning: str = Field(
        description="Brief explanation for why this agent was chosen"
    )


class ExecutionPlan(BaseModel):
    """Simple execution plan - just the steps."""

    model_config = {"extra": "forbid"}  # This generates additionalProperties: false

    steps: List[PlanStep] = Field(description="Ordered list of execution steps")


class PlanBasedOrchestratorConfig(BaseModel):
    """Configuration for PlanBasedOrchestrator serialization."""

    agents: List[ComponentModel] = Field(default_factory=list)
    termination: ComponentModel
    model_client: ComponentModel
    max_iterations: int = 50
    max_step_retries: int = 3


class PlanBasedOrchestrator(Component[PlanBasedOrchestratorConfig], BaseOrchestrator):
    """
    Plan-based orchestrator that creates explicit plans and retries failed steps.

    This orchestrator works within the BaseOrchestrator framework by mapping
    step retries to orchestrator iterations. Uses LLM-based planning similar
    to AI orchestrator's agent selection approach.

    Key features:
    - LLM-generated execution plans with agent assignments
    - Step retry logic with enhanced instructions
    - Context curation for focused execution
    - Runtime state tracking separate from plan data
    """

    component_config_schema = PlanBasedOrchestratorConfig
    component_type = "orchestrator"
    component_provider_override = "forla.orchestration.PlanBasedOrchestrator"

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        termination: BaseTermination,
        model_client: BaseChatCompletionClient,
        max_iterations: int = 50,
        max_step_retries: int = 3,
    ):
        super().__init__(agents, termination, max_iterations)
        self.model_client = model_client
        self.max_step_retries = max_step_retries

        # Plan execution state (separate from plan data)
        self.execution_plan: Optional[ExecutionPlan] = None
        self.current_step_index = 0
        self.current_step_retry_count = 0
        self.initial_task: Optional[str] = None

        # Runtime tracking (not part of plan) - orchestrator state only
        self.step_attempts: Dict[
            int, List[AgentResponse]
        ] = {}  # step_index -> attempts
        self.step_results: Dict[int, AgentResponse] = {}  # step_index -> final result
        self.retry_instructions: Dict[int, str] = {}  # step_index -> retry instructions

        # Performance optimization - cache agent capabilities
        self.agent_capabilities_cache: Optional[str] = None

    async def select_next_agent(self) -> BaseAgent:
        """Select agent for the current step."""
        # Create plan on first iteration
        if not self.execution_plan:
            # Extract task from first message
            if self.shared_messages:
                self.initial_task = self.shared_messages[0].content
                self.execution_plan = await self.create_plan(self.initial_task)
            else:
                raise ValueError("No initial task found to create plan")

        # Check if we've completed all steps
        if self.current_step_index >= len(self.execution_plan.steps):
            # All steps completed - return fallback agent
            # BaseOrchestrator will handle termination through termination conditions
            return self.agents[0]

        current_step = self.execution_plan.steps[self.current_step_index]
        return self._find_agent_by_name(current_step.agent_name)

    async def prepare_context_for_agent(
        self, agent: BaseAgent
    ) -> Union[str, UserMessage, List[Message]]:
        """Prepare context including step-specific instructions and retry context."""
        if not self.execution_plan or self.current_step_index >= len(
            self.execution_plan.steps
        ):
            return self.shared_messages.copy()

        current_step = self.execution_plan.steps[self.current_step_index]

        # Get base context for this step
        context = self.extract_relevant_context(current_step)

        # Add step-specific task message
        step_message = UserMessage(
            content=self._format_step_task(current_step), source="plan_orchestrator"
        )
        context.append(step_message)

        return context

    async def update_shared_state(self, result: AgentResponse) -> None:
        """Update state and evaluate step progress for retry logic."""
        # Extract new messages from agent response (excluding UserMessage context)
        new_messages = [
            msg for msg in result.messages if not isinstance(msg, UserMessage)
        ]
        self.shared_messages.extend(new_messages)

        if not self.execution_plan or self.current_step_index >= len(
            self.execution_plan.steps
        ):
            return

        # Track attempt in runtime state (not in plan)
        if self.current_step_index not in self.step_attempts:
            self.step_attempts[self.current_step_index] = []
        self.step_attempts[self.current_step_index].append(result)

        # Evaluate step progress
        current_step = self.execution_plan.steps[self.current_step_index]
        progress_eval = await self.evaluate_step_progress(current_step, result)

        if progress_eval.step_completed:
            # Step completed successfully - move to next step
            self.step_results[self.current_step_index] = result
            self.current_step_index += 1
            self.current_step_retry_count = 0
        else:
            # Step failed - check if we should retry
            self.current_step_retry_count += 1

            if self.current_step_retry_count <= self.max_step_retries:
                # Prepare for retry
                self.retry_instructions[
                    self.current_step_index
                ] = self._create_retry_instructions(current_step, progress_eval)
            else:
                # Max retries exceeded - move on to next step or let termination handle it
                self.current_step_index += 1
                self.current_step_retry_count = 0

    async def create_plan(self, task: str) -> ExecutionPlan:
        """Create execution plan using LLM with structured output."""
        # Get agent capabilities summary
        capabilities = self.get_agent_capabilities_summary()

        planning_prompt = f"""You are a helpful assistant that breaks down tasks into executable steps.

Available agents and their capabilities:
{capabilities}

User task: {task}

Generate a concise set of step-by-step execution plan. For each step:
- Assign it to the agent best suited for that type of work
- Provide clear, actionable task description  
- Explain briefly why that agent was chosen

Keep it simple and focused. Multiple steps can use the same agent if appropriate.
The plans need not be too long - if only 2 or 3 steps are needed, that's perfectly fine.
"""

        try:
            # Use structured output like AI orchestrator does
            result = await self.model_client.create(
                messages=[UserMessage(content=planning_prompt, source="planner")],
                output_format=ExecutionPlan,
            )

            if result.structured_output and isinstance(
                result.structured_output, ExecutionPlan
            ):
                return result.structured_output
            else:
                # Fallback to simple plan
                return self._create_fallback_plan(task)

        except Exception as e:
            print(f"Warning: Plan generation failed: {e}")
            return self._create_fallback_plan(task)

    def get_agent_capabilities_summary(self) -> str:
        """Get agent capabilities summary with caching."""
        if self.agent_capabilities_cache is None:
            self.agent_capabilities_cache = super().get_agent_capabilities_summary()
        return self.agent_capabilities_cache

    def extract_relevant_context(self, _step: PlanStep) -> List[Message]:
        """Curate context for focused execution."""
        # For now, return recent messages - could be made more intelligent
        return (
            self.shared_messages[-5:]
            if len(self.shared_messages) > 5
            else self.shared_messages.copy()
        )

    async def evaluate_step_progress(
        self, step: PlanStep, result: AgentResponse
    ) -> StepProgressEvaluation:
        """Evaluate if step was completed successfully using LLM with structured output."""
        # Extract agent's output for evaluation
        agent_output = ""
        for msg in result.messages:
            if hasattr(msg, "content") and not isinstance(msg, UserMessage):
                agent_output += f"{msg.content}\n"

        if not agent_output.strip():
            return StepProgressEvaluation(
                step_completed=False,
                failure_reason="No meaningful output detected",
                confidence_score=0.9,
                suggested_improvements=[
                    "Provide more specific instructions",
                    "Add examples of expected output",
                ],
            )

        evaluation_prompt = f"""Evaluate whether the following step was successfully completed based on the agent's output.

Step Task: {step.task}
Expected Agent: {step.agent_name}
Reasoning: {step.reasoning}

Agent's Output:
{agent_output}

Evaluate:
1. Was the step task completed successfully?
2. If not, what was the main failure reason?
3. How confident are you in this assessment (0.0 to 1.0)?
4. If the step failed, provide 2-3 specific suggestions for improvement.

Consider the step successful if the agent made meaningful progress toward the stated goal, even if not perfect."""

        try:
            # Use LLM with structured output for evaluation
            eval_result = await self.model_client.create(
                messages=[
                    UserMessage(content=evaluation_prompt, source="step_evaluator")
                ],
                output_format=StepProgressEvaluation,
            )

            if eval_result.structured_output and isinstance(
                eval_result.structured_output, StepProgressEvaluation
            ):
                return eval_result.structured_output
            else:
                # Fallback evaluation
                return self._fallback_step_evaluation(agent_output)

        except Exception as e:
            print(f"Warning: LLM step evaluation failed: {e}")
            return self._fallback_step_evaluation(agent_output)

    def _fallback_step_evaluation(self, agent_output: str) -> StepProgressEvaluation:
        """Fallback heuristic evaluation when LLM fails."""
        # Simple heuristics
        output_lower = agent_output.lower()
        has_meaningful_content = len(agent_output.strip()) > 20
        has_error_indicators = any(
            word in output_lower
            for word in ["error", "failed", "cannot", "unable", "sorry"]
        )
        has_positive_indicators = any(
            word in output_lower
            for word in ["completed", "found", "created", "analyzed", "wrote"]
        )

        if has_meaningful_content and not has_error_indicators:
            return StepProgressEvaluation(
                step_completed=True,
                failure_reason="None",
                confidence_score=0.7,
                suggested_improvements=[],
            )
        else:
            return StepProgressEvaluation(
                step_completed=False,
                failure_reason="Output suggests task was not completed successfully",
                confidence_score=0.6,
                suggested_improvements=[
                    "Provide clearer instructions",
                    "Break task into smaller parts",
                    "Add specific examples",
                ],
            )

    def _create_fallback_plan(self, task: str) -> ExecutionPlan:
        """Create simple fallback plan when LLM planning fails."""
        return ExecutionPlan(
            steps=[
                PlanStep(
                    task=f"Complete the task: {task}",
                    agent_name=self.agents[0].name,
                    reasoning="Single step plan fallback",
                )
            ]
        )

    def _find_agent_by_name(self, name: str) -> BaseAgent:
        """Find agent by name with fuzzy matching."""
        name_lower = name.lower().strip()

        # Exact match first
        for agent in self.agents:
            if agent.name.lower() == name_lower:
                return agent

        # Partial match fallback
        for agent in self.agents:
            if name_lower in agent.name.lower() or agent.name.lower() in name_lower:
                return agent

        # No match found - return first agent as fallback
        print(
            f"Warning: Agent '{name}' not found, using fallback: {self.agents[0].name}"
        )
        return self.agents[0]

    def _format_step_task(self, step: PlanStep) -> str:
        """Format step task with retry context if applicable."""
        base_task = f"STEP {self.current_step_index + 1}: {step.task}"

        # Add retry instructions if this is a retry
        if (
            self.current_step_retry_count > 0
            and self.current_step_index in self.retry_instructions
        ):
            retry_info = self.retry_instructions[self.current_step_index]
            base_task += f"\n\nRETRY INSTRUCTIONS (Attempt {self.current_step_retry_count + 1}):\n{retry_info}"

        return base_task

    def _create_retry_instructions(
        self, _step: PlanStep, progress_eval: StepProgressEvaluation
    ) -> str:
        """Create enhanced instructions for retry attempts."""
        instructions = f"Previous attempt failed: {progress_eval.failure_reason or 'Unknown reason'}\n"

        # Add context about previous attempts
        attempt_count = len(self.step_attempts.get(self.current_step_index, []))
        if attempt_count > 0:
            instructions += f"This is retry attempt {attempt_count + 1}. Please try a different approach."

        return instructions

    def _get_pattern_metadata(self) -> Dict[str, Any]:
        """Get plan-specific metadata."""
        base_metadata = super()._get_pattern_metadata()

        if self.execution_plan:
            completed_steps = len(self.step_results)
            failed_steps = max(0, self.current_step_index - completed_steps)

            plan_metadata = {
                "plan": self.execution_plan,
                "current_step_index": self.current_step_index,
                "steps_completed": completed_steps,
                "steps_failed": failed_steps,
                "total_retries": sum(
                    len(attempts) - 1 for attempts in self.step_attempts.values()
                ),
                "current_step_retry_count": self.current_step_retry_count,
            }
            base_metadata.update(plan_metadata)

        return base_metadata

    def _reset_for_run(self) -> None:
        """Reset plan-based orchestrator state."""
        super()._reset_for_run()
        self.execution_plan = None
        self.current_step_index = 0
        self.current_step_retry_count = 0
        self.initial_task = None
        self.step_attempts = {}
        self.step_results = {}
        self.retry_instructions = {}
        self.agent_capabilities_cache = None

    def _to_config(self) -> PlanBasedOrchestratorConfig:
        """Convert to configuration for serialization."""
        # Serialize agents
        agent_configs = []
        for agent in self.agents:
            try:
                agent_configs.append(agent.dump_component())
            except NotImplementedError:
                continue

        # Serialize termination condition and model client
        termination_config = self.termination.dump_component()
        model_client_config = self.model_client.dump_component()

        return PlanBasedOrchestratorConfig(
            agents=agent_configs,
            termination=termination_config,
            model_client=model_client_config,
            max_iterations=self.max_iterations,
            max_step_retries=self.max_step_retries,
        )

    @classmethod
    def _from_config(
        cls, config: PlanBasedOrchestratorConfig
    ) -> "PlanBasedOrchestrator":
        """Create from configuration."""
        from ..agents import BaseAgent

        # Deserialize agents
        agents = []
        for agent_config in config.agents:
            try:
                agent = BaseAgent.load_component(agent_config)
                agents.append(agent)
            except Exception:
                continue

        # Deserialize termination condition and model client
        termination = BaseTermination.load_component(config.termination)
        model_client = BaseChatCompletionClient.load_component(config.model_client)

        return cls(
            agents=agents,
            termination=termination,
            model_client=model_client,
            max_iterations=config.max_iterations,
            max_step_retries=config.max_step_retries,
        )
