from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    """A structured evaluation score from the LLM judge.
    
    WHY Pydantic? Because we need reliable, parseable scores from the LLM.
    'overall: 7.5' is machine-readable. 'this is pretty good' is not.
    """
    score: float = Field(..., ge=0.0, le=10.0, description="Overall score 0-10")
    reasoning: str = Field(..., description="Explanation for the score")
    strengths: List[str] = Field(default_factory=list, description="What was done well")
    weaknesses: List[str] = Field(default_factory=list, description="What could improve")


class LLMJudge:
    """Uses a strong LLM to evaluate agent responses.
    
    WHY use an LLM as a judge?
    Because agent outputs are natural language and require nuanced evaluation.
    Traditional metrics like BLEU or ROUGE measure word overlap, not quality.
    An LLM judge can assess accuracy, helpfulness, clarity, and domain-specific criteria
    in the same way a human expert would.
    
    RUBRIC (from Chapter 10 of the book):
    - 9-10: Perfect. Completely correct, well-formatted, no issues.
    - 7-8:  Good. Minor issues but fundamentally correct.
    - 5-6:  Acceptable. Correct intent but significant problems.
    - 3-4:  Poor. Partially correct but major issues.
    - 1-2:  Bad. Mostly incorrect or unhelpful.
    - 0:    Completely wrong or harmful.
    
    IMPORTANT LESSON FROM THE BOOK (Section 10.6.3):
    Always add custom instructions to prevent the judge from penalizing
    multi-agent 'verbosity' — the judge should reward collaborative transparency,
    not penalize it as wordiness.
    """

    def __init__(
        self,
        judge_model_client,
        custom_instructions: str = "",
    ):
        self._judge = judge_model_client
        self._custom_instructions = custom_instructions or (
            "EVALUATION GUIDELINES:\n"
            "- If the answer is CORRECT and COMPLETE, score 8-10 regardless of length.\n"
            "- DO NOT PENALIZE multi-agent collaborative process visibility.\n"
            "- DO NOT PENALIZE showing reasoning steps or intermediate work.\n"
            "- Reward accuracy and helpfulness above brevity.\n"
        )

    async def score(
        self,
        task: str,
        response: str,
        expected_output: Optional[str] = None,
        criteria: Optional[List[str]] = None,
    ) -> JudgeScore:
        """Score an agent response on a task.
        
        Args:
            task: The original task given to the agent
            response: The agent's response to evaluate
            expected_output: The expected correct answer (if known)
            criteria: Specific aspects to evaluate (e.g., ["accuracy", "clarity"])
        """
        from ...messages import SystemMessage, UserMessage
        
        criteria_str = "\n".join(f"- {c}" for c in (criteria or ["accuracy", "helpfulness", "clarity"]))
        
        prompt_parts = [
            f"Task: {task}",
            f"Agent Response: {response}",
        ]
        
        if expected_output:
            prompt_parts.append(f"Expected Answer: {expected_output}")
        
        prompt_parts.append(f"\nEvaluate on:\n{criteria_str}")
        prompt_parts.append(
            "\nReturn JSON with: score (float 0-10), reasoning (string), "
            "strengths (list of strings), weaknesses (list of strings)."
        )
        
        messages = [
            SystemMessage(
                content=f"You are an expert evaluator. {self._custom_instructions}",
                source="system",
            ),
            UserMessage(
                content="\n".join(prompt_parts),
                source="user",
            ),
        ]
        
        result = await self._judge.create(messages=messages, output_format=JudgeScore)
        
        if result.structured_output:
            return result.structured_output
        
        # Fallback if structured output failed
        return JudgeScore(score=5.0, reasoning="Could not parse evaluation")
