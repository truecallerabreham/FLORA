from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .judges._llm_judge import LLMJudge, JudgeScore


@dataclass
class EvalTask:
    """A single test case for evaluation."""
    task_id: str
    input: str
    expected_output: Optional[str] = None
    criteria: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """The result of evaluating a single task."""
    task_id: str
    score: float
    agent_response: str
    reasoning: str
    strengths: List[str]
    weaknesses: List[str]
    usage_tokens: int = 0
    duration_ms: int = 0


class EvaluationRunner:
    """Runs evaluation suites against agents and scores the results.
    
    USAGE:
        runner = EvaluationRunner(judge=my_judge)
        tasks = [
            EvalTask("q1", "What is 2+2?", expected_output="4"),
            EvalTask("q2", "What is the capital of France?", expected_output="Paris"),
        ]
        results = await runner.evaluate(agent, tasks)
        summary = runner.summarize(results)
        print(f"Average score: {summary['avg_score']:.1f}/10")
    """

    def __init__(self, judge: LLMJudge):
        self._judge = judge

    async def evaluate(
        self,
        agent,
        tasks: List[EvalTask],
        concurrency: int = 3,
    ) -> List[EvalResult]:
        """Evaluate the agent on all tasks.
        
        'concurrency' limits how many tasks run in parallel.
        Higher concurrency is faster but uses more API rate limit.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(task: EvalTask) -> EvalResult:
            async with semaphore:
                import time
                start = time.time()
                
                # Run the agent
                response = await agent.run(task.input)
                
                duration_ms = int((time.time() - start) * 1000)
                usage = response.usage.tokens_input + response.usage.tokens_output

                # Score the response
                score = await self._judge.score(
                    task=task.input,
                    response=response.content,
                    expected_output=task.expected_output,
                    criteria=task.criteria,
                )

                return EvalResult(
                    task_id=task.task_id,
                    score=score.score,
                    agent_response=response.content,
                    reasoning=score.reasoning,
                    strengths=score.strengths,
                    weaknesses=score.weaknesses,
                    usage_tokens=usage,
                    duration_ms=duration_ms,
                )

        results = await asyncio.gather(*[run_one(t) for t in tasks])
        return list(results)

    def summarize(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Generate a summary of evaluation results."""
        if not results:
            return {}
        
        scores = [r.score for r in results]
        
        return {
            "total_tasks": len(results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "pass_rate": sum(1 for s in scores if s >= 7.0) / len(scores),
            "total_tokens": sum(r.usage_tokens for r in results),
            "avg_duration_ms": sum(r.duration_ms for r in results) / len(results),
            "scores_by_task": {r.task_id: r.score for r in results},
        }
