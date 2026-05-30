"""
Evaluation system for forla.

This module provides a complete evaluation framework for testing
and comparing different forla components (agents, models, orchestrators).

    >>> from forla.eval import (
    ...     AgentConfig, EvalRunner, Dataset, EvalResults,
    ...     ForlaAgentTarget, LLMEvalJudge, load_builtin_dataset,
    ... )
    >>>
    >>> # Load dataset
    >>> dataset = load_builtin_dataset("coding_v1")
    >>>
    >>> # Define configurations to compare
    >>> configs = [
    ...     AgentConfig(name="baseline", compaction=None),
    ...     AgentConfig(name="head_tail", compaction="head_tail"),
    ... ]
    >>>
    >>> # Run evaluation
    >>> runner = EvalRunner(judge=LLMEvalJudge(model_client))
    >>> results = await runner.run(dataset, configs)
    >>>
    >>> # Analyze results
    >>> print_results(results)
"""

# Base classes
from ._base import EvalJudge, Target

# Runner
from ._runner import EvalRunner, Runnable

# Targets
from ._targets import (
    AgentEvalTarget,
    CallableTarget,
    ClaudeCodeTarget,
    ModelEvalTarget,
    OrchestratorEvalTarget,
    ForlaAgentTarget,
)

# Judges
from .judges import (
    BaseEvalJudge,
    CompositeJudge,
    ContainsJudge,
    ExactMatchJudge,
    FuzzyMatchJudge,
    LLMEvalJudge,
)

# Dataset
from ._dataset import Dataset, list_builtin_datasets, load_builtin_dataset

# Config
from ._config import AgentConfig

# Results
from ._results import (
    EvalResults,
    TaskResult,
    TargetSummary,
    list_eval_results,
    load_eval_results,
)

# Middleware
from ._middleware import RunMiddleware

# Analysis
from ._analysis import (
    format_file_read_analysis,
    format_summary_table,
    format_task_breakdown,
    format_token_growth,
    print_results,
)

__all__ = [
    # Base classes
    "Target",
    "EvalJudge",
    # Runner
    "EvalRunner",
    "Runnable",
    # Targets
    "AgentEvalTarget",
    "ModelEvalTarget",
    "OrchestratorEvalTarget",
    "ForlaAgentTarget",
    "ClaudeCodeTarget",
    "CallableTarget",
    # Judges
    "BaseEvalJudge",
    "LLMEvalJudge",
    "ExactMatchJudge",
    "FuzzyMatchJudge",
    "ContainsJudge",
    "CompositeJudge",
    # Dataset
    "Dataset",
    "load_builtin_dataset",
    "list_builtin_datasets",
    # Config
    "AgentConfig",
    # Results
    "TaskResult",
    "TargetSummary",
    "EvalResults",
    "load_eval_results",
    "list_eval_results",
    # Middleware
    "RunMiddleware",
    # Analysis
    "format_summary_table",
    "format_task_breakdown",
    "format_file_read_analysis",
    "format_token_growth",
    "print_results",
]
