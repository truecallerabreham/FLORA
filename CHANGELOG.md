# Changelog

All notable changes to Forla will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-05

### Added

- **Deterministic loop hooks** (`_hooks.py`): `BaseStartHook` and `BaseEndHook` for injecting instructions before the first LLM call and resuming the loop when the agent would stop. Includes `PlanningHook` (forces task planning) and `CompletionCheckHook` (verifies todo completion before allowing stop). Composable `TerminationCondition` with `|` / `&` operators.
- **Context compaction strategies** (`compaction.py`): `CompactionStrategy` protocol called before each LLM call. `HeadTailCompaction` preserves system prompt + recent work while dropping middle messages, respecting atomic groups (tool calls + results stay together). `SlidingWindowCompaction` keeps system message + most recent messages. `NoCompaction` baseline for benchmarking. Token counting via tiktoken with character-estimation fallback.
- **Context engineering tools** (`tools/_context_tools.py`): `TaskTool` for spawning sub-agents in isolated contexts (explore, research, general types). `TodoWriteTool` / `TodoReadTool` / `TodoListSessionsTool` for file-backed task tracking with session management. `SkillsTool` for progressive disclosure via SKILL.md files. `MultiEditTool` for atomic multi-file edits. Factory function `create_context_engineering_tools()`.
- **Benchmark CLI system** (`eval/benchmarks/`): `forla benchmark list|run|results` CLI commands. Built-in datasets (`coding_v1`, `repo_analysis_v1`). `AgentConfig` for declarative agent configuration comparison. `BenchmarkRunner`, `BenchmarkMiddleware`, and analysis formatters. Targets: `ForlaAgentTarget`, `ClaudeCodeTarget`, `CallableTarget`.
- **Instruction presets** (`_instructions.py`): `get_instructions()` for loading reusable system prompt templates.
- **Built-in skills** (`skills/`): Shipped code-review skill with SKILL.md frontmatter format.
- **Progressive implementation examples** (`code_along/`): Four files building a minimal agent from zero - core loop, tools, memory, streaming - with API compatibility test.

### Changed

- Agent `__init__` accepts `compaction`, `start_hooks`, and `end_hooks` parameters
- Agent tool loop integrates compaction (before each LLM call) and hooks (before first call, on exit)
- Renamed `context_strategy` parameter to `compaction` for clarity
- Renamed strategy classes: `HeadTailStrategy` → `HeadTailCompaction`, `SlidingWindowStrategy` → `SlidingWindowCompaction`, `NoCompactionStrategy` → `NoCompaction`, `ContextStrategy` → `CompactionStrategy`

## [0.3.2] - 2025-12-28

### Fixed

- Fixed `AgentResponse.__str__()` to display full conversation content instead of only metadata. Previously, printing a response would only show usage statistics; now it shows all messages followed by usage stats.

### Changed

- Improved user experience when printing agent responses - users can now simply `print(response)` to see the complete conversation history.

## [0.3.1] - 2025-11-23

### Added

- Anthropic Claude model client with full API support
- GitHub Models integration example
- Agent-as-tool result strategies for flexible output handling
- Context engineering examples

### Changed

- Updated model client tests for better coverage and reliability
- Improved workflow integration tests with proper Pydantic model handling
- Enhanced WebUI workflow view components

### Fixed

- Fixed Anthropic model name in tests (claude-3-5-haiku-20241022)
- Fixed CancellationToken import path in workflow tests
- Fixed workflow step progress events test with correct function signatures
- Updated OpenAI test to use gpt-4.1-mini model

## [0.3.0] - 2025-11-10

### Added

- Model Context Protocol (MCP) integration with complete client implementation
  - MCP client, configuration, and transport layers
  - MCP tool wrapper for seamless integration with Forla tools
  - Examples and comprehensive test coverage
- Software Engineering (SWE) agent implementation with full documentation
- Enhanced evaluation framework with comprehensive evaluation system
  - Expected answer generation utilities
  - Results tracking and visualization
  - Updated composite and LLM judges
- YouTube caption tool for extracting transcripts
- List memory example demonstrating memory management patterns
- Context inspector component in Web UI for debugging agent context
- Message handling and entity execution hooks in Web UI frontend
- Workflow progress tracking with dedicated test coverage
- Premium samples collection with documentation

### Changed

- Enhanced research tools with improved capabilities
- Improved Web UI execution handling and state management
- Updated agent and orchestration examples with better patterns
- Refined LLM client implementations (OpenAI and Azure OpenAI) for better error handling
- Improved workflow runner with enhanced progress reporting
- Updated evaluation results with new metrics and visualizations
- Enhanced memory tool with better examples

### Fixed

- Test mocks now properly use AgentContext matching real Agent behavior
- Web UI frontend dependency updates for security and compatibility
- Tool initialization and registration improvements
- Message handling in agent communication

## [0.2.3] - 2025-10-22

### Added

- OpenTelemetry integration following Gen-AI semantic conventions with automatic instrumentation
- Workflow checkpoint system with file and memory storage backends for state persistence
- Tool approval system with `@tool` decorator and `ApprovalMode` support for human-in-the-loop workflows
- Enhanced middleware pipeline with approval flow hooks and tool execution monitoring
- Context management improvements with agent-specific context support
- Memory tools for persistent agent memory across sessions
- Poethepoet task automation (run `poe test`, `poe check`, etc.)
- Example tasks display component in Web UI
- Tool approval banner in Web UI for interactive approval workflows
- Comprehensive examples for memory management, OpenTelemetry, tool approval, and checkpointing

### Changed

- Improved Web UI debug panel with detailed execution traces
- Enhanced middleware system with better error handling and event emission
- Reorganized test structure: workflow tests moved to `tests/workflow/`
- Updated frontend build artifacts with latest React components

### Removed

- Deprecated planning tools module (functionality moved to core tools)
- Old workflow test files from `src/forla/workflow/tests/`

## [0.2.2] - 2025-10-11

### Changed

- Update default model to gpt-4.1-mini
- Add examples gallery to web UI for browsing and loading sample agents

## [0.2.1] - 2025-10-11

### Changed

- Update README documentation

## [0.2.0] - 2025-10-11

### Added

- Web UI integration with auto-discovery of agents, workflows, and orchestrators
- Updated examples directory structure

### Changed

- Moved examples from forla/examples to root examples/ directory for better organization

## [0.1.2] - 2024

### Initial Release

- Core agent implementation with tool support
- Workflow engine with DAG-based execution
- Orchestration patterns (round-robin, AI-driven, plan-based)
- Memory management system
- Evaluation framework
- Comprehensive test suite
