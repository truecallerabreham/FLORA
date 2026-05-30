# Memory Tool for Forla

## Overview

The `MemoryTool` provides file-based memory storage that persists across conversations, similar to Anthropic's memory tool. Agents can actively manage their own memory by creating, reading, editing, and organizing information in files.

## Key Features

- **Cross-conversation learning**: Memory persists between agent sessions
- **Agent-controlled**: Agents actively manage their memory through tool calls
- **File-based operations**: Create, view, edit, delete, and rename memory files
- **Organized storage**: Directory structure for categorizing information
- **Security**: Path traversal protection prevents unauthorized access
- **Streaming support**: Compatible with Forla' streaming architecture

## Installation

The memory tool is included with Forla. No additional dependencies required.

```python
from forla.tools import MemoryTool
```

## Quick Start

### Basic Usage

```python
from forla import Agent
from forla.llm import AzureOpenAIChatCompletionClient
from forla.tools import MemoryTool

# Create memory tool
memory = MemoryTool(base_path="./agent_memory")

# Create agent with memory
agent = Agent(
    name="assistant",
    instructions="""You are a helpful assistant.

IMPORTANT: ALWAYS check your memory directory first:
  Use memory(command="view", path="/memories")

Store important patterns and insights for future reference.""",
    model_client=AzureOpenAIChatCompletionClient(model="gpt-4.1-mini"),
    tools=[memory]
)

# Agent can now use memory across conversations
response = await agent.run("Help me debug this code")
```

### Cross-Conversation Learning

```python
# Session 1: Agent learns pattern
response1 = await agent.run("""
Review this code with a race condition:
```python
self.results = []
for future in as_completed(futures):
    self.results.append(future.result())  # RACE!
```
""")

# Session 2: New conversation, agent applies learned pattern
agent.context = AgentContext()  # Reset context
response2 = await agent.run("""
Review this async code:
```python
self.responses = []
async for item in items:
    self.responses.append(item)  # Similar pattern?
```
""")
# Agent will check memory first and apply stored knowledge
```

## Operations

### 1. View Directory or File

Show directory contents or file contents with optional line ranges.

```python
# View directory
memory(command="view", path="/memories")
# Output: Directory: /memories
#         - notes.md
#         - patterns/

# View file
memory(command="view", path="/memories/notes.md")
# Output:    1: # Meeting Notes
#            2: - Discussed timeline
#            3: - Next steps

# View specific lines
memory(command="view", path="/memories/notes.md", view_range=[2, 3])
# Output:    2: - Discussed timeline
#            3: - Next steps
```

### 2. Create File

Create or overwrite a file.

```python
memory(
    command="create",
    path="/memories/patterns/singleton.md",
    file_text="# Singleton Pattern\n\nEnsure only one instance..."
)
```

### 3. Edit File (str_replace)

Replace text in a file.

```python
memory(
    command="str_replace",
    path="/memories/notes.md",
    old_str="- Discussed timeline",
    new_str="- Discussed project timeline and deadlines"
)
```

### 4. Insert Text

Insert text at a specific line number.

```python
memory(
    command="insert",
    path="/memories/notes.md",
    insert_line=2,
    insert_text="- Team: Alice, Bob, Carol\n"
)
```

### 5. Delete File or Directory

Remove files or empty directories.

```python
# Delete file
memory(command="delete", path="/memories/temp.md")

# Delete empty directory
memory(command="delete", path="/memories/old_patterns")
```

### 6. Rename or Move

Rename or move files and directories.

```python
memory(
    command="rename",
    old_path="/memories/draft.md",
    new_path="/memories/final.md"
)

# Move to subdirectory
memory(
    command="rename",
    old_path="/memories/note.md",
    new_path="/memories/archive/note.md"
)
```

## Agent Instructions

For best results, include memory instructions in your agent's system prompt:

```python
agent = Agent(
    instructions="""You are an expert code reviewer.

MEMORY PROTOCOL:
1. ALWAYS check your memory directory FIRST using:
   memory(command="view", path="/memories")

2. Review relevant files before starting work

3. Store important patterns and insights:
   memory(command="create", path="/memories/patterns/bug_type.md", file_text="...")

4. Update memory as you learn:
   memory(command="str_replace", path="...", old_str="...", new_str="...")

5. Organize memory into directories:
   - /memories/patterns/ - Code patterns and best practices
   - /memories/bugs/ - Known bugs and fixes
   - /memories/projects/ - Project-specific notes

ASSUME INTERRUPTION: Your conversation may be reset at any time.
Only information in memory will persist.""",
    tools=[memory]
)
```

## Memory Organization Best Practices

### Directory Structure

Organize memory into logical categories:

```
/memories/
  ├── patterns/          # Code patterns and best practices
  │   ├── singleton.md
  │   └── race_conditions.md
  ├── bugs/             # Known bugs and fixes
  │   ├── thread_safety.md
  │   └── async_issues.md
  ├── projects/         # Project-specific context
  │   ├── project_a.md
  │   └── project_b.md
  └── users/            # User preferences
      └── preferences.md
```

### File Naming

- Use descriptive names: `race_condition_fix.md` not `fix.md`
- Use underscores or hyphens: `user_preferences.md`
- Include dates for versioned info: `meeting_2025_01_15.md`

### Content Format

Use structured formats for easy parsing:

```markdown
# Bug Pattern: Race Condition

## Symptom
Inconsistent results in concurrent operations

## Cause
Multiple threads/coroutines modifying shared state without synchronization

## Solution
1. Use thread-safe data structures (Queue, Lock)
2. Avoid shared mutable state
3. Use futures to collect results

## Examples
- Web scraper with self.results.append()
- API client with self.error_count += 1
```

## Security Considerations

### Path Traversal Protection

The memory backend validates all paths to prevent directory traversal attacks:

```python
# These are blocked:
memory(command="view", path="../../etc/passwd")  # Error!
memory(command="view", path="/memories/../../../secret")  # Error!

# These work:
memory(command="view", path="/memories/notes.md")  # ✓
memory(command="view", path="notes.md")  # ✓
```

### Approval Mode

For sensitive applications, require approval for all memory operations:

```python
memory = MemoryTool(
    base_path="./memory",
    approval_mode=ApprovalMode.ALWAYS  # Require user approval
)
```

### Sensitive Information

Agents should avoid storing sensitive data. Add instructions:

```python
instructions="""
DO NOT store in memory:
- Passwords or API keys
- Personal identifiable information (PII)
- Credit card numbers
- Confidential business data

Store only:
- Patterns and insights
- Non-sensitive user preferences
- Generic knowledge and best practices
"""
```

## Advanced Usage

### Custom Memory Backend

Extend `MemoryBackend` for custom storage:

```python
from forla.tools import MemoryBackend

class DatabaseMemoryBackend(MemoryBackend):
    """Store memory in a database instead of files."""

    def __init__(self, db_connection):
        super().__init__(base_path="./memories")
        self.db = db_connection

    def view(self, path, view_range=None):
        # Query database instead of file
        return self.db.query(path)

    def create(self, path, file_text):
        # Store in database
        self.db.insert(path, file_text)
        return f"Stored in database: {path}"
```

### Integration with Orchestration

Memory tools work with all orchestration patterns:

```python
from forla.orchestration import RoundRobinOrchestrator

# Create agents with shared memory backend
shared_memory = MemoryBackend(base_path="./team_memory")

researcher = Agent(
    name="researcher",
    tools=[MemoryTool(base_path=shared_memory.base_path)]
)

writer = Agent(
    name="writer",
    tools=[MemoryTool(base_path=shared_memory.base_path)]
)

# Orchestrator coordinates agents with shared memory
orchestrator = RoundRobinOrchestrator(
    agents=[researcher, writer],
    termination=MaxMessageTermination(10)
)
```

## Comparison with Anthropic's Memory Tool

| Feature | Anthropic | Forla MemoryTool |
|---------|-----------|----------------------|
| **Storage** | Client-side (you manage) | File-based (local) |
| **Operations** | 6 commands | 6 commands (same) |
| **Persistence** | Cross-session | Cross-session |
| **Path Security** | Required | Built-in |
| **Agent Control** | Full (creates/edits files) | Full (creates/edits files) |
| **Organization** | Directory-based | Directory-based |
| **Auto-prompting** | Built-in system prompt | Manual (add to instructions) |
| **Backend** | Your implementation | FileSystem (extensible) |

## Examples

See [examples/tools/memory_tool_example.py](../examples/tools/memory_tool_example.py) for complete examples:

- Code review with cross-session learning
- All memory operations demonstrated
- Memory organization patterns

## API Reference

### MemoryTool

```python
class MemoryTool(BaseTool):
    def __init__(
        self,
        base_path: Union[str, Path] = "./memories",
        approval_mode: ApprovalMode = ApprovalMode.NEVER
    )
```

**Parameters:**
- `base_path`: Root directory for memory storage (default: "./memories")
- `approval_mode`: Require approval for operations (NEVER or ALWAYS)

### MemoryBackend

```python
class MemoryBackend:
    def __init__(self, base_path: Union[str, Path] = "./memories")

    def view(self, path: str, view_range: Optional[List[int]] = None) -> str
    def create(self, path: str, file_text: str) -> str
    def str_replace(self, path: str, old_str: str, new_str: str) -> str
    def insert(self, path: str, insert_line: int, insert_text: str) -> str
    def delete(self, path: str) -> str
    def rename(self, old_path: str, new_path: str) -> str
```

**Security:**
- All paths validated to prevent traversal attacks
- Paths resolved relative to `base_path`
- Attempts to access parent directories raise `ValueError`

## Testing

Run the test suite:

```bash
pytest tests/test_memory_tool.py -v
```

Tests cover:
- All memory operations
- Path security validation
- Error handling
- Tool integration

## Best Practices

1. **Always check memory first**: Instruct agents to view `/memories` before tasks
2. **Organize by category**: Use directories for different types of information
3. **Structured formats**: Use markdown or XML for easy parsing
4. **Regular cleanup**: Delete outdated information
5. **Security**: Never store credentials or PII
6. **Error handling**: Agents should gracefully handle missing files
7. **Versioning**: Include dates or version numbers in filenames

## Troubleshooting

### Memory Not Persisting

**Problem**: Agent doesn't remember previous conversations

**Solutions**:
- Ensure agent instructions include memory check
- Verify `base_path` is consistent across sessions
- Check that files are actually created (inspect directory)

### Path Errors

**Problem**: `ValueError: Access denied: path outside memory directory`

**Solution**: Use relative paths or paths starting with `/memories`:
```python
# Good
memory(command="view", path="/memories/notes.md")
memory(command="view", path="notes.md")

# Bad
memory(command="view", path="../../notes.md")
```

### Performance

**Problem**: Large memory directories slow down view operations

**Solutions**:
- Organize into subdirectories
- Implement custom backend with indexing
- Periodically archive old information

## License

Part of Forla framework. See repository LICENSE for details.
