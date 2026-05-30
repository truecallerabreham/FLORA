from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from ._base import BaseTool, ToolResult


class MemoryBackend:
    """Sandboxed file operations for agent memory storage.
    
    CRITICAL SECURITY REQUIREMENT: All paths are validated to stay within
    the base_path directory. Without this, an agent (or injected content)
    could try to access files outside the memory sandbox with paths like
    '../../../../etc/passwd'. This is a real attack vector called
    "directory traversal" or "path traversal."
    
    The _validate_path() method prevents this by:
    1. Resolving the path to its absolute form
    2. Checking it is still within base_path
    3. Raising an error if it has escaped
    """

    def __init__(self, base_path: Path):
        # .resolve() converts to absolute path and resolves symlinks
        self.base_path = base_path.resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, path: str) -> Path:
        """Prevent directory traversal attacks."""
        # Strip leading slash, join with base
        clean = path.lstrip("/")
        full_path = (self.base_path / clean).resolve()
        
        try:
            # relative_to() raises ValueError if full_path is not inside base_path
            full_path.relative_to(self.base_path)
        except ValueError:
            raise ValueError(
                f"Access denied: path '{path}' would escape the memory directory"
            )
        
        return full_path

    def view(self, path: str) -> str:
        """Read a file or list a directory's contents."""
        target = self._validate_path(path)
        
        if target.is_dir():
            items = sorted(target.iterdir())
            if not items:
                return f"Directory '{path}' is empty."
            lines = []
            for item in items:
                prefix = "[DIR] " if item.is_dir() else "[FILE]"
                lines.append(f"{prefix} {item.name}")
            return "\n".join(lines)
        
        elif target.is_file():
            return target.read_text(encoding="utf-8")
        
        else:
            return f"Path '{path}' does not exist."

    def create(self, path: str, file_text: str) -> str:
        """Create a new file with the given content."""
        target = self._validate_path(path)
        
        if target.exists():
            return f"Error: '{path}' already exists. Use str_replace to modify it."
        
        # Create parent directories if needed
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_text, encoding="utf-8")
        return f"Created '{path}' ({len(file_text)} characters)."

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Replace a unique string within a file."""
        target = self._validate_path(path)
        
        if not target.is_file():
            return f"Error: '{path}' is not a file."
        
        content = target.read_text(encoding="utf-8")
        
        if old_str not in content:
            return f"Error: The string to replace was not found in '{path}'."
        
        count = content.count(old_str)
        if count > 1:
            return (
                f"Error: The string appears {count} times in '{path}'. "
                f"Be more specific to avoid ambiguity."
            )
        
        target.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
        return f"Successfully replaced in '{path}'."

    def delete(self, path: str) -> str:
        """Delete a file."""
        target = self._validate_path(path)
        
        if target.is_file():
            target.unlink()
            return f"Deleted '{path}'."
        elif target.is_dir():
            return f"Error: '{path}' is a directory. Delete files individually."
        else:
            return f"Error: '{path}' does not exist."


class MemoryTool(BaseTool):
    """File-system memory for agents that need to learn across sessions.
    
    WHY this tool exists:
    Sometimes you want the AGENT to decide what to remember — not the application.
    A code reviewer that discovers a critical bug pattern should actively store it,
    organize it in its knowledge base, and retrieve it in future reviews.
    
    This tool gives the agent file-system operations (view, create, str_replace, delete)
    within a sandboxed directory. The agent can organize its memory however it wants:
    directories for different topics, markdown files for notes, etc.
    
    HOW the agent uses it:
    The agent's instructions should include something like:
    "ALWAYS check /memories on startup. Store important discoveries."
    
    On the first run, the agent creates memory files.
    On subsequent runs (with a fresh context but the same file system),
    it reads those files and applies the learned knowledge.
    
    This is different from BaseMemory (Part 6) which is application-managed.
    """

    def __init__(self, base_path: str = "./agent_memory"):
        super().__init__(
            name="memory",
            description=(
                "Manage your persistent memory. "
                "Commands: view (read file/directory), create (new file), "
                "str_replace (edit file), delete (remove file). "
                "ALWAYS check /memories at the start of every session."
            ),
        )
        self._backend = MemoryBackend(Path(base_path))

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "delete"],
                    "description": "The operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path (e.g., '/memories' or '/memories/bugs/race_condition.md')",
                },
                "file_text": {
                    "type": "string",
                    "description": "Content for the 'create' command",
                },
                "old_str": {
                    "type": "string",
                    "description": "Text to replace (for 'str_replace' command)",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text (for 'str_replace' command)",
                },
            },
            "required": ["command", "path"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        command = parameters.get("command")
        path = parameters.get("path", "/")
        
        try:
            if command == "view":
                result = self._backend.view(path)
            elif command == "create":
                file_text = parameters.get("file_text", "")
                result = self._backend.create(path, file_text)
            elif command == "str_replace":
                old_str = parameters.get("old_str", "")
                new_str = parameters.get("new_str", "")
                result = self._backend.str_replace(path, old_str, new_str)
            elif command == "delete":
                result = self._backend.delete(path)
            else:
                return ToolResult(success=False, error=f"Unknown command: '{command}'")
            
            return ToolResult(success=True, result=result)
        
        except ValueError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"Memory operation failed: {e}")
