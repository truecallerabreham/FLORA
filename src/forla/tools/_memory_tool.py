"""
Memory tool implementation for Forla.

Provides file-based memory storage similar to Anthropic's memory tool,
allowing agents to store and retrieve information across conversations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..types import ToolResult
from ._base import ApprovalMode, BaseTool


class MemoryBackend:
    """File-based memory storage backend with security controls."""

    def __init__(self, base_path: Union[str, Path] = "./memories"):
        """
        Initialize memory backend.

        Args:
            base_path: Root directory for memory storage
        """
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve a memory path.

        Args:
            path: Path to validate (should start with /memories or be relative)

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or attempts traversal attack
        """
        # Normalize path (remove /memories prefix if present)
        if path.startswith("/memories"):
            path = path[len("/memories") :]
        path = path.lstrip("/")

        # Resolve to absolute path
        full_path = (self.base_path / path).resolve()

        # Security: Ensure path is within base_path
        try:
            full_path.relative_to(self.base_path)
        except ValueError:
            raise ValueError(
                f"Access denied: path '{path}' is outside memory directory"
            )

        return full_path

    def view(
        self, path: str, view_range: Optional[List[int]] = None
    ) -> str:
        """
        View directory contents or file contents.

        Args:
            path: Path to view
            view_range: Optional [start_line, end_line] for file viewing

        Returns:
            String representation of directory listing or file contents
        """
        full_path = self._validate_path(path)

        # Directory listing
        if full_path.is_dir():
            contents = []
            contents.append(f"Directory: {path}")
            items = sorted(full_path.iterdir(), key=lambda x: x.name)

            if not items:
                contents.append("(empty)")
            else:
                for item in items:
                    prefix = "  - "
                    suffix = "/" if item.is_dir() else ""
                    contents.append(f"{prefix}{item.name}{suffix}")

            return "\n".join(contents)

        # File contents
        if full_path.is_file():
            content = full_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            # Apply line range if specified
            if view_range:
                start, end = view_range
                start = max(1, start)  # Line numbers are 1-indexed
                end = min(len(lines), end)
                lines = lines[start - 1 : end]
                # Add line numbers
                numbered_lines = [
                    f"{i + start:5d}: {line}"
                    for i, line in enumerate(lines)
                ]
                return "\n".join(numbered_lines)

            # Return full file with line numbers
            numbered_lines = [
                f"{i + 1:5d}: {line}" for i, line in enumerate(lines)
            ]
            return "\n".join(numbered_lines)

        # Path doesn't exist
        raise FileNotFoundError(f"Path not found: {path}")

    def create(self, path: str, file_text: str) -> str:
        """
        Create or overwrite a file.

        Args:
            path: Path to file
            file_text: Content to write

        Returns:
            Success message
        """
        full_path = self._validate_path(path)

        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        full_path.write_text(file_text, encoding="utf-8")

        return f"File created successfully at {path}"

    def str_replace(
        self, path: str, old_str: str, new_str: str
    ) -> str:
        """
        Replace text in a file.

        Args:
            path: Path to file
            old_str: Text to find
            new_str: Replacement text

        Returns:
            Success message
        """
        full_path = self._validate_path(path)

        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        content = full_path.read_text(encoding="utf-8")

        if old_str not in content:
            raise ValueError(
                f"Text not found in file: '{old_str[:50]}...'"
            )

        new_content = content.replace(old_str, new_str, 1)
        full_path.write_text(new_content, encoding="utf-8")

        return f"File {path} has been edited successfully"

    def insert(
        self, path: str, insert_line: int, insert_text: str
    ) -> str:
        """
        Insert text at a specific line.

        Args:
            path: Path to file
            insert_line: Line number to insert at (1-indexed)
            insert_text: Text to insert

        Returns:
            Success message
        """
        full_path = self._validate_path(path)

        if not full_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        lines = full_path.read_text(encoding="utf-8").splitlines(
            keepends=True
        )

        # Ensure insert_text ends with newline
        if not insert_text.endswith('\n'):
            insert_text += '\n'

        # Insert at specified line (1-indexed)
        insert_line = max(1, min(insert_line, len(lines) + 1))
        lines.insert(insert_line - 1, insert_text)

        full_path.write_text("".join(lines), encoding="utf-8")

        return f"Text inserted at line {insert_line} in {path}"

    def delete(self, path: str) -> str:
        """
        Delete a file or directory.

        Args:
            path: Path to delete

        Returns:
            Success message
        """
        full_path = self._validate_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if full_path.is_file():
            full_path.unlink()
            return f"File deleted: {path}"
        elif full_path.is_dir():
            # Remove directory (must be empty for safety)
            if any(full_path.iterdir()):
                raise ValueError(
                    f"Directory not empty: {path}. "
                    "Delete contents first."
                )
            full_path.rmdir()
            return f"Directory deleted: {path}"

    def rename(self, old_path: str, new_path: str) -> str:
        """
        Rename or move a file/directory.

        Args:
            old_path: Current path
            new_path: New path

        Returns:
            Success message
        """
        old_full_path = self._validate_path(old_path)
        new_full_path = self._validate_path(new_path)

        if not old_full_path.exists():
            raise FileNotFoundError(f"Path not found: {old_path}")

        if new_full_path.exists():
            raise ValueError(f"Destination already exists: {new_path}")

        # Create parent directories for destination
        new_full_path.parent.mkdir(parents=True, exist_ok=True)

        old_full_path.rename(new_full_path)

        return f"Renamed {old_path} to {new_path}"

    def search(self, query: str, path: str = "/memories") -> str:
        """
        Search for text across memory files.

        Args:
            query: Text to search for (case-insensitive)
            path: Directory to search in (default: /memories)

        Returns:
            Formatted string with search results
        """
        full_path = self._validate_path(path)

        if not full_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if not full_path.is_dir():
            raise ValueError(f"Path must be a directory: {path}")

        matches = []
        query_lower = query.lower()

        # Search all files recursively
        for file_path in full_path.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    lines = content.splitlines()

                    for line_num, line in enumerate(lines, 1):
                        if query_lower in line.lower():
                            rel_path = file_path.relative_to(self.base_path)
                            matches.append({
                                "file": str(rel_path),
                                "line": line_num,
                                "content": line.strip()
                            })
                except (UnicodeDecodeError, PermissionError):
                    # Skip files that can't be read
                    continue

        if not matches:
            return f"No matches found for '{query}' in {path}"

        # Format results
        result_lines = [f"Found {len(matches)} match(es) for '{query}':\n"]
        for match in matches[:50]:  # Limit to 50 results
            result_lines.append(
                f"  {match['file']}:{match['line']} - {match['content'][:80]}"
            )

        if len(matches) > 50:
            result_lines.append(f"\n... and {len(matches) - 50} more matches")

        return "\n".join(result_lines)

    def append(self, path: str, text: str) -> str:
        """
        Append text to the end of a file.

        Args:
            path: Path to file
            text: Text to append

        Returns:
            Success message
        """
        full_path = self._validate_path(path)

        # Create file if it doesn't exist
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text("", encoding="utf-8")

        # Ensure text starts with newline if file isn't empty
        if full_path.stat().st_size > 0:
            existing_content = full_path.read_text(encoding="utf-8")
            if existing_content and not existing_content.endswith('\n'):
                text = '\n' + text

        # Ensure text ends with newline
        if not text.endswith('\n'):
            text += '\n'

        # Append to file
        with full_path.open('a', encoding='utf-8') as f:
            f.write(text)

        return f"Text appended to {path}"


class MemoryTool(BaseTool):
    """
    Memory tool for storing and retrieving information across conversations.

    Provides file-based operations similar to Anthropic's memory tool:
    - view: Show directory/file contents
    - create: Create or overwrite files
    - str_replace: Edit file contents
    - insert: Insert text at specific line
    - delete: Remove files/directories
    - rename: Rename or move files/directories
    - search: Search for text across all files
    - append: Append text to end of file

    Example:
        ```python
        from forla import Agent
        from forla.tools import MemoryTool

        memory = MemoryTool(base_path="./agent_memory")

        agent = Agent(
            name="assistant",
            instructions=(
                "IMPORTANT: ALWAYS check your memory directory "
                "at the start of each task using the memory tool. "
                "Store important patterns and insights for future reference."
            ),
            model_client=client,
            tools=[memory]
        )
        ```
    """

    def __init__(
        self,
        base_path: Union[str, Path] = "./memories",
        approval_mode: ApprovalMode = ApprovalMode.NEVER,
    ):
        """
        Initialize memory tool.

        Args:
            base_path: Root directory for memory storage
            approval_mode: Whether to require approval for operations
                          (NEVER = no approval, ALWAYS = require for all ops)
        """
        super().__init__(
            name="memory",
            description=(
                "Store and retrieve information in memory files that "
                "persist across conversations. Use this to remember "
                "important patterns, insights, and context. "
                "Operations: view (show directory/file), create (new file), "
                "str_replace (edit file), insert (add text at line), "
                "delete (remove file/dir), rename (move/rename file), "
                "search (find text in files), append (add to end of file)."
            ),
            version="1.0.0",
            approval_mode=approval_mode,
        )
        self.backend = MemoryBackend(base_path)

    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for memory tool parameters."""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": [
                        "view",
                        "create",
                        "str_replace",
                        "insert",
                        "delete",
                        "rename",
                        "search",
                        "append",
                    ],
                    "description": "Operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Path to file or directory "
                        "(e.g., '/memories/notes.md' or 'patterns/bugs.xml')"
                    ),
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": (
                        "Optional [start_line, end_line] "
                        "for viewing specific lines (1-indexed)"
                    ),
                },
                "file_text": {
                    "type": "string",
                    "description": (
                        "Content to write when creating a file"
                    ),
                },
                "old_str": {
                    "type": "string",
                    "description": "Text to find and replace",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text",
                },
                "insert_line": {
                    "type": "integer",
                    "description": (
                        "Line number to insert text at (1-indexed)"
                    ),
                },
                "insert_text": {
                    "type": "string",
                    "description": "Text to insert",
                },
                "old_path": {
                    "type": "string",
                    "description": "Current path (for rename operation)",
                },
                "new_path": {
                    "type": "string",
                    "description": "New path (for rename operation)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query text (for search operation)",
                },
                "append_text": {
                    "type": "string",
                    "description": "Text to append (for append operation)",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self, parameters: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute memory operation.

        Args:
            parameters: Operation parameters matching JSON schema

        Returns:
            ToolResult with operation outcome
        """
        command = parameters["command"]

        try:
            if command == "view":
                path = parameters.get("path", "/memories")
                view_range = parameters.get("view_range")
                result = self.backend.view(path, view_range)
                metadata = {
                    "command": "view",
                    "path": path,
                    "lines": len(result.splitlines()),
                }

            elif command == "create":
                path = parameters["path"]
                file_text = parameters["file_text"]
                result = self.backend.create(path, file_text)
                metadata = {
                    "command": "create",
                    "path": path,
                    "size": len(file_text),
                }

            elif command == "str_replace":
                path = parameters["path"]
                old_str = parameters["old_str"]
                new_str = parameters["new_str"]
                result = self.backend.str_replace(path, old_str, new_str)
                metadata = {
                    "command": "str_replace",
                    "path": path,
                }

            elif command == "insert":
                path = parameters["path"]
                insert_line = parameters["insert_line"]
                insert_text = parameters["insert_text"]
                result = self.backend.insert(
                    path, insert_line, insert_text
                )
                metadata = {
                    "command": "insert",
                    "path": path,
                    "line": insert_line,
                }

            elif command == "delete":
                path = parameters["path"]
                result = self.backend.delete(path)
                metadata = {"command": "delete", "path": path}

            elif command == "rename":
                old_path = parameters["old_path"]
                new_path = parameters["new_path"]
                result = self.backend.rename(old_path, new_path)
                metadata = {
                    "command": "rename",
                    "old_path": old_path,
                    "new_path": new_path,
                }

            elif command == "search":
                query = parameters["query"]
                path = parameters.get("path", "/memories")
                result = self.backend.search(query, path)
                metadata = {
                    "command": "search",
                    "query": query,
                    "path": path,
                }

            elif command == "append":
                path = parameters["path"]
                append_text = parameters["append_text"]
                result = self.backend.append(path, append_text)
                metadata = {
                    "command": "append",
                    "path": path,
                    "text_length": len(append_text),
                }

            else:
                raise ValueError(f"Unknown command: {command}")

            return ToolResult(
                success=True,
                result=result,
                error=None,
                metadata=metadata,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Memory operation failed: {str(e)}",
                metadata={"command": command},
            )


# Export
__all__ = ["MemoryTool", "MemoryBackend"]
