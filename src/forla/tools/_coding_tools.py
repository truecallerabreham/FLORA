"""
Coding tools for file operations, code execution, and development tasks.

These tools enable agents to read/write files, execute code, run tests,
and perform common development operations with workspace isolation.
"""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..types import ToolResult
from ._base import BaseTool


class ReadFileTool(BaseTool):
    """Read content from a file."""

    def __init__(self, workspace: Optional[Path] = None) -> None:
        super().__init__(
            name="read_file",
            description="Read the contents of a file. Returns the file content as a string.",
        )
        self.workspace = workspace or Path.cwd()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to workspace)",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        file_path = parameters["file_path"]
        encoding = parameters.get("encoding", "utf-8")

        try:
            full_path = (self.workspace / file_path).resolve()

            if not str(full_path).startswith(str(self.workspace.resolve())):
                raise ValueError("Access denied: path outside workspace")

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if not full_path.is_file():
                raise ValueError(f"Not a file: {file_path}")

            content = full_path.read_text(encoding=encoding)

            return ToolResult(
                success=True,
                result=content,
                error=None,
                metadata={
                    "file_path": file_path,
                    "size": len(content),
                    "lines": len(content.splitlines()),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Failed to read file: {str(e)}",
                metadata={"file_path": file_path},
            )


class WriteFileTool(BaseTool):
    """Write content to a file."""

    def __init__(self, workspace: Optional[Path] = None) -> None:
        super().__init__(
            name="write_file",
            description=(
                "Write or edit file content with granular operations. "
                "Supports: (1) full file write with 'content', "
                "(2) str_replace to replace specific text, "
                "(3) insert_at_line to add content at a specific line. "
                "Creates parent directories if needed."
            ),
        )
        self.workspace = workspace or Path.cwd()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write/edit (relative to workspace)",
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write (for complete file write)",
                },
                "old_str": {
                    "type": "string",
                    "description": "Text to replace (for str_replace operation)",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text (for str_replace operation)",
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Line number to insert at (1-indexed, for insert operation)",
                },
                "insert_content": {
                    "type": "string",
                    "description": "Content to insert (for insert_at_line operation)",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        file_path = parameters["file_path"]
        encoding = parameters.get("encoding", "utf-8")

        # Determine operation type
        has_content = "content" in parameters
        has_replace = "old_str" in parameters and "new_str" in parameters
        has_insert = "insert_line" in parameters and "insert_content" in parameters

        try:
            full_path = (self.workspace / file_path).resolve()

            if not str(full_path).startswith(str(self.workspace.resolve())):
                raise ValueError("Access denied: path outside workspace")

            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Operation 1: Full file write
            if has_content:
                content = parameters["content"]
                full_path.write_text(content, encoding=encoding)
                return ToolResult(
                    success=True,
                    result=f"Successfully wrote {len(content)} characters to {file_path}",
                    error=None,
                    metadata={
                        "file_path": file_path,
                        "operation": "write",
                        "size": len(content),
                        "lines": len(content.splitlines()),
                    },
                )

            # Operation 2: str_replace
            elif has_replace:
                if not full_path.exists():
                    raise FileNotFoundError(
                        f"File not found for str_replace: {file_path}"
                    )

                old_str = parameters["old_str"]
                new_str = parameters["new_str"]
                current_content = full_path.read_text(encoding=encoding)

                if old_str not in current_content:
                    raise ValueError(
                        f"String to replace not found in file: {old_str[:50]}..."
                    )

                new_content = current_content.replace(
                    old_str, new_str, 1
                )  # Replace first occurrence
                full_path.write_text(new_content, encoding=encoding)

                return ToolResult(
                    success=True,
                    result=f"Successfully replaced text in {file_path}",
                    error=None,
                    metadata={
                        "file_path": file_path,
                        "operation": "str_replace",
                        "old_length": len(old_str),
                        "new_length": len(new_str),
                    },
                )

            # Operation 3: insert_at_line
            elif has_insert:
                insert_line = parameters["insert_line"]
                insert_content = parameters["insert_content"]

                # Read existing content or start fresh
                if full_path.exists():
                    current_content = full_path.read_text(encoding=encoding)
                    lines = current_content.splitlines(keepends=True)
                else:
                    lines = []

                # Insert at line (1-indexed)
                insert_index = insert_line - 1
                if insert_index < 0 or insert_index > len(lines):
                    raise ValueError(
                        f"Invalid line number: {insert_line}. File has {len(lines)} lines."
                    )

                # Ensure insert_content ends with newline if needed
                if not insert_content.endswith("\n"):
                    insert_content += "\n"

                lines.insert(insert_index, insert_content)
                new_content = "".join(lines)
                full_path.write_text(new_content, encoding=encoding)

                return ToolResult(
                    success=True,
                    result=f"Successfully inserted content at line {insert_line} in {file_path}",
                    error=None,
                    metadata={
                        "file_path": file_path,
                        "operation": "insert_at_line",
                        "line": insert_line,
                        "insert_length": len(insert_content),
                    },
                )

            else:
                raise ValueError(
                    "Must provide either 'content', 'old_str' + 'new_str', or 'insert_line' + 'insert_content'"
                )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Failed to write file: {str(e)}",
                metadata={"file_path": file_path},
            )


class ListDirectoryTool(BaseTool):
    """List files and directories in a path."""

    def __init__(self, workspace: Optional[Path] = None) -> None:
        super().__init__(
            name="list_directory",
            description="List files and directories at a given path. Returns names, types, and sizes.",
        )
        self.workspace = workspace or Path.cwd()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to directory to list (relative to workspace, default: '.')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List subdirectories recursively (default: false)",
                },
            },
            "required": [],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        directory_path = parameters.get("directory_path", ".")
        recursive = parameters.get("recursive", False)

        try:
            full_path = (self.workspace / directory_path).resolve()

            if not str(full_path).startswith(str(self.workspace.resolve())):
                raise ValueError("Access denied: path outside workspace")

            if not full_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")

            if not full_path.is_dir():
                raise ValueError(f"Not a directory: {directory_path}")

            entries = []

            if recursive:
                for item in full_path.rglob("*"):
                    rel_path = item.relative_to(full_path)
                    entries.append(
                        {
                            "name": str(rel_path),
                            "type": "file" if item.is_file() else "directory",
                            "size": item.stat().st_size if item.is_file() else None,
                        }
                    )
            else:
                for item in full_path.iterdir():
                    entries.append(
                        {
                            "name": item.name,
                            "type": "file" if item.is_file() else "directory",
                            "size": item.stat().st_size if item.is_file() else None,
                        }
                    )

            entries.sort(key=lambda x: (x["type"] != "directory", x["name"]))

            return ToolResult(
                success=True,
                result=entries,
                error=None,
                metadata={"directory": directory_path, "count": len(entries)},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Failed to list directory: {str(e)}",
                metadata={"directory_path": directory_path},
            )


class GrepSearchTool(BaseTool):
    """Search for patterns in files using grep/ripgrep."""

    def __init__(self, workspace: Optional[Path] = None) -> None:
        super().__init__(
            name="grep_search",
            description="Search for text patterns in files. Uses ripgrep if available, falls back to grep.",
        )
        self.workspace = workspace or Path.cwd()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Path to search in (relative to workspace, default: '.')",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "File pattern to filter (e.g., '*.py', '*.js')",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case sensitive search (default: true)",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        pattern = parameters["pattern"]
        path = parameters.get("path", ".")
        file_pattern = parameters.get("file_pattern")
        case_sensitive = parameters.get("case_sensitive", True)

        try:
            full_path = (self.workspace / path).resolve()

            if not str(full_path).startswith(str(self.workspace.resolve())):
                raise ValueError("Access denied: path outside workspace")

            cmd = ["rg", "--json"]
            if not case_sensitive:
                cmd.append("-i")
            if file_pattern:
                cmd.extend(["-g", file_pattern])
            cmd.append(pattern)
            cmd.append(str(full_path))

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.workspace),
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode == 1:
                    return ToolResult(
                        success=True,
                        result=[],
                        error=None,
                        metadata={"pattern": pattern, "matches": 0},
                    )

                if proc.returncode != 0 and proc.returncode is not None:
                    raise subprocess.CalledProcessError(
                        proc.returncode, cmd, stdout, stderr
                    )

                import json

                matches = []
                for line in stdout.decode().splitlines():
                    try:
                        data = json.loads(line)
                        if data.get("type") == "match":
                            match_data = data.get("data", {})
                            matches.append(
                                {
                                    "file": match_data.get("path", {}).get("text"),
                                    "line": match_data.get("line_number"),
                                    "text": match_data.get("lines", {})
                                    .get("text", "")
                                    .strip(),
                                }
                            )
                    except json.JSONDecodeError:
                        continue

                return ToolResult(
                    success=True,
                    result=matches,
                    error=None,
                    metadata={"pattern": pattern, "matches": len(matches)},
                )

            except FileNotFoundError:
                cmd = ["grep", "-rn"]
                if not case_sensitive:
                    cmd.append("-i")
                if file_pattern:
                    cmd.extend(["--include", file_pattern])
                cmd.append(pattern)
                cmd.append(str(full_path))

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.workspace),
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode == 1:
                    return ToolResult(
                        success=True,
                        result=[],
                        error=None,
                        metadata={"pattern": pattern, "matches": 0},
                    )

                matches = []
                for line in stdout.decode().splitlines():
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        matches.append(
                            {
                                "file": parts[0],
                                "line": parts[1],
                                "text": parts[2].strip(),
                            }
                        )

                return ToolResult(
                    success=True,
                    result=matches,
                    error=None,
                    metadata={"pattern": pattern, "matches": len(matches)},
                )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Search failed: {str(e)}",
                metadata={"pattern": pattern},
            )


class BashExecuteTool(BaseTool):
    """Execute bash commands safely."""

    def __init__(self, workspace: Optional[Path] = None, timeout: int = 30) -> None:
        super().__init__(
            name="bash_execute",
            description="Execute bash commands in the workspace. Returns stdout and stderr.",
        )
        self.workspace = workspace or Path.cwd()
        self.timeout = timeout

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {
                    "type": "integer",
                    "description": f"Command timeout in seconds (default: {self.timeout})",
                },
            },
            "required": ["command"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        command = parameters["command"]
        timeout = parameters.get("timeout", self.timeout)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"Command timed out after {timeout} seconds")

            stdout_str = stdout.decode()
            stderr_str = stderr.decode()

            return ToolResult(
                success=proc.returncode == 0,
                result={
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "returncode": proc.returncode,
                },
                error=stderr_str if proc.returncode != 0 else None,
                metadata={"command": command, "returncode": proc.returncode},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Command execution failed: {str(e)}",
                metadata={"command": command},
            )


class PythonREPLTool(BaseTool):
    """Execute Python code in an isolated namespace."""

    def __init__(self, workspace: Optional[Path] = None) -> None:
        super().__init__(
            name="python_repl",
            description="Execute Python code in an isolated environment. Returns the output or any errors.",
        )
        self.workspace = workspace or Path.cwd()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        code = parameters["code"]

        try:
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            redirected_output = StringIO()
            redirected_error = StringIO()
            sys.stdout = redirected_output
            sys.stderr = redirected_error

            namespace = {"__builtins__": __builtins__}

            try:
                exec(code, namespace)
                output = redirected_output.getvalue()
                error = redirected_error.getvalue()

                result = output if output else None
                success = not error

                return ToolResult(
                    success=success,
                    result=result,
                    error=error if error else None,
                    metadata={"code_length": len(code)},
                )

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Execution failed: {str(e)}",
                metadata={"code_length": len(code)},
            )


def create_coding_tools(
    workspace: Optional[Path] = None, bash_timeout: int = 30
) -> Sequence[BaseTool]:
    """
    Create a list of coding tools for file operations and code execution.

    Args:
        workspace: Root directory for file operations (default: current directory)
        bash_timeout: Default timeout for bash commands in seconds

    Returns:
        List of coding tool instances
    """
    workspace = workspace or Path.cwd()

    return [
        ReadFileTool(workspace=workspace),
        WriteFileTool(workspace=workspace),
        ListDirectoryTool(workspace=workspace),
        GrepSearchTool(workspace=workspace),
        BashExecuteTool(workspace=workspace, timeout=bash_timeout),
        PythonREPLTool(workspace=workspace),
    ]
