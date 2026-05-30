"""Tests for MemoryTool functionality."""

import asyncio
from pathlib import Path

import pytest

from forla.tools import MemoryBackend, MemoryTool
from forla.types import ToolResult


@pytest.fixture
def temp_memory_path(tmp_path):
    """Provide a temporary directory for memory storage."""
    return tmp_path / "test_memory"


@pytest.fixture
def memory_backend(temp_memory_path):
    """Create a MemoryBackend instance for testing."""
    return MemoryBackend(base_path=temp_memory_path)


@pytest.fixture
def memory_tool(temp_memory_path):
    """Create a MemoryTool instance for testing."""
    return MemoryTool(base_path=temp_memory_path)


class TestMemoryBackend:
    """Test MemoryBackend functionality."""

    def test_initialization(self, memory_backend, temp_memory_path):
        """Test backend initializes and creates directory."""
        assert memory_backend.base_path == temp_memory_path.resolve()
        assert memory_backend.base_path.exists()
        assert memory_backend.base_path.is_dir()

    def test_path_validation_security(self, memory_backend):
        """Test path traversal attack prevention."""
        with pytest.raises(ValueError, match="outside memory directory"):
            memory_backend._validate_path("../../etc/passwd")

        with pytest.raises(ValueError, match="outside memory directory"):
            memory_backend._validate_path("/memories/../../../etc/passwd")

    def test_view_empty_directory(self, memory_backend):
        """Test viewing empty memory directory."""
        result = memory_backend.view("/memories")
        assert "Directory: /memories" in result
        assert "(empty)" in result

    def test_create_file(self, memory_backend):
        """Test creating a file."""
        content = "Test content\nLine 2"
        result = memory_backend.create("/memories/test.md", content)

        assert "created successfully" in result
        assert (memory_backend.base_path / "test.md").exists()

        # Verify content
        created_content = (
            memory_backend.base_path / "test.md"
        ).read_text()
        assert created_content == content

    def test_create_file_with_directories(self, memory_backend):
        """Test creating file in nested directory."""
        result = memory_backend.create(
            "/memories/patterns/bug.md", "Bug notes"
        )

        assert "created successfully" in result
        assert (memory_backend.base_path / "patterns" / "bug.md").exists()

    def test_view_file(self, memory_backend):
        """Test viewing file contents."""
        memory_backend.create("/memories/notes.txt", "Line 1\nLine 2\nLine 3")

        result = memory_backend.view("/memories/notes.txt")

        assert "1: Line 1" in result
        assert "2: Line 2" in result
        assert "3: Line 3" in result

    def test_view_file_with_range(self, memory_backend):
        """Test viewing specific line range."""
        memory_backend.create(
            "/memories/notes.txt", "Line 1\nLine 2\nLine 3\nLine 4"
        )

        result = memory_backend.view("/memories/notes.txt", view_range=[2, 3])

        assert "2: Line 2" in result
        assert "3: Line 3" in result
        assert "Line 1" not in result
        assert "Line 4" not in result

    def test_view_directory_with_files(self, memory_backend):
        """Test viewing directory listing."""
        memory_backend.create("/memories/file1.txt", "Content 1")
        memory_backend.create("/memories/file2.txt", "Content 2")

        result = memory_backend.view("/memories")

        assert "Directory: /memories" in result
        assert "- file1.txt" in result
        assert "- file2.txt" in result

    def test_str_replace(self, memory_backend):
        """Test string replacement in file."""
        memory_backend.create("/memories/test.txt", "Hello world\nHello again")

        result = memory_backend.str_replace(
            "/memories/test.txt", "Hello", "Hi"
        )

        assert "edited successfully" in result

        # Verify replacement (only first occurrence)
        content = (memory_backend.base_path / "test.txt").read_text()
        assert content == "Hi world\nHello again"

    def test_str_replace_not_found(self, memory_backend):
        """Test str_replace with non-existent text."""
        memory_backend.create("/memories/test.txt", "Hello world")

        with pytest.raises(ValueError, match="Text not found"):
            memory_backend.str_replace(
                "/memories/test.txt", "Goodbye", "Hi"
            )

    def test_insert(self, memory_backend):
        """Test inserting text at line."""
        memory_backend.create("/memories/test.txt", "Line 1\nLine 3\n")

        result = memory_backend.insert("/memories/test.txt", 2, "Line 2\n")

        assert "inserted" in result

        # Verify insertion
        content = (memory_backend.base_path / "test.txt").read_text()
        assert content == "Line 1\nLine 2\nLine 3\n"

    def test_delete_file(self, memory_backend):
        """Test deleting a file."""
        memory_backend.create("/memories/temp.txt", "Temporary")

        result = memory_backend.delete("/memories/temp.txt")

        assert "File deleted" in result
        assert not (memory_backend.base_path / "temp.txt").exists()

    def test_delete_empty_directory(self, memory_backend):
        """Test deleting empty directory."""
        (memory_backend.base_path / "empty_dir").mkdir()

        result = memory_backend.delete("/memories/empty_dir")

        assert "Directory deleted" in result
        assert not (memory_backend.base_path / "empty_dir").exists()

    def test_delete_non_empty_directory_fails(self, memory_backend):
        """Test that deleting non-empty directory raises error."""
        memory_backend.create("/memories/dir/file.txt", "Content")

        with pytest.raises(ValueError, match="not empty"):
            memory_backend.delete("/memories/dir")

    def test_rename_file(self, memory_backend):
        """Test renaming a file."""
        memory_backend.create("/memories/old.txt", "Content")

        result = memory_backend.rename("/memories/old.txt", "/memories/new.txt")

        assert "Renamed" in result
        assert not (memory_backend.base_path / "old.txt").exists()
        assert (memory_backend.base_path / "new.txt").exists()

    def test_rename_to_nested_path(self, memory_backend):
        """Test moving file to nested directory."""
        memory_backend.create("/memories/file.txt", "Content")

        result = memory_backend.rename(
            "/memories/file.txt", "/memories/archive/file.txt"
        )

        assert "Renamed" in result
        assert (memory_backend.base_path / "archive" / "file.txt").exists()


class TestMemoryTool:
    """Test MemoryTool integration."""

    @pytest.mark.asyncio
    async def test_tool_initialization(self, memory_tool):
        """Test MemoryTool initializes correctly."""
        assert memory_tool.name == "memory"
        assert "persist across conversations" in memory_tool.description
        assert memory_tool.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_tool_parameters_schema(self, memory_tool):
        """Test tool has correct parameter schema."""
        params = memory_tool.parameters

        assert params["type"] == "object"
        assert "command" in params["properties"]
        assert set(params["properties"]["command"]["enum"]) == {
            "view",
            "create",
            "str_replace",
            "insert",
            "delete",
            "rename",
            "search",  # Added in 0.3.0
            "append",  # Added in 0.3.0
        }

    @pytest.mark.asyncio
    async def test_execute_view(self, memory_tool):
        """Test executing view command."""
        result = await memory_tool.execute(
            {"command": "view", "path": "/memories"}
        )

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "Directory" in result.result
        assert result.metadata["command"] == "view"

    @pytest.mark.asyncio
    async def test_execute_create(self, memory_tool):
        """Test executing create command."""
        result = await memory_tool.execute(
            {
                "command": "create",
                "path": "/memories/test.md",
                "file_text": "# Test\nContent",
            }
        )

        assert result.success is True
        assert "created successfully" in result.result
        assert result.metadata["command"] == "create"
        assert result.metadata["size"] == len("# Test\nContent")

    @pytest.mark.asyncio
    async def test_execute_str_replace(self, memory_tool):
        """Test executing str_replace command."""
        # Create file first
        await memory_tool.execute(
            {
                "command": "create",
                "path": "/memories/test.txt",
                "file_text": "Hello world",
            }
        )

        # Replace text
        result = await memory_tool.execute(
            {
                "command": "str_replace",
                "path": "/memories/test.txt",
                "old_str": "Hello",
                "new_str": "Hi",
            }
        )

        assert result.success is True
        assert "edited successfully" in result.result

    @pytest.mark.asyncio
    async def test_execute_insert(self, memory_tool):
        """Test executing insert command."""
        # Create file first
        await memory_tool.execute(
            {
                "command": "create",
                "path": "/memories/test.txt",
                "file_text": "Line 1\nLine 3",
            }
        )

        # Insert line
        result = await memory_tool.execute(
            {
                "command": "insert",
                "path": "/memories/test.txt",
                "insert_line": 2,
                "insert_text": "Line 2\n",
            }
        )

        assert result.success is True
        assert "inserted" in result.result

    @pytest.mark.asyncio
    async def test_execute_delete(self, memory_tool):
        """Test executing delete command."""
        # Create file first
        await memory_tool.execute(
            {
                "command": "create",
                "path": "/memories/temp.txt",
                "file_text": "Temporary",
            }
        )

        # Delete file
        result = await memory_tool.execute(
            {"command": "delete", "path": "/memories/temp.txt"}
        )

        assert result.success is True
        assert "deleted" in result.result

    @pytest.mark.asyncio
    async def test_execute_rename(self, memory_tool):
        """Test executing rename command."""
        # Create file first
        await memory_tool.execute(
            {
                "command": "create",
                "path": "/memories/old.txt",
                "file_text": "Content",
            }
        )

        # Rename file
        result = await memory_tool.execute(
            {
                "command": "rename",
                "old_path": "/memories/old.txt",
                "new_path": "/memories/new.txt",
            }
        )

        assert result.success is True
        assert "Renamed" in result.result

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, memory_tool):
        """Test error handling for invalid operations."""
        # Try to view non-existent file
        result = await memory_tool.execute(
            {"command": "view", "path": "/memories/nonexistent.txt"}
        )

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_unknown_command(self, memory_tool):
        """Test handling of unknown command."""
        result = await memory_tool.execute({"command": "invalid_command"})

        assert result.success is False
        assert "Unknown command" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
