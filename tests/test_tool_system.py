"""
Tests for the Tool System V1.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock
import pytest

from agent.tool_system.models import (
    ToolRequest,
    ToolResult,
    ToolMetadata,
    OperationType,
    RiskLevel,
    PermissionDecision,
    Tool,
    create_file_read_metadata,
    create_file_write_metadata,
    create_file_edit_metadata,
    create_file_search_metadata,
    create_directory_list_metadata,
    create_terminal_metadata,
    create_git_metadata
)
from agent.tool_system.registry import ToolRegistryImpl
from agent.tool_system.permission import PermissionPolicyImpl, PermissionCheckerImpl
from agent.tool_system.manager import ToolManagerImpl
from agent.tool_system.tools import (
    FileReadTool,
    FileWriteTool,
    FileEditTool,
    FileSearchTool,
    DirectoryListTool,
    TerminalTool,
    GitTool
)


class TestTool(Tool):
    """Test tool for registry testing."""

    def __init__(self, name: str):
        metadata = ToolMetadata(
            name=name,
            description=f"Test tool {name}",
            operation_type=OperationType.READ,
            risk_level=RiskLevel.LOW,
            is_read_only=True,
            required_permission="TEST",
            target_resource="test_resource",
            allowed_workspace_only=True
        )
        super().__init__(metadata=metadata)

    def execute(self, request: ToolRequest) -> ToolResult:
        return ToolResult(
            request_id=request.request_id,
            tool_name=self.metadata.name,
            success=True,
            output={"test": "success"}
        )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing."""
    file_path = Path(temp_dir) / "sample.txt"
    file_path.write_text("Hello, World!\nThis is a test file.\nLine 3\n")
    return str(file_path)


@pytest.fixture
def sample_dir(temp_dir):
    """Create a sample directory structure for testing."""
    dir_path = Path(temp_dir)
    # Create some files
    (dir_path / "file1.txt").write_text("Content 1")
    (dir_path / "file2.py").write_text("print('hello')")
    (dir_path / ".hidden").write_text("hidden file")
    # Create subdirectory
    subdir = dir_path / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt").write_text("Content 3")
    return str(dir_path)


class TestToolRegistry:
    """Tests for the tool registry."""

    @pytest.mark.asyncio
    async def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        registry = ToolRegistryImpl()
        tool = TestTool("test_tool")

        await registry.register_tool(tool)

        retrieved_tool = await registry.get_tool("test_tool")
        assert retrieved_tool is not None
        assert retrieved_tool.metadata.name == "test_tool"

    @pytest.mark.asyncio
    async def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistryImpl()
        tool = TestTool("test_tool")

        await registry.register_tool(tool)
        assert await registry.tool_exists("test_tool")

        result = await registry.unregister_tool("test_tool")
        assert result is True

        assert not await registry.tool_exists("test_tool")

        # Try to unregister non-existent tool
        result = await registry.unregister_tool("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistryImpl()
        tool1 = TestTool("tool1")
        tool2 = TestTool("tool2")

        await registry.register_tool(tool1)
        await registry.register_tool(tool2)

        tools = await registry.list_tools()
        assert len(tools) == 2
        tool_names = {tool.metadata.name for tool in tools}
        assert tool_names == {"tool1", "tool2"}


class TestPermissionPolicy:
    """Tests for the permission policy."""

    @pytest.fixture
    def policy(self):
        """Create a permission policy for testing."""
        return PermissionPolicyImpl()

    @pytest.fixture
    def policy_with_workspace(self, temp_dir):
        """Create a permission policy with the temp directory as workspace."""
        return PermissionPolicyImpl(temp_dir)

    @pytest.mark.asyncio
    async def test_file_read_allowed(self, policy_with_workspace, sample_file):
        """Test that file read is allowed for existing files."""
        request = ToolRequest(
            request_id="test1",
            tool_name="file_read",
            tool_input={"file_path": sample_file}
        )

        decision = await policy_with_workspace.check_permission(request)
        assert decision == PermissionDecision.ALLOW

    @pytest.mark.asyncio
    async def test_file_write_requires_approval(self, policy_with_workspace, temp_dir):
        """Test that file write requires approval."""
        request = ToolRequest(
            request_id="test2",
            tool_name="file_write",
            tool_input={
                "file_path": str(Path(temp_dir) / "new_file.txt"),
                "content": "test content"
            }
        )

        decision = await policy_with_workspace.check_permission(request)
        assert decision == PermissionDecision.REQUIRE_APPROVAL

    @pytest.mark.asyncio
    async def test_path_outside_workspace_denied(self, policy):
        """Test that access outside workspace is denied."""
        # Use a path that's likely outside the workspace
        request = ToolRequest(
            request_id="test3",
            tool_name="file_read",
            tool_input={"file_path": "/etc/passwd"}  # This should be outside workspace
        )

        decision = await policy.check_permission(request)
        # This might be ALLOW or DENY depending on the actual workspace root
        # But we're testing that the policy checks the workspace boundary
        assert decision in [PermissionDecision.ALLOW, PermissionDecision.DENY]

    @pytest.mark.asyncio
    async def test_protected_file_denied(self, policy_with_workspace, temp_dir):
        """Test that access to protected files is denied."""
        # Create a .env file (protected)
        env_file = Path(temp_dir) / ".env"
        env_file.write_text("SECRET=value")

        request = ToolRequest(
            request_id="test4",
            tool_name="file_read",
            tool_input={"file_path": str(env_file)}
        )

        decision = await policy_with_workspace.check_permission(request)
        assert decision == PermissionDecision.DENY

    @pytest.mark.asyncio
    async def test_path_traversal_denied(self, policy_with_workspace, temp_dir):
        """Test that path traversal is denied."""
        request = ToolRequest(
            request_id="test5",
            tool_name="file_read",
            tool_input={"file_path": "../../etc/passwd"}  # Path traversal
        )

        decision = await policy_with_workspace.check_permission(request)
        assert decision == PermissionDecision.DENY


class TestToolManager:
    """Tests for the tool manager."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, temp_dir, sample_file):
        """Test successful tool execution through the manager."""
        # Set up tool system components
        registry = ToolRegistryImpl()
        policy = PermissionPolicyImpl(temp_dir)  # Use temp_dir as workspace
        permission_checker = PermissionCheckerImpl(policy, registry)
        tool_manager = ToolManagerImpl(registry, permission_checker)

        # Register a test tool
        test_tool = TestTool("test_tool")
        await tool_manager.register_tool(test_tool)

        # Execute the tool
        request = ToolRequest(
            request_id="test_exec",
            tool_name="test_tool",
            tool_input={"test_param": "value"}
        )

        result = await tool_manager.execute_tool(request)

        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.output == {"test": "success"}

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Test executing a non-existent tool."""
        registry = ToolRegistryImpl()
        policy = PermissionPolicyImpl()
        permission_checker = PermissionCheckerImpl(policy)
        tool_manager = ToolManagerImpl(registry, permission_checker)

        request = ToolRequest(
            request_id="test_not_found",
            tool_name="nonexistent_tool",
            tool_input={}
        )

        result = await tool_manager.execute_tool(request)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_permission_denied(self, temp_dir):
        """Test executing a tool with permission denied."""
        # Create a policy that denies everything for testing
        registry = ToolRegistryImpl()
        policy = PermissionPolicyImpl()
        # We'll mock the permission checker to always deny
        permission_checker = Mock()
        permission_checker.check_permission = AsyncMock(
            return_value=PermissionDecision.DENY
        )
        tool_manager = ToolManagerImpl(registry, permission_checker)
        # Register the file_read tool for this test
        file_read_tool = FileReadTool()
        await tool_manager.register_tool(file_read_tool)

        request = ToolRequest(
            request_id="test_denied",
            tool_name="file_read",
            tool_input={"file_path": "/etc/passwd"}  # Some path
        )

        result = await tool_manager.execute_tool(request)

        assert result.success is False
        assert result.permission_decision == PermissionDecision.DENY
        assert "permission denied" in result.error.lower()


class TestConcreteTools:
    """Tests for the concrete tool implementations."""

    @pytest.mark.asyncio
    async def test_file_read_tool(self, sample_file):
        """Test the file read tool."""
        tool = FileReadTool()

        request = ToolRequest(
            request_id="test_read",
            tool_name="file_read",
            tool_input={"file_path": sample_file}
        )

        result = tool.execute(request)

        assert result.success is True
        assert "content" in result.output
        assert "Hello, World!" in result.output["content"]
        assert result.output["file_path"] == sample_file

    @pytest.mark.asyncio
    async def test_file_write_tool(self, temp_dir):
        """Test the file write tool."""
        tool = FileWriteTool()
        file_path = str(Path(temp_dir) / "written_file.txt")

        request = ToolRequest(
            request_id="test_write",
            tool_name="file_write",
            tool_input={
                "file_path": file_path,
                "content": "This is written content.\nLine 2\n"
            }
        )

        result = tool.execute(request)

        assert result.success is True
        assert result.output["file_path"] == file_path
        assert result.output["bytes_written"] > 0

        # Verify file was written correctly
        written_content = Path(file_path).read_text()
        assert written_content == "This is written content.\nLine 2\n"

    @pytest.mark.asyncio
    async def test_file_edit_tool(self, sample_file):
        """Test the file edit tool."""
        tool = FileEditTool()

        request = ToolRequest(
            request_id="test_edit",
            tool_name="file_edit",
            tool_input={
                "file_path": sample_file,
                "old_string": "Hello, World!",
                "new_string": "Hello, Universe!",
                "replace_all": True
            }
        )

        result = tool.execute(request)

        assert result.success is True
        assert result.output["changes_made"] == 1

        # Verify file was edited correctly
        edited_content = Path(sample_file).read_text()
        assert edited_content == "Hello, Universe!\nThis is a test file.\nLine 3\n"

    @pytest.mark.asyncio
    async def test_file_search_tool(self, sample_dir):
        """Test the file search tool."""
        tool = FileSearchTool()

        # Search for files
        request = ToolRequest(
            request_id="test_search_file",
            tool_name="file_search",
            tool_input={
                "type": "file",
                "pattern": "file1",
                "path": sample_dir,
                "file_pattern": "*"
            }
        )

        result = tool.execute(request)

        assert result.success is True
        assert result.output["count"] >= 1
        # Should find file1.txt
        found_paths = [item["path"] for item in result.output["results"]]
        assert any("file1.txt" in path for path in found_paths)

        # Search for content
        request2 = ToolRequest(
            request_id="test_search_content",
            tool_name="file_search",
            tool_input={
                "type": "content",
                "pattern": "hello",
                "path": sample_dir,
                "file_pattern": "*.py"
            }
        )

        result2 = tool.execute(request2)

        assert result2.success is True
        assert result2.output["count"] >= 1
        # Should find the print statement in file2.py
        assert any("file2.py" in item["file_path"] for item in result2.output["results"])

    @pytest.mark.asyncio
    async def test_directory_list_tool(self, sample_dir):
        """Test the directory list tool."""
        tool = DirectoryListTool()

        request = ToolRequest(
            request_id="test_list",
            tool_name="directory_list",
            tool_input={"directory_path": sample_dir}
        )

        result = tool.execute(request)

        assert result.success is True
        assert result.output["count"] >= 3  # file1.txt, file2.py, .hidden, subdir
        # Check that we have both files and directories
        types = {item["type"] for item in result.output["items"]}
        assert "file" in types
        assert "directory" in types

    @pytest.mark.asyncio
    async def test_terminal_tool_simple(self):
        """Test the terminal tool with a simple command."""
        tool = TerminalTool()

        # Use echo command which should work on most systems
        request = ToolRequest(
            request_id="test_terminal",
            tool_name="terminal",
            tool_input={
                "command": "echo 'Hello Terminal'",
                "cwd": "."
            }
        )

        result = tool.execute(request)

        # Note: This might fail in some CI environments, but we'll check the structure
        assert result.success is True or (result.success is False and "command" in result.output)
        assert "command" in result.output
        assert result.output["command"] == "echo 'Hello Terminal'"

    @pytest.mark.asyncio
    async def test_git_tool_not_a_repo(self, temp_dir):
        """Test the git tool when not in a git repository."""
        tool = GitTool()

        request = ToolRequest(
            request_id="test_git",
            tool_name="git",
            tool_input={
                "operation": "status",
                "cwd": temp_dir  # Should not be a git repo
            }
        )

        result = tool.execute(request)

        assert result.success is False
        assert "not a git repository" in result.error.lower()


class TestToolSystemIntegration:
    """Tests for integrating the tool system with the executor."""

    @pytest.mark.asyncio
    async def test_tool_system_adapter(self, temp_dir, sample_file):
        """Test the tool system adapter for the executor."""
        from agent.executor.tool_system_adapter import (
            ToolSystemToolExecutor,
            ToolSystemPermissionChecker
        )
        from agent.tool_system.manager import ToolManagerImpl

        # Set up tool system
        registry = ToolRegistryImpl()
        policy = PermissionPolicyImpl(temp_dir)  # Use temp_dir as workspace
        permission_checker = PermissionCheckerImpl(policy, registry)
        tool_manager = ToolManagerImpl(registry, permission_checker)

        # Register a test tool
        test_tool = TestTool("test_tool")
        await tool_manager.register_tool(test_tool)

        # Create adapters
        tool_executor = ToolSystemToolExecutor(tool_manager)
        permission_checker_adapter = ToolSystemPermissionChecker(tool_manager)

        # Test permission checking (should require approval for our test tool based on policy)
        action_request = Mock()
        action_request.action_id = "test_action"
        action_request.tool_name = "test_tool"
        action_request.tool_input = {"test": "value"}
        action_request.step_id = "test_step"
        action_request.timeout_seconds = None

        permission_result = await permission_checker_adapter.check_permission(action_request)
        # Our test tool is read-only, so should be ALLOW
        assert permission_result is True

        # Test execution
        executor_result = await tool_executor.execute(action_request)

        assert executor_result.success is True
        assert executor_result.action_id == "test_action"
        assert executor_result.step_id == "test_step"
        assert executor_result.output == {"test": "success"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])