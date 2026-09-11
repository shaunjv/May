"""Tests for CLI composition of the real executor and Tool System."""

import pytest

from agent.cli import _load_executor
from agent.config import Settings
from agent.executor.executor import Executor
from agent.executor.tool_system_adapter import (
    ToolSystemPermissionChecker,
    ToolSystemToolExecutor,
)


@pytest.mark.asyncio
async def test_load_executor_wires_real_tool_system_adapters():
    """The CLI must use registered Tool System tools rather than placeholders."""
    executor = await _load_executor()

    assert isinstance(executor, Executor)
    assert isinstance(executor.tool_executor, ToolSystemToolExecutor)
    assert isinstance(executor.permission_checker, ToolSystemPermissionChecker)

    registered_tools = await executor.tool_executor.tool_manager.registry.list_tools()
    assert {tool.metadata.name for tool in registered_tools} == {
        "file_read",
        "file_write",
        "file_edit",
        "file_search",
        "directory_list",
        "terminal",
        "git",
    }

    policy = executor.tool_executor.tool_manager.permission_checker.permission_policy
    assert policy.workspace_root.is_dir()


@pytest.mark.asyncio
async def test_cli_rejects_missing_nvidia_credentials_before_runtime():
    """CLI setup must fail clearly rather than silently using a mock provider."""
    from agent.cli import _load_llm_provider

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        await _load_llm_provider(Settings(_env_file=None, nvidia_model="test-model"))
