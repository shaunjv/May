"""Dependency-injected composition root for the local desktop application."""

import os
from pathlib import Path

from agent.config import Settings
from agent.context_manager.context_manager import ContextManager
from agent.executor.executor import Executor
from agent.executor.tool_system_adapter import ToolSystemPermissionChecker, ToolSystemToolExecutor
from agent.intent_manager.intent_manager import IntentManager
from agent.intent_manager.nvidia_provider import NVIDIAProvider
from agent.planner.planner import Planner
from agent.tool_system.manager import ToolManagerImpl
from agent.tool_system.permission import PermissionCheckerImpl, PermissionPolicyImpl
from agent.tool_system.registry import ToolRegistryImpl
from agent.tool_system.tools import DirectoryListTool, FileEditTool, FileReadTool, FileSearchTool, FileWriteTool, GitTool, TerminalTool

from .persistence import SQLiteStore
from .approval import ApprovedActionAuthorizer, ApprovedActionPermissionChecker
from .resolver import LLMActionResolver
from .services import ConversationService, TaskService
from .workspace import WorkspaceGuard


class _TokenCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class DesktopRuntime:
    def __init__(self, conversations: ConversationService, tasks: TaskService, store: SQLiteStore, workspace: Path):
        self.conversations, self.tasks, self.store, self.workspace = conversations, tasks, store, workspace

    @classmethod
    async def create(cls, workspace: str | Path, settings: Settings) -> "DesktopRuntime":
        guard = WorkspaceGuard(workspace)
        provider = NVIDIAProvider.from_settings(settings)
        registry = ToolRegistryImpl()
        policy = PermissionPolicyImpl(workspace_root=str(guard.root))
        base_checker = PermissionCheckerImpl(policy, registry=registry)
        authorizer = ApprovedActionAuthorizer()
        checker = ApprovedActionPermissionChecker(base_checker, authorizer)
        manager = ToolManagerImpl(registry, checker, workspace_guard=guard)
        for tool in (FileReadTool(), FileWriteTool(), FileEditTool(), FileSearchTool(), DirectoryListTool(), TerminalTool(), GitTool()):
            await manager.register_tool(tool)
        executor = Executor(ToolSystemToolExecutor(manager), ToolSystemPermissionChecker(manager))
        database = cls.default_database_path()
        store = SQLiteStore(database)
        resolver = LLMActionResolver(provider, registry, guard)
        context_manager = ContextManager(token_counter=_TokenCounter())
        return cls(
            ConversationService(provider),
            TaskService(IntentManager(provider), Planner(provider), resolver, executor, store, context_manager, authorizer),
            store, guard.root,
        )

    @staticmethod
    def default_database_path() -> Path:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local")) / "PersonalAIAgent"
        return base / "agent.db"
