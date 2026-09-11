"""
Abstract interfaces for the Executor component.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .models import ActionRequest, ToolResult


class ToolExecutor(ABC):
    """Abstract interface for executing tools/actions."""

    @abstractmethod
    async def execute(self, action_request: ActionRequest) -> ToolResult:
        """
        Execute an action/request.

        Args:
            action_request: The action to execute

        Returns:
            ToolResult: Result of the execution

        Note:
            Implementations must NOT directly access filesystem, execute subprocesses,
            or execute shell commands. They should delegate to appropriate services.
        """
        pass


class PermissionChecker(ABC):
    """Abstract interface for checking permissions/approvals."""

    @abstractmethod
    async def check_permission(self, action_request: ActionRequest) -> bool:
        """
        Check if an action is permitted/approved.

        Args:
            action_request: The action to check

        Returns:
            bool: True if permitted, False if denied

        Note:
            Implementations should handle approval workflows and return False
            when approval is required but not yet granted.
        """
        pass