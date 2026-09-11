"""
Abstract interfaces for the Tool System V1.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from .models import Tool, ToolMetadata, ToolRequest, ToolResult, PermissionDecision


class ToolRegistry(ABC):
    """Abstract interface for tool registry."""

    @abstractmethod
    async def register_tool(self, tool: Tool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: The tool to register
        """
        pass

    @abstractmethod
    async def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            bool: True if tool was found and removed, False otherwise
        """
        pass

    @abstractmethod
    async def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Optional[Tool]: The tool if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_tools(self) -> List[Tool]:
        """
        List all registered tools.

        Returns:
            List[Tool]: List of all registered tools
        """
        pass

    @abstractmethod
    async def tool_exists(self, tool_name: str) -> bool:
        """
        Check if a tool exists in the registry.

        Args:
            tool_name: Name of the tool to check

        Returns:
            bool: True if tool exists, False otherwise
        """
        pass


class PermissionPolicy(ABC):
    """Abstract interface for permission policy."""

    @abstractmethod
    async def check_permission(self, request: ToolRequest) -> PermissionDecision:
        """
        Check if a tool request is permitted according to the policy.

        Args:
            request: The tool request to check

        Returns:
            PermissionDecision: Decision on whether the request is permitted
        """
        pass


class PermissionChecker(ABC):
    """Abstract interface for checking permissions."""

    @abstractmethod
    async def check_permission(self, request: ToolRequest) -> PermissionDecision:
        """
        Check if a tool request is permitted.

        Args:
            request: The tool request to check

        Returns:
            PermissionDecision: Decision on whether the request is permitted
        """
        pass


class ToolExecutor(ABC):
    """Abstract interface for executing tools."""

    @abstractmethod
    async def execute(self, request: ToolRequest) -> ToolResult:
        """
        Execute a tool request.

        Args:
            request: The tool request to execute

        Returns:
            ToolResult: Result of the tool execution
        """
        pass


class ToolManager(ABC):
    """Abstract interface for managing tools."""

    @abstractmethod
    async def execute_tool(self, request: ToolRequest) -> ToolResult:
        """
        Execute a tool request through the complete tool system pipeline.

        Args:
            request: The tool request to execute

        Returns:
            ToolResult: Result of the tool execution
        """
        pass

    @abstractmethod
    async def register_tool(self, tool: Tool) -> None:
        """
        Register a tool with the tool manager.

        Args:
            tool: The tool to register
        """
        pass

    @abstractmethod
    async def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the tool manager.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            bool: True if tool was found and removed, False otherwise
        """
        pass

    @abstractmethod
    async def list_tools(self) -> List[Tool]:
        """
        List all registered tools.

        Returns:
            List[Tool]: List of all registered tools
        """
        pass