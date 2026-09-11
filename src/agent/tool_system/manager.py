"""
Tool manager implementation for the Tool System V1.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from .interfaces import ToolManager, ToolRegistry, PermissionChecker
from .models import ToolRequest, ToolResult, Tool, ToolMetadata, PermissionDecision

logger = logging.getLogger(__name__)


class ToolManagerImpl(ToolManager):
    """Concrete implementation of the ToolManager interface."""

    def __init__(self, registry: ToolRegistry, permission_checker: PermissionChecker, workspace_guard: Any = None):
        """
        Initialize the tool manager.

        Args:
            registry: The tool registry to use
            permission_checker: The permission checker to use
        """
        self.registry = registry
        self.permission_checker = permission_checker
        self.workspace_guard = workspace_guard
        logger.info("Tool manager initialized")

    async def execute_tool(self, request: ToolRequest) -> ToolResult:
        """
        Execute a tool request through the complete tool system pipeline.

        Args:
            request: The tool request to execute

        Returns:
            ToolResult: Result of the tool execution
        """
        import time
        start_time = time.time()

        try:
            # Step 1: Check if tool exists in registry
            tool = await self.registry.get_tool(request.tool_name)
            if not tool:
                logger.warning(f"Tool not found: {request.tool_name}")
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=request.tool_name,
                    success=False,
                    error=f"Tool '{request.tool_name}' not found",
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )

            if self.workspace_guard is not None:
                self.workspace_guard.validate(request, tool.metadata)

            # Step 2: Check permissions
            permission_decision = await self.permission_checker.check_permission(request)
            logger.debug(f"Permission decision for {request.tool_name}: {permission_decision}")

            # Step 3: Handle permission decision
            if permission_decision == PermissionDecision.DENY:
                logger.warning(f"Permission denied for tool: {request.tool_name}")
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=request.tool_name,
                    success=False,
                    error=f"Permission denied for tool '{request.tool_name}'",
                    permission_decision=permission_decision,
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )

            if permission_decision == PermissionDecision.REQUIRE_APPROVAL:
                logger.warning(f"Approval required for tool: {request.tool_name}")
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=request.tool_name,
                    success=False,
                    error=f"Approval required for tool '{request.tool_name}'",
                    permission_decision=permission_decision,
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )

            # Step 4: Execute the tool
            logger.debug(f"Executing tool: {request.tool_name}")
            # Bundled tools are synchronous. Run them off the event loop so a
            # cancellation request can be observed before subsequent actions.
            result = await asyncio.to_thread(tool.execute, request)

            # Step 5: Add execution time and permission decision to result
            execution_time_ms = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time_ms
            result.permission_decision = permission_decision

            logger.debug(f"Tool {request.tool_name} executed successfully in {execution_time_ms}ms")
            return result

        except Exception as e:
            logger.error(f"Error executing tool {request.tool_name}: {e}", exc_info=True)
            return ToolResult(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=f"Internal error executing tool: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

    async def register_tool(self, tool: Tool) -> None:
        """
        Register a tool with the tool manager.

        Args:
            tool: The tool to register
        """
        await self.registry.register_tool(tool)

    async def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the tool manager.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            bool: True if tool was found and removed, False otherwise
        """
        return await self.registry.unregister_tool(tool_name)

    async def list_tools(self) -> List[Tool]:
        """
        List all registered tools.

        Returns:
            List[Tool]: List of all registered tools
        """
        return await self.registry.list_tools()
