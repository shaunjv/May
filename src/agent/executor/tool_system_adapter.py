"""
Adapter to connect the Executor with the Tool System V1.
"""

import logging
from typing import Dict, Any
from agent.executor.interfaces import ToolExecutor, PermissionChecker
from agent.executor.models import ActionRequest, ToolResult
from agent.tool_system.models import ToolRequest, ToolResult as SystemToolResult, PermissionDecision
from agent.tool_system.manager import ToolManagerImpl

logger = logging.getLogger(__name__)


class ToolSystemToolExecutor(ToolExecutor):
    """
    Adapter that implements the executor's ToolExecutor interface
    using the tool system V1.
    """

    def __init__(self, tool_manager: ToolManagerImpl):
        """
        Initialize the adapter.

        Args:
            tool_manager: The tool system manager to use for execution
        """
        self.tool_manager = tool_manager
        logger.info("Tool system tool executor adapter initialized")

    async def execute(self, action_request: ActionRequest) -> ToolResult:
        """
        Execute an action using the tool system.

        Args:
            action_request: The action request from the executor

        Returns:
            ToolResult: Result in the executor's format
        """
        try:
            # Convert executor's ActionRequest to tool system's ToolRequest
            tool_request = ToolRequest(
                request_id=action_request.action_id,
                tool_name=action_request.tool_name,
                tool_input=action_request.tool_input,
                timeout_seconds=action_request.timeout_seconds
            )

            # Execute using the tool system
            system_result = await self.tool_manager.execute_tool(tool_request)

            # Convert tool system's ToolResult to executor's ToolResult
            executor_result = ToolResult(
                action_id=system_result.request_id,
                step_id=action_request.step_id,  # Preserve step_id from original request
                success=system_result.success,
                output=system_result.output,
                error=system_result.error,
                execution_time_ms=system_result.execution_time_ms
            )

            return executor_result

        except Exception as e:
            logger.error(f"Error in tool system adapter: {e}")
            return ToolResult(
                action_id=action_request.action_id,
                step_id=action_request.step_id,
                success=False,
                error=f"Tool system execution failed: {str(e)}"
            )


class ToolSystemPermissionChecker(PermissionChecker):
    """
    Adapter that implements the executor's PermissionChecker interface
    using the tool system V1's permission checking.
    """

    def __init__(self, tool_manager: ToolManagerImpl):
        """
        Initialize the adapter.

        Args:
            tool_manager: The tool system manager to use for permission checking
        """
        self.tool_manager = tool_manager
        logger.info("Tool system permission checker adapter initialized")

    async def check_permission(self, action_request: ActionRequest) -> bool:
        """
        Check if an action is permitted using the tool system.

        Args:
            action_request: The action request to check

        Returns:
            bool: True if permitted, False otherwise
        """
        try:
            # Convert executor's ActionRequest to tool system's ToolRequest
            tool_request = ToolRequest(
                request_id=action_request.action_id,
                tool_name=action_request.tool_name,
                tool_input=action_request.tool_input,
                timeout_seconds=action_request.timeout_seconds
            )

            # Check permission using the tool system
            permission_decision = await self.tool_manager.permission_checker.check_permission(tool_request)

            # Convert PermissionDecision to bool
            # ALLOW -> True, DENY -> False, REQUIRE_APPROVAL -> False (not yet approved)
            return permission_decision == PermissionDecision.ALLOW

        except Exception as e:
            logger.error(f"Error in tool system permission checker: {e}")
            # Default to False (deny) on error for security
            return False