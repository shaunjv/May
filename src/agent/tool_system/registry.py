"""
Tool registry implementation for the Tool System V1.
"""

import logging
from typing import Dict, List, Optional
from .interfaces import ToolRegistry
from .models import Tool

logger = logging.getLogger(__name__)


class ToolRegistryImpl(ToolRegistry):
    """Concrete implementation of the ToolRegistry interface."""

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Tool] = {}
        logger.info("Tool registry initialized")

    async def register_tool(self, tool: Tool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: The tool to register
        """
        tool_name = tool.metadata.name
        if tool_name in self._tools:
            logger.warning(f"Tool '{tool_name}' is already registered, overwriting")

        self._tools[tool_name] = tool
        logger.debug(f"Registered tool: {tool_name}")

    async def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            bool: True if tool was found and removed, False otherwise
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.debug(f"Unregistered tool: {tool_name}")
            return True

        logger.debug(f"Tool '{tool_name}' not found for unregistration")
        return False

    async def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Optional[Tool]: The tool if found, None otherwise
        """
        tool = self._tools.get(tool_name)
        if tool:
            logger.debug(f"Retrieved tool: {tool_name}")
        else:
            logger.debug(f"Tool '{tool_name}' not found")
        return tool

    async def list_tools(self) -> List[Tool]:
        """
        List all registered tools.

        Returns:
            List[Tool]: List of all registered tools
        """
        tools = list(self._tools.values())
        logger.debug(f"Listed {len(tools)} tools")
        return tools

    async def tool_exists(self, tool_name: str) -> bool:
        """
        Check if a tool exists in the registry.

        Args:
            tool_name: Name of the tool to check

        Returns:
            bool: True if tool exists, False otherwise
        """
        exists = tool_name in self._tools
        logger.debug(f"Tool '{tool_name}' exists: {exists}")
        return exists