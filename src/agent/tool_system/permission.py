"""
Permission policy and checker for the Tool System V1.
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from .interfaces import PermissionPolicy, PermissionChecker
from .models import ToolRequest, PermissionDecision, ToolMetadata, OperationType, RiskLevel

logger = logging.getLogger(__name__)


class PermissionPolicyImpl(PermissionPolicy):
    """Concrete implementation of the PermissionPolicy interface."""

    def __init__(self, workspace_root: Optional[str] = None):
        """
        Initialize the permission policy.

        Args:
            workspace_root: Root directory of the allowed workspace. If None,
                          uses current working directory.
        """
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self._protected_files: Set[str] = {
            # Environment and credential files
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "id_rsa",
            "id_dsa",
            "id_ecdsa",
            "id_ed25519",
            # AWS credentials
            "~/.aws/credentials",
            # Docker credentials
            "~/.docker/config.json",
            # Git credentials
            "~/.git-credentials",
            # npm/yarn/pip credentials
            "~/.npmrc",
            "~/.yarnrc",
            "~/.pypirc",
            # General secrets patterns
            "*secret*",
            "*credential*",
            "*password*",
            "*token*",
            "*api_key*",
            "*access_key*",
            "*private*",
        }
        # Compile regex patterns for protected files
        self._protected_patterns = [
            re.compile(self._glob_to_regex(pattern))
            for pattern in self._protected_files
        ]

        # Define which operations require approval
        self._approval_required: Set[str] = {
            "file_write",
            "file_edit",
            "terminal",
            "git_commit",
            "git_push"
        }

        logger.info(f"Permission policy initialized with workspace root: {self.workspace_root}")

    def _glob_to_regex(self, pattern: str) -> str:
        """Convert a glob pattern to a regex pattern."""
        # Handle home directory expansion
        if pattern.startswith("~/"):
            pattern = ".+" + pattern[1:]  # Match any path that ends with the pattern after ~/

        # Escape special regex characters except * and ?
        pattern = re.escape(pattern)
        # Convert * to .* and ? to .
        pattern = pattern.replace(r"\*", ".*").replace(r"\?", ".")
        # Ensure it matches the full path or just the filename for basename matching
        return f"^{pattern}$"

    def _is_path_in_workspace(self, path: str) -> bool:
        """Check if a path is within the configured workspace."""
        try:
            abs_path = Path(path).resolve()
            # Check if path is within workspace root
            return self.workspace_root in abs_path.parents or abs_path == self.workspace_root
        except Exception:
            return False

    def _is_path_traversal(self, path: str) -> bool:
        """Check if a path contains path traversal attempts."""
        # Normalize path and check for parent directory references
        # Convert to absolute path first to properly check for traversal
        try:
            abs_path = Path(path).resolve()
            # Check if any part of the path is '..' after resolving
            parts = abs_path.parts
            return ".." in parts
        except Exception:
            # If we can't resolve the path, check the string representation
            normalized = os.path.normpath(path)
            return ".." in normalized or normalized.startswith("../")

    def _is_protected_file(self, path: str) -> bool:
        """Check if a file matches protected file patterns."""
        try:
            filename = os.path.basename(path)
            # Also check if the full path matches patterns (for paths like ~/.env)
            for pattern in self._protected_patterns:
                if pattern.match(filename):
                    return True
                # Also try matching against the full path for patterns like ~/.env
                if pattern.match(path):
                    return True
            return False
        except Exception:
            return False

    async def _check_permission_with_metadata(self, tool_metadata: ToolMetadata, request: ToolRequest) -> PermissionDecision:
        """
        Check permission using the provided tool metadata.

        Args:
            tool_metadata: The metadata of the tool to check permissions for.
            request: The tool request to check

        Returns:
            PermissionDecision: Decision on whether the request is permitted
        """
        try:
            # Extract target resource from tool_input if available
            target_resource = None

            if tool_metadata and tool_metadata.target_resource:
                target_resource = request.tool_input.get(tool_metadata.target_resource)

            # Check 1: Default deny if we can't determine permissions
            if not tool_metadata:
                logger.warning(f"No metadata found for tool: {request.tool_name}")
                return PermissionDecision.DENY

            # Check 2: Workspace boundary enforcement
            if tool_metadata.allowed_workspace_only and target_resource:
                if isinstance(target_resource, str):
                    if not self._is_path_in_workspace(target_resource):
                        logger.warning(f"Access denied: {target_resource} is outside workspace")
                        return PermissionDecision.DENY

                    # Check for path traversal
                    if self._is_path_traversal(target_resource):
                        logger.warning(f"Access denied: Path traversal detected in {target_resource}")
                        return PermissionDecision.DENY

            # Check 3: Protected file protection
            if tool_metadata.target_resource and target_resource:
                if isinstance(target_resource, str):
                    if self._is_protected_file(target_resource):
                        logger.warning(f"Access denied: {target_resource} is a protected file")
                        return PermissionDecision.DENY

            # Check 4: Determine if approval is required
            if tool_metadata.name in self._approval_required:
                logger.info(f"Approval required for tool: {tool_metadata.name}")
                return PermissionDecision.REQUIRE_APPROVAL

            # Check 5: Default allow for read-only operations within workspace
            if tool_metadata.is_read_only:
                return PermissionDecision.ALLOW

            # For write operations that don't require explicit approval, allow but log
            logger.info(f"Permission granted for tool: {tool_metadata.name}")
            return PermissionDecision.ALLOW

        except Exception as e:
            logger.error(f"Error checking permission for {request.tool_name}: {e}")
            # Default deny on error for security
            return PermissionDecision.DENY

    async def check_permission(self, request: ToolRequest) -> PermissionDecision:
        """
        Check if a tool request is permitted according to the policy.

        Args:
            request: The tool request to check

        Returns:
            PermissionDecision: Decision on whether the request is permitted
        """
        try:
            # Extract target resource from tool_input if available
            target_resource = None
            tool_metadata = None

            # Try to get tool metadata from request (this would come from registry in practice)
            # For now, we'll create basic metadata based on tool name
            if request.tool_name:
                tool_metadata = self._create_tool_metadata_from_name(request.tool_name)

            if tool_metadata and tool_metadata.target_resource:
                target_resource = request.tool_input.get(tool_metadata.target_resource)

            # Check 1: Default deny if we can't determine permissions
            if not tool_metadata:
                logger.warning(f"No metadata found for tool: {request.tool_name}")
                return PermissionDecision.DENY

            # If we have tool_metadata, delegate to the new method
            return await self._check_permission_with_metadata(tool_metadata, request)

        except Exception as e:
            logger.error(f"Error checking permission for {request.tool_name}: {e}")
            # Default deny on error for security
            return PermissionDecision.DENY

    def _create_tool_metadata_from_name(self, tool_name: str) -> Optional[ToolMetadata]:
        """Create basic tool metadata from tool name for permission checking."""
        # This is a simplified version - in practice, this would come from the registry
        # Map tool names to their correct target_resource values
        target_resource_map = {
            "file_read": "file_path",
            "file_write": "file_path",
            "file_edit": "file_path",
            "file_search": "pattern",  # or could be "search_pattern"
            "directory_list": "directory_path",
            "terminal": "command",
            "git": "operation"
        }

        tool_map = {
            "file_read": OperationType.READ,
            "file_write": OperationType.WRITE,
            "file_edit": OperationType.WRITE,
            "file_search": OperationType.SEARCH,
            "directory_list": OperationType.LIST,
            "terminal": OperationType.EXECUTE,
            "git": OperationType.GIT_READ,  # Base type
        }

        if tool_name not in tool_map:
            return None

        operation_type = tool_map[tool_name]

        # Determine risk level based on operation type
        risk_map = {
            OperationType.READ: RiskLevel.LOW,
            OperationType.SEARCH: RiskLevel.LOW,
            OperationType.LIST: RiskLevel.LOW,
            OperationType.WRITE: RiskLevel.MEDIUM,
            OperationType.EXECUTE: RiskLevel.HIGH,
            OperationType.GIT_READ: RiskLevel.LOW,
        }

        risk_level = risk_map.get(operation_type, RiskLevel.MEDIUM)

        # Determine if approval is required
        approval_required_tools = {"file_write", "file_edit", "terminal", "git"}
        requires_approval = tool_name in approval_required_tools

        return ToolMetadata(
            name=tool_name,
            description=f"{tool_name} tool",
            operation_type=operation_type,
            risk_level=risk_level,
            is_read_only=operation_type in [OperationType.READ, OperationType.SEARCH, OperationType.LIST],
            required_permission=tool_name.upper(),
            target_resource=target_resource_map.get(tool_name, tool_name.split("_")[-1] if "_" in tool_name else tool_name),
            allowed_workspace_only=True
        )


class PermissionCheckerImpl(PermissionChecker):
    """Concrete implementation of the PermissionChecker interface."""

    def __init__(self, permission_policy: PermissionPolicy, registry: Optional[Any] = None):
        """
        Initialize the permission checker.

        Args:
            permission_policy: The permission policy to use for checking
            registry: Optional tool registry to look up tool metadata
        """
        self.permission_policy = permission_policy
        self.registry = registry
        logger.info("Permission checker initialized")

    async def check_permission(self, request: ToolRequest) -> PermissionDecision:
        """
        Check if a tool request is permitted.

        Args:
            request: The tool request to check

        Returns:
            PermissionDecision: Decision on whether the request is permitted
        """
        try:
            # If we have a registry, try to get the tool from the registry
            if self.registry is not None:
                tool = await self.registry.get_tool(request.tool_name)
                if tool is not None:
                    # Use the tool's metadata for permission checking
                    return await self.permission_policy._check_permission_with_metadata(tool.metadata, request)

            # Fall back to inferring metadata from tool name (original behavior)
            return await self.permission_policy.check_permission(request)

        except Exception as e:
            logger.error(f"Error checking permission for {request.tool_name}: {e}")
            # Default deny on error for security
            return PermissionDecision.DENY