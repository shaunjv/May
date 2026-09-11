"""
Concrete tool implementations for the Tool System V1.
"""

import asyncio
import logging
import os
import subprocess
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
from .models import Tool, ToolMetadata, ToolRequest, ToolResult, OperationType, RiskLevel, PermissionDecision

logger = logging.getLogger(__name__)


class BaseTool(Tool):
    """Base class for all tools."""

    def __init__(self, metadata: ToolMetadata):
        """
        Initialize the base tool.

        Args:
            metadata: The tool metadata
        """
        super().__init__(metadata=metadata)
        self.metadata = metadata

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Execute the tool with the given request.
        This method should be overridden by concrete implementations.

        Args:
            request: The tool request to execute

        Returns:
            ToolResult: Result of the tool execution
        """
        raise NotImplementedError("Tool.execute() must be implemented by subclasses")


class FileReadTool(BaseTool):
    """Tool for reading file contents."""

    def __init__(self):
        metadata = ToolMetadata(
            name="file_read",
            description="Read contents of a file",
            operation_type=OperationType.READ,
            risk_level=RiskLevel.LOW,
            is_read_only=True,
            required_permission="READ",
            target_resource="file_path",
            allowed_workspace_only=True
        )
        super().__init__(metadata)

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Read contents of a file.

        Args:
            request: The tool request containing file_path in tool_input

        Returns:
            ToolResult: Result containing file contents or error
        """
        try:
            file_path = request.tool_input.get("file_path")
            if not file_path:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="file_path is required in tool_input"
                )

            # Security: Basic path validation (more thorough validation in permission checker)
            path = Path(file_path)
            if not path.is_absolute():
                # Make relative paths relative to current working directory
                path = Path.cwd() / path

            # Check if file exists
            if not path.exists():
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"File not found: {file_path}"
                )

            # Check if it's actually a file
            if not path.is_file():
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"Path is not a file: {file_path}"
                )

            # Read file contents
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=True,
                output={
                    "content": content,
                    "file_path": str(path),
                    "size_bytes": len(content.encode('utf-8'))
                }
            )

        except UnicodeDecodeError:
            # Try to read as binary if UTF-8 fails
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=True,
                    output={
                        "content": content.hex(),  # Return as hex string for binary
                        "file_path": str(path),
                        "size_bytes": len(content),
                        "is_binary": True
                    }
                )
            except Exception as e:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"Could not read file (binary read failed): {str(e)}"
                )
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=False,
                error=f"Error reading file: {str(e)}"
            )


class FileWriteTool(BaseTool):
    """Tool for writing file contents."""

    def __init__(self):
        metadata = ToolMetadata(
            name="file_write",
            description="Write contents to a file",
            operation_type=OperationType.WRITE,
            risk_level=RiskLevel.MEDIUM,
            is_read_only=False,
            required_permission="WRITE",
            target_resource="file_path",
            allowed_workspace_only=True
        )
        super().__init__(metadata)

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Write contents to a file.

        Args:
            request: The tool request containing file_path and content in tool_input

        Returns:
            ToolResult: Result indicating success or error
        """
        try:
            file_path = request.tool_input.get("file_path")
            content = request.tool_input.get("content")

            if file_path is None:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="file_path is required in tool_input"
                )

            if content is None:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="content is required in tool_input"
                )

            # Security: Basic path validation (more thorough validation in permission checker)
            path = Path(file_path)
            if not path.is_absolute():
                # Make relative paths relative to current working directory
                path = Path.cwd() / path

            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file contents
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=True,
                output={
                    "file_path": str(path),
                    "size_bytes": len(content.encode('utf-8')),
                    "bytes_written": len(content.encode('utf-8'))
                }
            )

        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=False,
                error=f"Error writing file: {str(e)}"
            )


class FileEditTool(BaseTool):
    """Tool for editing file contents."""

    def __init__(self):
        metadata = ToolMetadata(
            name="file_edit",
            description="Edit contents of a file",
            operation_type=OperationType.WRITE,
            risk_level=RiskLevel.MEDIUM,
            is_read_only=False,
            required_permission="WRITE",
            target_resource="file_path",
            allowed_workspace_only=True
        )
        super().__init__(metadata)

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Edit contents of a file by replacing text.

        Args:
            request: The tool request containing file_path, old_string, and new_string in tool_input

        Returns:
            ToolResult: Result indicating success or error
        """
        try:
            file_path = request.tool_input.get("file_path")
            old_string = request.tool_input.get("old_string")
            new_string = request.tool_input.get("new_string")
            replace_all = request.tool_input.get("replace_all", False)

            if file_path is None:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="file_path is required in tool_input"
                )

            if old_string is None:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="old_string is required in tool_input"
                )

            if new_string is None:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="new_string is required in tool_input"
                )

            # Security: Basic path validation (more thorough validation in permission checker)
            path = Path(file_path)
            if not path.is_absolute():
                # Make relative paths relative to current working directory
                path = Path.cwd() / path

            # Check if file exists
            if not path.exists():
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"File not found: {file_path}"
                )

            # Check if it's actually a file
            if not path.is_file():
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"Path is not a file: {file_path}"
                )

            # Read file contents
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                # Replace only first occurrence
                new_content = content.replace(old_string, new_string, 1)

            # Check if any changes were made
            if new_content == content:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=True,
                    output={
                        "file_path": str(path),
                        "changes_made": 0,
                        "message": "No matches found for replacement"
                    }
                )

            # Write back to file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            # Count occurrences for reporting
            occurrences = content.count(old_string)
            if replace_all:
                changes_made = occurrences
            else:
                changes_made = 1 if occurrences > 0 else 0

            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=True,
                output={
                    "file_path": str(path),
                    "changes_made": changes_made,
                    "old_string": old_string,
                    "new_string": new_string,
                    "size_bytes": len(new_content.encode('utf-8'))
                }
            )

        except Exception as e:
            logger.error(f"Error editing file {file_path}: {e}")
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=False,
                error=f"Error editing file: {str(e)}"
            )


class FileSearchTool(BaseTool):
    """Tool for searching files or content within files."""

    def __init__(self):
        metadata = ToolMetadata(
            name="file_search",
            description="Search for files or content within files",
            operation_type=OperationType.SEARCH,
            risk_level=RiskLevel.LOW,
            is_read_only=True,
            required_permission="READ",
            target_resource="search_pattern",
            allowed_workspace_only=True
        )
        super().__init__(metadata)

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Search for files or content within files.

        Args:
            request: The tool request containing search parameters in tool_input

        Returns:
            ToolResult: Result containing search results
        """
        try:
            pattern = request.tool_input.get("pattern")
            search_type = request.tool_input.get("type", "file")  # "file" or "content"
            path_str = request.tool_input.get("path", ".")
            file_pattern = request.tool_input.get("file_pattern", "*")

            if pattern is None:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="pattern is required in tool_input"
                )

            # Security: Basic path validation (more thorough validation in permission checker)
            base_path = Path(path_str)
            if not base_path.is_absolute():
                # Make relative paths relative to current working directory
                base_path = Path.cwd() / base_path

            # Check if path exists
            if not base_path.exists():
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"Path not found: {path_str}"
                )

            results = []

            if search_type == "file":
                # Search for files by name pattern
                search_pattern = str(base_path / "**" / file_pattern)
                for file_path in glob.glob(search_pattern, recursive=True):
                    if Path(file_path).is_file():
                        # Check if filename matches the search pattern
                        if pattern.lower() in Path(file_path).name.lower():
                            results.append({
                                "type": "file",
                                "path": file_path,
                                "name": Path(file_path).name
                            })

            elif search_type == "content":
                # Search for content within files
                for file_path in glob.glob(str(base_path / "**" / file_pattern), recursive=True):
                    if Path(file_path).is_file():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()

                            # Search for pattern in content
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern.lower() in line.lower():
                                    results.append({
                                        "type": "content",
                                        "file_path": file_path,
                                        "line_number": i + 1,
                                        "line_content": line,
                                        "match": pattern
                                    })
                        except UnicodeDecodeError:
                            # Skip binary files
                            continue
                        except Exception:
                            # Skip files that can't be read
                            continue
            else:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"Invalid search type: {search_type}. Must be 'file' or 'content'"
                )

            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=True,
                output={
                    "results": results,
                    "count": len(results),
                    "search_pattern": pattern,
                    "search_type": search_type,
                    "path": str(base_path)
                }
            )

        except Exception as e:
            logger.error(f"Error searching with pattern {pattern}: {e}")
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=False,
                error=f"Error searching: {str(e)}"
            )


class DirectoryListTool(BaseTool):
    """Tool for listing directory contents."""

    def __init__(self):
        metadata = ToolMetadata(
            name="directory_list",
            description="List contents of a directory",
            operation_type=OperationType.LIST,
            risk_level=RiskLevel.LOW,
            is_read_only=True,
            required_permission="READ",
            target_resource="directory_path",
            allowed_workspace_only=True
        )
        super().__init__(metadata)

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        List contents of a directory.

        Args:
            request: The tool request containing directory_path in tool_input

        Returns:
            ToolResult: Result containing directory listing
        """
        try:
            directory_path = request.tool_input.get("directory_path")
            if directory_path is None:
                directory_path = "."  # Default to current directory

            # Security: Basic path validation (more thorough validation in permission checker)
            path = Path(directory_path)
            if not path.is_absolute():
                # Make relative paths relative to current working directory
                path = Path.cwd() / path

            # Check if directory exists
            if not path.exists():
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"Directory not found: {directory_path}"
                )

            # Check if it's actually a directory
            if not path.is_dir():
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"Path is not a directory: {directory_path}"
                )

            # List directory contents
            items = []
            for item in path.iterdir():
                item_info = {
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                }
                items.append(item_info)

            # Sort: directories first, then files, both alphabetically
            items.sort(key=lambda x: (x["type"] == "file", x["name"].lower()))

            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=True,
                output={
                    "directory": str(path),
                    "items": items,
                    "count": len(items)
                }
            )

        except Exception as e:
            logger.error(f"Error listing directory {directory_path}: {e}")
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=False,
                error=f"Error listing directory: {str(e)}"
            )


class TerminalTool(BaseTool):
    """Tool for executing terminal commands."""

    def __init__(self):
        metadata = ToolMetadata(
            name="terminal",
            description="Execute terminal commands",
            operation_type=OperationType.EXECUTE,
            risk_level=RiskLevel.HIGH,
            is_read_only=False,
            required_permission="EXECUTE",
            target_resource="command",
            allowed_workspace_only=True
        )
        super().__init__(metadata)

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Execute a terminal command.

        Args:
            request: The tool request containing command in tool_input

        Returns:
            ToolResult: Result containing command output or error
        """
        try:
            command = request.tool_input.get("command")
            if command is None:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="command is required in tool_input"
                )

            # Get working directory (default to current)
            cwd_str = request.tool_input.get("cwd", ".")
            cwd = Path(cwd_str)
            if not cwd.is_absolute():
                cwd = Path.cwd() / cwd

            # Security checks would be in permission checker - this is just execution
            logger.info(f"Executing command: {command}")

            # Desktop-approved commands are argument lists and never invoke a shell.
            # String commands retain legacy behaviour for direct V1 callers; the
            # desktop WorkspaceGuard rejects them before this point.
            use_shell = isinstance(command, str)
            if not use_shell and not (isinstance(command, list) and all(isinstance(part, str) for part in command)):
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="command must be a string or a list of argument strings",
                )
            result = subprocess.run(
                command,
                shell=use_shell,
                capture_output=True,
                text=True,
                cwd=str(cwd),
                timeout=request.timeout_seconds or 30  # Default timeout
            )

            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=result.returncode == 0,
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "command": command,
                    "cwd": str(cwd)
                },
                error=result.stderr if result.returncode != 0 else None
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=False,
                error=f"Command timed out after {request.timeout_seconds or 30} seconds"
            )
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=False,
                error=f"Error executing command: {str(e)}"
            )


class GitTool(BaseTool):
    """Tool for executing git operations."""

    def __init__(self):
        # Initialize with placeholder metadata - will be updated in execute method
        metadata = ToolMetadata(
            name="git",
            description="Execute git operations",
            operation_type=OperationType.GIT_READ,  # Placeholder
            risk_level=RiskLevel.LOW,  # Placeholder
            is_read_only=True,  # Placeholder
            required_permission="GIT_READ",  # Placeholder
            target_resource="git_operation",
            allowed_workspace_only=True
        )
        super().__init__(metadata)

    def _get_git_metadata_for_operation(self, operation: str) -> ToolMetadata:
        """
        Get the appropriate metadata for a specific git operation.

        Args:
            operation: The git operation (status, diff, log, add, commit, push, etc.)

        Returns:
            ToolMetadata: Metadata appropriate for the operation
        """
        # Define metadata for different git operations
        if operation in ["status", "diff", "log"]:
            return ToolMetadata(
                name="git",
                description=f"Execute git {operation} operation",
                operation_type=OperationType.GIT_READ,
                risk_level=RiskLevel.LOW,
                is_read_only=True,
                required_permission="GIT_READ",
                target_resource="git_operation",
                allowed_workspace_only=True
            )
        elif operation == "add":
            return ToolMetadata(
                name="git",
                description="Execute git add operation",
                operation_type=OperationType.GIT_WRITE,
                risk_level=RiskLevel.MEDIUM,
                is_read_only=False,
                required_permission="GIT_WRITE",
                target_resource="git_operation",
                allowed_workspace_only=True
            )
        elif operation == "commit":
            return ToolMetadata(
                name="git",
                description="Execute git commit operation",
                operation_type=OperationType.GIT_COMMIT,
                risk_level=RiskLevel.MEDIUM,
                is_read_only=False,
                required_permission="GIT_COMMIT",
                target_resource="git_operation",
                allowed_workspace_only=True
            )
        elif operation == "push":
            return ToolMetadata(
                name="git",
                description="Execute git push operation",
                operation_type=OperationType.GIT_PUSH,
                risk_level=RiskLevel.HIGH,
                is_read_only=False,
                required_permission="GIT_PUSH",
                target_resource="git_operation",
                allowed_workspace_only=True
            )
        elif operation in ["branch", "checkout", "merge", "pull"]:
            # These are generally read-only or medium risk
            return ToolMetadata(
                name="git",
                description=f"Execute git {operation} operation",
                operation_type=OperationType.GIT_READ,
                risk_level=RiskLevel.LOW,
                is_read_only=True,
                required_permission="GIT_READ",
                target_resource="git_operation",
                allowed_workspace_only=True
            )
        else:
            # Default fallback for unknown operations
            return ToolMetadata(
                name="git",
                description=f"Execute git {operation} operation",
                operation_type=OperationType.GIT_READ,
                risk_level=RiskLevel.LOW,
                is_read_only=True,
                required_permission="GIT_READ",
                target_resource="git_operation",
                allowed_workspace_only=True
            )

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Execute a git operation.

        Args:
            request: The tool request containing git parameters in tool_input

        Returns:
            ToolResult: Result containing git output or error
        """
        try:
            operation = request.tool_input.get("operation")
            if operation is None:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error="operation is required in tool_input (e.g., 'status', 'diff', 'log', 'add', 'commit', 'push')"
                )

            # Update metadata based on the operation being performed
            self.metadata = self._get_git_metadata_for_operation(operation)

            # Get working directory (default to current)
            cwd_str = request.tool_input.get("cwd", ".")
            cwd = Path(cwd_str)
            if not cwd.is_absolute():
                cwd = Path.cwd() / cwd

            # Check if we're in a git repository
            git_dir = cwd / ".git"
            if not git_dir.exists():
                # Check parent directories
                parent = cwd.parent
                while parent != parent.parent:  # Stop at root
                    git_dir = parent / ".git"
                    if git_dir.exists():
                        break
                    parent = parent.parent
                else:
                    return ToolResult(
                        request_id=request.request_id,
                        tool_name=self.metadata.name,
                        success=False,
                        error=f"Not a git repository: {cwd}"
                    )

            # Build git command
            git_command = ["git", "-c", "color.ui=false"]  # Disable color for cleaner output

            # Add operation-specific arguments
            if operation == "status":
                git_command.append("status")
                git_command.append("--porcelain")  # Machine-readable format

            elif operation == "diff":
                git_command.append("diff")
                # Add any additional args
                args = request.tool_input.get("args", [])
                if isinstance(args, list):
                    git_command.extend(args)
                elif isinstance(args, str):
                    git_command.append(args)

            elif operation == "log":
                git_command.append("log")
                git_command.append("--oneline")
                git_command.append("-n")  # Limit number of commits
                limit = request.tool_input.get("limit", 10)
                git_command.append(str(limit))

            elif operation == "add":
                git_command.append("add")
                paths = request.tool_input.get("paths", [])
                if isinstance(paths, str):
                    paths = [paths]
                git_command.extend(paths)

            elif operation == "commit":
                git_command.append("commit")
                message = request.tool_input.get("message")
                if message:
                    git_command.extend(["-m", message])
                # Add --all flag if specified
                if request.tool_input.get("all", False):
                    git_command.append("--all")

            elif operation == "push":
                git_command.append("push")
                remote = request.tool_input.get("remote", "origin")
                branch = request.tool_input.get("branch")
                git_command.append(remote)
                if branch:
                    git_command.append(branch)

            elif operation == "branch":
                git_command.append("branch")

            elif operation == "checkout":
                git_command.append("checkout")
                branch = request.tool_input.get("branch")
                if branch:
                    git_command.append(branch)

            elif operation == "merge":
                git_command.append("merge")
                branch = request.tool_input.get("branch")
                if branch:
                    git_command.append(branch)

            elif operation == "pull":
                git_command.append("pull")
                remote = request.tool_input.get("remote", "origin")
                branch = request.tool_input.get("branch")
                git_command.append(remote)
                if branch:
                    git_command.append(branch)

            else:
                return ToolResult(
                    request_id=request.request_id,
                    tool_name=self.metadata.name,
                    success=False,
                    error=f"Unsupported git operation: {operation}"
                )

            # Execute git command
            logger.info(f"Executing git command: {' '.join(git_command)}")
            result = subprocess.run(
                git_command,
                capture_output=True,
                text=True,
                cwd=str(cwd),
                timeout=request.timeout_seconds or 30  # Default timeout
            )

            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=result.returncode == 0,
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "operation": operation,
                    "command": " ".join(git_command),
                    "cwd": str(cwd)
                },
                error=result.stderr if result.returncode != 0 else None
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=False,
                error=f"Git command timed out after {request.timeout_seconds or 30} seconds"
            )
        except Exception as e:
            logger.error(f"Error executing git operation {operation}: {e}")
            return ToolResult(
                request_id=request.request_id,
                tool_name=self.metadata.name,
                success=False,
                error=f"Error executing git operation: {str(e)}"
            )
