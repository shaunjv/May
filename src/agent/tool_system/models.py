"""
Data models for the Tool System V1.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, validator, model_validator


class OperationType(str, Enum):
    """Types of operations that tools can perform."""
    READ = "READ"
    WRITE = "WRITE"
    EXECUTE = "EXECUTE"
    SEARCH = "SEARCH"
    LIST = "LIST"
    GIT_READ = "GIT_READ"
    GIT_WRITE = "GIT_WRITE"
    GIT_COMMIT = "GIT_COMMIT"
    GIT_PUSH = "GIT_PUSH"


class RiskLevel(str, Enum):
    """Risk levels for tool operations."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PermissionDecision(str, Enum):
    """Permission decisions for tool requests."""
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class ToolMetadata(BaseModel):
    """Metadata for a tool that describes its properties and requirements."""

    name: str = Field(description="Unique name of the tool")
    description: str = Field(description="Human-readable description of what the tool does")
    operation_type: OperationType = Field(description="Type of operation the tool performs")
    risk_level: RiskLevel = Field(description="Risk level associated with the tool")
    is_read_only: bool = Field(description="Whether the tool is read-only (does not modify state)")
    required_permission: str = Field(description="Permission required to use the tool")
    target_resource: Optional[str] = Field(
        default=None,
        description="Target resource pattern or identifier (e.g., file path pattern)"
    )
    allowed_workspace_only: bool = Field(
        default=True,
        description="Whether the tool is restricted to the configured workspace"
    )
    capabilities: Set[str] = Field(
        default_factory=set,
        description="Machine-checkable capabilities implemented by this tool"
    )
    argument_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-schema-like description of accepted tool input"
    )

    @model_validator(mode="after")
    def add_builtin_contracts(self) -> "ToolMetadata":
        """Supply explicit contracts for bundled V1 tools without breaking plugins."""
        contracts = {
            "file_read": ({"filesystem.read"}, {"required": ["file_path"]}),
            "file_write": ({"filesystem.write"}, {"required": ["file_path", "content"]}),
            "file_edit": ({"filesystem.write"}, {"required": ["file_path"]}),
            "file_search": ({"filesystem.read"}, {"required": ["pattern"]}),
            "directory_list": ({"filesystem.list"}, {"required": ["directory_path"]}),
            "terminal": ({"terminal.execute"}, {"required": ["command", "cwd"]}),
            "git": ({"git.inspect"}, {"required": ["operation", "cwd"]}),
        }
        if self.name in contracts:
            capabilities, schema = contracts[self.name]
            if not self.capabilities:
                self.capabilities = capabilities
            if not self.argument_schema:
                self.argument_schema = schema
        return self

    class Config:
        use_enum_values = True


class ToolRequest(BaseModel):
    """Request to execute a specific tool."""

    request_id: str = Field(description="Unique identifier for the request")
    tool_name: str = Field(description="Name of the tool to execute")
    tool_input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters for the tool"
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        description="Optional timeout for the tool execution in seconds"
    )

    @validator('tool_name')
    def tool_name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('tool_name must not be empty')
        return v.strip()


class ToolResult(BaseModel):
    """Result of executing a tool."""

    request_id: str = Field(description="Identifier of the request that was executed")
    tool_name: str = Field(description="Name of the tool that was executed")
    success: bool = Field(description="Whether the tool executed successfully")
    output: Optional[Any] = Field(
        default=None,
        description="Output data from the tool execution"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    execution_time_ms: Optional[int] = Field(
        default=None,
        description="Execution time in milliseconds"
    )
    permission_decision: Optional[PermissionDecision] = Field(
        default=None,
        description="Permission decision made for this request"
    )


class Tool(BaseModel):
    """A tool that can be executed by the tool system."""

    metadata: ToolMetadata = Field(description="Metadata describing the tool")

    # The actual implementation of the tool - this would be a callable/function
    # In practice, this would be set to an instance that implements the tool logic
    # For the interface definition, we'll use a generic callable type

    class Config:
        arbitrary_types_allowed = True

    def execute(self, request: ToolRequest) -> ToolResult:
        """
        Execute the tool with the given request.
        This is a placeholder - actual implementations would override this
        or provide their own execution logic.

        Args:
            request: The tool request to execute

        Returns:
            ToolResult: Result of the tool execution
        """
        # This would be implemented by concrete tool classes
        raise NotImplementedError("Tool.execute() must be implemented by subclasses")


# Convenience functions for creating common tool metadata
def create_file_read_metadata() -> ToolMetadata:
    """Create metadata for a file read tool."""
    return ToolMetadata(
        name="file_read",
        description="Read contents of a file",
        operation_type=OperationType.READ,
        risk_level=RiskLevel.LOW,
        is_read_only=True,
        required_permission="READ",
        target_resource="file_path",
        allowed_workspace_only=True
    )


def create_file_write_metadata() -> ToolMetadata:
    """Create metadata for a file write tool."""
    return ToolMetadata(
        name="file_write",
        description="Write contents to a file",
        operation_type=OperationType.WRITE,
        risk_level=RiskLevel.MEDIUM,
        is_read_only=False,
        required_permission="WRITE",
        target_resource="file_path",
        allowed_workspace_only=True
    )


def create_file_edit_metadata() -> ToolMetadata:
    """Create metadata for a file edit tool."""
    return ToolMetadata(
        name="file_edit",
        description="Edit contents of a file",
        operation_type=OperationType.WRITE,
        risk_level=RiskLevel.MEDIUM,
        is_read_only=False,
        required_permission="WRITE",
        target_resource="file_path",
        allowed_workspace_only=True
    )


def create_file_search_metadata() -> ToolMetadata:
    """Create metadata for a file search tool."""
    return ToolMetadata(
        name="file_search",
        description="Search for files or content within files",
        operation_type=OperationType.SEARCH,
        risk_level=RiskLevel.LOW,
        is_read_only=True,
        required_permission="READ",
        target_resource="search_pattern",
        allowed_workspace_only=True
    )


def create_directory_list_metadata() -> ToolMetadata:
    """Create metadata for a directory listing tool."""
    return ToolMetadata(
        name="directory_list",
        description="List contents of a directory",
        operation_type=OperationType.LIST,
        risk_level=RiskLevel.LOW,
        is_read_only=True,
        required_permission="READ",
        target_resource="directory_path",
        allowed_workspace_only=True
    )


def create_terminal_metadata() -> ToolMetadata:
    """Create metadata for a terminal execution tool."""
    return ToolMetadata(
        name="terminal",
        description="Execute terminal commands",
        operation_type=OperationType.EXECUTE,
        risk_level=RiskLevel.HIGH,
        is_read_only=False,
        required_permission="EXECUTE",
        target_resource="command",
        allowed_workspace_only=True
    )


def create_git_metadata() -> ToolMetadata:
    """Create metadata for a git tool."""
    return ToolMetadata(
        name="git",
        description="Execute git operations",
        operation_type=OperationType.GIT_READ,  # Base type - specific ops will be refined
        risk_level=RiskLevel.LOW,
        is_read_only=True,  # Base type - specific ops will refine this
        required_permission="GIT_READ",
        target_resource="git_operation",
        allowed_workspace_only=True
    )
