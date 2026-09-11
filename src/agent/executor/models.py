"""
Data models for the Executor component.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ActionRequest(BaseModel):
    """Request to execute a specific action/tool."""
    action_id: str = Field(description="Unique identifier for the action")
    step_id: str = Field(description="ID of the plan step this action belongs to")
    tool_name: str = Field(description="Name of the tool to execute")
    tool_input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters for the tool"
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        description="Optional timeout for the action execution"
    )


class ToolResult(BaseModel):
    """Result of executing a tool/action."""
    action_id: str = Field(description="Identifier of the action that was executed")
    step_id: str = Field(description="ID of the plan step this action belongs to")
    success: bool = Field(description="Whether the action executed successfully")
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


class ExecutionState(str, Enum):
    """Execution states for plan steps."""
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class ExecutorConfig(BaseModel):
    """Configuration for the Executor."""
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for failed steps"
    )
    retry_delay_ms: int = Field(
        default=1000,
        description="Delay between retry attempts in milliseconds"
    )
    default_timeout_seconds: int = Field(
        default=30,
        description="Default timeout for actions in seconds"
    )


class ExecutionResult(BaseModel):
    """Result of executing a plan or portion of a plan."""
    execution_id: str = Field(description="Unique identifier for this execution")
    plan_id: str = Field(description="ID of the plan that was executed")
    task_id: str = Field(description="ID of the task this plan belongs to")
    status: ExecutionState = Field(description="Final execution status")
    started_at: Optional[str] = Field(
        default=None,
        description="Timestamp when execution started (ISO format)"
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="Timestamp when execution completed (ISO format)"
    )
    step_results: Dict[str, ToolResult] = Field(
        default_factory=dict,
        description="Results for each executed step, keyed by step_id"
    )
    failed_steps: List[str] = Field(
        default_factory=list,
        description="List of step IDs that failed during execution"
    )
    cancelled_steps: List[str] = Field(
        default_factory=list,
        description="List of step IDs that were cancelled"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if execution failed overall"
    )