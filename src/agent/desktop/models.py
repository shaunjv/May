"""Immutable, serialisable domain models for approved desktop-agent work."""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from agent.executor.models import ActionRequest


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    """Hash canonical JSON so approval is bound to contents, not object identity."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class TaskState(str, Enum):
    PLANNING = "PLANNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class TaskRecord(BaseModel):
    task_id: str
    conversation_id: str
    workspace_path: str
    state: TaskState = TaskState.PLANNING
    created_at: str = Field(default_factory=utcnow)
    planning_started_at: Optional[str] = None
    planning_completed_at: Optional[str] = None
    approval_requested_at: Optional[str] = None
    approved_at: Optional[str] = None
    execution_started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    cancellation_reason: Optional[str] = None
    plan_id: Optional[str] = None
    plan_version: Optional[int] = None
    plan_hash: Optional[str] = None
    action_manifest_hash: Optional[str] = None


class ApprovedExecutionPlan(BaseModel):
    """The exact immutable action manifest approved by the user."""
    model_config = ConfigDict(frozen=True)

    task_id: str
    plan_id: str
    plan_version: int
    plan_hash: str
    action_manifest_hash: str
    actions: List[ActionRequest]

    @classmethod
    def create(cls, task_id: str, plan: BaseModel, actions: List[ActionRequest]) -> "ApprovedExecutionPlan":
        plan_data = plan.model_dump(mode="json")
        return cls(
            task_id=task_id,
            plan_id=plan_data["plan_id"],
            plan_version=plan_data["version"],
            plan_hash=canonical_hash(plan_data),
            action_manifest_hash=canonical_hash([a.model_dump(mode="json") for a in actions]),
            actions=actions,
        )

    def verify(self, plan: BaseModel) -> bool:
        return (
            self.plan_id == plan.plan_id
            and self.plan_version == plan.version
            and self.plan_hash == canonical_hash(plan)
            and self.action_manifest_hash == canonical_hash([a.model_dump(mode="json") for a in self.actions])
        )
