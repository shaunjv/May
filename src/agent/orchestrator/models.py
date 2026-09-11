"""Models for the Orchestrator component."""

from pydantic import BaseModel
from agent.intent_manager.models import Intent
from agent.planner.models import Plan
from agent.executor.models import ExecutionResult


class OrchestrationResult(BaseModel):
    """Result of an orchestration run."""

    task_id: str
    user_input: str
    intent: Intent | None = None
    plan: Plan | None = None
    execution_result: ExecutionResult | None = None
    status: str  # e.g., "completed", "failed_intent", "failed_planning", "failed_execution"
    error: str | None = None