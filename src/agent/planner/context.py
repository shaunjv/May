"""
Context model for the Planner component.
"""

from typing import List, Optional, Set
from pydantic import BaseModel, Field, model_validator
from ..intent_manager.models import Intent


class PlannerContext(BaseModel):
    """Context provided to the Planner for making planning decisions."""
    intent: Intent = Field(description="The user's structured intent")
    task_id: str = Field(description="Identifier for the task being planned")
    task_state: dict = Field(
        default_factory=dict,
        description="Current state of the task (managed by Task State Manager)"
    )
    active_plan: Optional[dict] = Field(
        default=None,
        description="Currently active plan (if any), as a dictionary for flexibility"
    )
    recent_observations: List[str] = Field(
        default_factory=list,
        description="Recent observations from task execution"
    )
    available_capabilities: List[str] = Field(
        default_factory=list,
        description="Abstract capabilities available for planning (e.g., 'filesystem.read')"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Constraints that must be respected during planning"
    )
    relevant_context: dict = Field(
        default_factory=dict,
        description="Additional relevant context information"
    )
    planning_mode: str = Field(
        description="Planning mode: INITIAL, CONTINUE, or REPLAN"
    )

    @model_validator(mode='after')
    def validate_context(self) -> 'PlannerContext':
        """Validate context after all fields are parsed."""
        # Validate planning_mode
        valid_modes = {"INITIAL", "CONTINUE", "REPLAN"}
        if self.planning_mode not in valid_modes:
            raise ValueError(f'planning_mode must be one of {valid_modes}')

        return self