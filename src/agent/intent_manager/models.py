"""
Data models for the Intent Manager.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class IntentType(str, Enum):
    """Top-level intent types."""
    CONVERSATION = "CONVERSATION"
    INFORMATION = "INFORMATION"
    TASK = "TASK"
    AUTOMATION = "AUTOMATION"
    COMPUTER_ACTION = "COMPUTER_ACTION"
    CLARIFICATION = "CLARIFICATION"


class TaskObjective(str, Enum):
    """Possible objectives for TASK intent type."""
    CREATE = "CREATE"
    MODIFY = "MODIFY"
    FIX = "FIX"
    ANALYZE = "ANALYZE"
    RESEARCH = "RESEARCH"
    EXECUTE = "EXECUTE"


class Priority(str, Enum):
    """Priority levels."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Intent(BaseModel):
    """
    Structured representation of user intent.

    This model represents the output of the Intent Manager after processing
    raw user input through either deterministic or LLM classification paths.
    """
    primary_intent: IntentType = Field(
        description="The primary intent classification"
    )
    sub_intents: List[IntentType] = Field(
        default_factory=list,
        description="Additional intents present in compound requests"
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain or topic area (e.g., 'code', 'email', 'calendar')"
    )
    objective: Optional[TaskObjective] = Field(
        default=None,
        description="Specific objective when primary_intent is TASK"
    )
    goal: Optional[str] = Field(
        default=None,
        description="High-level goal or desired outcome"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Constraints or limitations mentioned in the request"
    )
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria for determining when the intent is successfully fulfilled"
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Explicit deadline or time expression (e.g., 'today', 'by 5 PM')"
    )
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Priority level extracted from urgency signals"
    )
    requires_clarification: bool = Field(
        default=False,
        description="Whether the Intent Manager needs clarification from the user"
    )
    clarification_reason: Optional[str] = Field(
        default=None,
        description="Reason why clarification is needed (when requires_clarification=True)"
    )

    @model_validator(mode='after')
    def validate_intent(self) -> 'Intent':
        """Validate the intent after all fields are parsed."""
        # Validate objective for TASK intent
        if self.primary_intent == IntentType.TASK and self.objective is None:
            raise ValueError('objective is required when primary_intent is TASK')

        # Validate clarification_reason when requires_clarification is True
        if self.requires_clarification and not self.clarification_reason:
            raise ValueError('clarification_reason is required when requires_clarification is True')

        return self