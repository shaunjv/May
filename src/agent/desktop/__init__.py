"""Desktop Agent application domain and presentation boundary."""

from .models import TaskState, TaskRecord, ApprovedExecutionPlan
from .state_machine import InvalidTaskTransition, TaskStateMachine

__all__ = ["TaskState", "TaskRecord", "ApprovedExecutionPlan", "InvalidTaskTransition", "TaskStateMachine"]
