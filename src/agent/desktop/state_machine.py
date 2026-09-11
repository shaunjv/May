"""Task lifecycle transition rules, intentionally independent of Qt."""

from .models import TaskRecord, TaskState, utcnow


class InvalidTaskTransition(ValueError):
    pass


class TaskStateMachine:
    _allowed = {
        TaskState.PLANNING: {TaskState.AWAITING_APPROVAL, TaskState.FAILED, TaskState.CANCELLED},
        TaskState.AWAITING_APPROVAL: {TaskState.EXECUTING, TaskState.REJECTED, TaskState.CANCELLED},
        TaskState.EXECUTING: {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED},
    }
    terminal = {TaskState.COMPLETED, TaskState.FAILED, TaskState.REJECTED, TaskState.CANCELLED}

    @classmethod
    def transition(cls, task: TaskRecord, target: TaskState, *, reason: str | None = None) -> bool:
        """Apply a legal transition. Repeating an already-terminal action is a no-op."""
        if task.state == target and target in cls.terminal:
            return False
        if target not in cls._allowed.get(task.state, set()):
            raise InvalidTaskTransition(f"Cannot transition {task.state} to {target}.")
        task.state = target
        now = utcnow()
        if target == TaskState.AWAITING_APPROVAL:
            task.planning_completed_at = now
            task.approval_requested_at = now
        elif target == TaskState.EXECUTING:
            task.approved_at = now
            task.execution_started_at = now
        elif target in cls.terminal:
            task.completed_at = now
            if target == TaskState.CANCELLED:
                task.cancellation_reason = reason or "Cancelled by user"
            elif target == TaskState.FAILED:
                task.error = reason or "Task failed"
        return True
