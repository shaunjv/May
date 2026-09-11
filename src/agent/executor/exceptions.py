"""
Custom exceptions for the Executor component.
"""


class ExecutorError(Exception):
    """Base exception for executor-related errors."""
    pass


class ExecutionCancelledError(ExecutorError):
    """Raised when execution is cancelled."""
    pass


class PermissionDeniedError(ExecutorError):
    """Raised when permission is denied for an action."""
    pass


class MaxRetriesExceededError(ExecutorError):
    """Raised when maximum retry attempts are exceeded."""
    pass


class DependencyError(ExecutorError):
    """Raised when step dependencies are not met."""
    pass


class InvalidPlanError(ExecutorError):
    """Raised when the plan is invalid or malformed."""
    pass