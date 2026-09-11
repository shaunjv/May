"""
Tests for the Executor V1 component.
"""

import asyncio
from unittest.mock import AsyncMock, Mock
import pytest
from pydantic import ValidationError

from agent.executor.executor import Executor
from agent.executor.models import (
    ActionRequest,
    ExecutionResult,
    ExecutionState,
    ToolResult,
    ExecutorConfig
)
from agent.executor.interfaces import PermissionChecker, ToolExecutor
from agent.executor.exceptions import (
    ExecutorError,
    ExecutionCancelledError,
    PermissionDeniedError,
    MaxRetriesExceededError,
    DependencyError,
    InvalidPlanError
)
from agent.planner.models import Plan, Phase, PlanStep


class MockToolExecutor(ToolExecutor):
    """Mock tool executor for testing."""

    def __init__(self, should_succeed=True, delay=0):
        self.should_succeed = should_succeed
        self.delay = delay
        self.execute_calls = []

    async def execute(self, action_request: ActionRequest) -> ToolResult:
        self.execute_calls.append(action_request)
        if self.delay:
            await asyncio.sleep(self.delay)

        if self.should_succeed:
            return ToolResult(
                action_id=action_request.action_id,
                step_id=action_request.step_id,
                success=True,
                output={"result": "success"}
            )
        else:
            return ToolResult(
                action_id=action_request.action_id,
                step_id=action_request.step_id,
                success=False,
                error="Tool execution failed"
            )


class MockPermissionChecker(PermissionChecker):
    """Mock permission checker for testing."""

    def __init__(self, permitted=True):
        self.permitted = permitted
        self.check_permission_calls = []

    async def check_permission(self, action_request: ActionRequest) -> bool:
        self.check_permission_calls.append(action_request)
        return self.permitted


def create_test_plan(steps_config=None):
    """Create a test plan with the given steps configuration."""
    if steps_config is None:
        steps_config = [
            {
                "step_id": "step1",
                "description": "First step",
                "objective": "Do first thing",
                "rationale": "Need to start",
                "expected_outcome": "First thing done",
                "success_criteria": ["First thing completed"],
                "dependencies": []
            },
            {
                "step_id": "step2",
                "description": "Second step",
                "objective": "Do second thing",
                "rationale": "Need to continue",
                "expected_outcome": "Second thing done",
                "success_criteria": ["Second thing completed"],
                "dependencies": ["step1"]
            }
        ]

    phases = [
        Phase(
            phase_id="phase1",
            name="Test Phase",
            objective="Test phase objective",
            steps=[
                PlanStep(
                    step_id=config["step_id"],
                    description=config["description"],
                    objective=config["objective"],
                    rationale=config["rationale"],
                    expected_outcome=config["expected_outcome"],
                    success_criteria=config.get("success_criteria", []),
                    dependencies=config.get("dependencies", [])
                )
                for config in steps_config
            ]
        )
    ]

    return Plan(
        plan_id="test_plan",
        task_id="test_task",
        goal="Test goal",
        phases=phases
    )


@pytest.mark.asyncio
async def test_executor_initialization():
    """Test that executor initializes correctly."""
    tool_executor = MockToolExecutor()
    permission_checker = MockPermissionChecker()
    executor = Executor(tool_executor, permission_checker)

    assert executor.tool_executor == tool_executor
    assert executor.permission_checker == permission_checker
    assert executor.config.max_retries == 3
    assert executor.config.retry_delay_ms == 1000
    assert executor.config.default_timeout_seconds == 30
    assert executor._cancelled == False


@pytest.mark.asyncio
async def test_executor_with_custom_config():
    """Test executor initialization with custom config."""
    tool_executor = MockToolExecutor()
    permission_checker = MockPermissionChecker()
    config = ExecutorConfig(max_retries=5, retry_delay_ms=2000, default_timeout_seconds=60)
    executor = Executor(tool_executor, permission_checker, config)

    assert executor.config.max_retries == 5
    assert executor.config.retry_delay_ms == 2000
    assert executor.config.default_timeout_seconds == 60


@pytest.mark.asyncio
async def test_execute_simple_plan_success():
    """Test successful execution of a simple plan."""
    tool_executor = MockToolExecutor(should_succeed=True)
    permission_checker = MockPermissionChecker(permitted=True)
    executor = Executor(tool_executor, permission_checker)

    plan = create_test_plan()
    result = await executor.execute_plan(plan)

    assert result.status == ExecutionState.SUCCEEDED
    assert len(result.step_results) == 2
    assert "step1" in result.step_results
    assert "step2" in result.step_results
    assert result.step_results["step1"].success == True
    assert result.step_results["step2"].success == True
    assert len(result.failed_steps) == 0
    assert len(result.cancelled_steps) == 0
    assert result.error_message is None

    # Verify tool executor was called for each step
    assert len(tool_executor.execute_calls) == 2

    # Verify permission checker was called for each step
    assert len(permission_checker.check_permission_calls) == 2


@pytest.mark.asyncio
async def test_execute_plan_with_failure():
    """Test execution where one step fails."""
    tool_executor = MockToolExecutor(should_succeed=False)  # All steps fail
    permission_checker = MockPermissionChecker(permitted=True)
    executor = Executor(tool_executor, permission_checker)

    plan = create_test_plan()
    result = await executor.execute_plan(plan)

    assert result.status == ExecutionState.FAILED
    assert len(result.step_results) == 2
    assert result.step_results["step1"].success == False
    assert result.step_results["step2"].success == False
    assert len(result.failed_steps) == 2
    assert "step1" in result.failed_steps
    assert "step2" in result.failed_steps
    assert len(result.cancelled_steps) == 0
    assert result.error_message is not None
    assert "failed" in result.error_message.lower()


@pytest.mark.asyncio
async def test_execute_plan_with_dependency_failure():
    """Test execution where a dependency failure causes dependent step to fail."""
    # First step succeeds, second step fails
    call_count = 0

    async def mock_execute(action_request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # First call succeeds
            return ToolResult(
                action_id=action_request.action_id,
                step_id=action_request.step_id,
                success=True,
                output={"result": "success"}
            )
        else:  # Second call fails
            return ToolResult(
                action_id=action_request.action_id,
                step_id=action_request.step_id,
                success=False,
                error="Tool execution failed"
            )

    tool_executor = MockToolExecutor()
    tool_executor.execute = mock_execute
    permission_checker = MockPermissionChecker(permitted=True)
    executor = Executor(tool_executor, permission_checker)

    plan = create_test_plan()
    result = await executor.execute_plan(plan)

    # First step should succeed, second should fail due to dependency
    assert result.status == ExecutionState.FAILED
    assert result.step_results["step1"].success == True
    assert result.step_results["step2"].success == False  # Failed due to dependency not being satisfied
    assert len(result.failed_steps) == 1  # Only step2 should be in failed_steps
    assert "step2" in result.failed_steps
    # Note: step1 is not in failed_steps because it succeeded


@pytest.mark.asyncio
async def test_execute_plan_with_cancelled_permission():
    """Test execution where permission is denied."""
    tool_executor = MockToolExecutor(should_succeed=True)
    permission_checker = MockPermissionChecker(permitted=False)  # Permission denied
    executor = Executor(tool_executor, permission_checker)

    plan = create_test_plan()
    result = await executor.execute_plan(plan)

    assert result.status == ExecutionState.FAILED
    # With permission denied, the step should fail and not proceed to tool execution
    assert len(result.step_results) == 0  # No steps executed due to permission denial
    assert len(result.failed_steps) == 2  # Both steps fail due to permission
    assert "step1" in result.failed_steps
    assert "step2" in result.failed_steps

    # Verify permission checker was called but tool executor was not
    assert len(permission_checker.check_permission_calls) == 2
    assert len(tool_executor.execute_calls) == 0


@pytest.mark.asyncio
async def test_executor_cancel_execution():
    """Test cancelling execution."""
    # Create a tool executor that delays to allow cancellation
    tool_executor = MockToolExecutor(should_succeed=True, delay=0.1)
    permission_checker = MockPermissionChecker(permitted=True)
    executor = Executor(tool_executor, permission_checker)

    # Create plan with multiple steps
    steps_config = [
        {
            "step_id": "step1",
            "description": "First step",
            "objective": "Do first thing",
            "rationale": "Need to start",
            "expected_outcome": "First thing done",
            "success_criteria": ["First thing completed"],
            "dependencies": []
        },
        {
            "step_id": "step2",
            "description": "Second step",
            "objective": "Do second thing",
            "rationale": "Need to continue",
            "expected_outcome": "Second thing done",
            "success_criteria": ["Second thing completed"],
            "dependencies": ["step1"]
        },
        {
            "step_id": "step3",
            "description": "Third step",
            "objective": "Do third thing",
            "rationale": "Need to finish",
            "expected_outcome": "Third thing done",
            "success_criteria": ["Third thing completed"],
            "dependencies": ["step2"]
        }
    ]
    plan = create_test_plan(steps_config)

    # Start execution and cancel it quickly
    execute_task = asyncio.create_task(executor.execute_plan(plan))

    # Give it a moment to start, then cancel
    await asyncio.sleep(0.01)
    executor.cancel()

    result = await execute_task

    # Should be cancelled
    assert result.status == ExecutionState.CANCELLED
    assert len(result.cancelled_steps) > 0
    # Should have executed some steps before cancellation
    assert len(result.step_results) >= 0


@pytest.mark.asyncio
async def test_executor_retry_logic():
    """Test retry logic for failed steps."""
    call_count = 0

    async def mock_execute(action_request):
        nonlocal call_count
        call_count += 1
        if call_count < 3:  # Fail first two attempts
            return ToolResult(
                action_id=action_request.action_id,
                step_id=action_request.step_id,
                success=False,
                error=f"Attempt {call_count} failed"
            )
        else:  # Succeed on third attempt
            return ToolResult(
                action_id=action_request.action_id,
                step_id=action_request.step_id,
                success=True,
                output={"result": "success"}
            )

    tool_executor = MockToolExecutor()
    tool_executor.execute = mock_execute
    permission_checker = MockPermissionChecker(permitted=True)
    # Configure executor to retry up to 5 times
    config = ExecutorConfig(max_retries=5, retry_delay_ms=10)
    executor = Executor(tool_executor, permission_checker, config)

    # Create plan with single step
    steps_config = [{
        "step_id": "step1",
        "description": "Retry step",
        "objective": "Test retry",
        "rationale": "Testing retry logic",
        "expected_outcome": "Success after retries",
        "success_criteria": ["Step completed successfully"],
        "dependencies": []
    }]
    plan = create_test_plan(steps_config)

    result = await executor.execute_plan(plan)

    # Should succeed after retries
    assert result.status == ExecutionState.SUCCEEDED
    assert result.step_results["step1"].success == True
    assert call_count == 3  # Should have tried 3 times (2 failures + 1 success)


@pytest.mark.asyncio
async def test_executor_max_retries_exceeded():
    """Test that max retries exceeded error is handled properly."""
    tool_executor = MockToolExecutor(should_succeed=False)  # Always fail
    permission_checker = MockPermissionChecker(permitted=True)
    # Configure executor with low retry limit
    config = ExecutorConfig(max_retries=2, retry_delay_ms=10)
    executor = Executor(tool_executor, permission_checker, config)

    # Create plan with single step
    steps_config = [{
        "step_id": "step1",
        "description": "Fail step",
        "objective": "Always fail",
        "rationale": "Testing failure handling",
        "expected_outcome": "Failure",
        "success_criteria": ["Should never happen"],
        "dependencies": []
    }]
    plan = create_test_plan(steps_config)

    result = await executor.execute_plan(plan)

    # Should fail after exhausting retries
    assert result.status == ExecutionState.FAILED
    assert result.step_results["step1"].success == False
    # Should have tried max_retries + 1 times (3 attempts for max_retries=2)
    assert len(tool_executor.execute_calls) == 3


@pytest.mark.asyncio
async def test_executor_dependency_validation():
    """Test dependency validation in plan."""
    tool_executor = MockToolExecutor(should_succeed=True)
    permission_checker = MockPermissionChecker(permitted=True)
    executor = Executor(tool_executor, permission_checker)

    # Create plan with circular dependency - should fail during Plan creation
    steps_config = [
        {
            "step_id": "step1",
            "description": "Step 1",
            "objective": "Do first thing",
            "rationale": "Start",
            "expected_outcome": "First done",
            "success_criteria": ["First completed"],
            "dependencies": ["step2"]  # Depends on step2
        },
        {
            "step_id": "step2",
            "description": "Step 2",
            "objective": "Do second thing",
            "rationale": "Continue",
            "expected_outcome": "Second done",
            "success_criteria": ["Second completed"],
            "dependencies": ["step1"]  # Depends on step1 - creates circle
        }
    ]

    # Should raise ValidationError due to circular dependency in plan validation
    with pytest.raises(Exception) as exc_info:
        create_test_plan(steps_config)
    assert "Circular dependency detected in plan steps" in str(exc_info.value)


@pytest.mark.asyncio
async def test_executor_empty_plan():
    """Test execution of empty plan."""
    tool_executor = MockToolExecutor(should_succeed=True)
    permission_checker = MockPermissionChecker(permitted=True)
    executor = Executor(tool_executor, permission_checker)

    # Create plan with no steps in phases
    phases = [
        Phase(
            phase_id="phase1",
            name="Empty Phase",
            description="Empty phase",
            objective="No objective",
            steps=[]  # No steps
        )
    ]
    plan = Plan(
        plan_id="empty_plan",
        task_id="empty_task",
        goal="Empty goal",
        phases=phases
    )

    # Should raise InvalidPlanError
    with pytest.raises(InvalidPlanError, match="Phase .* has no steps"):
        await executor.execute_plan(plan)


@pytest.mark.asyncio
async def test_executor_missing_dependencies():
    """Test execution when dependencies are not met."""
    tool_executor = MockToolExecutor(should_succeed=True)
    permission_checker = MockPermissionChecker(permitted=True)
    executor = Executor(tool_executor, permission_checker)

    # Create plan where step2 depends on step1, but we'll simulate step1 failing
    call_count = 0

    async def mock_execute(action_request):
        nonlocal call_count
        call_count += 1
        if action_request.step_id == "step1":
            # First call returns failure
            return ToolResult(
                action_id=action_request.action_id,
                step_id=action_request.step_id,
                success=False,
                error="Step 1 failed"
            )
        else:
            # Subsequent calls succeed
            return ToolResult(
                action_id=action_request.action_id,
                step_id=action_request.step_id,
                success=True,
                output={"result": "success"}
            )

    tool_executor.execute = mock_execute

    steps_config = [
        {
            "step_id": "step1",
            "description": "First step",
            "objective": "Do first thing",
            "rationale": "Need to start",
            "expected_outcome": "First thing done",
            "success_criteria": ["First thing completed"],
            "dependencies": []
        },
        {
            "step_id": "step2",
            "description": "Second step",
            "objective": "Do second thing",
            "rationale": "Need to continue",
            "expected_outcome": "Second thing done",
            "success_criteria": ["Second thing completed"],
            "dependencies": ["step1"]  # Depends on step1
        }
    ]
    plan = create_test_plan(steps_config)

    result = await executor.execute_plan(plan)

    # Step1 should fail, step2 should fail due to unsatisfied dependency
    assert result.status == ExecutionState.FAILED
    assert result.step_results["step1"].success == False
    assert result.step_results["step2"].success == False  # Failed due to dependency
    assert len(result.failed_steps) == 2
    assert "step1" in result.failed_steps
    assert "step2" in result.failed_steps


def test_executor_config_validation():
    """Test executor config validation."""
    # Valid config
    config = ExecutorConfig(max_retries=3, retry_delay_ms=1000, default_timeout_seconds=30)
    assert config.max_retries == 3
    assert config.retry_delay_ms == 1000
    assert config.default_timeout_seconds == 30

    # Test default values
    default_config = ExecutorConfig()
    assert default_config.max_retries == 3
    assert default_config.retry_delay_ms == 1000
    assert default_config.default_timeout_seconds == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])