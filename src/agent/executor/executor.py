"""
Executor V1 - Sequential plan execution engine.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from .models import (
    ActionRequest,
    ToolResult,
    ExecutionState,
    ExecutorConfig,
    ExecutionResult
)
from .interfaces import ToolExecutor, PermissionChecker
from .exceptions import (
    ExecutorError,
    ExecutionCancelledError,
    PermissionDeniedError,
    MaxRetriesExceededError,
    DependencyError,
    InvalidPlanError
)
from agent.planner.models import Plan, PlanStep, Phase
if TYPE_CHECKING:
    from agent.desktop.models import ApprovedExecutionPlan


class Executor:
    """
    Sequential plan execution engine.

    Responsibilities:
    - Orchestrate execution of plan steps in sequence
    - Respect step dependencies
    - Handle retries with configurable limits
    - Interface with tools through ToolExecutor abstraction
    - Check permissions through PermissionChecker abstraction
    - Track execution state and results
    - Support cancellation
    """

    def __init__(
        self,
        tool_executor: ToolExecutor,
        permission_checker: PermissionChecker,
        config: Optional[ExecutorConfig] = None
    ):
        """
        Initialize the executor.

        Args:
            tool_executor: Implementation for executing tools/actions
            permission_checker: Implementation for checking permissions
            config: Optional configuration (defaults used if not provided)
        """
        self.tool_executor = tool_executor
        self.permission_checker = permission_checker
        self.config = config or ExecutorConfig()
        self._cancelled = False

    async def execute_plan(self, plan: Plan) -> ExecutionResult:
        """
        Execute a plan sequentially.

        Args:
            plan: The plan to execute

        Returns:
            ExecutionResult: Result of the execution

        Raises:
            InvalidPlanError: If the plan is invalid
        """
        # Validate plan
        self._validate_plan(plan)

        # Initialize execution tracking
        execution_id = f"exec_{int(time.time()*1000)}"
        start_time = time.time()
        step_results: Dict[str, ToolResult] = {}
        failed_steps: List[str] = []
        cancelled_steps: List[str] = []
        self._cancelled = False

        # Get all steps in dependency order
        ordered_steps = self._get_steps_in_dependency_order(plan)

        # Execute each step in order
        for step in ordered_steps:
            if self._cancelled:
                cancelled_steps.append(step.step_id)
                step.status = ExecutionState.CANCELLED
                continue

            # Create action request for permission checking
            action_request = self._create_action_request(step)

            # Check permissions
            try:
                permitted = await self._check_permissions(action_request)
            except Exception:
                # If permission checker fails, deny by default for security
                permitted = False

            if not permitted:
                # Permission denied - step fails without tool execution
                failed_steps.append(step.step_id)
                step.status = ExecutionState.FAILED
                # Do not add to step_results as no tool was executed
                continue

            # Check if dependencies are satisfied
            if not self._are_dependencies_satisfied(step, step_results):
                # Dependencies not satisfied - mark step as failed
                failed_steps.append(step.step_id)
                step.status = ExecutionState.FAILED
                # Add a failed ToolResult for dependency failure
                step_results[step.step_id] = ToolResult(
                    action_id=f"action_{step.step_id}",
                    step_id=step.step_id,
                    success=False,
                    error="Dependencies not satisfied"
                )
                continue

            # Execute the step
            try:
                result = await self._execute_step_with_retry(step)
                step_results[step.step_id] = result
                step.status = ExecutionState.SUCCEEDED if result.success else ExecutionState.FAILED

                if not result.success:
                    failed_steps.append(step.step_id)

            except ExecutionCancelledError:
                cancelled_steps.append(step.step_id)
                step.status = ExecutionState.CANCELLED
                break
            except PermissionDeniedError:
                # Permission denied - step fails without tool execution
                failed_steps.append(step.step_id)
                step.status = ExecutionState.FAILED
                # Do not add to step_results as no tool was executed
                continue
            except Exception as e:
                # Handle unexpected errors
                failed_steps.append(step.step_id)
                step.status = ExecutionState.FAILED
                step_results[step.step_id] = ToolResult(
                    action_id=f"action_{step.step_id}",
                    step_id=step.step_id,
                    success=False,
                    error=str(e)
                )

        # Determine overall execution status
        if self._cancelled:
            final_status = ExecutionState.CANCELLED
        elif failed_steps:
            final_status = ExecutionState.FAILED
        else:
            final_status = ExecutionState.SUCCEEDED

        # Create execution result
        execution_time = time.time() - start_time
        return ExecutionResult(
            execution_id=execution_id,
            plan_id=plan.plan_id,
            task_id=plan.task_id,
            status=final_status,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time + execution_time)),
            step_results=step_results,
            failed_steps=failed_steps,
            cancelled_steps=cancelled_steps,
            error_message=(
                f"Execution failed with {len(failed_steps)} failed steps"
                if failed_steps and not self._cancelled
                else None
            )
        )

    async def execute_approved_plan(self, approved: "ApprovedExecutionPlan", plan: Plan, cancellation_event=None) -> ExecutionResult:
        """Execute only an immutable, hash-verified manifest approved by the user."""
        if not approved.verify(plan):
            raise InvalidPlanError("Approved plan identity or action manifest no longer matches the current plan.")
        start_time = time.time()
        results: Dict[str, ToolResult] = {}
        failed: List[str] = []
        cancelled: List[str] = []
        for action in approved.actions:
            if (cancellation_event is not None and cancellation_event.is_set()) or self._cancelled:
                cancelled.append(action.step_id)
                continue
            try:
                if not await self._check_permissions(action):
                    failed.append(action.step_id)
                    break
                result = await self.tool_executor.execute(action)
                results[action.step_id] = result
                if not result.success:
                    failed.append(action.step_id)
                    break
            except Exception as error:
                failed.append(action.step_id)
                results[action.step_id] = ToolResult(action_id=action.action_id, step_id=action.step_id, success=False, error=str(error))
                break
        state = ExecutionState.CANCELLED if cancelled else (ExecutionState.FAILED if failed else ExecutionState.SUCCEEDED)
        return ExecutionResult(
            execution_id=f"exec_{int(time.time()*1000)}", plan_id=plan.plan_id, task_id=plan.task_id,
            status=state, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), step_results=results,
            failed_steps=failed, cancelled_steps=cancelled,
            error_message="Execution cancelled" if cancelled else ("Execution failed" if failed else None),
        )

    def cancel(self) -> None:
        """Cancel the current execution."""
        self._cancelled = True

    def _validate_plan(self, plan: Plan) -> None:
        """Validate that the plan is suitable for execution."""
        if not plan.plan_id:
            raise InvalidPlanError("Plan ID is required")
        if not plan.task_id:
            raise InvalidPlanError("Task ID is required")
        if not plan.goal:
            raise InvalidPlanError("Goal is required")
        if not plan.phases:
            raise InvalidPlanError("Plan must have at least one phase")

        # Check that all phases have steps
        for phase in plan.phases:
            if not phase.steps:
                raise InvalidPlanError(f"Phase {phase.phase_id} has no steps")

    def _get_steps_in_dependency_order(self, plan: Plan) -> List[PlanStep]:
        """
        Get all steps in topological order based on dependencies.

        Returns:
            List of PlanStep objects in executable order
        """
        # Collect all steps
        all_steps: List[PlanStep] = []
        for phase in plan.phases:
            all_steps.extend(phase.steps)

        # Build dependency graph
        dependencies: Dict[str, List[str]] = {}
        dependents: Dict[str, List[str]] = {}

        for step in all_steps:
            dependencies[step.step_id] = step.dependencies.copy()
            dependents[step.step_id] = []

        # Build reverse dependency map
        for step in all_steps:
            for dep_id in step.dependencies:
                if dep_id not in dependents:
                    dependents[dep_id] = []
                dependents[dep_id].append(step.step_id)

        # Find steps with no dependencies (ready to execute)
        ready_steps: List[PlanStep] = [
            step for step in all_steps if not step.dependencies
        ]

        # Topological sort (Kahn's algorithm)
        ordered_steps: List[PlanStep] = []
        remaining_deps: Dict[str, int] = {
            step.step_id: len(step.dependencies) for step in all_steps
        }

        while ready_steps:
            # Take the first ready step
            current_step = ready_steps.pop(0)
            ordered_steps.append(current_step)

            # Decrease dependency count for dependents
            for dependent_id in dependents.get(current_step.step_id, []):
                remaining_deps[dependent_id] -= 1
                if remaining_deps[dependent_id] == 0:
                    # Find the step object for this dependent
                    dependent_step = next(
                        s for s in all_steps if s.step_id == dependent_id
                    )
                    ready_steps.append(dependent_step)

        # Check for circular dependencies
        if len(ordered_steps) != len(all_steps):
            raise InvalidPlanError("Circular dependencies detected in plan")

        return ordered_steps

    def _are_dependencies_satisfied(
        self,
        step: PlanStep,
        completed_results: Dict[str, ToolResult]
    ) -> bool:
        """
        Check if all dependencies for a step have been satisfied.

        Args:
            step: The step to check
            completed_results: Results from completed steps

        Returns:
            bool: True if all dependencies succeeded
        """
        for dep_id in step.dependencies:
            if dep_id not in completed_results:
                return False
            if not completed_results[dep_id].success:
                return False
        return True

    async def _execute_step_with_retry(self, step: PlanStep) -> ToolResult:
        """
        Execute a step with retry logic.

        Args:
            step: The step to execute

        Returns:
            ToolResult: Result of the execution

        Raises:
            ExecutionCancelledError: If execution is cancelled
            MaxRetriesExceededError: If max retries exceeded
        """
        last_error: Optional[Exception] = None
        last_result: Optional[ToolResult] = None

        for attempt in range(self.config.max_retries + 1):
            # Check for cancellation before each attempt
            if self._cancelled:
                raise ExecutionCancelledError("Execution cancelled")

            try:
                # Create action request from step
                action_request = self._create_action_request(step)

                # Execute the action
                result = await self.tool_executor.execute(action_request)

                # If successful, return immediately
                if result.success:
                    return result

                # If not successful, save result and retry if we have attempts left
                last_result = result
                if attempt < self.config.max_retries:
                    # Wait before retry (if not cancelled)
                    if not self._cancelled:
                        await asyncio.sleep(self.config.retry_delay_ms / 1000.0)
                else:
                    # Max retries exceeded - return the last result
                    return last_result

            except ExecutionCancelledError:
                # Don't retry if cancelled
                raise
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    # Wait before retry (if not cancelled)
                    if not self._cancelled:
                        await asyncio.sleep(self.config.retry_delay_ms / 1000.0)
                else:
                    # Max retries exceeded
                    raise MaxRetriesExceededError(
                        f"Step {step.step_id} failed after {self.config.max_retries + 1} attempts"
                    ) from e

        # This should not be reached, but just in case
        if last_error:
            raise last_error
        elif last_result:
            return last_result
        else:
            raise ExecutorError(f"Step {step.step_id} failed unexpectedly")

    def _create_action_request(self, step: PlanStep) -> ActionRequest:
        """
        Create an ActionRequest from a PlanStep.

        Args:
            step: The plan step to convert

        Returns:
            ActionRequest: The action request
        """
        # For V1, we'll use the step's objective as the tool input
        # In a real implementation, this would be more sophisticated
        tool_input = {
            "step_description": step.description,
            "step_objective": step.objective,
            "step_rationale": step.rationale,
            "expected_outcome": step.expected_outcome,
            "success_criteria": step.success_criteria
        }

        return ActionRequest(
            action_id=f"action_{step.step_id}",
            step_id=step.step_id,
            tool_name="abstract_tool",  # Placeholder - real tools would be injected
            tool_input=tool_input,
            timeout_seconds=self.config.default_timeout_seconds
        )

    async def _check_permissions(self, action_request: ActionRequest) -> bool:
        """
        Check if an action is permitted.

        Args:
            action_request: The action to check

        Returns:
            bool: True if permitted
        """
        try:
            return await self.permission_checker.check_permission(action_request)
        except Exception:
            # If permission checker fails, deny by default for security
            return False
